from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################################################################################
# Discriminator
##################################################################################

class Discriminator(nn.Module):
    def __init__(self, input_dim, params):
        super(Discriminator, self).__init__()
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.input_dim = input_dim
        self.use_sigmoid = params.get('use_sigmoid', True)
        self.label_smoothing = params.get('label_smoothing', 0.0)
        self.dropout_rate = params.get('dropout_rate', 0.3)
        
        # Build the network
        self.net = self._make_net()
        
        # Optional gradient penalty for WGAN-GP
        self.use_gradient_penalty = params.get('use_gradient_penalty', False)
        self.gp_lambda = params.get('gp_lambda', 10.0)

    def _make_net(self):
        """
        Build enhanced discriminator network with better capacity and regularization
        """
        layers = []
        
        # Input layer with expanded capacity
        layers.append(nn.Linear(self.input_dim, self.dim * 2))
        if self.norm == 'batch':
            layers.append(nn.BatchNorm1d(self.dim * 2))
        elif self.norm == 'layer':
            layers.append(nn.LayerNorm(self.dim * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout(self.dropout_rate))
        
        # Second hidden layer
        layers.append(nn.Linear(self.dim * 2, self.dim))
        if self.norm == 'batch':
            layers.append(nn.BatchNorm1d(self.dim))
        elif self.norm == 'layer':
            layers.append(nn.LayerNorm(self.dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout(self.dropout_rate))
        
        # Third hidden layer
        layers.append(nn.Linear(self.dim, self.dim // 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer
        layers.append(nn.Linear(self.dim // 2, 1))
        
        # Apply sigmoid for certain GAN types
        if self.use_sigmoid and self.gan_type in ['nsgan', 'lsgan']:
            layers.append(nn.Sigmoid())
            
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through discriminator"""
        return self.net(x)
    
    def gradient_penalty(self, real_samples, fake_samples):
        """
        Calculate gradient penalty for WGAN-GP
        """
        batch_size = real_samples.size(0)
        device = real_samples.device
        
        # Random interpolation factor
        alpha = torch.rand(batch_size, 1).to(device)
        alpha = alpha.expand_as(real_samples)
        
        # Interpolated samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        disc_interpolated = self.forward(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def calc_dis_loss(self, input_fake, input_real):
        """
        Enhanced discriminator loss calculation with improved stability
        """
        out_fake = self.forward(input_fake)
        out_real = self.forward(input_real)
        
        if self.gan_type == 'lsgan':
            # LSGAN: Least squares loss
            real_target = 1.0 - self.label_smoothing
            fake_target = 0.0 + self.label_smoothing
            
            loss_real = torch.mean((out_real - real_target) ** 2)
            loss_fake = torch.mean((out_fake - fake_target) ** 2)
            loss = (loss_real + loss_fake) * 0.5
            
        elif self.gan_type == 'nsgan':
            # Standard GAN: Binary cross entropy loss
            real_target = torch.ones_like(out_real) * (1.0 - self.label_smoothing)
            fake_target = torch.zeros_like(out_fake) + self.label_smoothing
            
            # Use BCEWithLogitsLoss for numerical stability if sigmoid is not used
            if not self.use_sigmoid:
                loss_real = F.binary_cross_entropy_with_logits(out_real, real_target)
                loss_fake = F.binary_cross_entropy_with_logits(out_fake, fake_target)
            else:
                loss_real = F.binary_cross_entropy(out_real, real_target)
                loss_fake = F.binary_cross_entropy(out_fake, fake_target)
            
            loss = (loss_real + loss_fake) * 0.5
            
        elif self.gan_type == 'wgan':
            # Wasserstein GAN loss
            loss = torch.mean(out_fake) - torch.mean(out_real)
            
            # Add gradient penalty if enabled
            if self.use_gradient_penalty:
                gp = self.gradient_penalty(input_real, input_fake)
                loss += self.gp_lambda * gp
                
        else:
            raise ValueError(f"Unsupported GAN type: {self.gan_type}")
        
        return loss

    def calc_gen_loss(self, input_fake):
        """
        Generator loss for fooling discriminator (modality A -> B direction)
        """
        out_fake = self.forward(input_fake)
        
        if self.gan_type == 'lsgan':
            # LSGAN: Generator wants discriminator to output 1
            target = 1.0 - self.label_smoothing
            loss = torch.mean((out_fake - target) ** 2)
            
        elif self.gan_type == 'nsgan':
            # Standard GAN: Generator wants discriminator to output 1
            target = torch.ones_like(out_fake) * (1.0 - self.label_smoothing)
            
            if not self.use_sigmoid:
                loss = F.binary_cross_entropy_with_logits(out_fake, target)
            else:
                loss = F.binary_cross_entropy(out_fake, target)
                
        elif self.gan_type == 'wgan':
            # WGAN: Generator wants to minimize discriminator output
            loss = -torch.mean(out_fake)
            
        return loss

    def calc_gen_loss_reverse(self, input_real):
        """
        Generator loss for reverse direction (modality B -> A)
        The generator wants to make B samples look like they come from A's distribution
        """
        out_real = self.forward(input_real)
        
        if self.gan_type == 'lsgan':
            # Generator wants discriminator to be confused (output close to 0.5 or 0)
            target = 0.0 + self.label_smoothing
            loss = torch.mean((out_real - target) ** 2)
            
        elif self.gan_type == 'nsgan':
            # Generator wants discriminator to output 0 (think B samples are from A)
            target = torch.zeros_like(out_real) + self.label_smoothing
            
            if not self.use_sigmoid:
                loss = F.binary_cross_entropy_with_logits(out_real, target)
            else:
                loss = F.binary_cross_entropy(out_real, target)
                
        elif self.gan_type == 'wgan':
            # WGAN: Generator wants to maximize discriminator output for B
            loss = torch.mean(out_real)
            
        return loss
    
    def get_discriminator_accuracy(self, input_fake, input_real):
        """
        Calculate discriminator accuracy for monitoring training progress
        """
        with torch.no_grad():
            out_fake = self.forward(input_fake)
            out_real = self.forward(input_real)
            
            if self.gan_type in ['lsgan', 'nsgan']:
                # For LSGAN and NSGAN, use 0.5 as decision boundary
                fake_acc = torch.mean((out_fake < 0.5).float())
                real_acc = torch.mean((out_real > 0.5).float())
            else:
                # For WGAN, check if real > fake
                fake_acc = torch.mean((out_fake < out_real.mean()).float())
                real_acc = torch.mean((out_real > out_fake.mean()).float())
            
            total_acc = (fake_acc + real_acc) / 2.0
            return total_acc.item(), fake_acc.item(), real_acc.item()
    

##################################################################################
# Generator
##################################################################################

class VAEGen_MORE_LAYERS(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params, shared_layer=False):
        super(VAEGen_MORE_LAYERS, self).__init__()
        self.dim = params['dim']  # Usually 256
        self.latent = params['latent']  # Usually 16
        self.input_dim = input_dim
        
        # Calculate intermediate dimensions for gradual compression
        # Always compress, never expand in encoder
        
        if input_dim >= 2000:  # Gene expression data (e.g., 2000 genes)
            self.encoder_dims = [input_dim, 1024, 512, self.dim]
        elif input_dim >= 1000:  # Large data
            self.encoder_dims = [input_dim, 512, 256, self.dim]
        elif input_dim >= 500:  # Medium data (e.g., 645 morphology features)
            self.encoder_dims = [input_dim, max(256, self.dim*2), self.dim]
        else:  # Small data
            # For very small input, use fewer layers
            if input_dim > self.dim * 4:
                self.encoder_dims = [input_dim, self.dim*2, self.dim]
            else:
                self.encoder_dims = [input_dim, self.dim]
        
        # Build encoder with gradual compression
        encoder_layers = []
        for i in range(len(self.encoder_dims) - 1):
            in_dim = self.encoder_dims[i]
            out_dim = self.encoder_dims[i + 1]
            
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1)
            ])
        
        self.enc_base = nn.Sequential(*encoder_layers)
        
        # Latent space projections
        if shared_layer and isinstance(shared_layer, dict) and "enc" in shared_layer:
            self.enc_mean = shared_layer["enc"]
            self.enc_logvar = nn.Linear(self.dim, self.latent)
        else:
            self.enc_mean = nn.Linear(self.dim, self.latent)
            self.enc_logvar = nn.Linear(self.dim, self.latent)
        
        # Build decoder with gradual expansion
        decoder_dims = self.encoder_dims[::-1]  # Reverse the encoder dimensions
        
        decoder_layers = []
        
        # First layer from latent to first hidden
        if shared_layer and isinstance(shared_layer, dict) and "dec" in shared_layer:
            decoder_layers.append(shared_layer["dec"])
        else:
            decoder_layers.append(nn.Linear(self.latent, decoder_dims[0]))
        
        decoder_layers.extend([
            nn.BatchNorm1d(decoder_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        ])
        
        # Remaining decoder layers
        for i in range(len(decoder_dims) - 1):
            in_dim = decoder_dims[i]
            out_dim = decoder_dims[i + 1]
            
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if i < len(decoder_dims) - 2 else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True) if i < len(decoder_dims) - 2 else nn.Identity(),
                nn.Dropout(0.1) if i < len(decoder_dims) - 2 else nn.Identity()
            ])
        
        self.dec = nn.Sequential(*decoder_layers)
        
        # Print architecture info
        print(f"VAE Architecture for input_dim={input_dim}:")
        print(f"  Encoder dims: {' -> '.join(map(str, self.encoder_dims))} -> {self.latent}")
        print(f"  Decoder dims: {self.latent} -> {' -> '.join(map(str, decoder_dims))}")

    def encode(self, x):
        """Encoder: outputs mean and log-variance of the distribution"""
        h = self.enc_base(x)
        mean = self.enc_mean(h)
        logvar = self.enc_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        Reparameterize: z = μ + σ * ε, where ε ~ N(0,1)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def decode(self, z):
        """Decoder: reconstruct data from latent representation"""
        return self.dec(z)

    def forward(self, x):
        """Forward pass: encode, sample, decode"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    def enc(self, x):
        """Compatibility method: returns mean of latent representation"""
        mean, _ = self.encode(x)
        return mean

##################################################################################
# Classifier
##################################################################################

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.net = self._make_net()

        self.cel = nn.CrossEntropyLoss()

    def _make_net(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, 3)
        )

    def forward(self, x):
        return self.net(x)

    def class_loss(self, input, target):
        return self.cel(input, target)
