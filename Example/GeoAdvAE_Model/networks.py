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
        self.net = self._make_net()

    def _make_net(self):
        """
        input_dim -> dim -> dim -> 1
        """
        layers = [
            nn.Linear(self.input_dim, self.dim),     # latent_dim -> hidden_dim
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim, self.dim),           # hidden_dim -> hidden_dim  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim, 1)                   # hidden_dim -> 1
        ]
        
        if self.use_sigmoid:
            layers.append(nn.Sigmoid())
            
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def calc_dis_loss(self, input_fake, input_real):
        outs0 = [self.forward(input_fake)]
        outs1 = [self.forward(input_real)]
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                # LSGAN: MSE loss
                real_target = 1.0 - self.label_smoothing 
                fake_target = 0.0 + self.label_smoothing
                loss += torch.mean((out0 - fake_target) ** 2) + torch.mean((out1 - real_target) ** 2)
                
            elif self.gan_type == 'nsgan':
                # Standard GAN: BCE loss
                real_target = torch.ones_like(out1) * (1.0 - self.label_smoothing)
                fake_target = torch.zeros_like(out0) + self.label_smoothing
                loss += F.binary_cross_entropy(out0, fake_target) + F.binary_cross_entropy(out1, real_target)
                
            elif self.gan_type == 'wgan':
                # Wasserstein GAN: Wasserstein distance
                loss += torch.mean(out0) - torch.mean(out1)
                
            else:
                raise ValueError(f"Unsupported GAN type: {self.gan_type}")
                
        return loss

    def calc_gen_loss(self, input_fake):
        outs0 = [self.forward(input_fake)]
        loss = 0
        
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                target = 1.0 - self.label_smoothing
                loss += torch.mean((out0 - target) ** 2)
                
            elif self.gan_type == 'nsgan':
                target = torch.ones_like(out0) * (1.0 - self.label_smoothing)
                loss += F.binary_cross_entropy(out0, target)
                
            elif self.gan_type == 'wgan':
                loss += -torch.mean(out0)
                
        return loss

    def calc_gen_loss_reverse(self, input_real):
        outs0 = [self.forward(input_real)]
        loss = 0
        
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                target = 0.0 + self.label_smoothing
                loss += torch.mean((out0 - target) ** 2)
                
            elif self.gan_type == 'nsgan':
                target = torch.zeros_like(out0) + self.label_smoothing
                loss += F.binary_cross_entropy(out0, target)
                
            elif self.gan_type == 'wgan':
                loss += torch.mean(out0)
                
        return loss
    

##################################################################################
# Generator
##################################################################################

class VAEGen_MORE_LAYERS(nn.Module):
    def __init__(self, input_dim, params, shared_layer=False):
        super(VAEGen_MORE_LAYERS, self).__init__()
        self.dim = params['dim']
        self.latent = params['latent']
        self.input_dim = input_dim

        # Encoder backbone network
        encoder_base = nn.Sequential(
            nn.Linear(self.input_dim, self.dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Separate outputs for mean and log-variance
        if shared_layer:
            self.enc_mean = shared_layer["enc"]
            self.enc_logvar = nn.Linear(self.dim, self.latent)
        else:
            self.enc_mean = nn.Linear(self.dim, self.latent)
            self.enc_logvar = nn.Linear(self.dim, self.latent)
        

        if shared_layer:
            decoder_layers = [
                shared_layer["dec"],  # latent -> dim
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.dim, self.input_dim),  
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else:
            decoder_layers = [
                nn.Linear(self.latent, self.dim),    
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.dim, self.input_dim),  
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        self.enc_base = encoder_base
        self.dec = nn.Sequential(*decoder_layers)

    def encode(self, images):
        h = self.enc_base(images)
        mean = self.enc_mean(h)
        logvar = self.enc_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def decode(self, z):
        return self.dec(z)

    def forward(self, images):
        mean, logvar = self.encode(images)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    def enc(self, images):
        mean, _ = self.encode(images)
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