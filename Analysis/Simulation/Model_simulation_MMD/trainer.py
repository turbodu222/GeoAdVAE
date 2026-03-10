"""
Trainer class for model training with configurable loss scheduling.
This code is based on https://github.com/NVlabs/MUNIT.

FIXED VERSION: Added numerical stability to MMD loss computation
"""
from networks import Discriminator
from networks import VAEGen_MORE_LAYERS as VAEGen
from utils import weights_init, get_model_list, get_scheduler
from utils import load_prior_correlation_matrix, get_cluster_assignments
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import ot 
import numpy as np  
import sys  
import hashlib
import copy
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(nn.Module):
    def __init__(self, hyperparameters, dataset=None):
        super(Trainer, self).__init__()
        self.dataset = dataset
        lr = hyperparameters['lr']
        
        # Parse loss scheduling configuration from hyperparameters
        self.loss_schedule = self._parse_loss_schedule(hyperparameters)
        print(f"Loss schedule configuration: {self.loss_schedule}")
        
        # Initiate the networks
        shared_layer = False
        if "shared_layer" in hyperparameters and hyperparameters["shared_layer"]:
            shared_layer = {}
            shared_layer["enc"] = nn.Linear(hyperparameters['gen']['dim'], hyperparameters['gen']['latent'])
            shared_layer["dec"] = nn.Linear(hyperparameters['gen']['latent'], hyperparameters['gen']['dim'])

        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'],
                            shared_layer).to(device)
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'],
                            shared_layer).to(device)
        self.dis_latent = Discriminator(hyperparameters['gen']['latent'],
                                        hyperparameters['dis']).to(device)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_latent.parameters())
        
        if shared_layer:
            all_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
            seen = set()
            gen_params = []
            for param in all_params:
                if id(param) not in seen:
                    seen.add(id(param))
                    gen_params.append(param)
        else:
            gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_latent.apply(weights_init('gaussian'))
        
        # Training phases configuration based on hyperparameters
        self.training_phase = 1
        self.phase_durations = self._parse_phase_durations(hyperparameters)
        self.current_iteration = 0
        
        print(f"Training phase durations: {self.phase_durations}")
        print(f"Total training phases: {len(self.phase_durations)}")

        # Loss history for logging - MODIFIED: replaced gw_loss with mmd_loss
        self.loss_history = {
            'kl_loss': [],
            'mmd_loss': [],  # CHANGED from gw_loss
            'recon_loss': [],
            'recon_loss_a': [],  
            'recon_loss_b': [],  
            'gan_loss': [],
            'prior_loss': [], 
            'total_loss': [],
            'iterations': []
        }
        self.weighted_loss_history = {
            'kl_loss': [],
            'mmd_loss': [],  # CHANGED from gw_loss
            'recon_loss': [],
            'recon_loss_a': [],  
            'recon_loss_b': [],  
            'gan_loss': [],
            'prior_loss': [], 
            'total_loss': [],
            'iterations': []
        }

        # MMD loss configuration - NEW
        self.mmd_bandwidth = hyperparameters.get('mmd_bandwidth', 1.0)
        print(f"MMD bandwidth: {self.mmd_bandwidth}")

        # Prior loss setup with improvements
        if dataset is not None and dataset.get_prior_matrix() is not None:
            self.prior_correlation_matrix = dataset.get_prior_matrix(device=device)
            self.use_prior = True
            self.lambda_p = hyperparameters.get('lambda_p', 1.0)
            self.epsilon = hyperparameters.get('epsilon', 1e-6)
            
            # New prior loss parameters
            self.prior_temperature = hyperparameters.get('prior_temperature', 0.1)
            self.prior_loss_warmup = hyperparameters.get('prior_loss_warmup', 100)
            
            # Use dataset's cluster labels
            self.gex_cluster_labels = dataset.gex_cluster_labels.numpy()
            self.morpho_cluster_labels = dataset.morpho_cluster_labels.numpy()
            
            print(f"Prior loss enabled with λ_p = {self.lambda_p}")
            print(f"Prior temperature = {self.prior_temperature}")
            print(f"Prior matrix shape: {self.prior_correlation_matrix.shape}")
            print(f"Prior loss will be activated in phase: {self.loss_schedule.get('prior_loss', 'N/A')}")
        else:
            self.use_prior = False
            self.gex_cluster_labels = None
            self.morpho_cluster_labels = None
            print("Prior loss disabled - no dataset provided or no prior matrix available")

        self.unfreeze_discriminator()
        self.unfreeze_generators()

        # Debugging for weights and gradients in GAN training
        self.debug_mode = hyperparameters.get('debug_gan_training', True)
        self.weight_snapshots = {}
        self.freeze_verification_interval = 50  # Check every 50 iterations
        
        # Initialize weight tracking
        if self.debug_mode:
            self._initialize_weight_tracking()

    def _parse_loss_schedule(self, hyperparameters):
        """
        Parse loss scheduling configuration from hyperparameters.
        
        Expected format in YAML:
        loss_schedule:
          kl_loss: 1          # Start from phase 1
          recon_loss: 1       # Start from phase 1  
          gan_loss: 2         # Start from phase 2
          mmd_loss: 3         # Start from phase 3 (CHANGED from gw_loss)
          prior_loss: 1       # Start from phase 1
        """
        default_schedule = {
            'kl_loss': 1,
            'recon_loss': 1,
            'gan_loss': 2,
            'mmd_loss': 3,  # CHANGED from gw_loss
            'prior_loss': 3  # Default to phase 3 for backward compatibility
        }
        
        if 'loss_schedule' in hyperparameters:
            schedule = hyperparameters['loss_schedule']
            # Update defaults with user configuration
            default_schedule.update(schedule)
        
        return default_schedule
    
    def _parse_phase_durations(self, hyperparameters):
        """
        Parse training phase durations from hyperparameters.
        
        Expected format in YAML:
        phase_durations:
          phase_1: 200        # Phase 1 lasts 200 iterations
          phase_2: 400        # Phase 2 lasts 400 iterations  
          phase_3: 400        # Phase 3 lasts 400 iterations (optional)
        """
        default_durations = {
            1: 200,    # Extended reconstruction phase
            2: 400,    # Extended adversarial phase
            3: 400     # Final phase with all losses
        }
        
        if 'phase_durations' in hyperparameters:
            duration_config = hyperparameters['phase_durations']
            for phase_name, duration in duration_config.items():
                # Parse phase number from "phase_X" format
                if isinstance(phase_name, str) and phase_name.startswith('phase_'):
                    phase_num = int(phase_name.split('_')[1])
                    default_durations[phase_num] = duration
                elif isinstance(phase_name, int):
                    default_durations[phase_name] = duration
        
        return default_durations
    
    def _should_use_loss(self, loss_name):
        """
        Check if a specific loss should be used in the current training phase.
        
        Args:
            loss_name (str): Name of the loss ('kl_loss', 'gan_loss', 'mmd_loss', etc.)
            
        Returns:
            bool: True if loss should be used in current phase
        """
        required_phase = self.loss_schedule.get(loss_name, 1)
        return self.training_phase >= required_phase

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def super_criterion(self, input, target):
        """
        Supervised reconstruction loss, can be L1 or L2
        We don't use this in the current implementation, because we don't have labels.
        """
        return torch.mean(torch.abs(input - target))

    # ========================================================================
    # NEW: MMD Loss Implementation with Numerical Stability
    # ========================================================================
    
    def _laplacian_kernel(self, x, y, bandwidth=1.0):
        """
        Compute Laplacian kernel k(x, y) = exp(-||x - y|| / bandwidth)
        with numerical stability improvements
        
        Args:
            x: Tensor of shape (n, d)
            y: Tensor of shape (m, d)
            bandwidth: Kernel bandwidth (default: 1.0)
            
        Returns:
            Kernel matrix of shape (n, m)
        """
        # Compute pairwise L2 distances
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (n, 1)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (m, 1)
        
        # Compute distances: (n, 1) + (1, m) - 2 * (n, m) = (n, m)
        distances_squared = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t())
        
        # Numerical stability: ensure non-negative before sqrt
        distances_squared = torch.clamp(distances_squared, min=0.0)
        distances = torch.sqrt(distances_squared + 1e-8)
        
        # Apply Laplacian kernel
        kernel_values = torch.exp(-distances / bandwidth)
        
        return kernel_values
    
    def compute_mmd_loss(self, z_a, z_b, bandwidth=None):
        """
        Compute Maximum Mean Discrepancy (MMD) loss using Laplacian kernel.
        FIXED: Added numerical stability checks and latent clipping
        
        MMD²(X,Y) = 1/(m(m-1)) ∑∑(i≠j) k(xi,xj) - 2/(m·m) ∑∑ k(xi,yj) + 1/(m(m-1)) ∑∑(i≠j) k(yi,yj)
        
        Since higher MMD indicates more correlation, we use -MMD as loss to minimize.
        
        Args:
            z_a: Latent representations from modality A (batch_size, latent_dim)
            z_b: Latent representations from modality B (batch_size, latent_dim)
            bandwidth: Kernel bandwidth (default: use self.mmd_bandwidth)
            
        Returns:
            MMD loss (scalar tensor)
        """
        if bandwidth is None:
            bandwidth = self.mmd_bandwidth
        
        # NUMERICAL STABILITY FIX 1: Clip latent values to prevent overflow
        max_latent_value = 10.0  # Reasonable range for latent space
        z_a = torch.clamp(z_a, min=-max_latent_value, max=max_latent_value)
        z_b = torch.clamp(z_b, min=-max_latent_value, max=max_latent_value)
        
        # NUMERICAL STABILITY FIX 2: Check for NaN/Inf in input
        if torch.isnan(z_a).any() or torch.isinf(z_a).any():
            print(f"WARNING: NaN or Inf detected in z_a at iteration {self.current_iteration}")
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)
        
        if torch.isnan(z_b).any() or torch.isinf(z_b).any():
            print(f"WARNING: NaN or Inf detected in z_b at iteration {self.current_iteration}")
            return torch.tensor(0.0, device=z_b.device, requires_grad=True)
        
        m = z_a.size(0)  # batch size
        
        # NUMERICAL STABILITY FIX 3: Ensure batch size > 1
        if m <= 1:
            print(f"WARNING: Batch size {m} too small for MMD computation")
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)
        
        # Compute kernel matrices
        k_aa = self._laplacian_kernel(z_a, z_a, bandwidth)  # (m, m)
        k_bb = self._laplacian_kernel(z_b, z_b, bandwidth)  # (m, m)
        k_ab = self._laplacian_kernel(z_a, z_b, bandwidth)  # (m, m)
        
        # NUMERICAL STABILITY FIX 4: Check kernel matrices for NaN/Inf
        if torch.isnan(k_aa).any() or torch.isinf(k_aa).any():
            print(f"WARNING: NaN or Inf in k_aa at iteration {self.current_iteration}")
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)
        
        if torch.isnan(k_bb).any() or torch.isinf(k_bb).any():
            print(f"WARNING: NaN or Inf in k_bb at iteration {self.current_iteration}")
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)
        
        if torch.isnan(k_ab).any() or torch.isinf(k_ab).any():
            print(f"WARNING: NaN or Inf in k_ab at iteration {self.current_iteration}")
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)
        
        # Term A: 1/(m(m-1)) ∑∑(i≠j) k(xi,xj)
        # Remove diagonal (i==j) by subtracting diagonal sum
        k_aa_sum = k_aa.sum()
        k_aa_diag = k_aa.diagonal().sum()
        term_a = (k_aa_sum - k_aa_diag) / (m * (m - 1) + 1e-8)  # Add epsilon to denominator
        
        # Term B: -2/(m·m) ∑∑ k(xi,yj)
        # Include all pairs
        term_b = -2.0 * k_ab.sum() / (m * m + 1e-8)
        
        # Term C: 1/(m(m-1)) ∑∑(i≠j) k(yi,yj)
        # Remove diagonal (i==j)
        k_bb_sum = k_bb.sum()
        k_bb_diag = k_bb.diagonal().sum()
        term_c = (k_bb_sum - k_bb_diag) / (m * (m - 1) + 1e-8)
        
        # MMD² = A + B + C
        mmd_squared = term_a + term_b + term_c
        
        # NUMERICAL STABILITY FIX 5: Check final MMD value
        if torch.isnan(mmd_squared) or torch.isinf(mmd_squared):
            print(f"WARNING: NaN or Inf in MMD at iteration {self.current_iteration}")
            print(f"  term_a: {term_a.item()}, term_b: {term_b.item()}, term_c: {term_c.item()}")
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)
        
        # Return MMD as loss (minimizing MMD = maximizing correlation)
        mmd_loss = mmd_squared
        
        # Debug output
        if self.current_iteration % 100 == 0:
            print(f"\nMMD Loss Debug (iter {self.current_iteration}):")
            print(f"  Latent A range: [{z_a.min().item():.4f}, {z_a.max().item():.4f}]")
            print(f"  Latent B range: [{z_b.min().item():.4f}, {z_b.max().item():.4f}]")
            print(f"  Term A (k_aa): {term_a.item():.6f}")
            print(f"  Term B (k_ab): {term_b.item():.6f}")
            print(f"  Term C (k_bb): {term_c.item():.6f}")
            print(f"  MMD² value: {mmd_squared.item():.6f}")
            print(f"  MMD loss: {mmd_loss.item():.6f}")
        
        return mmd_loss

    # ========================================================================
    # Prior Loss (Unchanged)
    # ========================================================================

    def compute_improved_prior_loss(self, z_a, z_b, batch_indices_a=None, batch_indices_b=None):
        """
        Simplified contrastive prior loss for identity matrix
        Pulls same-cluster pairs together, pushes different-cluster pairs apart
        """
        if not self.use_prior:
            return torch.tensor(0.0, device=z_a.device)
        
        # Get cluster assignments
        if batch_indices_a is not None and batch_indices_b is not None:
            cluster_a = self.morpho_cluster_labels[batch_indices_a]
            cluster_b = self.gex_cluster_labels[batch_indices_b]
        else:
            batch_size_a = z_a.shape[0]
            batch_size_b = z_b.shape[0]
            cluster_a = self.morpho_cluster_labels[:batch_size_a]
            cluster_b = self.gex_cluster_labels[:batch_size_b]
        
        # L2 normalize
        z_a_norm = F.normalize(z_a, p=2, dim=1)
        z_b_norm = F.normalize(z_b, p=2, dim=1)
        
        # Compute similarity matrix (cosine similarity)
        similarity_matrix = torch.mm(z_a_norm, z_b_norm.t()) / self.prior_temperature
        
        # Create mask for same-cluster pairs (positive pairs)
        cluster_a = torch.from_numpy(cluster_a).long().to(z_a.device)
        cluster_b = torch.from_numpy(cluster_b).long().to(z_a.device)
        
        # Broadcasting: cluster_a[:, None] vs cluster_b[None, :]
        positive_mask = (cluster_a[:, None] == cluster_b[None, :]).float()
        
        # InfoNCE-style contrastive loss
        # For each sample in z_a, we want high similarity with same-cluster z_b
        exp_sim = torch.exp(similarity_matrix)
        
        # Positive pairs: same cluster
        positive_sim = torch.sum(exp_sim * positive_mask, dim=1)
        
        # All pairs (including positives and negatives)
        all_sim = torch.sum(exp_sim, dim=1)
        
        # Loss: -log(positive / all)
        loss = -torch.log(positive_sim / (all_sim + 1e-8) + 1e-8).mean()
        
        # Warmup (optional, but helps stability)
        if self.current_iteration < self.prior_loss_warmup:
            warmup_factor = 0.5 + 0.5 * (self.current_iteration / self.prior_loss_warmup)
            loss = loss * warmup_factor
        
        # Debug output
        if self.current_iteration % 100 == 0:
            n_positive = positive_mask.sum().item()
            n_total = positive_mask.numel()
            print(f"Prior loss (iter {self.current_iteration}):")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Positive pairs: {n_positive}/{n_total}")
            print(f"  Avg similarity (same cluster): {(similarity_matrix * positive_mask).sum() / (n_positive + 1e-8):.4f}")
            print(f"  Avg similarity (diff cluster): {(similarity_matrix * (1 - positive_mask)).sum() / (n_total - n_positive + 1e-8):.4f}")
        
        return loss

    def compute_prior_loss(self, z_a, z_b, batch_indices_a=None, batch_indices_b=None):
        """
        Compute prior loss using Wasserstein transport with debugging
        """
        if not self.use_prior:
            return torch.tensor(0.0, device=z_a.device)
        
        # Convert to numpy for POT library
        z_a_np = z_a.detach().cpu().numpy()
        z_b_np = z_b.detach().cpu().numpy()
        
        # Get precomputed cluster assignments for current batch
        if batch_indices_a is not None and batch_indices_b is not None:
            cluster_a_np = self.morpho_cluster_labels[batch_indices_a]
            cluster_b_np = self.gex_cluster_labels[batch_indices_b]
        else:
            batch_size_a = z_a_np.shape[0]
            batch_size_b = z_b_np.shape[0]
            cluster_a_np = self.morpho_cluster_labels[:batch_size_a]
            cluster_b_np = self.gex_cluster_labels[:batch_size_b]
        
        # Force conversion to int32 to prevent dtype issues
        cluster_a_np = np.asarray(cluster_a_np, dtype=np.int32)
        cluster_b_np = np.asarray(cluster_b_np, dtype=np.int32)
        
        prior_matrix_np = self.prior_correlation_matrix.detach().cpu().numpy().astype(np.float32)
        
        # DEBUG: Print cluster information
        if self.current_iteration % 100 == 0:
            print(f"DEBUG - Iteration {self.current_iteration}:")
            print(f"  Cluster A dtype: {cluster_a_np.dtype}, range: {cluster_a_np.min()}-{cluster_a_np.max()}")
            print(f"  Cluster B dtype: {cluster_b_np.dtype}, range: {cluster_b_np.min()}-{cluster_b_np.max()}")
            print(f"  Prior matrix shape: {prior_matrix_np.shape}")
            print(f"  Prior matrix range: {prior_matrix_np.min():.6f}-{prior_matrix_np.max():.6f}")
            print(f"  Batch indices A: {batch_indices_a[:5] if batch_indices_a is not None else 'None'}")
            print(f"  Batch indices B: {batch_indices_b[:5] if batch_indices_b is not None else 'None'}")
        
        # Compute cost matrix
        n_a, n_b = len(cluster_a_np), len(cluster_b_np)
        phi_matrix = np.zeros((n_a, n_b))
        
        valid_pairs = 0
        invalid_pairs = 0
        cost_sum = 0.0
        
        try:
            for i in range(n_a):
                for j in range(n_b):
                    cluster_i = int(cluster_a_np[i])  # morphology cluster
                    cluster_j = int(cluster_b_np[j])  # gene expression cluster
                    
                    # Check bounds
                    if (cluster_j < prior_matrix_np.shape[0] and 
                        cluster_i < prior_matrix_np.shape[1] and
                        cluster_j >= 0 and cluster_i >= 0):
                        prob = float(prior_matrix_np[cluster_j, cluster_i])  # Explicit float conversion
                        phi_matrix[i, j] = -np.log(prob + float(self.epsilon))
                        cost_sum += phi_matrix[i, j]
                        valid_pairs += 1
                    else:
                        phi_matrix[i, j] = -np.log(float(self.epsilon))
                        invalid_pairs += 1
        except Exception as e:
            print(f"Error in cost matrix computation: {e}")
            print(f"  i={i if 'i' in locals() else 'unknown'}, j={j if 'j' in locals() else 'unknown'}")
            print(f"  cluster_i={cluster_i if 'cluster_i' in locals() else 'unknown'}")
            print(f"  cluster_j={cluster_j if 'cluster_j' in locals() else 'unknown'}")
            print(f"  prob type: {type(prob) if 'prob' in locals() else 'unknown'}")
            print(f"  prob value: {prob if 'prob' in locals() else 'unknown'}")
            return torch.tensor(0.0, device=z_a.device)
        
        # DEBUG: Print cost matrix statistics
        if self.current_iteration % 100 == 0:
            print(f"  Valid pairs: {valid_pairs}, Invalid pairs: {invalid_pairs}")
            print(f"  Cost matrix range: {phi_matrix.min():.6f}-{phi_matrix.max():.6f}")
            print(f"  Average cost: {cost_sum/max(valid_pairs, 1):.6f}")
        
        # Compute optimal transport
        p = ot.unif(n_a)
        q = ot.unif(n_b)
        
        try:
            # Ensure phi_matrix is proper numpy float array
            phi_matrix = np.asarray(phi_matrix, dtype=np.float64)
            
            if self.current_iteration % 100 == 0:
                print(f"  Phi matrix dtype: {phi_matrix.dtype}")
                print(f"  Phi matrix shape: {phi_matrix.shape}")
                print(f"  p dtype: {p.dtype}, q dtype: {q.dtype}")
            
            transport_matrix = ot.emd(p, q, phi_matrix)
            prior_loss_value = np.sum(transport_matrix * phi_matrix)
            
            # DEBUG: Print final loss
            if self.current_iteration % 100 == 0:
                print(f"  Transport sum: {transport_matrix.sum():.6f}")
                print(f"  Prior loss value: {prior_loss_value:.6f}")
            
            return torch.tensor(float(prior_loss_value), device=z_a.device, requires_grad=True)
            
        except Exception as e:
            print(f"Warning: Optimal transport computation failed: {e}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  phi_matrix dtype: {phi_matrix.dtype if 'phi_matrix' in locals() else 'unknown'}")
            print(f"  phi_matrix shape: {phi_matrix.shape if 'phi_matrix' in locals() else 'unknown'}")
            return torch.tensor(0.0, device=z_a.device)

    # ========================================================================
    # Training Phase Management
    # ========================================================================

    def update_training_phase(self, iteration):
        """Update training phase based on iteration with configurable phase durations"""
        cumulative_duration = 0
        current_phase = 1
        
        for phase, duration in sorted(self.phase_durations.items()):
            cumulative_duration += duration
            if iteration < cumulative_duration:
                current_phase = phase
                break
            current_phase = max(current_phase, phase)  # Use the highest phase if beyond all durations
        
        self.training_phase = current_phase
        self.current_iteration = iteration

    # ========================================================================
    # Generator Update - MODIFIED: replaced GW loss with MMD loss
    # ========================================================================

    def gen_update(self, x_a, x_b, hyperparameters, variational=True, 
                   batch_indices_a=None, batch_indices_b=None):
        """
        Enhanced generator update with configurable loss scheduling
        MODIFIED: Replaced GW loss with MMD loss
        """
        # Debug: Take discriminator weight snapshot before training
        if self.debug_mode and self.current_iteration % 100 == 0:
            disc_before, _ = self._snapshot_weights(self.dis_latent, "discriminator")
        
        # Step 1: Unfreeze generators, freeze discriminator
        self.unfreeze_generators()
        self.freeze_discriminator()
        
        self.gen_opt.zero_grad()
        
        # Step 2: Encode to get distribution parameters
        mean_a, logvar_a = self.gen_a.encode(x_a)
        mean_b, logvar_b = self.gen_b.encode(x_b)

        # Step 3: Sample from latent distribution
        if variational:
            z_a = self.gen_a.reparameterize(mean_a, logvar_a)
            z_b = self.gen_b.reparameterize(mean_b, logvar_b)
            
            # KL divergence loss - normalized by latent dimension to prevent excessive scaling
            kl_a = -0.5 * torch.sum(1 + logvar_a - mean_a.pow(2) - logvar_a.exp(), dim=1)
            kl_b = -0.5 * torch.sum(1 + logvar_b - mean_b.pow(2) - logvar_b.exp(), dim=1)
            kl_loss = (torch.mean(kl_a) + torch.mean(kl_b)) / hyperparameters['gen']['latent']
        else:
            z_a, z_b = mean_a, mean_b
            kl_loss = torch.tensor(0.0, device=mean_a.device)

        # Step 4: Reconstruction loss (separate for debug)
        x_a_recon = self.gen_a.decode(z_a)
        x_b_recon = self.gen_b.decode(z_b)
        recon_loss_a = self.recon_criterion(x_a_recon, x_a)
        recon_loss_b = self.recon_criterion(x_b_recon, x_b)
        recon_loss = recon_loss_a + recon_loss_b

        # Step 5: Initialize losses
        gan_loss = torch.tensor(0.0, device=z_a.device)
        mmd_loss = torch.tensor(0.0, device=z_a.device)  # CHANGED from gw_loss
        prior_loss = torch.tensor(0.0, device=z_a.device)

        # Step 6: Initialize total loss with always-active losses
        total_loss = torch.tensor(0.0, device=z_a.device)
        
        # Add KL loss if enabled in current phase
        if self._should_use_loss('kl_loss') and variational:
            total_loss += hyperparameters.get('kl_w', 1.0) * kl_loss

        # Add reconstruction loss if enabled in current phase
        if self._should_use_loss('recon_loss'):
            total_loss += hyperparameters['recon_x_w'] * recon_loss

        # Step 7: Add adversarial loss if enabled in current phase
        if self._should_use_loss('gan_loss'):
            loss_latent_a = self.dis_latent.calc_gen_loss(z_a)
            loss_latent_b = self.dis_latent.calc_gen_loss_reverse(z_b)
            gan_loss = loss_latent_a + loss_latent_b
            total_loss += hyperparameters['gan_w'] * gan_loss

        # Step 8: Add MMD loss if enabled in current phase - MODIFIED
        if self._should_use_loss('mmd_loss'):
            try:
                mmd_loss = self.compute_mmd_loss(z_a, z_b)
                total_loss += hyperparameters.get('mmd_w', 1.0) * mmd_loss
            except Exception as e:
                print(f"MMD loss computation failed: {e}")
                mmd_loss = torch.tensor(0.0, device=z_a.device)

        # Step 9: Add prior loss if enabled in current phase
        if self._should_use_loss('prior_loss') and self.use_prior:
            try:
                prior_loss = self.compute_improved_prior_loss(z_a, z_b, batch_indices_a, batch_indices_b)
                total_loss += hyperparameters.get('lambda_p', 1.0) * prior_loss
            except Exception as e:
                print(f"Prior loss computation failed: {e}")
                prior_loss = torch.tensor(0.0, device=z_a.device)

        # Store losses for logging (separate reconstruction losses) - MODIFIED
        self.raw_losses = {
            'kl_loss': kl_loss.item(),
            'mmd_loss': mmd_loss.item(),  # CHANGED from gw_loss
            'recon_loss': recon_loss.item(),
            'recon_loss_a': recon_loss_a.item(),
            'recon_loss_b': recon_loss_b.item(),
            'gan_loss': gan_loss.item(),
            'prior_loss': prior_loss.item(),
            'total_loss': total_loss.item()
        }

        self.weighted_losses = {
            'kl_loss': kl_loss.item() * hyperparameters.get('kl_w', 1.0) if self._should_use_loss('kl_loss') else 0.0,
            'mmd_loss': mmd_loss.item() * hyperparameters.get('mmd_w', 1.0) if self._should_use_loss('mmd_loss') else 0.0,  # CHANGED
            'recon_loss': recon_loss.item() * hyperparameters['recon_x_w'] if self._should_use_loss('recon_loss') else 0.0,
            'recon_loss_a': recon_loss_a.item() * hyperparameters['recon_x_w'] if self._should_use_loss('recon_loss') else 0.0,
            'recon_loss_b': recon_loss_b.item() * hyperparameters['recon_x_w'] if self._should_use_loss('recon_loss') else 0.0,
            'gan_loss': gan_loss.item() * hyperparameters.get('gan_w', 1.0) if self._should_use_loss('gan_loss') else 0.0,
            'prior_loss': prior_loss.item() * hyperparameters.get('lambda_p', 1.0) if self._should_use_loss('prior_loss') else 0.0,
            'total_loss': total_loss.item()
        }

        # Step 10: Backward and optimize
        total_loss.backward()
        
        # GRADIENT CLIPPING to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.gen_a.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(self.gen_b.parameters(), max_norm=5.0)
        
        self.gen_opt.step()

        # Debug: Verify discriminator weights didn't change
        if self.debug_mode and self.current_iteration % 100 == 0:
            self._verify_weights_unchanged(self.dis_latent, "Discriminator", disc_before, "generator update")
        
        # Run comprehensive freeze verification
        #self._debug_freeze_verification(self.current_iteration)

        # Store scalar attributes for TensorBoard - MODIFIED
        self.loss_scalar_kl = kl_loss.item()
        self.loss_scalar_mmd = mmd_loss.item()  # CHANGED from loss_scalar_gw
        self.loss_scalar_recon = recon_loss.item()
        self.loss_scalar_gan = gan_loss.item()
        self.loss_scalar_prior = prior_loss.item()
        self.loss_scalar_total = total_loss.item()

        # Detailed loss debugging information - MODIFIED
        if self.current_iteration % 50 == 0:
            print(f"\n=== Loss Breakdown at Iter {self.current_iteration} Phase {self.training_phase} ===")
            active_losses = []
            for loss_name in ['kl_loss', 'recon_loss', 'gan_loss', 'mmd_loss', 'prior_loss']:  # CHANGED
                if self._should_use_loss(loss_name):
                    active_losses.append(loss_name)
            print(f"Active losses in phase {self.training_phase}: {active_losses}")
            
            if self._should_use_loss('kl_loss'):
                print(f"Raw KL Loss: {kl_loss.item():.6f}")
                print(f"Weighted KL: {kl_loss.item() * hyperparameters.get('kl_w', 1.0):.6f} (weight: {hyperparameters.get('kl_w', 1.0)})")
            if self._should_use_loss('recon_loss'):
                print(f"Raw Recon: {recon_loss.item():.6f}")
                print(f"Weighted Recon: {recon_loss.item() * hyperparameters['recon_x_w']:.6f} (weight: {hyperparameters['recon_x_w']})")
            if self._should_use_loss('gan_loss'):
                print(f"Raw GAN Loss: {gan_loss.item():.6f}")
                print(f"Weighted GAN: {gan_loss.item() * hyperparameters.get('gan_w', 1.0):.6f} (weight: {hyperparameters.get('gan_w', 1.0)})")
            if self._should_use_loss('mmd_loss'):  # CHANGED from gw_loss
                print(f"Raw MMD Loss: {mmd_loss.item():.6f}")
                print(f"Weighted MMD: {mmd_loss.item() * hyperparameters.get('mmd_w', 1.0):.6f} (weight: {hyperparameters.get('mmd_w', 1.0)})")
            if self._should_use_loss('prior_loss'):
                print(f"Raw Prior Loss: {prior_loss.item():.6f}")
                print(f"Weighted Prior: {prior_loss.item() * hyperparameters.get('lambda_p', 1.0):.6f} (weight: {hyperparameters.get('lambda_p', 1.0)})")
            print(f"Total Loss: {total_loss.item():.6f}")
            print("=" * 60)

    # ========================================================================
    # Discriminator Update (Unchanged)
    # ========================================================================

    def dis_update(self, x_a, x_b, hyperparameters):
        """
        Enhanced discriminator update with phase awareness
        """
        # Only update discriminator if GAN loss is active
        if not self._should_use_loss('gan_loss'):
            return
        
        # Debug: Take weight snapshots before training
        if self.debug_mode and self.current_iteration % 100 == 0:
            gen_a_before, _ = self._snapshot_weights(self.gen_a, "generator_a")
            gen_b_before, _ = self._snapshot_weights(self.gen_b, "generator_b")
        
        # Step 1: Freeze generators, unfreeze discriminator
        self.freeze_generators()
        self.unfreeze_discriminator()
        
        self.dis_opt.zero_grad()
        
        # Step 2: Generate latent variables without gradients for generators
        with torch.no_grad():
            mean_a, logvar_a = self.gen_a.encode(x_a)
            mean_b, logvar_b = self.gen_b.encode(x_b)
            z_a = self.gen_a.reparameterize(mean_a, logvar_a)
            z_b = self.gen_b.reparameterize(mean_b, logvar_b)
        
        # Step 3: Discriminator loss
        dis_loss = self.dis_latent.calc_dis_loss(z_a, z_b)
        total_dis_loss = hyperparameters['gan_w'] * dis_loss
        
        # Step 4: Backward and optimize
        total_dis_loss.backward()
        self.dis_opt.step()
        
        # Store discriminator loss
        self.loss_dis_total = total_dis_loss.item()
        
        # Debug: Verify generator weights didn't change
        if self.debug_mode and self.current_iteration % 100 == 0:
            self._verify_weights_unchanged(self.gen_a, "Generator A", gen_a_before, "discriminator update")
            self._verify_weights_unchanged(self.gen_b, "Generator B", gen_b_before, "discriminator update")
        
        # Debug discriminator outputs periodically
        if self.current_iteration % 200 == 0:
            with torch.no_grad():
                disc_out_a = self.dis_latent(z_a)
                disc_out_b = self.dis_latent(z_b)
                print(f"Discriminator outputs - A: {disc_out_a.mean().item():.3f}±{disc_out_a.std().item():.3f}, "
                      f"B: {disc_out_b.mean().item():.3f}±{disc_out_b.std().item():.3f}")

    # ========================================================================
    # Utility Functions (Unchanged)
    # ========================================================================

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_latent.load_state_dict(state_dict['latent'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'latent': self.dis_latent.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def record_losses(self, iteration):
        """Record losses for logging including prior loss. MODIFIED: replaced gw_loss with mmd_loss"""
        # Make sure raw_losses and weighted_losses exist
        if not hasattr(self, 'raw_losses'):
            self.raw_losses = {
                'kl_loss': 0.0,
                'mmd_loss': 0.0,  # CHANGED from gw_loss
                'recon_loss': 0.0,
                'recon_loss_a': 0.0,  
                'recon_loss_b': 0.0,  
                'gan_loss': 0.0,
                'prior_loss': 0.0,     
                'total_loss': 0.0
            }

        if not hasattr(self, 'weighted_losses'):
            self.weighted_losses = {
                'kl_loss': 0.0,
                'mmd_loss': 0.0,  # CHANGED from gw_loss
                'recon_loss': 0.0,
                'recon_loss_a': 0.0,  
                'recon_loss_b': 0.0,  
                'gan_loss': 0.0,
                'prior_loss': 0.0,     
                'total_loss': 0.0
            }
        
        # Log raw losses
        for key, value in self.raw_losses.items():
            self.loss_history[key].append(value)
        self.loss_history['iterations'].append(iteration)
        
        # Log weighted losses  
        for key, value in self.weighted_losses.items():
            self.weighted_loss_history[key].append(value)
        self.weighted_loss_history['iterations'].append(iteration)

    def freeze_discriminator(self):
        """Freeze discriminator parameters"""
        for param in self.dis_latent.parameters():
            param.requires_grad = False

    def unfreeze_discriminator(self):
        """Unfreeze discriminator parameters"""
        for param in self.dis_latent.parameters():
            param.requires_grad = True

    def freeze_generators(self):
        """Freeze generator parameters"""
        for param in self.gen_a.parameters():
            param.requires_grad = False
        for param in self.gen_b.parameters():
            param.requires_grad = False

    def unfreeze_generators(self):
        """Unfreeze generator parameters"""
        for param in self.gen_a.parameters():
            param.requires_grad = True
        for param in self.gen_b.parameters():
            param.requires_grad = True

    # ========================================================================
    # Debug Functions (Unchanged)
    # ========================================================================

    def _initialize_weight_tracking(self):
        """Initialize weight tracking for debug purposes"""
        print("Initializing GAN training debug system...")
        self.weight_snapshots = {
            'discriminator': {},
            'generator_a': {},
            'generator_b': {}
        }

    def _get_param_hash(self, param_dict):
        """Calculate hash of parameter dictionary for change detection"""
        param_str = ""
        for name, param in param_dict.items():
            if param.requires_grad:
                param_str += f"{name}:{param.data.cpu().numpy().tobytes().hex()}"
        return hashlib.md5(param_str.encode()).hexdigest()

    def _snapshot_weights(self, model, model_name):
        """Take a snapshot of model weights"""
        snapshot = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                snapshot[name] = param.data.clone().detach()
        
        param_hash = self._get_param_hash(snapshot)
        return snapshot, param_hash

    def _verify_weights_unchanged(self, model, model_name, expected_snapshot, operation_name):
        """Verify that weights haven't changed during an operation"""
        current_snapshot, current_hash = self._snapshot_weights(model, model_name)
        expected_hash = self._get_param_hash(expected_snapshot)
        
        if current_hash != expected_hash:
            print(f"❌ ERROR: {model_name} weights changed during {operation_name}!")
            
            # Find which parameters changed
            changed_params = []
            for name in expected_snapshot.keys():
                if not torch.equal(expected_snapshot[name], current_snapshot[name]):
                    old_norm = expected_snapshot[name].norm().item()
                    new_norm = current_snapshot[name].norm().item()
                    change = abs(new_norm - old_norm)
                    changed_params.append(f"{name}: Δ={change:.6f}")
            
            print(f"  Changed parameters: {changed_params[:5]}")  # Show first 5
            return False
        else:
            print(f"✓ VERIFIED: {model_name} weights unchanged during {operation_name}")
            return True

    def _debug_freeze_verification(self, iteration):
        """Comprehensive freeze mechanism verification"""
        if not self.debug_mode or iteration % self.freeze_verification_interval != 0:
            return
        
        print(f"\n=== GAN Freeze Debug Verification at iteration {iteration} ===")
        
        # Test discriminator training with frozen generators
        print("Testing discriminator training (generators should be frozen)...")
        
        # Take snapshots before discriminator training
        gen_a_snapshot, _ = self._snapshot_weights(self.gen_a, "generator_a")
        gen_b_snapshot, _ = self._snapshot_weights(self.gen_b, "generator_b")
        disc_snapshot_before, _ = self._snapshot_weights(self.dis_latent, "discriminator")
        
        # Create dummy data for testing
        dummy_a = torch.randn(8, self.gen_a.input_dim).to(device)
        dummy_b = torch.randn(8, self.gen_b.input_dim).to(device)
        
        # Perform one discriminator update
        self.freeze_generators()
        self.unfreeze_discriminator()
        
        # Check requires_grad states
        print("Checking requires_grad states during discriminator training:")
        gen_a_frozen = all(not p.requires_grad for p in self.gen_a.parameters())
        gen_b_frozen = all(not p.requires_grad for p in self.gen_b.parameters())
        disc_unfrozen = any(p.requires_grad for p in self.dis_latent.parameters())
        
        print(f"  Generator A frozen: {gen_a_frozen}")
        print(f"  Generator B frozen: {gen_b_frozen}")
        print(f"  Discriminator unfrozen: {disc_unfrozen}")
        
        if not (gen_a_frozen and gen_b_frozen and disc_unfrozen):
            print("❌ ERROR: Freeze states are incorrect!")
            return
        
        # Perform discriminator training step
        self.dis_opt.zero_grad()
        with torch.no_grad():
            mean_a, logvar_a = self.gen_a.encode(dummy_a)
            mean_b, logvar_b = self.gen_b.encode(dummy_b)
            z_a = self.gen_a.reparameterize(mean_a, logvar_a)
            z_b = self.gen_b.reparameterize(mean_b, logvar_b)
        
        dis_loss = self.dis_latent.calc_dis_loss(z_a, z_b)
        dis_loss.backward()
        self.dis_opt.step()
        
        # Verify generators didn't change
        self._verify_weights_unchanged(self.gen_a, "Generator A", gen_a_snapshot, "discriminator training")
        self._verify_weights_unchanged(self.gen_b, "Generator B", gen_b_snapshot, "discriminator training")
        
        # Verify discriminator did change
        disc_snapshot_after, _ = self._snapshot_weights(self.dis_latent, "discriminator")
        disc_changed = self._get_param_hash(disc_snapshot_before) != self._get_param_hash(disc_snapshot_after)
        if disc_changed:
            print("✓ VERIFIED: Discriminator weights updated during discriminator training")
        else:
            print("❌ ERROR: Discriminator weights did not change during training!")
        
        # Test generator training with frozen discriminator
        print("\nTesting generator training (discriminator should be frozen)...")
        
        # Take snapshots before generator training
        gen_a_snapshot_before, _ = self._snapshot_weights(self.gen_a, "generator_a")
        gen_b_snapshot_before, _ = self._snapshot_weights(self.gen_b, "generator_b")
        disc_snapshot_before_gen, _ = self._snapshot_weights(self.dis_latent, "discriminator")
        
        # Switch freeze states
        self.unfreeze_generators()
        self.freeze_discriminator()
        
        # Check requires_grad states
        print("Checking requires_grad states during generator training:")
        gen_a_unfrozen = any(p.requires_grad for p in self.gen_a.parameters())
        gen_b_unfrozen = any(p.requires_grad for p in self.gen_b.parameters())
        disc_frozen = all(not p.requires_grad for p in self.dis_latent.parameters())
        
        print(f"  Generator A unfrozen: {gen_a_unfrozen}")
        print(f"  Generator B unfrozen: {gen_b_unfrozen}")
        print(f"  Discriminator frozen: {disc_frozen}")
        
        if not (gen_a_unfrozen and gen_b_unfrozen and disc_frozen):
            print("❌ ERROR: Freeze states are incorrect!")
            return
        
        # Perform generator training step
        self.gen_opt.zero_grad()
        
        mean_a, logvar_a = self.gen_a.encode(dummy_a)
        mean_b, logvar_b = self.gen_b.encode(dummy_b)
        z_a = self.gen_a.reparameterize(mean_a, logvar_a)
        z_b = self.gen_b.reparameterize(mean_b, logvar_b)
        
        # Reconstruction loss
        x_a_recon = self.gen_a.decode(z_a)
        x_b_recon = self.gen_b.decode(z_b)
        recon_loss = self.recon_criterion(x_a_recon, dummy_a) + self.recon_criterion(x_b_recon, dummy_b)
        
        # GAN loss (discriminator should not receive gradients)
        gan_loss_a = self.dis_latent.calc_gen_loss(z_a)
        gan_loss_b = self.dis_latent.calc_gen_loss_reverse(z_b)
        gan_loss = gan_loss_a + gan_loss_b
        
        total_loss = recon_loss + 0.1 * gan_loss
        total_loss.backward()
        self.gen_opt.step()
        
        # Verify discriminator didn't change
        self._verify_weights_unchanged(self.dis_latent, "Discriminator", disc_snapshot_before_gen, "generator training")
        
        # Verify generators did change
        gen_a_snapshot_after, _ = self._snapshot_weights(self.gen_a, "generator_a")
        gen_b_snapshot_after, _ = self._snapshot_weights(self.gen_b, "generator_b")
        
        gen_a_changed = self._get_param_hash(gen_a_snapshot_before) != self._get_param_hash(gen_a_snapshot_after)
        gen_b_changed = self._get_param_hash(gen_b_snapshot_before) != self._get_param_hash(gen_b_snapshot_after)
        
        if gen_a_changed:
            print("✓ VERIFIED: Generator A weights updated during generator training")
        else:
            print("❌ ERROR: Generator A weights did not change during training!")
            
        if gen_b_changed:
            print("✓ VERIFIED: Generator B weights updated during generator training")
        else:
            print("❌ ERROR: Generator B weights did not change during training!")
        
        print("=== GAN Freeze Debug Verification Complete ===\n")