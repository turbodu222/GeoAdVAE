"""
Main functionality for starting training with configurable loss scheduling.
This code is based on https://github.com/NVlabs/MUNIT.
"""
import torch

from utils import prepare_sub_folder, write_loss, get_config, save_plots, \
    write_knn, save_loss_curves, save_weighted_loss_curves, get_cross_modal_data_loader
import argparse
from torch.autograd import Variable
from trainer import Trainer
import torch.backends.cudnn as cudnn
import ot
import numpy as np

try:
    from itertools import izip as zip
except ImportError:
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']

# Create unified data loader and dataset
train_loader, dataset = get_cross_modal_data_loader(
    batch_size=config['batch_size'], 
    shuffle=True  # Now we can safely shuffle!
)

# Get full data for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_a, data_b = dataset.get_full_data(device=device)

# Pass dataset to trainer
trainer = Trainer(config, dataset=dataset)
trainer.to(device)
super_a, super_b = None, None

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
num_disc = config.get("num_disc", 2)  # Number of discriminator updates per iteration
num_gen = config.get("num_gen", 1)    # Number of generator updates per iteration

# Add flush to ensure output is shown
sys.stdout.flush()

print(f"Starting training with {max_iter} iterations")
print(f"Discriminator updates per iteration: {num_disc}")
print(f"Generator updates per iteration: {num_gen}")
print(f"Batch size: {config['batch_size']}")

# Print loss schedule configuration
print(f"\n=== Loss Schedule Configuration ===")
loss_schedule = trainer.loss_schedule
phase_durations = trainer.phase_durations
for loss_name, start_phase in loss_schedule.items():
    print(f"{loss_name:12}: starts from phase {start_phase}")

print(f"\n=== Phase Duration Configuration ===")
cumulative = 0
for phase, duration in sorted(phase_durations.items()):
    start_iter = cumulative
    end_iter = cumulative + duration
    print(f"Phase {phase}: iterations {start_iter:4d} - {end_iter:4d} (duration: {duration})")
    cumulative += duration
    
sys.stdout.flush()

def get_active_losses_for_phase(phase, loss_schedule):
    """Helper function to get active losses for a given phase"""
    return [loss_name for loss_name, start_phase in loss_schedule.items() if phase >= start_phase]

def print_phase_transition(old_phase, new_phase, iteration, trainer):
    """Print detailed phase transition information"""
    if old_phase != new_phase:
        print(f"\n{'='*80}")
        print(f"PHASE TRANSITION at iteration {iteration}")
        print(f"From Phase {old_phase} → Phase {new_phase}")
        print(f"{'='*80}")
        
        # Show what losses are now active
        active_losses = get_active_losses_for_phase(new_phase, trainer.loss_schedule)
        newly_activated = [loss for loss in active_losses 
                          if trainer.loss_schedule[loss] == new_phase]
        
        print(f"Active losses in Phase {new_phase}: {active_losses}")
        if newly_activated:
            print(f"Newly activated losses: {newly_activated}")
        
        # Show phase duration
        if new_phase in trainer.phase_durations:
            duration = trainer.phase_durations[new_phase]
            end_iter = iteration + duration
            print(f"Phase {new_phase} will last until iteration {end_iter}")
        
        print(f"{'='*80}\n")
        sys.stdout.flush()

while True:
    for it, batch in enumerate(train_loader):
        # Store previous phase for transition detection
        previous_phase = trainer.training_phase
        
        # Update training phase
        trainer.update_training_phase(iterations)
        
        # Print phase transition information if phase changed
        print_phase_transition(previous_phase, trainer.training_phase, iterations, trainer)
        
        # Print phase information periodically with active losses
        if iterations % 100 == 0:
            active_losses = get_active_losses_for_phase(trainer.training_phase, trainer.loss_schedule)
            print(f"=== Iteration {iterations}/{max_iter} - Phase {trainer.training_phase} ===")
            print(f"Active losses: {active_losses}")
            sys.stdout.flush()
        
        # Extract batch data
        images_a = batch['morpho_data'].to(device)
        images_b = batch['gex_data'].to(device)
        indices_a = batch['index'].cpu().numpy()
        indices_b = batch['index'].cpu().numpy()  # Same indices since they're paired
        morpho_clusters = batch['morpho_cluster'].cpu().numpy()
        gex_clusters = batch['gex_cluster'].cpu().numpy()
        
        # Verify cluster label correspondence periodically
        if iterations % 500 == 0:
            print(f"\n=== Cluster Label Verification at iteration {iterations} ===")
            for i in range(min(3, len(indices_a))):
                idx = indices_a[i]
                morpho_cluster = morpho_clusters[i]
                gex_cluster = gex_clusters[i]
                rna_family = batch['rna_family'][i] if batch['rna_family'][i] is not None else "N/A"
                print(f"Batch sample {i} (global index {idx}): "
                      f"Morpho_cluster={morpho_cluster}, GEX_cluster={gex_cluster}, RNA_family={rna_family}")
            sys.stdout.flush()

        # Enhanced discriminator training (only when GAN loss is active)
        if trainer._should_use_loss('gan_loss'):
            # Train discriminator multiple times per generator update for stability
            for disc_iter in range(num_disc):
                trainer.dis_update(images_a, images_b, config)
            
            # Monitor discriminator performance periodically
            if iterations % 200 == 0:
                with torch.no_grad():
                    # Get latent representations
                    mean_a, logvar_a = trainer.gen_a.encode(images_a)
                    mean_b, logvar_b = trainer.gen_b.encode(images_b)
                    z_a = trainer.gen_a.reparameterize(mean_a, logvar_a)
                    z_b = trainer.gen_b.reparameterize(mean_b, logvar_b)
                    
                    # Check discriminator outputs
                    disc_out_a = trainer.dis_latent(z_a.detach())
                    disc_out_b = trainer.dis_latent(z_b.detach())
                    
                    print(f"Discriminator Analysis:")
                    print(f"  Output A (should be ~1): {disc_out_a.mean().item():.3f} ± {disc_out_a.std().item():.3f}")
                    print(f"  Output B (should be ~0): {disc_out_b.mean().item():.3f} ± {disc_out_b.std().item():.3f}")
                    
                    # Check if discriminator has accuracy info
                    if hasattr(trainer.dis_latent, 'get_discriminator_accuracy'):
                        total_acc, fake_acc, real_acc = trainer.dis_latent.get_discriminator_accuracy(z_a.detach(), z_b.detach())
                        print(f"  Discriminator accuracy: {total_acc:.3f} (fake: {fake_acc:.3f}, real: {real_acc:.3f})")
                    
                    sys.stdout.flush()
        
        # Generator updates (always) - pass indices for prior loss
        for gen_iter in range(num_gen):
            trainer.gen_update(images_a, images_b, config, variational=True, 
                             batch_indices_a=indices_a, batch_indices_b=indices_b)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Update learning rate
        trainer.update_learning_rate()
        trainer.record_losses(iterations)

        # Enhanced logging with detailed monitoring
        if (iterations + 1) % config['log_iter'] == 0:
            print(f"\n=== Detailed Logging at iteration {iterations + 1}/{max_iter} ===")
            
            # Print current phase and learning rates
            current_gen_lr = trainer.gen_opt.param_groups[0]['lr']
            current_dis_lr = trainer.dis_opt.param_groups[0]['lr']
            active_losses = get_active_losses_for_phase(trainer.training_phase, trainer.loss_schedule)
            
            print(f"Training Phase: {trainer.training_phase}")
            print(f"Active losses: {active_losses}")
            print(f"Learning Rates - Generator: {current_gen_lr:.6f}, Discriminator: {current_dis_lr:.6f}")
            
            # Print detailed loss breakdown with phase awareness
            if hasattr(trainer, 'raw_losses'):
                losses = trainer.raw_losses
                print(f"Raw Losses:")
                for loss_name, loss_value in losses.items():
                    status = "ACTIVE" if loss_name in [l.replace('_loss', '') + '_loss' for l in active_losses] else "INACTIVE"
                    print(f"  {loss_name:15}: {loss_value:8.6f} ({status})")
            
            if hasattr(trainer, 'weighted_losses'):
                weighted_losses = trainer.weighted_losses
                print(f"Weighted Losses:")
                for loss_name, loss_value in weighted_losses.items():
                    status = "ACTIVE" if loss_name in [l.replace('_loss', '') + '_loss' for l in active_losses] else "INACTIVE"
                    print(f"  {loss_name:15}: {loss_value:8.6f} ({status})")
            
            # Monitor latent space statistics
            with torch.no_grad():
                sample_size = min(100, data_a.shape[0])  # Sample for efficiency
                latent_a_sample = trainer.gen_a.enc(data_a[:sample_size]).cpu().numpy()
                latent_b_sample = trainer.gen_b.enc(data_b[:sample_size]).cpu().numpy()
                
                print(f"Latent Space Statistics:")
                print(f"  Modality A - mean: {latent_a_sample.mean():.3f}, std: {latent_a_sample.std():.3f}")
                print(f"  Modality B - mean: {latent_b_sample.mean():.3f}, std: {latent_b_sample.std():.3f}")
                
                # Compute cross-modal correlation
                if len(latent_a_sample) == len(latent_b_sample):
                    cross_correlations = []
                    for i in range(min(10, len(latent_a_sample))):
                        try:
                            corr = np.corrcoef(latent_a_sample[i], latent_b_sample[i])[0,1]
                            if not np.isnan(corr):
                                cross_correlations.append(corr)
                        except:
                            continue
                    
                    if cross_correlations:
                        avg_corr = np.mean(cross_correlations)
                        print(f"  Average cross-modal correlation: {avg_corr:.3f}")
                
                # Monitor latent space alignment (cosine similarity)
                latent_a_norm = latent_a_sample / (np.linalg.norm(latent_a_sample, axis=1, keepdims=True) + 1e-8)
                latent_b_norm = latent_b_sample / (np.linalg.norm(latent_b_sample, axis=1, keepdims=True) + 1e-8)
                cosine_sims = np.sum(latent_a_norm * latent_b_norm, axis=1)
                print(f"  Average cosine similarity: {cosine_sims.mean():.3f} ± {cosine_sims.std():.3f}")
            
            sys.stdout.flush()
            
            # Write to TensorBoard
            write_loss(iterations, trainer, train_writer)
            write_knn(trainer, data_a, data_b, image_directory, str(iterations), dataset)

        # Save checkpoints and generate visualizations
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            print(f"\n=== Saving checkpoint and plots at iteration {iterations + 1} ===")
            sys.stdout.flush()
            
            # Save model checkpoint
            trainer.save(checkpoint_directory, iterations)
            
            # Generate and save plots
            save_plots(trainer, data_a, data_b, image_directory, str(iterations), config)
            
            print(f"Checkpoint and plots saved successfully!")
            sys.stdout.flush()

        iterations += 1
        
        # Training completion
        if iterations >= max_iter:
            print(f"\n=== Training Completed at iteration {iterations} ===")
            print("Generating final visualizations and saving loss curves...")
            
            # Print final summary of loss schedule
            print(f"\n=== Final Training Summary ===")
            for phase, duration in sorted(trainer.phase_durations.items()):
                active_losses = get_active_losses_for_phase(phase, trainer.loss_schedule)
                print(f"Phase {phase} (duration {duration}): {active_losses}")
            
            sys.stdout.flush()
            
            # Final model save
            trainer.save(checkpoint_directory, iterations)
            
            # Final visualizations
            save_plots(trainer, data_a, data_b, image_directory, "final", config)
            
            # Save loss curves
            save_loss_curves(trainer.loss_history, output_directory)
            save_weighted_loss_curves(trainer.weighted_loss_history, output_directory)
            
            # Final evaluation
            print("Performing final evaluation...")
            write_knn(trainer, data_a, data_b, image_directory, "final", dataset)
            
            print("Training completed successfully!")
            print("All outputs saved to:", output_directory)
            sys.stdout.flush()
            
            sys.exit('Training finished successfully')