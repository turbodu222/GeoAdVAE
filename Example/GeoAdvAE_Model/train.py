import argparse
import torch
import numpy as np
import random
import os
import sys

# Suppress output utility
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
parser.add_argument('--output_path', type=str, default='.', help='Output path')
parser.add_argument('--resume', action="store_true", help='Resume training')
parser.add_argument('--seed', type=int, default=40, help='Random seed')
opts = parser.parse_args()

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

print(f"✓ Random seed set to: {opts.seed}")
set_seed(opts.seed)

# Import modules silently
with SuppressOutput():
    from utils import prepare_sub_folder, write_loss, get_config, save_plots_with_celltype_colors, \
        write_knn, save_loss_curves, save_weighted_loss_curves, get_cross_modal_data_loader
    from torch.autograd import Variable
    from trainer import Trainer
    import torch.backends.cudnn as cudnn
    import ot
    import tensorboardX
    import shutil
    from tqdm import tqdm
    import glob
    from sklearn.neighbors import NearestNeighbors

try:
    from itertools import izip as zip
except ImportError:
    pass

cudnn.benchmark = True

# Load configuration and dataset
print("Loading dataset...")
with SuppressOutput():
    config = get_config(opts.config)
    max_iter = config['max_iter']
    
    # Create data loader
    train_loader, dataset = get_cross_modal_data_loader(
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_a, data_b = dataset.get_full_data(device=device)
    
    # Initialize trainer
    trainer = Trainer(config, dataset=dataset)
    trainer.to(device)

print("✓ Dataset loaded")

# Setup output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# Training setup
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
num_disc = config.get("num_disc", 2)
num_gen = config.get("num_gen", 1)

print(f"\nStarting training...\n")

# Progress bar
pbar = tqdm(total=max_iter, initial=iterations, desc="Training Progress", 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

# Training loop - suppress all trainer outputs
with SuppressOutput():
    while True:
        for it, batch in enumerate(train_loader):
            # Update training phase
            trainer.update_training_phase(iterations)
            
            # Extract batch data
            images_a = batch['morpho_data'].to(device)
            images_b = batch['gex_data'].to(device)
            indices_a = batch['index'].cpu().numpy()
            indices_b = batch['index'].cpu().numpy()

            # Discriminator updates
            if trainer._should_use_loss('gan_loss'):
                for disc_iter in range(num_disc):
                    trainer.dis_update(images_a, images_b, config)
            
            # Generator updates
            for gen_iter in range(num_gen):
                trainer.gen_update(images_a, images_b, config, variational=True, 
                                 batch_indices_a=indices_a, batch_indices_b=indices_b)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Update learning rate and record losses
            trainer.update_learning_rate()
            trainer.record_losses(iterations)

            # Periodic logging
            if (iterations + 1) % config['log_iter'] == 0:
                write_loss(iterations, trainer, train_writer)
                write_knn(trainer, data_a, data_b, image_directory, str(iterations), dataset, config)

            # Save checkpoints
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)
                save_plots_with_celltype_colors(trainer, data_a, data_b, image_directory, str(iterations), config, dataset)

            iterations += 1
            pbar.update(1)
            
            # Check if training is complete
            if iterations >= max_iter:
                break
        
        if iterations >= max_iter:
            break

pbar.close()

# Save final results
print("\nSaving final results...")
with SuppressOutput():
    trainer.save(checkpoint_directory, iterations)
    save_plots_with_celltype_colors(trainer, data_a, data_b, image_directory, "final", config, dataset)
    save_loss_curves(trainer.loss_history, output_directory)
    save_weighted_loss_curves(trainer.weighted_loss_history, output_directory)

# ========== Calculate Final Accuracy ==========
print("\nCalculating final accuracy...")

# Get latent representations
with torch.no_grad():
    latent_a = trainer.gen_a.enc(data_a).cpu().numpy()
    latent_b = trainer.gen_b.enc(data_b).cpu().numpy()

# Get RNA family labels
rna_labels = dataset.rna_family_labels
if rna_labels is None:
    rna_labels = dataset.gex_cluster_labels.cpu().numpy()
else:
    rna_labels = np.array(rna_labels)

# Calculate KNN accuracy: Morphology -> Gene Expression
def calculate_knn_accuracy(source_latent, target_latent, labels, k=1):
    """Calculate KNN accuracy from source to target modality"""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(target_latent)
    distances, indices = nbrs.kneighbors(source_latent)
    
    correct = 0
    for i in range(len(source_latent)):
        neighbor_labels = labels[indices[i]]
        # Majority vote
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        predicted_label = unique[np.argmax(counts)]
        if predicted_label == labels[i]:
            correct += 1
    
    return correct / len(source_latent)

k = 1  # Use k=1 for accuracy calculation

# Morphology -> Gene Expression
acc_a_to_b = calculate_knn_accuracy(latent_a, latent_b, rna_labels, k=k)

# Gene Expression -> Morphology  
acc_b_to_a = calculate_knn_accuracy(latent_b, latent_a, rna_labels, k=k)

# ========== Display Final Results ==========
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print(f"\n✓ Cell Type Accuracy (k={k}):")
print(f"  Morphology → Gene Expression: {acc_a_to_b:.4f}")
print(f"  Gene Expression → Morphology: {acc_b_to_a:.4f}")
print(f"  Average: {(acc_a_to_b + acc_b_to_a) / 2:.4f}")

# Display Integrated Plot
print("\n✓ Integrated Plot:")

final_plot = os.path.join(image_directory, "_combined_final.png")

if os.path.exists(final_plot):
    print(f"  Plot saved at: {final_plot}")
    print(f"  Displaying plot...\n")
    
    # Print the path for user to manually check if needed
    from IPython.display import Image as IPImage, display
    display(IPImage(filename=final_plot, width=1000))
    
else:
    print("⚠ Plot file not found")
    print(f"Expected location: {final_plot}")

print("\n" + "="*70)
print(f"Seed: {opts.seed}")
print(f"Output directory: {output_directory}")
print("="*70)

sys.exit(0)