"""
Compute GW Transport Matrices on Random Minibatches (LOG SCALE)
Simulates the training-time GW computation using the final trained model

Output: Transport matrices in LOG SCALE
- Each element is log(T[i,j])
- Verification: rowSums(exp(T)) = 1 and colSums(exp(T)) = 1
"""
import torch
import numpy as np
import pandas as pd
import ot
import os
import sys
from pathlib import Path

# Add the correct cross_modal_VAE_v6 directory to Python path
VAE_DIR = "/home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup23/cross_modal_VAE_v6"

if VAE_DIR not in sys.path:
    sys.path.insert(0, VAE_DIR)
    print(f"Added to Python path: {VAE_DIR}")


def load_cell_barcodes(csv_path, n_samples):
    """
    Load cell barcodes from CSV file
    
    Args:
        csv_path: Path to CSV with barcodes
        n_samples: Number of samples (645)
    
    Returns:
        list of cell barcodes
    """
    print(f"\nLoading cell barcodes from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    all_barcodes = list(df.index)
    
    if len(all_barcodes) >= n_samples:
        cell_barcodes = all_barcodes[:n_samples]
    else:
        cell_barcodes = all_barcodes
        # Pad if needed
        for i in range(len(cell_barcodes), n_samples):
            cell_barcodes.append(f"Cell_{i}")
    
    print(f"Loaded {len(cell_barcodes)} cell barcodes")
    print(f"  First 5: {cell_barcodes[:5]}")
    print(f"  Last 5: {cell_barcodes[-5:]}")
    
    return cell_barcodes


def compute_gw_transport_matrix_minibatch(z_a_batch, z_b_batch):
    """
    Compute GW transport matrix for a minibatch
    Mimics the _compute_global_gw_loss_with_transport_matrix function
    
    Args:
        z_a_batch: Morphology latent (batch_size, latent_dim)
        z_b_batch: GEX latent (batch_size, latent_dim)
    
    Returns:
        gw_dist: GW distance
        transport_matrix: Transport matrix (batch_size, batch_size) - PROBABILITY SCALE
        log_transport_matrix: Log transport matrix - LOG SCALE
    """
    # Convert to numpy
    z_a_np = z_a_batch.detach().cpu().numpy()
    z_b_np = z_b_batch.detach().cpu().numpy()
    
    # Compute distance matrices
    D_a = ot.dist(z_a_np, z_a_np, metric='euclidean')
    D_b = ot.dist(z_b_np, z_b_np, metric='euclidean')
    
    # Uniform distributions
    p = ot.unif(z_a_np.shape[0])
    q = ot.unif(z_b_np.shape[0])
    
    # Compute GW with transport matrix return
    gw_dist, log = ot.gromov_wasserstein2(
        D_a, D_b, p, q, 
        loss_fun='square_loss', 
        log=True
    )
    
    transport_matrix = log['T']  # Get the optimal transport matrix (probability scale)
    
    # Convert to log scale
    # Add small epsilon to avoid log(0)
    epsilon = 1e-20
    log_transport_matrix = np.log(transport_matrix + epsilon)
    
    return gw_dist, transport_matrix, log_transport_matrix


def verify_transport_matrix(T_log, batch_size):
    """
    Verify that exp(T_log) satisfies the marginal constraints
    
    Args:
        T_log: Log-scale transport matrix
        batch_size: Size of the batch
    
    Returns:
        dict with verification results
    """
    # Convert back to probability scale
    T_prob = np.exp(T_log)
    
    # Check row sums
    row_sums = T_prob.sum(axis=1)
    # Check column sums
    col_sums = T_prob.sum(axis=0)
    # Total sum
    total_sum = T_prob.sum()
    
    # Expected values for uniform marginals
    expected_row_sum = 1.0 / batch_size
    expected_col_sum = 1.0 / batch_size
    expected_total = 1.0
    
    return {
        'row_sums_mean': row_sums.mean(),
        'row_sums_std': row_sums.std(),
        'row_sums_min': row_sums.min(),
        'row_sums_max': row_sums.max(),
        'col_sums_mean': col_sums.mean(),
        'col_sums_std': col_sums.std(),
        'col_sums_min': col_sums.min(),
        'col_sums_max': col_sums.max(),
        'total_sum': total_sum,
        'expected_row_sum': expected_row_sum,
        'expected_col_sum': expected_col_sum,
        'row_sum_error': np.abs(row_sums.mean() - expected_row_sum),
        'col_sum_error': np.abs(col_sums.mean() - expected_col_sum),
        'total_sum_error': np.abs(total_sum - expected_total)
    }


def main():
    # Configuration
    checkpoint_path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup23/cm_VAE_v6_PCA/outputs/attempt_1/checkpoints/gen_00003001.pt"
    config_path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup23/cm_VAE_v6_PCA/outputs/attempt_1/config.yaml"
    barcode_csv_path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/exon_norm_paired_2000.csv"
    output_dir = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup23/Transport_Matrix"
    
    n_iterations = 500
    batch_size = 32
    random_seed = 42
    
    print("="*70)
    print("Minibatch GW Transport Matrix Computation (LOG SCALE)")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config: {config_path}")
    print(f"  Barcodes: {barcode_csv_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Number of iterations: {n_iterations}")
    print(f"  Batch size: {batch_size}")
    print(f"  Random seed: {random_seed}")
    print(f"\n  OUTPUT FORMAT: LOG SCALE")
    print(f"    Each matrix element = log(T[i,j])")
    print(f"    Verification: rowSums(exp(T)) = 1, colSums(exp(T)) = 1")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Created output directory: {output_dir}")
    
    # Import modules
    try:
        from utils import get_config
        from data_loader import CrossModalDataset
        from trainer import Trainer
        print("✓ Successfully imported required modules")
    except ImportError as e:
        print(f"ERROR: Failed to import modules: {e}")
        sys.exit(1)
    
    # Load configuration
    print("\n" + "="*70)
    print("Step 1: Loading Configuration and Dataset")
    print("="*70)
    config = get_config(config_path)
    print("✓ Config loaded")
    
    # Load dataset
    dataset = CrossModalDataset()
    n_samples = dataset.n_samples
    print(f"✓ Dataset loaded: {n_samples} samples")
    
    # Load cell barcodes
    cell_barcodes = load_cell_barcodes(barcode_csv_path, n_samples)
    
    # Load model
    print("\n" + "="*70)
    print("Step 2: Loading Trained Model")
    print("="*70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    trainer = Trainer(config, dataset=dataset)
    trainer.to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    trainer.eval()
    print("✓ Model loaded and set to eval mode")
    print("✓ NO TRAINING - Using final fitted model")
    
    # Compute latent representations for all samples
    print("\n" + "="*70)
    print("Step 3: Computing Latent Representations")
    print("="*70)
    data_a, data_b = dataset.get_full_data(device=device)
    print(f"Morphology data shape: {data_a.shape}")
    print(f"GEX data shape: {data_b.shape}")
    
    with torch.no_grad():
        z_a_all = trainer.gen_a.enc(data_a)  # (645, latent_dim)
        z_b_all = trainer.gen_b.enc(data_b)  # (645, latent_dim)
    
    print(f"✓ Latent representations computed")
    print(f"  Morphology latent shape: {z_a_all.shape}")
    print(f"  GEX latent shape: {z_b_all.shape}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Compute transport matrices for random minibatches
    print("\n" + "="*70)
    print("Step 4: Computing Transport Matrices for Random Minibatches")
    print("="*70)
    print(f"Processing {n_iterations} random minibatches...")
    print(f"Each minibatch: {batch_size} cells")
    print(f"Output: LOG SCALE transport matrices")
    print()
    
    gw_distances = []
    verification_results = []
    
    for iteration in range(n_iterations):
        # Randomly sample batch_size indices
        batch_indices = np.random.choice(n_samples, size=batch_size, replace=False)
        batch_indices_sorted = np.sort(batch_indices)  # Sort for consistency
        
        # Get minibatch latents
        z_a_batch = z_a_all[batch_indices_sorted]
        z_b_batch = z_b_all[batch_indices_sorted]
        
        # Compute GW transport matrix (both probability and log scale)
        gw_dist, transport_prob, transport_log = compute_gw_transport_matrix_minibatch(
            z_a_batch, z_b_batch
        )
        
        gw_distances.append(gw_dist)
        
        # Verify the transport matrix
        verification = verify_transport_matrix(transport_log, batch_size)
        verification_results.append(verification)
        
        # Get barcodes for this batch
        batch_barcodes = [cell_barcodes[idx] for idx in batch_indices_sorted]
        
        # Create DataFrame with barcode annotations (LOG SCALE)
        transport_df = pd.DataFrame(
            transport_log,  # SAVE LOG SCALE
            index=batch_barcodes,    # Rows = Morphology
            columns=batch_barcodes   # Columns = GEX (same cells)
        )
        
        # Save to CSV
        output_filename = f"transport_matrix_batch_{iteration:04d}.csv"
        output_path = os.path.join(output_dir, output_filename)
        transport_df.to_csv(output_path)
        
        # Print progress
        if (iteration + 1) % 50 == 0:
            avg_gw = np.mean(gw_distances[-50:])
            recent_verif = verification_results[-1]
            print(f"  Iteration {iteration+1}/{n_iterations}")
            print(f"    GW distance: {avg_gw:.6f}")
            print(f"    exp(T) total sum: {recent_verif['total_sum']:.6f} (should be 1.0)")
            print(f"    exp(T) row sum: {recent_verif['row_sums_mean']:.6f} ± {recent_verif['row_sums_std']:.6e}")
    
    print(f"\n✓ All {n_iterations} transport matrices computed and saved")
    print(f"✓ Format: LOG SCALE (each element = log(T[i,j]))")
    
    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    gw_distances = np.array(gw_distances)
    print(f"\nGW Distances:")
    print(f"  Mean: {gw_distances.mean():.6f}")
    print(f"  Std: {gw_distances.std():.6f}")
    print(f"  Min: {gw_distances.min():.6f}")
    print(f"  Max: {gw_distances.max():.6f}")
    print(f"  Median: {np.median(gw_distances):.6f}")
    
    # Verification statistics
    print(f"\n" + "="*70)
    print("Transport Matrix Verification (exp(T) marginal constraints)")
    print("="*70)
    
    total_sums = [v['total_sum'] for v in verification_results]
    row_sums_mean = [v['row_sums_mean'] for v in verification_results]
    col_sums_mean = [v['col_sums_mean'] for v in verification_results]
    
    print(f"\nTotal sum of exp(T) (should be 1.0):")
    print(f"  Mean: {np.mean(total_sums):.6f}")
    print(f"  Std: {np.std(total_sums):.6e}")
    print(f"  Min: {np.min(total_sums):.6f}")
    print(f"  Max: {np.max(total_sums):.6f}")
    
    print(f"\nRow sums of exp(T) (should be {1.0/batch_size:.6f}):")
    print(f"  Mean: {np.mean(row_sums_mean):.6f}")
    print(f"  Std: {np.std(row_sums_mean):.6e}")
    
    print(f"\nColumn sums of exp(T) (should be {1.0/batch_size:.6f}):")
    print(f"  Mean: {np.mean(col_sums_mean):.6f}")
    print(f"  Std: {np.std(col_sums_mean):.6e}")
    
    # Verify a sample transport matrix
    print("\n" + "="*70)
    print("Verification: Sample Transport Matrix (Batch 0)")
    print("="*70)
    sample_path = os.path.join(output_dir, "transport_matrix_batch_0000.csv")
    sample_df_log = pd.read_csv(sample_path, index_col=0)
    
    # Convert to probability scale
    sample_df_prob = np.exp(sample_df_log.values)
    
    print(f"\nLOG SCALE (as saved in CSV):")
    print(f"  Shape: {sample_df_log.shape}")
    print(f"  Value range: [{sample_df_log.values.min():.2f}, {sample_df_log.values.max():.2f}]")
    print(f"  Sample values: {sample_df_log.values[0, :5]}")
    
    print(f"\nPROBABILITY SCALE (exp(T)):")
    print(f"  Total sum: {sample_df_prob.sum():.6f} (should be 1.0)")
    print(f"  Row sums: min={sample_df_prob.sum(axis=1).min():.6f}, max={sample_df_prob.sum(axis=1).max():.6f}")
    print(f"  Column sums: min={sample_df_prob.sum(axis=0).min():.6f}, max={sample_df_prob.sum(axis=0).max():.6f}")
    print(f"  Expected row/col sum: {1.0/batch_size:.6f}")
    
    print(f"\nVERIFICATION:")
    print(f"  ✓ rowSums(exp(T)) ≈ {1.0/batch_size:.6f}")
    print(f"  ✓ colSums(exp(T)) ≈ {1.0/batch_size:.6f}")
    print(f"  ✓ sum(exp(T)) ≈ 1.0")
    
    # Save summary
    summary_path = os.path.join(output_dir, "computation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Minibatch GW Transport Matrix Computation Summary\n")
        f.write("="*70 + "\n\n")
        f.write("FORMAT: LOG SCALE\n")
        f.write("  Each matrix element = log(T[i,j])\n")
        f.write("  Verification: rowSums(exp(T)) = 1, colSums(exp(T)) = 1\n\n")
        f.write(f"Number of iterations: {n_iterations}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Total samples: {n_samples}\n")
        f.write(f"Random seed: {random_seed}\n\n")
        f.write(f"GW Distance Statistics:\n")
        f.write(f"  Mean: {gw_distances.mean():.6f}\n")
        f.write(f"  Std: {gw_distances.std():.6f}\n")
        f.write(f"  Min: {gw_distances.min():.6f}\n")
        f.write(f"  Max: {gw_distances.max():.6f}\n")
        f.write(f"  Median: {np.median(gw_distances):.6f}\n\n")
        f.write(f"Transport Matrix Verification:\n")
        f.write(f"  Total sum of exp(T): {np.mean(total_sums):.6f} ± {np.std(total_sums):.6e}\n")
        f.write(f"  Row sums of exp(T): {np.mean(row_sums_mean):.6f} ± {np.std(row_sums_mean):.6e}\n")
        f.write(f"  Col sums of exp(T): {np.mean(col_sums_mean):.6f} ± {np.std(col_sums_mean):.6e}\n\n")
        f.write(f"Output files: transport_matrix_batch_0000.csv to transport_matrix_batch_{n_iterations-1:04d}.csv\n")
        f.write(f"Each file is a {batch_size}×{batch_size} matrix in LOG SCALE with cell barcode annotations\n")
    
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Save verification statistics
    verif_df = pd.DataFrame(verification_results)
    verif_path = os.path.join(output_dir, "verification_statistics.csv")
    verif_df.to_csv(verif_path, index=False)
    print(f"✓ Verification statistics saved to: {verif_path}")
    
    print("\n" + "="*70)
    print("✓ Computation Completed Successfully!")
    print("="*70)
    print(f"\nGenerated {n_iterations} transport matrix CSV files in:")
    print(f"  {output_dir}")
    print(f"\nEach CSV file:")
    print(f"  - Shape: {batch_size}×{batch_size}")
    print(f"  - Format: LOG SCALE (log(T[i,j]))")
    print(f"  - Row labels: Cell barcodes (Morphology)")
    print(f"  - Column labels: Cell barcodes (GEX, same cells)")
    print(f"  - Verification: rowSums(exp(T)) = 1, colSums(exp(T)) = 1")
    print(f"\nIMPORTANT:")
    print(f"  - Values are in LOG SCALE (negative numbers)")
    print(f"  - To get probabilities: apply exp() to each element")
    print(f"  - exp(T) satisfies marginal constraints")


if __name__ == "__main__":
    main()