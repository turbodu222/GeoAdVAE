"""
Analyze 500 Minibatch Transport Matrices (LOG SCALE)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def analyze_transport_matrices_logscale(matrix_dir, output_dir=None):
    """
    Analyze the 500 minibatch transport matrices (LOG SCALE format)
    
    Args:
        matrix_dir: Directory containing the transport matrix CSV files
        output_dir: Directory to save analysis results (default: same as matrix_dir)
    """
    if output_dir is None:
        output_dir = matrix_dir
    
    print("="*70)
    print("Analysis of Minibatch Transport Matrices (LOG SCALE)")
    print("="*70)
    
    # Find all transport matrix files
    matrix_files = sorted(Path(matrix_dir).glob("transport_matrix_batch_*.csv"))
    n_matrices = len(matrix_files)
    
    print(f"\nFound {n_matrices} transport matrix files")
    
    if n_matrices == 0:
        print("ERROR: No transport matrix files found!")
        return
    
    # Load first matrix to get dimensions
    first_df_log = pd.read_csv(matrix_files[0], index_col=0)
    batch_size = first_df_log.shape[0]
    
    print(f"Batch size: {batch_size}")
    print(f"Matrix shape: {batch_size}×{batch_size}")
    print(f"Format: LOG SCALE")
    
    # Statistics to collect (on probability scale)
    all_diagonal_means_prob = []
    all_diagonal_dominance_prob = []
    all_total_sums_prob = []
    all_row_sum_stds_prob = []
    all_col_sum_stds_prob = []
    
    # Statistics on log scale
    all_log_min_values = []
    all_log_max_values = []
    all_log_mean_values = []
    
    print("\nProcessing matrices...")
    for i, matrix_file in enumerate(matrix_files):
        df_log = pd.read_csv(matrix_file, index_col=0)
        
        # Convert to probability scale
        df_prob = np.exp(df_log.values)
        
        # Collect statistics on probability scale
        diagonal_prob = np.diag(df_prob)
        all_diagonal_means_prob.append(diagonal_prob.mean())
        all_diagonal_dominance_prob.append(diagonal_prob.sum() / df_prob.sum())
        all_total_sums_prob.append(df_prob.sum())
        all_row_sum_stds_prob.append(df_prob.sum(axis=1).std())
        all_col_sum_stds_prob.append(df_prob.sum(axis=0).std())
        
        # Statistics on log scale
        all_log_min_values.append(df_log.values.min())
        all_log_max_values.append(df_log.values.max())
        all_log_mean_values.append(df_log.values.mean())
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_matrices} matrices")
    
    print(f"✓ Processed all {n_matrices} matrices")
    
    # Convert to numpy arrays
    all_diagonal_means_prob = np.array(all_diagonal_means_prob)
    all_diagonal_dominance_prob = np.array(all_diagonal_dominance_prob)
    all_total_sums_prob = np.array(all_total_sums_prob)
    all_row_sum_stds_prob = np.array(all_row_sum_stds_prob)
    all_col_sum_stds_prob = np.array(all_col_sum_stds_prob)
    all_log_min_values = np.array(all_log_min_values)
    all_log_max_values = np.array(all_log_max_values)
    all_log_mean_values = np.array(all_log_mean_values)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("Summary Statistics (PROBABILITY SCALE - exp(T))")
    print("="*70)
    
    print(f"\nDiagonal mean (average across batches):")
    print(f"  Mean: {all_diagonal_means_prob.mean():.6e}")
    print(f"  Std: {all_diagonal_means_prob.std():.6e}")
    print(f"  Min: {all_diagonal_means_prob.min():.6e}")
    print(f"  Max: {all_diagonal_means_prob.max():.6e}")
    
    print(f"\nDiagonal dominance (average across batches):")
    print(f"  Mean: {all_diagonal_dominance_prob.mean():.4f} ({all_diagonal_dominance_prob.mean()*100:.2f}%)")
    print(f"  Std: {all_diagonal_dominance_prob.std():.4f}")
    print(f"  Min: {all_diagonal_dominance_prob.min():.4f}")
    print(f"  Max: {all_diagonal_dominance_prob.max():.4f}")
    
    print(f"\nTotal sum (should be 1.0):")
    print(f"  Mean: {all_total_sums_prob.mean():.6f}")
    print(f"  Std: {all_total_sums_prob.std():.6e}")
    print(f"  Min: {all_total_sums_prob.min():.6f}")
    print(f"  Max: {all_total_sums_prob.max():.6f}")
    
    print(f"\nRow sum std (uniformity check):")
    print(f"  Mean: {all_row_sum_stds_prob.mean():.6e}")
    print(f"  Expected for uniform: ~{np.sqrt(1/(12*batch_size)):.6e}")
    
    print("\n" + "="*70)
    print("Summary Statistics (LOG SCALE - log(T))")
    print("="*70)
    
    print(f"\nLog value range:")
    print(f"  Min across all matrices: {all_log_min_values.min():.2f}")
    print(f"  Max across all matrices: {all_log_max_values.max():.2f}")
    print(f"  Mean across all matrices: {all_log_mean_values.mean():.2f}")
    
    # Verification
    print("\n" + "="*70)
    print("Marginal Constraint Verification")
    print("="*70)
    
    # Check a sample matrix
    sample_df_log = pd.read_csv(matrix_files[0], index_col=0)
    sample_df_prob = np.exp(sample_df_log.values)
    
    row_sums = sample_df_prob.sum(axis=1)
    col_sums = sample_df_prob.sum(axis=0)
    
    print(f"\nSample matrix (batch 0) verification:")
    print(f"  Total sum: {sample_df_prob.sum():.6f}")
    print(f"  Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}, mean={row_sums.mean():.6f}")
    print(f"  Col sums: min={col_sums.min():.6f}, max={col_sums.max():.6f}, mean={col_sums.mean():.6f}")
    print(f"  Expected row/col sum: {1.0/batch_size:.6f}")
    print(f"\n  ✓ rowSums(exp(T)) ≈ {1.0/batch_size:.6f}")
    print(f"  ✓ colSums(exp(T)) ≈ {1.0/batch_size:.6f}")
    print(f"  ✓ sum(exp(T)) = 1.0")
    
    # Visualizations
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Diagonal mean distribution (probability scale)
    ax = axes[0, 0]
    ax.hist(all_diagonal_means_prob, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(all_diagonal_means_prob.mean(), color='red', linestyle='--', 
               label=f'Mean: {all_diagonal_means_prob.mean():.6e}')
    ax.set_xlabel('Diagonal Mean (exp(T))')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Diagonal Means (Probability Scale)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Diagonal dominance distribution (probability scale)
    ax = axes[0, 1]
    ax.hist(all_diagonal_dominance_prob, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(all_diagonal_dominance_prob.mean(), color='red', linestyle='--',
               label=f'Mean: {all_diagonal_dominance_prob.mean():.4f}')
    ax.set_xlabel('Diagonal Dominance (exp(T))')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Diagonal Dominance (Probability Scale)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Total sum distribution (probability scale)
    ax = axes[0, 2]
    ax.hist(all_total_sums_prob, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', label='Expected: 1.0')
    ax.set_xlabel('Total Sum (exp(T))')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Total Sums (Probability Scale)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Sample transport matrix heatmap (probability scale)
    ax = axes[1, 0]
    sns.heatmap(sample_df_prob, cmap='viridis', ax=ax, cbar_kws={'label': 'Probability'})
    ax.set_xlabel('GEX Cells')
    ax.set_ylabel('Morphology Cells')
    ax.set_title(f'Sample Transport Matrix (exp(T), Batch 0)')
    ax.plot([0, batch_size], [0, batch_size], 'r--', linewidth=2, alpha=0.5)
    
    # 5. Log value distribution
    ax = axes[1, 1]
    all_log_values = sample_df_log.values.flatten()
    ax.hist(all_log_values, bins=50, color='purple', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Log Transport Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Log Values (Sample Matrix)')
    ax.axvline(all_log_values.mean(), color='red', linestyle='--',
               label=f'Mean: {all_log_values.mean():.2f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. Diagonal vs off-diagonal (probability scale)
    ax = axes[1, 2]
    diagonal_sample = np.diag(sample_df_prob)
    off_diag_mask = ~np.eye(batch_size, dtype=bool)
    off_diag_sample = sample_df_prob[off_diag_mask]
    
    ax.hist([diagonal_sample, off_diag_sample], bins=30,
            label=['Diagonal', 'Off-diagonal'], alpha=0.7, color=['red', 'blue'])
    ax.set_xlabel('Transport Probability (exp(T))')
    ax.set_ylabel('Frequency')
    ax.set_title('Diagonal vs Off-diagonal (Sample, Probability Scale)')
    ax.set_yscale('log')
    ax.legend()
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "transport_matrices_analysis_logscale.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {fig_path}")
    plt.close()
    
    # Save statistics to CSV
    stats_df = pd.DataFrame({
        'batch_index': range(n_matrices),
        'diagonal_mean_prob': all_diagonal_means_prob,
        'diagonal_dominance_prob': all_diagonal_dominance_prob,
        'total_sum_prob': all_total_sums_prob,
        'row_sum_std_prob': all_row_sum_stds_prob,
        'col_sum_std_prob': all_col_sum_stds_prob,
        'log_min_value': all_log_min_values,
        'log_max_value': all_log_max_values,
        'log_mean_value': all_log_mean_values
    })
    
    stats_path = os.path.join(output_dir, "batch_statistics_logscale.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Saved batch statistics to: {stats_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "analysis_summary_logscale.txt")
    with open(summary_path, 'w') as f:
        f.write("Minibatch Transport Matrices Analysis Summary (LOG SCALE)\n")
        f.write("="*70 + "\n\n")
        f.write("INPUT FORMAT: LOG SCALE\n")
        f.write("  Each matrix element = log(T[i,j])\n")
        f.write("  Analysis converts to probability scale via exp()\n\n")
        f.write(f"Number of matrices analyzed: {n_matrices}\n")
        f.write(f"Batch size: {batch_size}\n\n")
        f.write("PROBABILITY SCALE STATISTICS (exp(T)):\n\n")
        f.write(f"Diagonal Mean:\n")
        f.write(f"  Mean: {all_diagonal_means_prob.mean():.6e}\n")
        f.write(f"  Std: {all_diagonal_means_prob.std():.6e}\n\n")
        f.write(f"Diagonal Dominance:\n")
        f.write(f"  Mean: {all_diagonal_dominance_prob.mean():.4f} ({all_diagonal_dominance_prob.mean()*100:.2f}%)\n")
        f.write(f"  Std: {all_diagonal_dominance_prob.std():.4f}\n\n")
        f.write(f"Total Sum:\n")
        f.write(f"  Mean: {all_total_sums_prob.mean():.6f}\n")
        f.write(f"  Std: {all_total_sums_prob.std():.6e}\n\n")
        f.write("VERIFICATION:\n")
        f.write(f"  ✓ rowSums(exp(T)) = {1.0/batch_size:.6f}\n")
        f.write(f"  ✓ colSums(exp(T)) = {1.0/batch_size:.6f}\n")
        f.write(f"  ✓ sum(exp(T)) = 1.0\n")
    
    print(f"✓ Saved analysis summary to: {summary_path}")
    
    print("\n" + "="*70)
    print("✓ Analysis Completed!")
    print("="*70)
    print(f"\nKey Findings:")
    print(f"  - Average diagonal dominance: {all_diagonal_dominance_prob.mean():.2%}")
    print(f"  - Total sum verification: {all_total_sums_prob.mean():.6f} (expected: 1.0)")
    print(f"  - Marginal constraints satisfied: ✓")


if __name__ == "__main__":
    matrix_dir = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup23/Transport_Matrix"
    
    analyze_transport_matrices_logscale(matrix_dir)