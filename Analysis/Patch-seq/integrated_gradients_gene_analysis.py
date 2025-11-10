"""
Integrated Gradients analysis for finding most influential genes in GEX encoder
"""
import torch
import numpy as np
import pandas as pd
import os
import argparse
from captum.attr import IntegratedGradients
from utils import get_config
from trainer import Trainer
from data_loader import CrossModalDataset
from tqdm import tqdm

def analyze_gene_importance_per_latent_dim(trainer, data_gex, gene_names, 
                                           baseline, latent_dim_idx, 
                                           n_steps=100, device='cuda'):
    """
    Analyze gene importance for a specific latent dimension using Integrated Gradients
    
    Args:
        trainer: Trained model
        data_gex: Gene expression data (n_samples, n_genes)
        gene_names: List of gene names
        baseline: Baseline for IG (same shape as data_gex)
        latent_dim_idx: Which latent dimension to analyze
        n_steps: Number of integration steps
        device: Device to run on
        
    Returns:
        gene_attributions: Array of attribution scores per gene
    """
    # Create wrapper function for specific latent dimension
    def encoder_latent_dim(x):
        """Extract specific latent dimension from encoder"""
        z = trainer.gen_b.enc(x)  # gen_b is the GEX encoder
        return z[:, latent_dim_idx]
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(encoder_latent_dim)
    
    # Compute attributions
    print(f"Computing attributions for latent dimension {latent_dim_idx}...")
    attributions, delta = ig.attribute(
        data_gex,
        baselines=baseline,
        n_steps=n_steps,
        method='gausslegendre',
        return_convergence_delta=True
    )
    
    # Check approximation error
    print(f"  Approximation error (delta): {delta.mean().item():.6e}")
    if abs(delta.mean().item()) > 0.01:
        print(f"  Warning: Large approximation error! Consider increasing n_steps.")
    
    # Average attributions across samples
    gene_attributions = attributions.abs().mean(dim=0).cpu().numpy()
    
    return gene_attributions


def analyze_all_latent_dimensions(trainer, data_gex, gene_names, 
                                  baseline, n_steps=100, device='cuda'):
    """
    Analyze gene importance for all latent dimensions
    
    Returns:
        results_df: DataFrame with genes ranked by importance for each latent dim
        aggregated_df: DataFrame with genes ranked by overall importance
    """
    latent_dim = trainer.gen_b.latent
    n_genes = data_gex.shape[1]
    
    # Store attributions for all dimensions
    all_attributions = np.zeros((latent_dim, n_genes))
    
    print(f"\n=== Analyzing {latent_dim} latent dimensions ===")
    
    for k in range(latent_dim):
        attributions = analyze_gene_importance_per_latent_dim(
            trainer, data_gex, gene_names, baseline, k, n_steps, device
        )
        all_attributions[k, :] = attributions
    
    # Create detailed results DataFrame
    results_list = []
    for k in range(latent_dim):
        for gene_idx, gene_name in enumerate(gene_names):
            results_list.append({
                'latent_dim': k,
                'gene': gene_name,
                'gene_index': gene_idx,
                'attribution': all_attributions[k, gene_idx]
            })
    
    results_df = pd.DataFrame(results_list)
    
    # Aggregate across all latent dimensions
    aggregated_attributions = all_attributions.mean(axis=0)  # Average across latent dims
    
    aggregated_df = pd.DataFrame({
        'gene': gene_names,
        'gene_index': np.arange(len(gene_names)),
        'mean_attribution': aggregated_attributions,
        'max_attribution': all_attributions.max(axis=0),
        'min_attribution': all_attributions.min(axis=0),
        'std_attribution': all_attributions.std(axis=0)
    })
    
    # Sort by mean attribution (descending)
    aggregated_df = aggregated_df.sort_values('mean_attribution', ascending=False).reset_index(drop=True)
    
    return results_df, aggregated_df, all_attributions


def create_baseline(data_gex, baseline_type='zeros', dataset=None):
    """
    Create baseline for Integrated Gradients
    
    Args:
        data_gex: Gene expression data
        baseline_type: Type of baseline ('zeros', 'median', 'mean')
        dataset: Dataset object (needed for some baseline types)
        
    Returns:
        baseline: Baseline tensor with same shape as data_gex
    """
    if baseline_type == 'zeros':
        baseline = torch.zeros_like(data_gex)
        print("Using zero baseline")
        
    elif baseline_type == 'median':
        # Per-gene median across all samples
        median_values = data_gex.median(dim=0).values
        baseline = median_values.unsqueeze(0).expand_as(data_gex)
        print("Using per-gene median baseline")
        
    elif baseline_type == 'mean':
        # Per-gene mean across all samples
        mean_values = data_gex.mean(dim=0)
        baseline = mean_values.unsqueeze(0).expand_as(data_gex)
        print("Using per-gene mean baseline")
        
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    return baseline


def load_gene_names(gene_expression_path):
    """
    Load gene names from the gene expression CSV file
    
    Returns:
        gene_names: List of gene names (or indices if no header)
    """
    try:
        # Try to load with header
        df = pd.read_csv(gene_expression_path, header=None, nrows=1)
        
        # Check if first column is index
        if df.iloc[0, 0] == 0 or pd.isna(df.iloc[0, 0]):
            # No gene names in header, use indices
            df_full = pd.read_csv(gene_expression_path, header=None)
            n_genes = df_full.shape[1] - 1  # Minus index column
            gene_names = [f"Gene_{i}" for i in range(n_genes)]
            print(f"No gene names found, using indices: Gene_0 to Gene_{n_genes-1}")
        else:
            # Gene names exist
            df_full = pd.read_csv(gene_expression_path, header=0)
            gene_names = df_full.columns[1:].tolist()  # Skip index column
            print(f"Loaded {len(gene_names)} gene names from header")
            
        return gene_names
        
    except Exception as e:
        print(f"Error loading gene names: {e}")
        print("Using default gene indices")
        # Fallback: load data to get dimensions
        df = pd.read_csv(gene_expression_path, header=None)
        n_genes = df.shape[1] - 1
        return [f"Gene_{i}" for i in range(n_genes)]


def main():
    parser = argparse.ArgumentParser(description='Integrated Gradients analysis for gene importance')
    parser.add_argument('--config', type=str,
                       default='/home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup14/cross_modal_VAE_v6/configs/attempt_1.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup14/cm_VAE_v6/outputs/attempt_1/checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str,
                       default='/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup14/cm_VAE_v6/outputs/attempt_1',
                       help='Directory to save results')
    parser.add_argument('--gene_expression_path', type=str,
                       default='/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/exon_data_top2000.csv',
                       help='Path to gene expression data')
    parser.add_argument('--baseline_type', type=str, default='zeros',
                       choices=['zeros', 'median', 'mean'],
                       help='Type of baseline for IG')
    parser.add_argument('--n_steps', type=int, default=100,
                       help='Number of integration steps')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for analysis (None = use all data)')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = get_config(args.config)
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = CrossModalDataset()
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data
    data_a, data_gex = dataset.get_full_data(device=device)
    print(f"Gene expression data shape: {data_gex.shape}")
    
    # Load gene names
    print("\nLoading gene names...")
    gene_names = load_gene_names(args.gene_expression_path)
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(config, dataset=dataset)
    trainer.to(device)
    
    # Load the final model checkpoint
    print("\nLoading model checkpoint...")
    from utils import get_model_list
    last_model_name = get_model_list(args.checkpoint_dir, "gen")
    
    if last_model_name is None:
        raise FileNotFoundError(f"No model checkpoint found in {args.checkpoint_dir}")
    
    state_dict = torch.load(last_model_name)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    
    print(f"Loaded checkpoint: {last_model_name}")
    
    # Set to evaluation mode
    trainer.eval()
    
    # Use subset if batch_size specified
    if args.batch_size is not None and args.batch_size < data_gex.shape[0]:
        print(f"\nUsing subset of {args.batch_size} samples for faster analysis")
        indices = torch.randperm(data_gex.shape[0])[:args.batch_size]
        data_gex_subset = data_gex[indices]
    else:
        data_gex_subset = data_gex
    
    # Create baseline
    print(f"\nCreating baseline ({args.baseline_type})...")
    baseline = create_baseline(data_gex_subset, args.baseline_type, dataset)
    
    # Ensure requires_grad is True for IG
    data_gex_subset.requires_grad_(True)
    
    # Analyze gene importance
    print(f"\nStarting Integrated Gradients analysis with {args.n_steps} steps...")
    print("This may take a while...")
    
    with torch.no_grad():
        results_df, aggregated_df, all_attributions = analyze_all_latent_dimensions(
            trainer, data_gex_subset, gene_names, baseline, 
            n_steps=args.n_steps, device=device
        )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save aggregated results (most important)
    output_path_agg = os.path.join(args.output_dir, 'gene_importance_aggregated.csv')
    aggregated_df.to_csv(output_path_agg, index=False)
    print(f"\n✓ Saved aggregated gene importance to: {output_path_agg}")
    
    # Save detailed per-dimension results
    output_path_detailed = os.path.join(args.output_dir, 'gene_importance_per_dimension.csv')
    results_df.to_csv(output_path_detailed, index=False)
    print(f"✓ Saved per-dimension results to: {output_path_detailed}")
    
    # Save top genes summary
    top_k = 50
    top_genes_df = aggregated_df.head(top_k)
    output_path_top = os.path.join(args.output_dir, f'gene_importance_top{top_k}.csv')
    top_genes_df.to_csv(output_path_top, index=False)
    print(f"✓ Saved top {top_k} genes to: {output_path_top}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total genes analyzed: {len(gene_names)}")
    print(f"Latent dimensions: {trainer.gen_b.latent}")
    print(f"Integration steps: {args.n_steps}")
    print(f"Baseline type: {args.baseline_type}")
    print(f"\nTop 10 most influential genes:")
    print("-" * 60)
    for idx, row in aggregated_df.head(10).iterrows():
        print(f"{idx+1:2d}. {row['gene']:20s} | Attribution: {row['mean_attribution']:.6f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()