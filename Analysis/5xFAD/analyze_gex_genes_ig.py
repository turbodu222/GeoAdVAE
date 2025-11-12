"""
Analyze gene importance using Integrated Gradients for the trained VAE model.
This script identifies which genes are most informative for the GEX encoder.
"""

import torch
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
import argparse
import os
import sys
from tqdm import tqdm

from utils import get_config, get_model_list
from trainer import Trainer
from data_loader import CrossModalDatasetUnpaired, create_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GEXEncoderWrapper(torch.nn.Module):
    """
    Wrapper for the GEX encoder to use with Integrated Gradients.
    Returns the sum of all latent dimensions for global importance.
    """
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        
    def forward(self, x):
        """Forward pass returning sum of latent representation."""
        mean, _ = self.generator.encode(x)
        # Sum all latent dimensions to get scalar output
        return mean.sum(dim=1)


def load_trained_model(config_path, checkpoint_dir):
    """
    Load a trained model from checkpoint.
    
    Args:
        config_path: Path to the config YAML file
        checkpoint_dir: Directory containing model checkpoints
        
    Returns:
        trainer: Trained Trainer object
        config: Configuration dictionary
        dataset: Dataset object
    """
    print("Loading configuration...")
    config = get_config(config_path)
    
    print("Initializing trainer...")
    dataset = CrossModalDatasetUnpaired(use_prior_loss=config.get('lambda_p', 0) > 0)
    trainer = Trainer(config, dataset=dataset)
    trainer.to(device)
    trainer.eval()
    
    print("Loading model weights...")
    gen_checkpoint = get_model_list(checkpoint_dir, "gen")
    if gen_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    state_dict = torch.load(gen_checkpoint, map_location=device, weights_only=False)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    
    print(f"Loaded checkpoint: {gen_checkpoint}")
    
    return trainer, config, dataset


def compute_gene_importance_with_statistics(
    generator,
    gene_expression_data,
    baseline,
    n_steps=50,
    batch_size=32
):
    """
    Compute gene importance with mean absolute attribution, mean attribution, and std.
    
    Args:
        generator: GEX generator model (gen_b)
        gene_expression_data: Full gene expression data (n_samples, n_genes)
        baseline: Baseline for IG
        n_steps: Number of integration steps
        batch_size: Batch size for processing
        
    Returns:
        dict with 'mean_abs_attribution', 'mean_attribution', 'std_attribution'
    """
    print(f"\nComputing Integrated Gradients...")
    
    # Wrap generator
    wrapper = GEXEncoderWrapper(generator)
    wrapper.eval()
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(wrapper)
    
    # Process in batches
    n_samples = gene_expression_data.shape[0]
    n_genes = gene_expression_data.shape[1]
    all_attributions = []
    
    print(f"Processing {n_samples} samples in batches of {batch_size}...")
    for i in tqdm(range(0, n_samples, batch_size), desc="Computing attributions"):
        batch_end = min(i + batch_size, n_samples)
        batch_data = gene_expression_data[i:batch_end].clone().detach().requires_grad_(True)
        batch_baseline = baseline.unsqueeze(0).expand(batch_data.shape[0], -1)
        
        try:
            attributions, delta = ig.attribute(
                batch_data,
                baselines=batch_baseline,
                n_steps=n_steps,
                method='gausslegendre',
                return_convergence_delta=True
            )
            
            all_attributions.append(attributions.detach().cpu())
            
            # Check convergence
            if delta.abs().mean() > 0.05:
                print(f"\n  Warning: Large approximation error (delta={delta.abs().mean():.6f}) for batch {i//batch_size}")
                
        except Exception as e:
            print(f"\n  Error processing batch {i//batch_size}: {e}")
            continue
    
    # Concatenate all attributions: shape (n_samples, n_genes)
    all_attributions = torch.cat(all_attributions, dim=0).numpy()
    
    print(f"\nAttribution matrix shape: {all_attributions.shape}")
    
    # Compute statistics across samples
    mean_abs_attribution = np.abs(all_attributions).mean(axis=0)  # (n_genes,)
    mean_attribution = all_attributions.mean(axis=0)              # (n_genes,)
    std_attribution = all_attributions.std(axis=0)                # (n_genes,)
    
    return {
        'mean_abs_attribution': mean_abs_attribution,
        'mean_attribution': mean_attribution,
        'std_attribution': std_attribution
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze gene importance using Integrated Gradients'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the config file used for training')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--baseline_type', type=str, default='zeros',
                       choices=['zeros', 'median', 'mean'],
                       help='Type of baseline to use')
    parser.add_argument('--n_steps', type=int, default=50,
                       help='Number of integration steps')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Number of samples to analyze (None for all)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trained model
    print("="*80)
    print("Integrated Gradients Analysis for GEX Encoder")
    print("="*80)
    trainer, config, dataset = load_trained_model(args.config, args.checkpoint_dir)
    
    # Get gene expression data
    print("\n" + "="*80)
    print("Loading gene expression data...")
    print("="*80)
    _, gex_data = dataset.get_full_data(device=device)
    
    # Optionally subsample
    if args.n_samples is not None and args.n_samples < gex_data.shape[0]:
        print(f"Subsampling {args.n_samples} out of {gex_data.shape[0]} samples...")
        indices = torch.randperm(gex_data.shape[0])[:args.n_samples]
        gex_data = gex_data[indices]
    
    print(f"Gene expression data shape: {gex_data.shape}")
    n_samples, n_genes = gex_data.shape
    
    # Load gene names from the original data file
    gene_names = None
    try:
        gex_path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/5xFAD/gex_matrix.csv"
        print(f"\nLoading gene names from: {gex_path}")
        gex_df = pd.read_csv(gex_path, header=None, nrows=1)
        gene_names = gex_df.iloc[0, 1:].values.tolist()
        print(f"Loaded {len(gene_names)} gene names")
        
        # Verify gene names match data dimensions
        if len(gene_names) != n_genes:
            print(f"Warning: Number of gene names ({len(gene_names)}) doesn't match data ({n_genes})")
            gene_names = gene_names[:n_genes] if len(gene_names) > n_genes else gene_names + [f"Gene_{i}" for i in range(len(gene_names), n_genes)]
            
    except Exception as e:
        print(f"Could not load gene names: {e}")
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
    
    # Create baseline
    print(f"\nCreating baseline ({args.baseline_type})...")
    if args.baseline_type == 'zeros':
        baseline = torch.zeros(n_genes, device=device)
    elif args.baseline_type == 'median':
        baseline = gex_data.median(dim=0)[0]
    elif args.baseline_type == 'mean':
        baseline = gex_data.mean(dim=0)
    
    print(f"Baseline shape: {baseline.shape}")
    
    # Get generator
    generator = trainer.gen_b
    generator.eval()
    
    # Compute gene importance
    print("\n" + "="*80)
    print("Computing Gene Importance Statistics")
    print("="*80)
    
    results = compute_gene_importance_with_statistics(
        generator,
        gex_data,
        baseline,
        n_steps=args.n_steps,
        batch_size=args.batch_size
    )
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'gene': gene_names,
        'mean_abs_attribution': results['mean_abs_attribution'],
        'mean_attribution': results['mean_attribution'],
        'std_attribution': results['std_attribution']
    })
    
    # Sort by mean absolute attribution (most important first)
    results_df = results_df.sort_values('mean_abs_attribution', ascending=False)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'gene_importance_ranked.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"\n" + "="*80)
    print("Results saved!")
    print("="*80)
    print(f"Output file: {output_file}")
    print(f"Total genes analyzed: {len(results_df)}")
    
    # Print top 20 genes
    print("\n" + "="*80)
    print("Top 20 Most Important Genes")
    print("="*80)
    print(f"{'Rank':<6}{'Gene':<20}{'Mean |Attr|':<15}{'Mean Attr':<15}{'Std Attr':<15}")
    print("-" * 80)
    
    for idx, row in results_df.head(20).iterrows():
        rank = results_df.index.get_loc(idx) + 1
        print(f"{rank:<6}{row['gene']:<20}{row['mean_abs_attribution']:<15.6f}"
              f"{row['mean_attribution']:<15.6f}{row['std_attribution']:<15.6f}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'gene_ig_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Integrated Gradients Analysis Summary - GEX Encoder\n")
        f.write("="*80 + "\n\n")
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"Number of genes: {n_genes}\n")
        f.write(f"Integration steps: {args.n_steps}\n")
        f.write(f"Baseline type: {args.baseline_type}\n\n")
        
        f.write("Top 20 Most Important Genes:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6}{'Gene':<20}{'Mean |Attr|':<15}{'Mean Attr':<15}{'Std Attr':<15}\n")
        f.write("-" * 80 + "\n")
        
        for idx, row in results_df.head(20).iterrows():
            rank = results_df.index.get_loc(idx) + 1
            f.write(f"{rank:<6}{row['gene']:<20}{row['mean_abs_attribution']:<15.6f}"
                   f"{row['mean_attribution']:<15.6f}{row['std_attribution']:<15.6f}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()