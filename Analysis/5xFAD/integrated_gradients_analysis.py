"""
Integrated Gradients analysis for Morphology encoder
Identifies which input dimensions are most influential for the learned embeddings
"""
import torch
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_config, get_model_list, get_cross_modal_data_loader
from trainer import Trainer


class IntegratedGradientsAnalyzer:
    """
    Compute Integrated Gradients for encoder input dimensions
    """
    def __init__(self, encoder, device='cuda'):
        """
        Args:
            encoder: The encoder model (gen_a or gen_b)
            device: 'cuda' or 'cpu'
        """
        self.encoder = encoder
        self.device = device
        self.encoder.eval()
    
    def compute_integrated_gradients(self, input_data, baseline=None, n_steps=50):
        """
        Compute integrated gradients for each input dimension
        
        Args:
            input_data: Input tensor [n_samples, n_features]
            baseline: Baseline tensor (default: zeros)
            n_steps: Number of integration steps
            
        Returns:
            attributions: [n_samples, n_features] attribution scores
        """
        if baseline is None:
            baseline = torch.zeros_like(input_data)
        
        # Store attributions
        attributions = torch.zeros_like(input_data)
        
        # Generate interpolated inputs
        for i in range(n_steps + 1):
            alpha = i / n_steps
            interpolated = baseline + alpha * (input_data - baseline)
            interpolated.requires_grad = True
            
            # Forward pass
            mean, _ = self.encoder.encode(interpolated)
            
            # Sum over latent dimensions to get scalar output
            output = mean.sum()
            
            # Backward pass
            output.backward()
            
            # Accumulate gradients
            if i > 0 and i < n_steps:
                attributions += interpolated.grad
            elif i == 0 or i == n_steps:
                attributions += interpolated.grad / 2
            
            # Zero gradients
            interpolated.grad.zero_()
        
        # Scale by (input - baseline) and normalize by steps
        attributions = attributions * (input_data - baseline) / n_steps
        
        return attributions.detach()
    
    def analyze_feature_importance(self, data, n_steps=50, batch_size=32):
        """
        Analyze feature importance across all samples
        
        Args:
            data: Input data tensor [n_samples, n_features]
            n_steps: Number of integration steps
            batch_size: Process in batches to save memory
            
        Returns:
            dict with analysis results
        """
        n_samples = data.shape[0]
        n_features = data.shape[1]
        
        print(f"Computing Integrated Gradients for {n_samples} samples...")
        print(f"Input dimensions: {n_features}")
        print(f"Integration steps: {n_steps}")
        
        all_attributions = []
        
        # Process in batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            # Compute baseline as mean of all samples
            baseline = data.mean(dim=0, keepdim=True)
        
        for i in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_data = data[start_idx:end_idx]
            batch_baseline = baseline.expand(batch_data.shape[0], -1)
            
            # Compute attributions for this batch
            batch_attributions = self.compute_integrated_gradients(
                batch_data, 
                baseline=batch_baseline,
                n_steps=n_steps
            )
            
            all_attributions.append(batch_attributions.cpu().numpy())
        
        # Concatenate all attributions
        attributions = np.concatenate(all_attributions, axis=0)
        
        # Compute statistics
        mean_abs_attributions = np.abs(attributions).mean(axis=0)
        std_attributions = attributions.std(axis=0)
        mean_attributions = attributions.mean(axis=0)
        
        # Sort by importance
        importance_order = np.argsort(mean_abs_attributions)[::-1]
        
        results = {
            'attributions': attributions,
            'mean_abs_attributions': mean_abs_attributions,
            'std_attributions': std_attributions,
            'mean_attributions': mean_attributions,
            'importance_order': importance_order,
            'n_features': n_features,
            'n_samples': n_samples
        }
        
        return results


def analyze_morphology_encoder(config_path, checkpoint_dir, output_dir, 
                               n_steps=50, batch_size=32, device='cuda'):
    """
    Main function to analyze morphology encoder
    """
    print("="*80)
    print("Integrated Gradients Analysis for Morphology Encoder")
    print("="*80)
    
    # Load configuration
    print(f"\nLoading config from: {config_path}")
    config = get_config(config_path)
    
    # Create dataset
    print("\nLoading dataset...")
    train_loader, dataset = get_cross_modal_data_loader(
        batch_size=config['batch_size'], 
        shuffle=False,
        config=config  
    )
    
    # Get morphology data
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_a, data_b = dataset.get_full_data(device=device)
    print(f"Morphology data shape: {data_a.shape}")
    
    # Initialize trainer and load model
    print("\nInitializing trainer...")
    trainer = Trainer(config, dataset=dataset)
    trainer.to(device)
    
    print(f"\nLoading trained model from: {checkpoint_dir}")
    last_model_name = get_model_list(checkpoint_dir, "gen")
    
    if last_model_name is None:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
    
    print(f"Loading checkpoint: {last_model_name}")
    state_dict = torch.load(last_model_name, map_location=device)
    trainer.gen_a.load_state_dict(state_dict['a'])
    
    # Create analyzer
    print("\nInitializing Integrated Gradients analyzer...")
    analyzer = IntegratedGradientsAnalyzer(trainer.gen_a, device=device)
    
    # Analyze feature importance
    print("\n" + "="*80)
    print("Computing Feature Importance")
    print("="*80)
    
    results = analyzer.analyze_feature_importance(
        data_a, 
        n_steps=n_steps, 
        batch_size=batch_size
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    # 1. Save ranked feature importance
    importance_df = pd.DataFrame({
        'dimension_index': results['importance_order'],
        'mean_abs_attribution': results['mean_abs_attributions'][results['importance_order']],
        'mean_attribution': results['mean_attributions'][results['importance_order']],
        'std_attribution': results['std_attributions'][results['importance_order']]
    })
    
    importance_path = os.path.join(output_dir, 'morphology_feature_importance_ranked.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Ranked feature importance saved to: {importance_path}")
    
    # 2. Save full attribution matrix
    attribution_path = os.path.join(output_dir, 'morphology_attributions_full.npy')
    np.save(attribution_path, results['attributions'])
    print(f"✓ Full attribution matrix saved to: {attribution_path}")
    
    # 3. Generate visualization
    print("\nGenerating visualizations...")
    
    # Plot top 30 features
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top features by absolute importance
    top_n = min(30, results['n_features'])
    top_indices = results['importance_order'][:top_n]
    top_values = results['mean_abs_attributions'][top_indices]
    
    axes[0].barh(range(top_n), top_values)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels([f"Dim {i}" for i in top_indices])
    axes[0].set_xlabel('Mean Absolute Attribution')
    axes[0].set_title(f'Top {top_n} Most Influential Morphology Dimensions')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Distribution of attributions (signed)
    top_values_signed = results['mean_attributions'][top_indices]
    colors = ['red' if v < 0 else 'blue' for v in top_values_signed]
    
    axes[1].barh(range(top_n), top_values_signed, color=colors)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels([f"Dim {i}" for i in top_indices])
    axes[1].set_xlabel('Mean Attribution (Signed)')
    axes[1].set_title(f'Top {top_n} Dimensions - Positive vs Negative Influence')
    axes[1].invert_yaxis()
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'morphology_feature_importance_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved to: {plot_path}")
    
    # 4. Generate heatmap for top features across samples
    print("\nGenerating attribution heatmap...")
    
    # Select top 20 features and random subset of samples for visualization
    top_20 = results['importance_order'][:20]
    n_samples_viz = min(50, results['n_samples'])
    sample_indices = np.linspace(0, results['n_samples']-1, n_samples_viz, dtype=int)
    
    heatmap_data = results['attributions'][sample_indices][:, top_20]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(heatmap_data.T, 
                cmap='RdBu_r', 
                center=0,
                yticklabels=[f"Dim {i}" for i in top_20],
                xticklabels=sample_indices,
                cbar_kws={'label': 'Attribution Score'},
                ax=ax)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Morphology Dimension')
    ax.set_title('Top 20 Morphology Dimensions - Attribution Heatmap')
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'morphology_attribution_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap saved to: {heatmap_path}")
    
    # 5. Save summary statistics
    summary_path = os.path.join(output_dir, 'morphology_ig_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Integrated Gradients Analysis Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model checkpoint: {last_model_name}\n")
        f.write(f"Config file: {config_path}\n")
        f.write(f"Number of samples: {results['n_samples']}\n")
        f.write(f"Number of features: {results['n_features']}\n")
        f.write(f"Integration steps: {n_steps}\n\n")
        
        f.write("Top 20 Most Influential Dimensions:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6}{'Dimension':<12}{'Mean |Attr|':<15}{'Mean Attr':<15}{'Std Attr':<15}\n")
        f.write("-" * 80 + "\n")
        
        for rank in range(min(20, results['n_features'])):
            dim_idx = results['importance_order'][rank]
            mean_abs = results['mean_abs_attributions'][dim_idx]
            mean_val = results['mean_attributions'][dim_idx]
            std_val = results['std_attributions'][dim_idx]
            
            f.write(f"{rank+1:<6}{dim_idx:<12}{mean_abs:<15.6f}{mean_val:<15.6f}{std_val:<15.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Output Files:\n")
        f.write(f"  {importance_path}\n")
        f.write(f"  {attribution_path}\n")
        f.write(f"  {plot_path}\n")
        f.write(f"  {heatmap_path}\n")
    
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Print top 10 to console
    print("\n" + "="*80)
    print("Top 10 Most Influential Morphology Dimensions")
    print("="*80)
    print(f"{'Rank':<6}{'Dimension':<12}{'Mean |Attribution|':<20}{'Direction':<10}")
    print("-" * 80)
    
    for rank in range(min(10, results['n_features'])):
        dim_idx = results['importance_order'][rank]
        mean_abs = results['mean_abs_attributions'][dim_idx]
        mean_val = results['mean_attributions'][dim_idx]
        direction = "Positive" if mean_val > 0 else "Negative"
        
        print(f"{rank+1:<6}{dim_idx:<12}{mean_abs:<20.6f}{direction:<10}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Integrated Gradients analysis for Morphology encoder'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save analysis results')
    parser.add_argument('--n_steps', type=int, default=50,
                       help='Number of integration steps (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Verify paths
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    
    analyze_morphology_encoder(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()