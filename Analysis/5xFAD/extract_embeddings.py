"""
Extract latent embeddings from trained cross-modal VAE model
"""
import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse

# Add the project directory to path if needed
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_config, get_model_list, get_cross_modal_data_loader
from trainer import Trainer

def extract_embeddings(config_path, checkpoint_dir, output_dir, device='cuda'):
    """
    Extract latent embeddings from trained model
    
    Args:
        config_path: Path to config YAML file
        checkpoint_dir: Directory containing model checkpoints
        output_dir: Directory to save embeddings
        device: 'cuda' or 'cpu'
    """
    print("="*80)
    print("Extracting Latent Embeddings from Trained Model")
    print("="*80)
    
    # Load configuration
    print(f"\nLoading config from: {config_path}")
    config = get_config(config_path)
    
    # Create dataset
    print("\nLoading dataset...")
    train_loader, dataset = get_cross_modal_data_loader(
        batch_size=config['batch_size'], 
        shuffle=False,  # No shuffle for embedding extraction
        config=config  
    )
    
    # Get full data
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_a, data_b = dataset.get_full_data(device=device)
    print(f"Morphology data shape: {data_a.shape}")
    print(f"Gene expression data shape: {data_b.shape}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(config, dataset=dataset)
    trainer.to(device)
    
    # Load trained model
    print(f"\nLoading trained model from: {checkpoint_dir}")
    last_model_name = get_model_list(checkpoint_dir, "gen")
    
    if last_model_name is None:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
    
    print(f"Loading checkpoint: {last_model_name}")
    state_dict = torch.load(last_model_name, map_location=device)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    
    # Set to evaluation mode
    trainer.gen_a.eval()
    trainer.gen_b.eval()
    
    print("\nModel loaded successfully!")

    # 在加载模型后添加（第 53 行附近）：
    print("\n=== Model Architecture Check ===")
    print(f"Generator A latent dimension: {trainer.gen_a.latent}")
    print(f"Generator B latent dimension: {trainer.gen_b.latent}")
    
    # Extract embeddings
    print("\n" + "="*80)
    print("Extracting Latent Embeddings")
    print("="*80)
    
    with torch.no_grad():
        # Extract morphology embeddings
        print("\nExtracting morphology embeddings...")
        mean_a, logvar_a = trainer.gen_a.encode(data_a)
        latent_a = mean_a  # Use mean as the embedding (deterministic)
        latent_a_np = latent_a.cpu().numpy()
        
        print(f"Morphology latent shape: {latent_a_np.shape}")
        print(f"Morphology tensor shape: {latent_a.shape}")  # 添加这行
        print(f"Expected latent dim: {config['gen']['latent']}")  # 添加这行
        
        # Extract gene expression embeddings
        print("\nExtracting gene expression embeddings...")
        mean_b, logvar_b = trainer.gen_b.encode(data_b)
        latent_b = mean_b  # Use mean as the embedding (deterministic)
        latent_b_np = latent_b.cpu().numpy()
        
        print(f"Gene expression latent shape: {latent_b_np.shape}")
        print(f"Gene expression tensor shape: {latent_b.shape}")  # 添加这行
        print(f"Expected latent dim: {config['gen']['latent']}")  # 添加这行


        print("\n=== Dimension Verification ===")
        print(f"Config latent dimension: {config['gen']['latent']}")
        print(f"Morphology encoder output: {mean_a.shape}")
        print(f"GEX encoder output: {mean_b.shape}")
        print(f"Morphology numpy shape: {latent_a_np.shape}")
        print(f"GEX numpy shape: {latent_b_np.shape}")
    
    # Get cluster labels
    morpho_cluster_labels = dataset.morpho_cluster_labels.cpu().numpy()
    gex_cluster_labels = dataset.gex_cluster_labels.cpu().numpy()
    
    print(f"\nCluster information:")
    print(f"Morphology unique clusters: {np.unique(morpho_cluster_labels)}")
    print(f"Gene expression unique clusters: {len(np.unique(gex_cluster_labels))} clusters")
    
    # Save embeddings
    print("\n" + "="*80)
    print("Saving Embeddings")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save morphology embeddings
    morpho_output_path = os.path.join(output_dir, "morphology_latent_embeddings.csv")
    morpho_df = pd.DataFrame(
        latent_a_np,
        columns=[f"latent_dim_{i}" for i in range(latent_a_np.shape[1])]
    )
    morpho_df.insert(0, 'sample_id', range(len(morpho_df)))
    morpho_df.insert(1, 'cluster_label', morpho_cluster_labels[:len(morpho_df)])
    morpho_df.to_csv(morpho_output_path, index=False)
    print(f"✓ Morphology embeddings saved to: {morpho_output_path}")
    print(f"  Shape: {morpho_df.shape}")
    print(f"  Columns: sample_id, cluster_label, {', '.join(morpho_df.columns[2:5].tolist())}...")
    
    # Save gene expression embeddings
    gex_output_path = os.path.join(output_dir, "gene_expression_latent_embeddings.csv")
    gex_df = pd.DataFrame(
        latent_b_np,
        columns=[f"latent_dim_{i}" for i in range(latent_b_np.shape[1])]
    )
    gex_df.insert(0, 'sample_id', range(len(gex_df)))
    gex_df.insert(1, 'cluster_label', gex_cluster_labels[:len(gex_df)])
    gex_df.to_csv(gex_output_path, index=False)
    print(f"✓ Gene expression embeddings saved to: {gex_output_path}")
    print(f"  Shape: {gex_df.shape}")
    print(f"  Columns: sample_id, cluster_label, {', '.join(gex_df.columns[2:5].tolist())}...")
    
    # Also save as numpy arrays for easier loading
    morpho_npy_path = os.path.join(output_dir, "morphology_latent_embeddings.npy")
    gex_npy_path = os.path.join(output_dir, "gene_expression_latent_embeddings.npy")
    
    np.save(morpho_npy_path, latent_a_np)
    np.save(gex_npy_path, latent_b_np)
    print(f"\n✓ NumPy arrays also saved:")
    print(f"  {morpho_npy_path}")
    print(f"  {gex_npy_path}")
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, "embedding_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Latent Embedding Extraction Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model checkpoint: {last_model_name}\n")
        f.write(f"Config file: {config_path}\n\n")
        
        f.write("Morphology Embeddings:\n")
        f.write(f"  Shape: {latent_a_np.shape}\n")
        f.write(f"  Mean: {latent_a_np.mean():.6f}\n")
        f.write(f"  Std: {latent_a_np.std():.6f}\n")
        f.write(f"  Min: {latent_a_np.min():.6f}\n")
        f.write(f"  Max: {latent_a_np.max():.6f}\n")
        f.write(f"  Number of clusters: {len(np.unique(morpho_cluster_labels))}\n")
        f.write(f"  Unique clusters: {np.unique(morpho_cluster_labels)}\n\n")
        
        f.write("Gene Expression Embeddings:\n")
        f.write(f"  Shape: {latent_b_np.shape}\n")
        f.write(f"  Mean: {latent_b_np.mean():.6f}\n")
        f.write(f"  Std: {latent_b_np.std():.6f}\n")
        f.write(f"  Min: {latent_b_np.min():.6f}\n")
        f.write(f"  Max: {latent_b_np.max():.6f}\n")
        f.write(f"  Number of clusters: {len(np.unique(gex_cluster_labels))}\n\n")
        
        f.write("Output Files:\n")
        f.write(f"  {morpho_output_path}\n")
        f.write(f"  {gex_output_path}\n")
        f.write(f"  {morpho_npy_path}\n")
        f.write(f"  {gex_npy_path}\n")
    
    print(f"\n✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("Embedding Extraction Complete!")
    print("="*80)
    print(f"\nAll files saved to: {output_dir}")
    
    return {
        'morpho_embeddings': latent_a_np,
        'gex_embeddings': latent_b_np,
        'morpho_clusters': morpho_cluster_labels,
        'gex_clusters': gex_cluster_labels,
        'morpho_path': morpho_output_path,
        'gex_path': gex_output_path
    }


def main():
    parser = argparse.ArgumentParser(description='Extract latent embeddings from trained model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save embeddings')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Verify paths exist
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    
    extract_embeddings(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()