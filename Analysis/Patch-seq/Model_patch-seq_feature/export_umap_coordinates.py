"""
Export UMAP coordinates for trained cross-modal model
"""
import torch
import numpy as np
import pandas as pd
import os
import argparse
from utils import get_config, apply_umap_reduction
from trainer import Trainer
from data_loader import CrossModalDataset
from sklearn.preprocessing import StandardScaler

def export_umap_coordinates(config_path, checkpoint_dir, output_dir, umap_params=None):
    """
    Export UMAP coordinates for both modalities
    
    Args:
        config_path: Path to config YAML file
        checkpoint_dir: Directory containing model checkpoints
        output_dir: Directory to save UMAP coordinates
        umap_params: Dict with UMAP parameters (n_neighbors, min_dist, metric)
    """
    # Load configuration
    config = get_config(config_path)
    
    # Create dataset
    print("Loading dataset...")
    dataset = CrossModalDataset()
    
    # Get full data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_a, data_b = dataset.get_full_data(device=device)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(config, dataset=dataset)
    trainer.to(device)
    
    # Load the final model checkpoint
    print("Loading model checkpoint...")
    from utils import get_model_list
    last_model_name = get_model_list(checkpoint_dir, "gen")
    
    if last_model_name is None:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
    
    state_dict = torch.load(last_model_name)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    
    print(f"Loaded checkpoint: {last_model_name}")
    
    # Get latent representations
    print("Generating latent representations...")
    with torch.no_grad():
        latent_a = trainer.gen_a.enc(data_a).cpu().numpy()  # Morphology
        latent_b = trainer.gen_b.enc(data_b).cpu().numpy()  # Gene expression
    
    print(f"Latent A (Morphology) shape: {latent_a.shape}")
    print(f"Latent B (Gene Expression) shape: {latent_b.shape}")
    
    # Set default UMAP parameters if not provided
    if umap_params is None:
        umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean'
        }
    
    print(f"Applying UMAP with parameters: {umap_params}")
    
    # Ensure same number of samples
    n_samples = min(latent_a.shape[0], latent_b.shape[0])
    latent_a = latent_a[:n_samples]
    latent_b = latent_b[:n_samples]
    
    # Combine and scale data for joint UMAP reduction
    matrix = np.vstack((latent_b, latent_a))  # Gene expression first, then morphology
    
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    
    # Apply UMAP
    print("Computing UMAP embeddings...")
    umap_coords = apply_umap_reduction(matrix_scaled, **umap_params)
    
    # Split back into two modalities
    umap_gex = umap_coords[:n_samples]  # Gene expression
    umap_morpho = umap_coords[n_samples:]  # Morphology
    
    print(f"UMAP Gene Expression shape: {umap_gex.shape}")
    print(f"UMAP Morphology shape: {umap_morpho.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get RNA family labels if available
    rna_labels = dataset.get_rna_family_labels()
    
    # Export Gene Expression UMAP coordinates
    gex_df = pd.DataFrame({
        'UMAP1': umap_gex[:, 0],
        'UMAP2': umap_gex[:, 1]
    })
    
    if rna_labels is not None:
        gex_df['RNA_Family'] = rna_labels[:n_samples]
    
    gex_output_path = os.path.join(output_dir, 'umap_gene_expression.csv')
    gex_df.to_csv(gex_output_path, index=False)
    print(f"Saved Gene Expression UMAP coordinates to: {gex_output_path}")
    
    # Export Morphology UMAP coordinates
    morpho_df = pd.DataFrame({
        'UMAP1': umap_morpho[:, 0],
        'UMAP2': umap_morpho[:, 1]
    })
    
    if rna_labels is not None:
        morpho_df['RNA_Family'] = rna_labels[:n_samples]
    
    morpho_output_path = os.path.join(output_dir, 'umap_morphology.csv')
    morpho_df.to_csv(morpho_output_path, index=False)
    print(f"Saved Morphology UMAP coordinates to: {morpho_output_path}")
    
    # Also save combined coordinates with modality labels
    combined_df = pd.DataFrame({
        'UMAP1': umap_coords[:, 0],
        'UMAP2': umap_coords[:, 1],
        'Modality': ['Gene_Expression'] * n_samples + ['Morphology'] * n_samples
    })
    
    if rna_labels is not None:
        combined_df['RNA_Family'] = np.concatenate([rna_labels[:n_samples], rna_labels[:n_samples]])
    
    combined_output_path = os.path.join(output_dir, 'umap_combined.csv')
    combined_df.to_csv(combined_output_path, index=False)
    print(f"Saved Combined UMAP coordinates to: {combined_output_path}")
    
    # Print summary statistics
    print("\n=== UMAP Coordinate Summary ===")
    print(f"Gene Expression UMAP1 range: [{umap_gex[:, 0].min():.3f}, {umap_gex[:, 0].max():.3f}]")
    print(f"Gene Expression UMAP2 range: [{umap_gex[:, 1].min():.3f}, {umap_gex[:, 1].max():.3f}]")
    print(f"Morphology UMAP1 range: [{umap_morpho[:, 0].min():.3f}, {umap_morpho[:, 0].max():.3f}]")
    print(f"Morphology UMAP2 range: [{umap_morpho[:, 1].min():.3f}, {umap_morpho[:, 1].max():.3f}]")
    
    return gex_output_path, morpho_output_path, combined_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export UMAP coordinates from trained model')
    parser.add_argument('--config', type=str, 
                       default='/home/users/turbodu/kzlinlab/projects/morpho_integration/git/morpho_integration/code/turbo/writeup14/cross_modal_VAE_v6/configs/attempt_1.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup14/cm_VAE_v6/outputs/attempt_1/checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str,
                       default='/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup14/cm_VAE_v6/outputs/attempt_1',
                       help='Directory to save UMAP coordinates')
    parser.add_argument('--n_neighbors', type=int, default=15,
                       help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1,
                       help='UMAP min_dist parameter')
    parser.add_argument('--metric', type=str, default='euclidean',
                       help='UMAP distance metric')
    
    args = parser.parse_args()
    
    umap_params = {
        'n_neighbors': args.n_neighbors,
        'min_dist': args.min_dist,
        'metric': args.metric
    }
    
    export_umap_coordinates(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        umap_params=umap_params
    )