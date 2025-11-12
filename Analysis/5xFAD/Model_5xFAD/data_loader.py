import torch
import torch.utils.data as utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CrossModalDatasetUnpaired(torch.utils.data.Dataset):
    """
    Unpaired dataset class for unequal sample sizes
    Supports morphology (98 samples) and gene expression (30k samples)
    """
    def __init__(self, 
                 morphology_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/5xFAD/gw_dist.csv",
                 gene_expression_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/5xFAD/gex_matrix.csv",
                 morpho_cluster_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/5xFAD/Cluster_id.csv",
                 gex_cluster_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/5xFAD/microgliaClusterID.csv",
                 prior_matrix_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/Corr_matrix.csv",
                 use_prior_loss=True):
        
        print("Loading unpaired cross-modal dataset...")
        self.use_prior_loss = use_prior_loss
        
        # Load morphology data (98 samples)
        print("Loading morphology data...")
        morpho_df = pd.read_csv(morphology_path, header=0)
        self.morpho_data = morpho_df.iloc[:, 1:].to_numpy().astype(np.float32)
        self.n_morpho = self.morpho_data.shape[0]
        
        # Load gene expression data (30k samples)
        print("Loading gene expression data...")
        gex_df = pd.read_csv(gene_expression_path, header=None)
        self.gex_data = gex_df.iloc[1:, 1:].to_numpy().astype(np.float32)
        self.n_gex = self.gex_data.shape[0]
        
        print(f"Morphology samples: {self.n_morpho}")
        print(f"Gene expression samples: {self.n_gex}")
        print(f"Morphology dimensions: {self.morpho_data.shape[1]}")
        print(f"Gene expression dimensions: {self.gex_data.shape[1]}")
        
        # Standardize data separately for each modality
        print("Standardizing data...")
        morpho_scaler = StandardScaler()
        gex_scaler = StandardScaler()
        self.morpho_data = morpho_scaler.fit_transform(self.morpho_data)
        self.gex_data = gex_scaler.fit_transform(self.gex_data)
        
        # Always load cluster labels and prior matrix (regardless of use_prior_loss)
        # The use_prior_loss flag only controls whether they are used in training
        print(f"\nLoading cluster labels and prior matrix (use_prior_loss={self.use_prior_loss})...")
        
        # Load morphology cluster labels
        print("\nLoading morphology cluster labels...")
        try:
            morpho_cluster_df = pd.read_csv(morpho_cluster_path, header=0)
            print(f"Morphology cluster file shape: {morpho_cluster_df.shape}")
            print(f"Morphology cluster file columns: {morpho_cluster_df.columns.tolist()}")
            print(f"First 5 rows:")
            print(morpho_cluster_df.head())
            
            # Extract 'x' column
            if 'x' in morpho_cluster_df.columns:
                labels = morpho_cluster_df['x'].values
            else:
                raise ValueError(f"Column 'x' not found. Available: {morpho_cluster_df.columns.tolist()}")
            
            print(f"Extracted {len(labels)} raw morphology labels")
            print(f"First 10 raw labels: {labels[:10]}")
            print(f"Unique raw labels: {np.unique(labels)}")
            
            # Convert to numeric
            self.morpho_cluster_labels = self._convert_to_numeric(labels)[:self.n_morpho]
            
            unique_clusters = np.unique(self.morpho_cluster_labels)
            print(f"Morphology clusters: {len(unique_clusters)} unique clusters: {unique_clusters}")
            print(f"First 10 converted labels: {self.morpho_cluster_labels[:10]}")
            print(f"Cluster range: {self.morpho_cluster_labels.min()}-{self.morpho_cluster_labels.max()}")
            
            if len(unique_clusters) > 1:
                print(f"Cluster distribution: {np.bincount(self.morpho_cluster_labels)}")
            else:
                print(f"WARNING: Only one unique cluster value found: {unique_clusters}")
                
        except Exception as e:
            print(f"ERROR loading morphology cluster labels: {e}")
            import traceback
            traceback.print_exc()
            self.morpho_cluster_labels = np.zeros(self.n_morpho, dtype=np.int32)
            print("Fallback: Using all zeros")
        
        # Load gene expression cluster labels
        print("\nLoading gene expression cluster labels...")
        try:
            gex_cluster_df = pd.read_csv(gex_cluster_path, header=0)
            print(f"GEX cluster file shape: {gex_cluster_df.shape}")
            print(f"GEX cluster file columns: {gex_cluster_df.columns.tolist()}")
            print(f"First 5 rows:")
            print(gex_cluster_df.head())
            
            # Extract 'x' column
            if 'x' in gex_cluster_df.columns:
                labels = gex_cluster_df['x'].values
            else:
                raise ValueError(f"Column 'x' not found. Available: {gex_cluster_df.columns.tolist()}")
            
            print(f"Extracted {len(labels)} raw GEX labels")
            print(f"First 10 raw labels: {labels[:10]}")
            print(f"Unique raw labels (first 20): {np.unique(labels)[:20]}")
            
            # Convert to numeric
            self.gex_cluster_labels = self._convert_to_numeric(labels)[:self.n_gex]
            
            unique_clusters = np.unique(self.gex_cluster_labels)
            print(f"GEX clusters: {len(unique_clusters)} unique clusters")
            print(f"First 10 converted labels: {self.gex_cluster_labels[:10]}")
            print(f"Cluster range: {self.gex_cluster_labels.min()}-{self.gex_cluster_labels.max()}")
            
            if len(unique_clusters) > 1:
                cluster_counts = np.bincount(self.gex_cluster_labels)
                print(f"Cluster distribution (first 20):")
                for i, count in enumerate(cluster_counts[:20]):
                    if count > 0:
                        print(f"  Cluster {i}: {count} cells")
            else:
                print(f"WARNING: Only one unique cluster value found: {unique_clusters}")
                
        except Exception as e:
            print(f"ERROR loading GEX cluster labels: {e}")
            import traceback
            traceback.print_exc()
            self.gex_cluster_labels = np.zeros(self.n_gex, dtype=np.int32)
            print("Fallback: Using all zeros")
        
        # Load prior correlation matrix
        print("\nLoading prior correlation matrix...")
        try:
            prior_df = pd.read_csv(prior_matrix_path, index_col=0)
            self.prior_matrix = torch.tensor(prior_df.values.astype(np.float32), dtype=torch.float32)
            print(f"Prior matrix shape: {self.prior_matrix.shape}")
            print(f"Prior matrix range: {self.prior_matrix.min():.6f} to {self.prior_matrix.max():.6f}")
            print(f"Row indices (GEX clusters): {prior_df.index.tolist()}")
            print(f"Column indices (Morpho clusters): {prior_df.columns.tolist()}")
        except Exception as e:
            print(f"ERROR loading prior correlation matrix: {e}")
            import traceback
            traceback.print_exc()
            self.prior_matrix = None
            print("Fallback: Prior matrix set to None")
        
        # Summary
        if self.use_prior_loss:
            print(f"\n✓ Prior loss ENABLED - cluster labels and prior matrix will be used in training")
        else:
            print(f"\n✓ Prior loss DISABLED - cluster labels and prior matrix loaded but NOT used in training")
        
        # Convert to torch tensors
        self.morpho_data = torch.from_numpy(self.morpho_data).float()
        self.gex_data = torch.from_numpy(self.gex_data).float()
        self.morpho_cluster_labels = torch.from_numpy(self.morpho_cluster_labels).long()
        self.gex_cluster_labels = torch.from_numpy(self.gex_cluster_labels).long()
        
        # Create indices for epoch-based iteration
        self.morpho_indices = np.arange(self.n_morpho)
        self.gex_indices = np.arange(self.n_gex)
        self.current_gex_position = 0
        
        print("\nUnpaired dataset initialization completed successfully!")
        self._verify_data_integrity()
    
    def _convert_to_numeric(self, labels):
        """Convert string labels to numeric format"""
        numeric_labels = []
        for label in labels:
            if isinstance(label, str):
                import re
                numbers = re.findall(r'\d+', str(label))
                if numbers:
                    numeric_labels.append(int(numbers[0]))
                else:
                    numeric_labels.append(0)
            else:
                numeric_labels.append(int(label))
        return np.array(numeric_labels, dtype=np.int32)
    
    def _verify_data_integrity(self):
        """Verify data integrity for unpaired dataset"""
        print("\n=== Unpaired Data Integrity Verification ===")
        print(f"Morphology samples: {self.n_morpho}")
        print(f"Gene expression samples: {self.n_gex}")
        print(f"Morphology data shape: {self.morpho_data.shape}")
        print(f"Gene expression data shape: {self.gex_data.shape}")
        print(f"Use prior loss: {self.use_prior_loss}")
        
        # Basic data consistency checks
        assert self.morpho_data.shape[0] == self.n_morpho
        assert self.gex_data.shape[0] == self.n_gex
        
        # Always verify cluster labels (they are always loaded now)
        print(f"\nCluster labels verification:")
        print(f"Morphology cluster labels shape: {self.morpho_cluster_labels.shape}")
        print(f"GEX cluster labels shape: {self.gex_cluster_labels.shape}")
        
        assert self.morpho_cluster_labels.shape[0] == self.n_morpho
        
        # Handle GEX cluster label size mismatch
        if self.gex_cluster_labels.shape[0] != self.n_gex:
            print(f"WARNING: GEX cluster labels mismatch: {self.gex_cluster_labels.shape[0]} != {self.n_gex}")
            if self.gex_cluster_labels.shape[0] < self.n_gex:
                padding = self.n_gex - self.gex_cluster_labels.shape[0]
                padded_labels = torch.cat([self.gex_cluster_labels, torch.zeros(padding, dtype=torch.long)])
                self.gex_cluster_labels = padded_labels
                print(f"Padded to: {self.gex_cluster_labels.shape}")
            else:
                self.gex_cluster_labels = self.gex_cluster_labels[:self.n_gex]
                print(f"Truncated to: {self.gex_cluster_labels.shape}")
        
        print(f"✓ All data sizes verified")
        
        # Show sample data
        print(f"\n=== Sample Data ===")
        print("First 3 morphology samples:")
        for i in range(min(3, self.n_morpho)):
            print(f"  Morpho {i}: cluster={self.morpho_cluster_labels[i].item()}")
        
        print("First 3 gene expression samples:")
        for i in range(min(3, self.n_gex)):
            print(f"  GEX {i}: cluster={self.gex_cluster_labels[i].item()}")
        
        if self.use_prior_loss:
            print(f"\n✓ Prior loss enabled - these labels will be used in training")
        else:
            print(f"\n✓ Prior loss disabled - these labels loaded but not used in training")
    
    def shuffle_modalities(self):
        """Shuffle indices at the beginning of each epoch"""
        np.random.shuffle(self.morpho_indices)
        np.random.shuffle(self.gex_indices)
        self.current_gex_position = 0
        print("Shuffled morphology and gene expression indices for new epoch")
    
    def get_batch(self, batch_size=32, morpho_sampling='with_replacement', gex_sampling='sequential'):
        """
        Get a batch from both modalities
        
        Args:
            batch_size: Size of the batch
            morpho_sampling: 'with_replacement' or 'without_replacement'
            gex_sampling: 'sequential' (iterate through all 30k) or 'random'
        
        Returns:
            Dictionary containing batch data
        """
        # Sample morphology batch
        if morpho_sampling == 'with_replacement':
            morpho_batch_idx = np.random.choice(self.n_morpho, batch_size, replace=True)
        else:
            morpho_batch_idx = np.random.choice(self.n_morpho, min(batch_size, self.n_morpho), replace=False)
        
        # Sample gene expression batch
        if gex_sampling == 'sequential':
            end_position = self.current_gex_position + batch_size
            if end_position <= self.n_gex:
                gex_batch_idx = self.gex_indices[self.current_gex_position:end_position]
                self.current_gex_position = end_position
            else:
                remaining = self.n_gex - self.current_gex_position
                gex_batch_idx = np.concatenate([
                    self.gex_indices[self.current_gex_position:],
                    self.gex_indices[:batch_size - remaining]
                ])
                self.current_gex_position = batch_size - remaining
        else:
            gex_batch_idx = np.random.choice(self.n_gex, batch_size, replace=False)
        
        return {
            'morpho_data': self.morpho_data[morpho_batch_idx],
            'gex_data': self.gex_data[gex_batch_idx],
            'morpho_cluster': self.morpho_cluster_labels[morpho_batch_idx],
            'gex_cluster': self.gex_cluster_labels[gex_batch_idx],
            'morpho_indices': morpho_batch_idx,
            'gex_indices': gex_batch_idx
        }
    
    def get_full_data(self, device='cuda'):
        """Get all data as tensors for evaluation"""
        morpho_data = self.morpho_data.to(device) if torch.cuda.is_available() and device == 'cuda' else self.morpho_data
        gex_data = self.gex_data.to(device) if torch.cuda.is_available() and device == 'cuda' else self.gex_data
        return morpho_data, gex_data
    
    def get_prior_matrix(self, device='cuda'):
        """Get prior correlation matrix"""
        if self.prior_matrix is not None:
            return self.prior_matrix.to(device) if torch.cuda.is_available() and device == 'cuda' else self.prior_matrix
        return None
    
    def get_iterations_per_epoch(self, batch_size=32):
        """Calculate number of iterations needed to cover all gene expression data once"""
        return int(np.ceil(self.n_gex / batch_size))
    
    def __len__(self):
        """Return the number of gene expression samples (larger modality)"""
        return self.n_gex

# Alias for compatibility
CrossModalDataset = CrossModalDatasetUnpaired

class CrossModalDataLoader:
    """
    Custom data loader for unpaired cross-modal training
    """
    def __init__(self, dataset, batch_size=32, morpho_sampling='with_replacement', gex_sampling='sequential', shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.morpho_sampling = morpho_sampling
        self.gex_sampling = gex_sampling
        self.shuffle = shuffle
        
        # Calculate iterations per epoch based on gene expression data (larger modality)
        self.iterations_per_epoch = dataset.get_iterations_per_epoch(batch_size)
        
        print(f"Data loader initialized:")
        print(f"  Batch size: {batch_size}")
        print(f"  Morphology sampling: {morpho_sampling}")
        print(f"  Gene expression sampling: {gex_sampling}")
        print(f"  Iterations per epoch: {self.iterations_per_epoch}")
    
    def __iter__(self):
        """Iterator for the data loader"""
        # Shuffle at the beginning of each epoch if requested
        if self.shuffle:
            self.dataset.shuffle_modalities()
        
        for i in range(self.iterations_per_epoch):
            # Get batch data
            batch_data = self.dataset.get_batch(
                batch_size=self.batch_size,
                morpho_sampling=self.morpho_sampling,
                gex_sampling=self.gex_sampling
            )
            
            # Convert to the expected format for training
            batch = {
                'morpho_data': batch_data['morpho_data'],
                'gex_data': batch_data['gex_data'],
                'morpho_cluster': batch_data['morpho_cluster'],
                'gex_cluster': batch_data['gex_cluster'],
                'index': torch.arange(self.batch_size),
                'rna_family': None
            }
            
            yield batch
    
    def __len__(self):
        """Return the number of iterations per epoch"""
        return self.iterations_per_epoch

def create_data_loader(dataset, batch_size=32, shuffle=True, morpho_sampling='with_replacement', gex_sampling='sequential'):
    """
    Create a data loader for unpaired cross-modal training
    
    Args:
        dataset: CrossModalDatasetUnpaired instance
        batch_size: Size of each batch
        shuffle: Whether to shuffle data at epoch start
        morpho_sampling: Sampling strategy for morphology data
        gex_sampling: Sampling strategy for gene expression data
    
    Returns:
        CrossModalDataLoader instance
    """
    return CrossModalDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        morpho_sampling=morpho_sampling,
        gex_sampling=gex_sampling,
        shuffle=shuffle
    )