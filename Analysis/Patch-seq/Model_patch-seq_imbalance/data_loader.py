import torch
import torch.utils.data as utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CrossModalDataset(torch.utils.data.Dataset):
    """
    Unified dataset class for unbalanced cross-modal data (different sample counts)
    Morphology: 645 samples, Gene Expression: 1329 samples
    """
    def __init__(self, 
                 morphology_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/gw_dist.csv",
                 gene_expression_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/exon_norm_full.csv",
                 rna_family_morpho_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/rna_family_morpho.csv",
                 rna_family_gex_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/rna_family_gex.csv",
                 morpho_cluster_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/cluster_label_morpho.csv",
                 gex_cluster_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/cluster_label_GEX_full.csv",
                 prior_matrix_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/Corr_matrix.csv"):
        
        print("Loading unbalanced cross-modal dataset...")
        
        # Load morphology data (645 samples)
        print("Loading morphology data...")
        morpho_df = pd.read_csv(morphology_path, header=0)
        self.morpho_data = morpho_df.iloc[:, 1:].to_numpy().astype(np.float32)
        self.n_morpho = self.morpho_data.shape[0]
        print(f"Morphology: {self.n_morpho} samples, {self.morpho_data.shape[1]} dimensions")
        
        # Load gene expression data (1329 samples) with NA handling
        print("Loading gene expression data...")
        gex_df = pd.read_csv(gene_expression_path, header=0, index_col=0)
        self.gex_data = gex_df.to_numpy().astype(np.float32)
        
        # Handle NA values in GEX data
        print(f"GEX data shape: {self.gex_data.shape}")
        nan_count = np.isnan(self.gex_data).sum()
        if nan_count > 0:
            total_elements = self.gex_data.size
            nan_percentage = (nan_count / total_elements) * 100
            print(f"Found {nan_count} NaN values ({nan_percentage:.2f}%)")
            print("Replacing NaN values with 0...")
            self.gex_data = np.nan_to_num(self.gex_data, nan=0.0)
            print("NaN values replaced successfully")
        
        self.n_gex = self.gex_data.shape[0]
        print(f"Gene expression: {self.n_gex} samples, {self.gex_data.shape[1]} dimensions")
        
        # Verify unbalanced setup
        print(f"\n=== Unbalanced Dataset Configuration ===")
        print(f"Morphology samples: {self.n_morpho}")
        print(f"Gene expression samples: {self.n_gex}")
        print(f"Ratio (GEX/Morpho): {self.n_gex / self.n_morpho:.2f}:1")
        
        # Standardize data separately
        print("\nStandardizing data...")
        morpho_scaler = StandardScaler()
        gex_scaler = StandardScaler()
        self.morpho_data = morpho_scaler.fit_transform(self.morpho_data)
        self.gex_data = gex_scaler.fit_transform(self.gex_data)
        
        # Load RNA family labels SEPARATELY for each modality
        print("\nLoading RNA family labels...")
        
        # Morphology RNA family labels (645 samples)
        try:
            rna_morpho_df = pd.read_csv(rna_family_morpho_path, header=0)
            if rna_morpho_df.shape[1] == 1:
                self.rna_family_morpho = rna_morpho_df.iloc[:, 0].values
            else:
                self.rna_family_morpho = rna_morpho_df.iloc[:, 1].values
            self.rna_family_morpho = self.rna_family_morpho[:self.n_morpho]
            print(f"Morphology RNA family labels: {len(self.rna_family_morpho)} samples, {len(np.unique(self.rna_family_morpho))} unique types")
        except Exception as e:
            print(f"Warning: Could not load morphology RNA family labels: {e}")
            self.rna_family_morpho = None
        
        # GEX RNA family labels (1329 samples)
        try:
            rna_gex_df = pd.read_csv(rna_family_gex_path, header=0)
            if rna_gex_df.shape[1] == 1:
                self.rna_family_gex = rna_gex_df.iloc[:, 0].values
            else:
                self.rna_family_gex = rna_gex_df.iloc[:, 1].values
            self.rna_family_gex = self.rna_family_gex[:self.n_gex]
            print(f"GEX RNA family labels: {len(self.rna_family_gex)} samples, {len(np.unique(self.rna_family_gex))} unique types")
        except Exception as e:
            print(f"Warning: Could not load GEX RNA family labels: {e}")
            self.rna_family_gex = None
        
        # Load morphology cluster labels (645 samples)
        print("\nLoading morphology cluster labels...")
        try:
            morpho_cluster_df = pd.read_csv(morpho_cluster_path, header=0)
            if morpho_cluster_df.shape[0] > 1 and morpho_cluster_df.shape[1] > 1:
                labels = morpho_cluster_df.iloc[1:, 1].values
            elif morpho_cluster_df.shape[1] == 1:
                labels = morpho_cluster_df.iloc[:, 0].values
            else:
                labels = morpho_cluster_df.iloc[:, 1].values
            
            self.morpho_cluster_labels = self._convert_to_numeric(labels)[:self.n_morpho]
            print(f"Morphology clusters: {len(np.unique(self.morpho_cluster_labels))} unique clusters")
            print(f"Morphology cluster range: {self.morpho_cluster_labels.min()}-{self.morpho_cluster_labels.max()}")
        except Exception as e:
            print(f"Warning: Could not load morphology cluster labels: {e}")
            self.morpho_cluster_labels = np.zeros(self.n_morpho, dtype=np.int32)
        
        # Load gene expression cluster labels (1329 samples)
        print("Loading gene expression cluster labels...")
        try:
            gex_cluster_df = pd.read_csv(gex_cluster_path, header=0)
            if gex_cluster_df.shape[1] == 1:
                labels = gex_cluster_df.iloc[:, 0].values
            else:
                labels = gex_cluster_df.iloc[:, 1].values if 'cluster' in gex_cluster_df.columns[1].lower() else gex_cluster_df.iloc[:, 0].values
            
            self.gex_cluster_labels = self._convert_to_numeric(labels)[:self.n_gex]
            print(f"Gene expression clusters: {len(np.unique(self.gex_cluster_labels))} unique clusters")
            print(f"GEX cluster range: {self.gex_cluster_labels.min()}-{self.gex_cluster_labels.max()}")
        except Exception as e:
            print(f"Warning: Could not load GEX cluster labels: {e}")
            self.gex_cluster_labels = np.zeros(self.n_gex, dtype=np.int32)
        
        # Load prior correlation matrix
        print("\nLoading prior correlation matrix...")
        try:
            prior_df = pd.read_csv(prior_matrix_path, index_col=0)
            self.prior_matrix = torch.tensor(prior_df.values.astype(np.float32), dtype=torch.float32)
            print(f"Prior matrix shape: {self.prior_matrix.shape}")
            print(f"Prior matrix range: {self.prior_matrix.min():.6f} to {self.prior_matrix.max():.6f}")
        except Exception as e:
            print(f"Warning: Could not load prior correlation matrix: {e}")
            self.prior_matrix = None
        
        # Convert to torch tensors
        self.morpho_data = torch.from_numpy(self.morpho_data).float()
        self.gex_data = torch.from_numpy(self.gex_data).float()
        self.morpho_cluster_labels = torch.from_numpy(self.morpho_cluster_labels).long()
        self.gex_cluster_labels = torch.from_numpy(self.gex_cluster_labels).long()
        
        print("\nDataset initialization completed successfully!")
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
        """Verify that all data components have correct sizes"""
        print("\n=== Data Integrity Verification ===")
        print(f"Morphology samples: {self.n_morpho}")
        print(f"Gene expression samples: {self.n_gex}")
        print(f"Morphology data shape: {self.morpho_data.shape}")
        print(f"Gene expression data shape: {self.gex_data.shape}")
        print(f"Morphology cluster labels shape: {self.morpho_cluster_labels.shape}")
        print(f"GEX cluster labels shape: {self.gex_cluster_labels.shape}")
        
        if self.rna_family_morpho is not None:
            print(f"RNA family labels (morpho) shape: {len(self.rna_family_morpho)}")
        if self.rna_family_gex is not None:
            print(f"RNA family labels (gex) shape: {len(self.rna_family_gex)}")
        
        # Verify consistency for each modality
        assert self.morpho_data.shape[0] == self.n_morpho
        assert self.gex_data.shape[0] == self.n_gex
        assert self.morpho_cluster_labels.shape[0] == self.n_morpho
        assert self.gex_cluster_labels.shape[0] == self.n_gex
        
        print("✓ All data components have correct sizes for unbalanced setup")
        
        # Print sample verification for both modalities
        print("\n=== Sample Data Verification ===")
        print("Morphology samples (first 3):")
        for i in range(min(3, self.n_morpho)):
            morpho_cluster = self.morpho_cluster_labels[i].item()
            rna_family = self.rna_family_morpho[i] if self.rna_family_morpho is not None else "N/A"
            print(f"  Morpho {i}: cluster={morpho_cluster}, RNA_family={rna_family}")
        
        print("Gene expression samples (first 3):")
        for i in range(min(3, self.n_gex)):
            gex_cluster = self.gex_cluster_labels[i].item()
            rna_family = self.rna_family_gex[i] if self.rna_family_gex is not None else "N/A"
            print(f"  GEX {i}: cluster={gex_cluster}, RNA_family={rna_family}")
    
    def __len__(self):
        """
        For unbalanced datasets, return the LARGER sample count to ensure all data is used
        """
        return max(self.n_morpho, self.n_gex)
    
    def __getitem__(self, idx):
        """
        Returns a sample containing data from both modalities
        For unbalanced data: cycle through the smaller dataset
        """
        # For morphology (smaller dataset), cycle through indices
        morpho_idx = idx % self.n_morpho
        
        # For GEX (larger dataset), use direct indexing up to available samples
        gex_idx = idx % self.n_gex
        
        return {
            'morpho_data': self.morpho_data[morpho_idx],
            'gex_data': self.gex_data[gex_idx],
            'morpho_cluster': self.morpho_cluster_labels[morpho_idx],
            'gex_cluster': self.gex_cluster_labels[gex_idx],
            'morpho_index': morpho_idx,
            'gex_index': gex_idx,
            'rna_family_morpho': self.rna_family_morpho[morpho_idx] if self.rna_family_morpho is not None else None,
            'rna_family_gex': self.rna_family_gex[gex_idx] if self.rna_family_gex is not None else None
        }
    
    def get_full_data(self, device='cuda'):
        """Get all data as tensors for full-batch operations"""
        morpho_data = self.morpho_data.to(device) if torch.cuda.is_available() and device == 'cuda' else self.morpho_data
        gex_data = self.gex_data.to(device) if torch.cuda.is_available() and device == 'cuda' else self.gex_data
        return morpho_data, gex_data
    
    def get_prior_matrix(self, device='cuda'):
        """Get prior correlation matrix"""
        if self.prior_matrix is not None:
            return self.prior_matrix.to(device) if torch.cuda.is_available() and device == 'cuda' else self.prior_matrix
        return None
    
    def get_rna_family_labels(self):
        """
        Get RNA family labels for both modalities
        Returns: dict with 'morpho' and 'gex' keys
        """
        return {
            'morpho': self.rna_family_morpho,
            'gex': self.rna_family_gex
        }

def create_data_loader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a data loader for the cross-modal dataset
    """
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )