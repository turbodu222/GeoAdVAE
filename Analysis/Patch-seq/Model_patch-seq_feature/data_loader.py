import torch
import torch.utils.data as utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CrossModalDataset(torch.utils.data.Dataset):
    """
    Unified dataset class that loads and manages all cross-modal data
    """
    def __init__(self, 
                 morphology_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/feature_paired_normalized.csv",
                 gene_expression_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/exon_norm_paired_2000.csv",
                 rna_family_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/rna_family_matched.csv",
                 morpho_cluster_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/cluster_label_morpho.csv",
                 gex_cluster_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/cluster_label_GEX.csv",
                 prior_matrix_path="/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/Corr_matrix.csv"):
        
        print("Loading cross-modal dataset...")
        
        # Load morphology data
        print("Loading morphology data...")
        morpho_df = pd.read_csv(morphology_path, header=0, index_col=0)
        self.morpho_data = morpho_df.to_numpy().astype(np.float32)
        
        # Load gene expression data
        print("Loading gene expression data...")
        gex_df = pd.read_csv(gene_expression_path, index_col=0)
        self.gex_data = gex_df.to_numpy().astype(np.float32)
        
        # === NEW: Handle NaN values BEFORE any processing ===
        print("Checking for NaN values...")
        
        # Check morphology data
        morpho_nan_count = np.isnan(self.morpho_data).sum()
        if morpho_nan_count > 0:
            print(f"  Warning: Found {morpho_nan_count} NaN values in morphology data")
            print(f"  Replacing NaN values with 0...")
            self.morpho_data = np.nan_to_num(self.morpho_data, nan=0.0)
            print(f"  ✓ Morphology NaN values replaced")
        else:
            print(f"  ✓ No NaN values in morphology data")
        
        # Check gene expression data
        gex_nan_count = np.isnan(self.gex_data).sum()
        if gex_nan_count > 0:
            print(f"  Warning: Found {gex_nan_count} NaN values in gene expression data")
            print(f"  Replacing NaN values with 0...")
            self.gex_data = np.nan_to_num(self.gex_data, nan=0.0)
            print(f"  ✓ Gene expression NaN values replaced")
        else:
            print(f"  ✓ No NaN values in gene expression data")
        
        # Ensure data sizes match
        min_samples = min(self.morpho_data.shape[0], self.gex_data.shape[0])
        self.morpho_data = self.morpho_data[:min_samples]
        self.gex_data = self.gex_data[:min_samples]
        self.n_samples = min_samples
        
        print(f"Dataset size: {self.n_samples} samples")
        print(f"Morphology dimensions: {self.morpho_data.shape[1]}")
        print(f"Gene expression dimensions: {self.gex_data.shape[1]}")
        
        # Standardize data (now safe after NaN removal)
        print("Standardizing data...")
        morpho_scaler = StandardScaler()
        gex_scaler = StandardScaler()
        self.morpho_data = morpho_scaler.fit_transform(self.morpho_data)
        self.gex_data = gex_scaler.fit_transform(self.gex_data)
        
        # === NEW: Verify no NaN after standardization ===
        if np.isnan(self.morpho_data).any():
            print("  ERROR: NaN values present after morphology standardization!")
            # This shouldn't happen, but handle it anyway
            self.morpho_data = np.nan_to_num(self.morpho_data, nan=0.0)
        
        if np.isnan(self.gex_data).any():
            print("  ERROR: NaN values present after gene expression standardization!")
            # This shouldn't happen, but handle it anyway
            self.gex_data = np.nan_to_num(self.gex_data, nan=0.0)
        
        print("  ✓ Standardization complete (no NaN values)")
        
        # Load RNA family labels (for evaluation)
        print("Loading RNA family labels...")
        try:
            rna_df = pd.read_csv(rna_family_path, header=0)
            if rna_df.shape[1] == 1:
                self.rna_family_labels = rna_df.iloc[:, 0].values
            else:
                self.rna_family_labels = rna_df.iloc[:, 1].values
            self.rna_family_labels = self.rna_family_labels[:min_samples]
            print(f"RNA family labels loaded: {len(np.unique(self.rna_family_labels))} unique types")
        except Exception as e:
            print(f"Warning: Could not load RNA family labels: {e}")
            self.rna_family_labels = None
        
        # Load morphology cluster labels
        print("Loading morphology cluster labels...")
        try:
            morpho_cluster_df = pd.read_csv(morpho_cluster_path, header=0)
            if morpho_cluster_df.shape[0] > 1 and morpho_cluster_df.shape[1] > 1:
                labels = morpho_cluster_df.iloc[1:, 1].values
            elif morpho_cluster_df.shape[1] == 1:
                labels = morpho_cluster_df.iloc[:, 0].values
            else:
                labels = morpho_cluster_df.iloc[:, 1].values
            
            # Convert to numeric
            self.morpho_cluster_labels = self._convert_to_numeric(labels)[:min_samples]
            print(f"Morphology clusters: {len(np.unique(self.morpho_cluster_labels))} unique clusters")
            print(f"Morphology cluster range: {self.morpho_cluster_labels.min()}-{self.morpho_cluster_labels.max()}")
        except Exception as e:
            print(f"Warning: Could not load morphology cluster labels: {e}")
            self.morpho_cluster_labels = np.zeros(min_samples, dtype=np.int32)
        
        # Load gene expression cluster labels
        print("Loading gene expression cluster labels...")
        try:
            gex_cluster_df = pd.read_csv(gex_cluster_path, header=0)
            if gex_cluster_df.shape[1] == 1:
                labels = gex_cluster_df.iloc[:, 0].values
            else:
                labels = gex_cluster_df.iloc[:, 1].values if 'cluster' in gex_cluster_df.columns[1].lower() else gex_cluster_df.iloc[:, 0].values
            
            # Convert to numeric
            self.gex_cluster_labels = self._convert_to_numeric(labels)[:min_samples]
            print(f"Gene expression clusters: {len(np.unique(self.gex_cluster_labels))} unique clusters")
            print(f"GEX cluster range: {self.gex_cluster_labels.min()}-{self.gex_cluster_labels.max()}")
        except Exception as e:
            print(f"Warning: Could not load GEX cluster labels: {e}")
            self.gex_cluster_labels = np.zeros(min_samples, dtype=np.int32)
        
        # Load prior correlation matrix
        print("Loading prior correlation matrix...")
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
        
        # === NEW: Final NaN check on tensors ===
        if torch.isnan(self.morpho_data).any():
            print("  ERROR: NaN in final morphology tensor!")
            self.morpho_data = torch.nan_to_num(self.morpho_data, nan=0.0)
        
        if torch.isnan(self.gex_data).any():
            print("  ERROR: NaN in final gene expression tensor!")
            self.gex_data = torch.nan_to_num(self.gex_data, nan=0.0)
        
        print("Dataset initialization completed successfully!")
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
        """Verify that all data components have consistent sizes"""
        print("\n=== Data Integrity Verification ===")
        print(f"Sample count: {self.n_samples}")
        print(f"Morphology data shape: {self.morpho_data.shape}")
        print(f"Gene expression data shape: {self.gex_data.shape}")
        print(f"Morphology cluster labels shape: {self.morpho_cluster_labels.shape}")
        print(f"GEX cluster labels shape: {self.gex_cluster_labels.shape}")
        
        if self.rna_family_labels is not None:
            print(f"RNA family labels shape: {len(self.rna_family_labels)}")
        
        # Verify consistency
        assert self.morpho_data.shape[0] == self.n_samples
        assert self.gex_data.shape[0] == self.n_samples
        assert self.morpho_cluster_labels.shape[0] == self.n_samples
        assert self.gex_cluster_labels.shape[0] == self.n_samples
        
        print("✓ All data components have consistent sizes")
        
        # === NEW: Verify no NaN values ===
        morpho_has_nan = torch.isnan(self.morpho_data).any().item()
        gex_has_nan = torch.isnan(self.gex_data).any().item()
        
        if morpho_has_nan or gex_has_nan:
            print("✗ WARNING: NaN values detected in final data!")
            if morpho_has_nan:
                print(f"  - Morphology data contains NaN")
            if gex_has_nan:
                print(f"  - Gene expression data contains NaN")
        else:
            print("✓ No NaN values in final data")
        
        # Print sample verification
        print("\n=== Sample Data Verification (First 5 samples) ===")
        for i in range(min(5, self.n_samples)):
            morpho_cluster = self.morpho_cluster_labels[i].item()
            gex_cluster = self.gex_cluster_labels[i].item()
            rna_family = self.rna_family_labels[i] if self.rna_family_labels is not None else "N/A"
            print(f"Sample {i}: Morpho_cluster={morpho_cluster}, GEX_cluster={gex_cluster}, RNA_family={rna_family}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Returns a sample containing all modalities and labels
        """
        return {
            'morpho_data': self.morpho_data[idx],
            'gex_data': self.gex_data[idx],
            'morpho_cluster': self.morpho_cluster_labels[idx],
            'gex_cluster': self.gex_cluster_labels[idx],
            'index': idx,
            'rna_family': self.rna_family_labels[idx] if self.rna_family_labels is not None else None
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
        """Get RNA family labels for evaluation"""
        return self.rna_family_labels

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