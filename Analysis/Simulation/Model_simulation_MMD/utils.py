"""
Utils for data loading and model training
This code is based on https://github.com/NVlabs/MUNIT
"""

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch

import os
import math
import yaml
import numpy as np
import torch.nn.init as init

from scipy import sparse
from scipy.stats import percentileofscore
import torch.utils.data as utils
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from random import sample
from sklearn import metrics
import math

CONF = {}
data_a = None
data_b = None


directory_prefix = "processed_data/"
DATA_DIRECTORY = directory_prefix + "transcription_factor/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sample_count = None  # global tracker to unify row count

def load_data_from_csv(is_modality1=True):
    global sample_count

    if is_modality1:
        path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/gw_dist.csv"
        df = pd.read_csv(path, header=0)  
        data = df.iloc[:, 1:].to_numpy().astype(np.float32)
    else:
        path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/exon_data_top2000.csv"
        df = pd.read_csv(path, header=None)  
        data = df.iloc[:, 1:].to_numpy().astype(np.float32)

    if sample_count is None:
        sample_count = data.shape[0]
    else:
        data = data[:sample_count, :]

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    tensor_data = torch.from_numpy(data)
    return tensor_data.cuda() if torch.cuda.is_available() else tensor_data


def load_rna_labels():
    """
    Load RNA family labels from meta data file
    Returns:
        numpy.ndarray: Array of RNA type labels (e.g., ET, IT, etc.)
    """
    try:
        label_path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/rna_family_matched.csv"
        df = pd.read_csv(label_path, header=0)
        # Assume the labels are in the first column after any index column
        # Adjust column selection based on your CSV structure
        if df.shape[1] == 1:
            labels = df.iloc[:, 0].values
        else:
            # If there are multiple columns, assume labels are in the second column (first is index)
            labels = df.iloc[:, 1].values
        
        # Ensure we use the same sample count as the data
        global sample_count
        if sample_count is not None:
            labels = labels[:sample_count]
            
        return labels
    except Exception as e:
        print(f"Error loading RNA labels: {e}")
        return None


def get_cross_modal_data_loader(batch_size=32, shuffle=True):
    """
    Create unified cross-modal data loader
    """
    from data_loader import CrossModalDataset, create_data_loader
    
    # Create dataset
    dataset = CrossModalDataset()
    
    # Create data loader
    data_loader = create_data_loader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader, dataset

def load_rna_labels():
    """
    Updated function to use the unified dataset
    This function is kept for backward compatibility
    """
    try:
        from data_loader import CrossModalDataset
        dataset = CrossModalDataset()
        return dataset.get_rna_family_labels()
    except Exception as e:
        print(f"Error loading RNA labels: {e}")
        return None



def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and (
                   'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        try:
            value = getattr(trainer, m)
            
            # Skip dictionaries, lists, and any complex data structures
            if isinstance(value, (dict, list)):
                continue
                
            # Skip None values
            if value is None:
                continue
                
            # Convert tensor to scalar
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            # Only write numeric values to TensorBoard
            if isinstance(value, (int, float)) and train_writer is not None:
                # Check for NaN or Inf
                if not (math.isnan(value) or math.isinf(value)):
                    train_writer.add_scalar(m, value, iterations + 1)
                    
        except Exception as e:
            # Silently skip problematic attributes
            continue



# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


# Code for plotting latent space.


def plot_pca(a, b, outname1=None, outname2=None, outname=None, scale=True):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Load celltype labels
    labels = load_rna_labels()
    
    # Automatically detect the number of cells
    n_cells_a = len(a)  # Morphology
    n_cells_b = len(b)  # Gene expression
    
    print(f"Detected {n_cells_a} morphology samples and {n_cells_b} gene expression samples")
    
    # Use the minimum number to ensure consistency
    n_cells = min(n_cells_a, n_cells_b)
    print(f"Using {n_cells} samples for PCA visualization")
    
    # Truncate data to consistent size
    a_truncated = a[:n_cells]
    b_truncated = b[:n_cells]
    
    matrix = np.vstack((b_truncated, a_truncated))  # b(gene expression) first, then a(morphology)
    pca = PCA(n_components=2)
    scaled = matrix.copy()
    if scale:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

    comp = pca.fit_transform(scaled)  # shape: (2*n_cells, 2)
    
    # comp[:n_cells] = gene expression PCA coordinates
    # comp[n_cells:] = morphology PCA coordinates  
    # labels[:n_cells] = celltype labels for n_cells samples
    
    # Create color mapping for celltypes
    if labels is not None:
        # Truncate labels to match data size
        labels_truncated = labels[:n_cells]
        unique_labels = np.unique(labels_truncated)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    else:
        labels_truncated = None

    # Plot Modality A only (Morphology - empty circles)
    if outname1:
        fig, ax = plt.subplots(figsize=(8, 6))
        if labels_truncated is not None:
            for label in unique_labels:
                # Find samples with this label
                mask = labels_truncated == label
                if np.any(mask):
                    # Get morphology PCA coordinates for these samples
                    morph_coords = comp[n_cells:][mask]  # comp[n_cells:] is morphology
                    ax.scatter(morph_coords[:, 0], morph_coords[:, 1], 
                             s=50, marker='o',
                             facecolors='none', edgecolors=color_map[label], 
                             linewidth=2, label=f'{label}')
        else:
            ax.scatter(comp[n_cells:, 0], comp[n_cells:, 1], s=50, marker='o',
                     facecolors='none', edgecolors='purple', linewidth=2, 
                     label='Morphology')
        
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("Morphology Embedding (PCA)")
        ax.legend()
        plt.savefig(outname1, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot Modality B only (Gene Expression - empty triangles)
    if outname2:
        fig, ax = plt.subplots(figsize=(8, 6))
        if labels_truncated is not None:
            for label in unique_labels:
                # Find samples with this label
                mask = labels_truncated == label
                if np.any(mask):
                    # Get gene expression PCA coordinates for these samples
                    gene_coords = comp[:n_cells][mask]  # comp[:n_cells] is gene expression
                    ax.scatter(gene_coords[:, 0], gene_coords[:, 1], 
                             s=50, marker='^',
                             facecolors='none', edgecolors=color_map[label], 
                             linewidth=2, label=f'{label}')
        else:
            ax.scatter(comp[:n_cells, 0], comp[:n_cells, 1], s=50, marker='^',
                     facecolors='none', edgecolors='orange', linewidth=2,
                     label='Gene Expression')
        
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("Gene Expression (PCA)")
        ax.legend()
        plt.savefig(outname2, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot both modalities together
    if outname:
        fig, ax = plt.subplots(figsize=(12, 8))
        if labels_truncated is not None:
            # Plot each celltype separately
            for label in unique_labels:
                mask = labels_truncated == label
                if np.any(mask):
                    # Morphology (empty circles)
                    morph_coords = comp[n_cells:][mask]
                    ax.scatter(morph_coords[:, 0], morph_coords[:, 1], 
                             s=50, marker='o',
                             facecolors='none', edgecolors=color_map[label], 
                             linewidth=2, label=f'{label} (Morphology)')
                    
                    # Gene Expression (empty triangles)
                    gene_coords = comp[:n_cells][mask]
                    ax.scatter(gene_coords[:, 0], gene_coords[:, 1], 
                             s=50, marker='^',
                             facecolors='none', edgecolors=color_map[label], 
                             linewidth=2, label=f'{label} (Gene Expression)')
        else:
            # Fallback without celltype labels
            ax.scatter(comp[n_cells:, 0], comp[n_cells:, 1], s=50, marker='o',
                     facecolors='none', edgecolors='purple', linewidth=2, 
                     label='Morphology')
            ax.scatter(comp[:n_cells, 0], comp[:n_cells, 1], s=50, marker='^',
                     facecolors='none', edgecolors='orange', linewidth=2,
                     label='Gene Expression')
        
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("Combined Morphology and Gene Expression (PCA)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(outname, dpi=300, bbox_inches='tight')
        plt.close()


def plot_pca_both_spaces(a, b, outname, scale=True):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Load celltype labels
    labels = load_rna_labels()
    
    # Automatically detect the number of cells
    n_cells_a = len(a)  # Morphology
    n_cells_b = len(b)  # Gene expression
    
    print(f"Detected {n_cells_a} morphology samples and {n_cells_b} gene expression samples")
    
    # Use the minimum number to ensure consistency
    n_cells = min(n_cells_a, n_cells_b)
    print(f"Using {n_cells} samples for PCA visualization")
    
    # Truncate data to consistent size
    a_truncated = a[:n_cells]
    b_truncated = b[:n_cells]
    
    matrix = np.vstack((b_truncated, a_truncated))  # b(gene expression) first, then a(morphology)
    pca = PCA(n_components=2)
    scaled = matrix.copy()
    if scale:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

    comp = pca.fit_transform(scaled)  # shape: (2*n_cells, 2)
    
    # comp[:n_cells] = gene expression PCA coordinates
    # comp[n_cells:] = morphology PCA coordinates  
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if labels is not None:
        # Truncate labels to match data size
        labels_truncated = labels[:n_cells]
        unique_labels = np.unique(labels_truncated)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # Plot each celltype separately
        for label in unique_labels:
            mask = labels_truncated == label
            if np.any(mask):
                # Morphology (empty circles) - comp[n_cells:]
                morph_coords = comp[n_cells:][mask]
                ax.scatter(morph_coords[:, 0], morph_coords[:, 1], 
                         c=[color_map[label]], s=50, marker='o',
                         facecolors='none', edgecolors=[color_map[label]], 
                         linewidth=2, label=f'{label} (Morphology)')
                
                # Gene Expression (empty triangles) - comp[:n_cells]
                gene_coords = comp[:n_cells][mask]
                ax.scatter(gene_coords[:, 0], gene_coords[:, 1], 
                         c=[color_map[label]], s=50, marker='^',
                         facecolors='none', edgecolors=[color_map[label]], 
                         linewidth=2, label=f'{label} (Gene Expression)')
    else:
        # Fallback without celltype labels
        ax.scatter(comp[n_cells:, 0], comp[n_cells:, 1], s=50, marker='o',
                 facecolors='none', edgecolors='purple', linewidth=2, 
                 label='Morphology')
        ax.scatter(comp[:n_cells, 0], comp[:n_cells, 1], s=50, marker='^',
                 facecolors='none', edgecolors='orange', linewidth=2,
                 label='Gene Expression')
    
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Combined Morphology and Gene Expression (PCA)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.close()




def knn_label_accuracy(query_latent, reference_latent, labels, k, direction, filter_low_quality=False):
    """
    Calculate label matching accuracy using k-nearest neighbors
    
    Args:
        query_latent: Latent representations for query modality (numpy array)
        reference_latent: Latent representations for reference modality (numpy array) 
        labels: RNA type labels for each sample (numpy array)
        k: Number of nearest neighbors to consider
        direction: Direction of matching (e.g., "A→B" or "B→A")
        filter_low_quality: Whether to exclude 'low quality' samples from calculation
    
    Returns:
        accuracy: Proportion of (valid) query samples where at least one of k-NN has matching label
    """
    if labels is None:
        return 0.0
    
    if filter_low_quality:
        # Filter out 'low quality' samples from both query and reference
        valid_mask = labels != 'low quality'
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            print(f"Warning: No valid samples found for {direction} accuracy calculation")
            return 0.0
        
        # Filter query and reference data to only include valid samples
        query_latent_filtered = query_latent[valid_indices]
        reference_latent_filtered = reference_latent[valid_indices]
        labels_filtered = labels[valid_indices]
        
        # Adjust k if we have fewer valid samples than requested neighbors
        effective_k = min(k, len(valid_indices))
        if effective_k != k:
            print(f"Warning: Adjusted k from {k} to {effective_k} due to limited valid samples")
    else:
        # Use all samples
        query_latent_filtered = query_latent
        reference_latent_filtered = reference_latent
        labels_filtered = labels
        effective_k = k
    
    # Build k-NN model on (filtered) reference modality
    nn = NearestNeighbors(n_neighbors=effective_k, metric="l1")
    nn.fit(reference_latent_filtered)
    
    # For each sample in query modality, find k nearest neighbors in reference modality
    knn_indices = nn.kneighbors(query_latent_filtered, effective_k, return_distance=False)
    
    match_count = 0
    total_samples = len(query_latent_filtered)
    
    for i in range(total_samples):
        # Get label of current query sample
        query_label = labels_filtered[i]
        
        # Skip 'low quality' samples in counting (even when not filtering the dataset)
        if filter_low_quality and query_label == 'low quality':
            continue
            
        # Get labels of k nearest neighbors in reference modality
        neighbor_labels = labels_filtered[knn_indices[i]]
        
        # Check if any neighbor has the same label
        if query_label in neighbor_labels:
            match_count += 1
    
    # Calculate accuracy based on whether we're filtering
    if filter_low_quality:
        # Accuracy = matched valid query samples / total valid query samples
        accuracy = match_count / total_samples if total_samples > 0 else 0.0
        print(f"  {direction} evaluation: {total_samples} valid samples, "
              f"{match_count} matches, accuracy: {accuracy:.4f}")
    else:
        # Original calculation including all samples
        accuracy = match_count / total_samples if total_samples > 0 else 0.0
    
    return accuracy


def write_knn(trainer, data_a, data_b, directory, suffix, dataset=None, config=None):
    """
    Calculate and write k-NN accuracy metrics with optional low quality filtering
    """
    latent_a = trainer.gen_a.enc(data_a).data.cpu().numpy()
    latent_b = trainer.gen_b.enc(data_b).data.cpu().numpy()
    
    if dataset is not None:
        rna_labels = dataset.get_rna_family_labels()
    else:
        rna_labels = load_rna_labels()  # Fallback to old method
    
    # Get filter setting from config
    filter_low_quality = False
    if config is not None and 'filter_low_quality' in config:
        filter_low_quality = config['filter_low_quality']
    
    output_lines = [f"Iteration: {suffix}"]
    
    # Display filter setting
    if filter_low_quality:
        output_lines.append("Low quality filtering: ENABLED (excluding 'low quality' from accuracy calculation)")
    else:
        output_lines.append("Low quality filtering: DISABLED (including all samples in accuracy calculation)")
    
    # Check if labels are available and count valid samples
    if rna_labels is not None:
        total_samples = len(rna_labels)
        valid_mask = rna_labels != 'low quality'
        valid_samples = np.sum(valid_mask)
        low_quality_samples = total_samples - valid_samples
        
        output_lines.append(f"Total samples: {total_samples}")
        output_lines.append(f"High quality samples: {valid_samples}")
        output_lines.append(f"Low quality samples: {low_quality_samples}")
        output_lines.append("")
        
        if filter_low_quality and valid_samples == 0:
            output_lines.append("WARNING: No valid samples found for evaluation!")
            # Write warning and return
            for line in output_lines:
                print(line)
            with open(os.path.join(directory, "knn_accuracy.txt"), "a") as myfile:
                for line in output_lines:
                    myfile.write(line + "\n")
            return
    
    # Original k-NN accuracy (position matching) - keeping for backward compatibility
    for k in [5, 50]:
        acc_ab, _ = knn_accuracy(latent_a, latent_b, k)
        acc_ba, _ = knn_accuracy(latent_b, latent_a, k)
        output_lines.append(f"{k}NN accuracy A→B: {acc_ab:.4f}")
        output_lines.append(f"{k}NN accuracy B→A: {acc_ba:.4f}")
    
    # New label matching accuracy with optional filtering
    if rna_labels is not None:
        if filter_low_quality:
            output_lines.append("--- RNA Label Matching Accuracy (excluding 'low quality') ---")
        else:
            output_lines.append("--- RNA Label Matching Accuracy (including all samples) ---")
        
        # A→B label matching (morphology to gene expression)
        for k in [1, 3, 5, 10, 20]:
            label_acc_ab = knn_label_accuracy(latent_a, latent_b, rna_labels, k, "A→B", filter_low_quality)
            output_lines.append(f"{k}NN label accuracy A→B: {label_acc_ab:.4f}")
        
        # B→A label matching (gene expression to morphology)
        for k in [1, 3, 5, 10, 20]:
            label_acc_ba = knn_label_accuracy(latent_b, latent_a, rna_labels, k, "B→A", filter_low_quality)
            output_lines.append(f"{k}NN label accuracy B→A: {label_acc_ba:.4f}")
            
        # Add summary of cell types
        if filter_low_quality:
            valid_labels = rna_labels[valid_mask]
            unique_valid_labels = np.unique(valid_labels)
            output_lines.append(f"Cell types evaluated: {list(unique_valid_labels)}")
        else:
            unique_labels = np.unique(rna_labels)
            output_lines.append(f"Cell types evaluated: {list(unique_labels)}")
            
    else:
        output_lines.append("Warning: RNA labels could not be loaded")
    
    output_lines.append("")  # Add empty line for separation
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Write to file
    with open(os.path.join(directory, "knn_accuracy.txt"), "a") as myfile:
        for line in output_lines:
            myfile.write(line + "\n")


def knn_accuracy(A, B, k):
    nn = NearestNeighbors(n_neighbors=k, metric="l1")
    nn.fit(A)
    transp_nearest_neighbor = nn.kneighbors(B, 1, return_distance=False)
    actual_nn = nn.kneighbors(A, k, return_distance=False)

    match_count = 0
    for i in range(len(transp_nearest_neighbor)):
        if transp_nearest_neighbor[i][0] in actual_nn[i]:
            match_count += 1

    accuracy = match_count / len(B)
    return accuracy, 0


def save_loss_curves(loss_history, output_directory):
    """
    Save training loss curves showing KL, GW, Recon, GAN, Prior, and Total losses
    """
    import matplotlib.pyplot as plt
    
    iterations = loss_history['iterations']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Updated colors and labels to include prior loss
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'lightgray']
    labels = ['KL Loss', 'GW Loss', 'Recon Loss (Total)', 'Recon Loss A (Morphology)', 'Recon Loss B (Gene Expression)', 'GAN Loss', 'Prior Loss', 'Total Loss']
    loss_keys = ['kl_loss', 'gw_loss', 'recon_loss', 'recon_loss_a', 'recon_loss_b', 'gan_loss', 'prior_loss', 'total_loss']
    
    
    for i, (key, label, color) in enumerate(zip(loss_keys, labels, colors)):
        if key in loss_history:  # Check if the loss exists
            losses = loss_history[key]
            
            # Determine starting index for plotting
            start_idx = 0
            if key in ['gw_loss', 'gan_loss', 'prior_loss']:
                for j, loss_val in enumerate(losses):
                    if loss_val > 0:
                        start_idx = j
                        break
            
            # Plot the loss curve starting from the first non-zero value
            if start_idx < len(iterations):
                ax.plot(iterations[start_idx:], losses[start_idx:], 
                       color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves (Raw Values)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    loss_curve_path = os.path.join(output_directory, 'training_loss_curves.png')
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to: {loss_curve_path}")
    
    # Log scale version
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (key, label, color) in enumerate(zip(loss_keys, labels, colors)):
        if key in loss_history:
            losses = loss_history[key]
            
            start_idx = 0
            if key in ['gw_loss', 'gan_loss', 'prior_loss']:
                for j, loss_val in enumerate(losses):
                    if loss_val > 0:
                        start_idx = j
                        break
            
            if start_idx < len(iterations):
                pos_losses = [max(l, 1e-8) for l in losses[start_idx:]]
                ax.semilogy(iterations[start_idx:], pos_losses, 
                           color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss Curves (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    log_loss_curve_path = os.path.join(output_directory, 'training_loss_curves_log.png')
    plt.savefig(log_loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Log-scale loss curves saved to: {log_loss_curve_path}")

def save_weighted_loss_curves(weighted_loss_history, output_directory):
    """
    Save weighted training loss curves including prior loss
    """
    import matplotlib.pyplot as plt
    
    iterations = weighted_loss_history['iterations']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'lightgray']
    labels = ['Weighted KL Loss', 'Weighted GW Loss', 'Weighted Recon Loss (Total)', 
              'Weighted Recon Loss A (Morphology)', 'Weighted Recon Loss B (Gene Expression)',
              'Weighted GAN Loss', 'Weighted Prior Loss', 'Total Loss']
    loss_keys = ['kl_loss', 'gw_loss', 'recon_loss', 'recon_loss_a', 'recon_loss_b', 'gan_loss', 'prior_loss', 'total_loss']
    
    for i, (key, label, color) in enumerate(zip(loss_keys, labels, colors)):
        if key in weighted_loss_history:
            losses = weighted_loss_history[key]
            
            start_idx = 0
            if key in ['gw_loss', 'gan_loss', 'prior_loss']:
                for j, loss_val in enumerate(losses):
                    if loss_val > 0:
                        start_idx = j
                        break
            
            if start_idx < len(iterations):
                ax.plot(iterations[start_idx:], losses[start_idx:], 
                       color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weighted Loss')
    ax.set_title('Training Loss Curves (Weighted Values)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    weighted_loss_curve_path = os.path.join(output_directory, 'training_weighted_loss_curves.png')
    plt.savefig(weighted_loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Weighted loss curves saved to: {weighted_loss_curve_path}")
    
    # Log scale version with prior loss
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (key, label, color) in enumerate(zip(loss_keys, labels, colors)):
        if key in weighted_loss_history:
            losses = weighted_loss_history[key]
            
            start_idx = 0
            if key in ['gw_loss', 'gan_loss', 'prior_loss']:
                for j, loss_val in enumerate(losses):
                    if loss_val > 0:
                        start_idx = j
                        break
            
            if start_idx < len(iterations):
                pos_losses = [max(l, 1e-8) for l in losses[start_idx:]]
                ax.semilogy(iterations[start_idx:], pos_losses, 
                           color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weighted Loss (log scale)')
    ax.set_title('Training Loss Curves (Weighted Values - Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    weighted_log_loss_curve_path = os.path.join(output_directory, 'training_weighted_loss_curves_log.png')
    plt.savefig(weighted_log_loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Weighted log-scale loss curves saved to: {weighted_log_loss_curve_path}")



####----------Prior Correction Matrix----------####

def load_gex_cluster_labels():
    """
    Load gene expression cluster labels for prior loss computation
    Returns:
        numpy.ndarray: Array of gene expression cluster labels
    """
    try:
        label_path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/cluster_label_GEX.csv"
        df = pd.read_csv(label_path, header=0)
        
        # Get the labels
        if df.shape[1] == 1:
            labels = df.iloc[:, 0].values
        else:
            labels = df.iloc[:, 1].values if 'cluster' in df.columns[1].lower() else df.iloc[:, 0].values
        
        # Convert to numeric - handle different string formats
        numeric_labels = []
        for label in labels:
            if isinstance(label, str):
                # Extract numbers from strings like "cluster_0", "0", etc.
                import re
                numbers = re.findall(r'\d+', str(label))
                if numbers:
                    numeric_labels.append(int(numbers[0]))
                else:
                    print(f"Warning: Could not extract number from label '{label}', using 0")
                    numeric_labels.append(0)
            else:
                # Already numeric
                numeric_labels.append(int(label))
        
        labels = np.array(numeric_labels, dtype=np.int32)
        
        # Ensure we use the same sample count as the data
        global sample_count
        if sample_count is not None:
            labels = labels[:sample_count]
            
        print(f"Loaded {len(labels)} gene expression cluster labels")
        print(f"Label range: {labels.min()}-{labels.max()}")
        print(f"First few labels: {labels[:5]}")
        print(f"Label dtype: {labels.dtype}")
        return labels
        
    except Exception as e:
        print(f"Error loading gene expression cluster labels: {e}")
        return None


def load_morpho_cluster_labels():
    """
    Load morphology cluster labels for prior loss computation
    Returns:
        numpy.ndarray: Array of morphology cluster labels
    """
    try:
        label_path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/cluster_label_morpho.csv"
        df = pd.read_csv(label_path, header=0)
        # Get the labels (adjust based on your CSV structure)
        if df.shape[0] > 1 and df.shape[1] > 1:
            labels = df.iloc[1:, 1].values
        elif df.shape[1] == 1:
            labels = df.iloc[:, 0].values
        else:
            labels = df.iloc[:, 1].values
        
        # Convert to numeric - handle different string formats
        numeric_labels = []
        for label in labels:
            if isinstance(label, str):
                # Extract numbers from strings like "cluster_0", "0", etc.
                import re
                numbers = re.findall(r'\d+', str(label))
                if numbers:
                    numeric_labels.append(int(numbers[0]))
                else:
                    print(f"Warning: Could not extract number from label '{label}', using 0")
                    numeric_labels.append(0)
            else:
                # Already numeric
                numeric_labels.append(int(label))
        
        labels = np.array(numeric_labels, dtype=np.int32)
        
        # Ensure we use the same sample count as the data
        global sample_count
        if sample_count is not None:
            labels = labels[:sample_count]
            
        print(f"Loaded {len(labels)} morphology cluster labels")
        print(f"Label range: {labels.min()}-{labels.max()}")
        print(f"First few labels: {labels[:5]}")
        print(f"Label dtype: {labels.dtype}")
        return labels
        
    except Exception as e:
        print(f"Error loading morphology cluster labels: {e}")
        return None

def load_prior_correlation_matrix(prior_matrix_path):
    """
    Load prior correlation matrix for cross-modal alignment
    Args:
        prior_matrix_path: Path to the prior correlation matrix CSV file
    Returns:
        torch.Tensor: Prior correlation matrix (n_gex_clusters, n_morpho_clusters)
    """
    import pandas as pd
    import numpy as np
    
    try:
        # Load the correlation matrix CSV with explicit parsing
        prior_df = pd.read_csv(prior_matrix_path, index_col=0)
        
        print(f"Prior matrix raw shape: {prior_df.shape}")
        print(f"Prior matrix columns: {prior_df.columns.tolist()}")
        print(f"Prior matrix index: {prior_df.index.tolist()}")
        
        # Ensure all values are numeric
        prior_matrix_np = prior_df.values.astype(np.float32)
        
        # Create torch tensor with explicit float32 type
        prior_matrix = torch.tensor(prior_matrix_np, dtype=torch.float32)
        
        print(f"Loaded prior correlation matrix with shape: {prior_matrix.shape}")
        print(f"Prior matrix dtype: {prior_matrix.dtype}")
        print(f"Prior matrix value range: {prior_matrix.min():.6f} to {prior_matrix.max():.6f}")
        print(f"GEX cluster names: {prior_df.index.tolist()}")
        print(f"Morpho cluster names: {prior_df.columns.tolist()}")
        
        # Verify no NaN or inf values
        if torch.isnan(prior_matrix).any():
            print("Warning: Prior matrix contains NaN values!")
        if torch.isinf(prior_matrix).any():
            print("Warning: Prior matrix contains infinite values!")
            
        return prior_matrix
        
    except Exception as e:
        print(f"Error loading prior correlation matrix: {e}")
        print(f"Error type: {type(e).__name__}")
        raise



#-----we don't use clustering in the current version, but keep the code here for future reference-----#
def get_cluster_assignments(latent_representations, n_clusters, method='kmeans'):
    """
    Get cluster assignments for latent representations
    Args:
        latent_representations: Latent representations (n_samples, latent_dim)
        n_clusters: Number of clusters
        method: Clustering method
    Returns:
        torch.Tensor: Cluster assignments (n_samples,)
    """
    from sklearn.cluster import KMeans
    import numpy as np
    
    if isinstance(latent_representations, torch.Tensor):
        latent_np = latent_representations.detach().cpu().numpy()
    else:
        latent_np = latent_representations
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_np)
    
    return torch.tensor(cluster_labels, dtype=torch.long)



def generate_color_pairs_for_simulation(unique_labels):
    """
    Generate color pairs for simulation data using Red, Green, Blue scheme
    Returns dictionary mapping labels to (light_color, dark_color) tuples
    For morphology: lighter shades, triangles
    For gene expression: darker shades, circles
    """
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Define RGB color scheme for simulation (cluster 0, 1, 2)
    # Darker shades for GEX, lighter shades for Morphology
    simulation_colors = {
        0: ('#FF6B6B', '#FFCCCC'),  # (darker red, lighter red)
        1: ('#4CAF50', '#C8E6C9'),  # (darker green, lighter green)
        2: ('#2196F3', '#BBDEFB'),  # (darker blue, lighter blue)
    }
    
    color_pairs = {}
    
    for label in unique_labels:
        # Try to convert label to int for simulation data
        try:
            label_int = int(label)
            if label_int in simulation_colors:
                dark_color, light_color = simulation_colors[label_int]
            else:
                # Fallback for unexpected cluster numbers
                dark_color = '#808080'  # gray
                light_color = '#D3D3D3'  # light gray
        except (ValueError, TypeError):
            # If label is not numeric, use default colors
            dark_color = '#808080'
            light_color = '#D3D3D3'
        
        # Return as (light_color, dark_color) - light for morphology, dark for GEX
        color_pairs[label] = (light_color, dark_color)
    
    return color_pairs


def generate_color_pairs_with_celltype_mapping(unique_labels):
    """
    Generate color pairs with specific mapping for known cell types
    Returns dictionary mapping labels to (pastel_color, solid_color) tuples
    """
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Specific color mapping for known cell types
    celltype_colors = {
        'CT': '#196343',
        'ET': '#02455B', 
        'IT': '#319D8F',
        'Lamp5': '#D76E8D',
        'NP': '#397D71',
        'Pvalb': '#A02F4A',
        'Sncg': '#933E94',
        'Sst': '#C37631',
        'Vip': '#B703AD',
        'low quality': 'lightgrey'
    }
    
    # Default colors for other labels
    default_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    ]
    
    color_pairs = {}
    default_color_idx = 0
    
    for label in unique_labels:
        if label in celltype_colors:
            # Use predefined colors for known cell types
            solid_color = celltype_colors[label]
            
            if solid_color == 'lightgrey':
                # Special handling for low quality
                pastel_color = 'lightgrey'
            else:
                # Convert hex to RGB for pastel version
                rgb = mcolors.hex2color(solid_color)
                # Create pastel version (mix with white) - less white mixing to maintain distinction
                pastel_rgb = tuple(0.6 * c + 0.4 for c in rgb)  # Mix with 40% white instead of 60%
                pastel_color = mcolors.to_hex(pastel_rgb)
        else:
            # Use default colors for unknown labels
            base_color = default_colors[default_color_idx % len(default_colors)]
            default_color_idx += 1
            
            # Convert hex to RGB
            rgb = mcolors.hex2color(base_color)
            # Create pastel version
            pastel_rgb = tuple(0.4 * c + 0.6 for c in rgb)
            pastel_color = mcolors.to_hex(pastel_rgb)
            solid_color = base_color
        
        color_pairs[label] = (pastel_color, solid_color)
    
    return color_pairs


def apply_umap_reduction(data, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """
    Apply UMAP dimensionality reduction
    """
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        embedding = reducer.fit_transform(data)
        return embedding
    except ImportError:
        print("Warning: UMAP not installed. Please install with: pip install umap-learn")
        print("Falling back to PCA...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        return pca.fit_transform(data)


def plot_latent_visualization_with_celltype_colors(a, b, outname1=None, outname2=None, outname=None, 
                                                   method='pca', umap_params=None, scale=True, 
                                                   use_simulation_colors=False, dataset=None):
    """
    Plotting function with specific cell type colors for UMAP visualization
    Shows ALL samples including 'low quality' in visualization
    
    Args:
        a: Morphology latent representations
        b: Gene expression latent representations  
        outname1: Output filename for morphology only plot
        outname2: Output filename for gene expression only plot
        outname: Output filename for combined plot
        method: 'pca' or 'umap'
        umap_params: dict with UMAP parameters
        scale: Whether to scale data before dimensionality reduction
        use_simulation_colors: If True, use red-green-blue color scheme for simulation data
        dataset: Dataset object to get cluster labels from
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Get cluster labels - prioritize dataset labels for simulation
    if dataset is not None and hasattr(dataset, 'gex_cluster_labels'):
        # Use data-driven cluster labels from dataset
        labels = dataset.gex_cluster_labels.cpu().numpy()
        print(f"Using cluster labels from dataset")
        use_simulation_colors = True  # Automatically use simulation colors for cluster data
    else:
        # Load RNA family labels for patch-seq data
        labels = load_rna_labels()
        print(f"Using RNA family labels")
    
    # Automatically detect the number of cells
    n_cells_a = len(a)  # Morphology
    n_cells_b = len(b)  # Gene expression
    
    print(f"Detected {n_cells_a} morphology samples and {n_cells_b} gene expression samples")
    
    # Use the minimum number to ensure consistency
    n_cells = min(n_cells_a, n_cells_b)
    print(f"Using {n_cells} samples for {method.upper()} visualization")
    
    # Truncate data to consistent size
    a_truncated = a[:n_cells]
    b_truncated = b[:n_cells]
    
    # Combine data for joint dimensionality reduction
    matrix = np.vstack((b_truncated, a_truncated))  # gene expression first, then morphology
    
    # Scale data if requested
    if scale:
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)
    
    # Apply dimensionality reduction
    if method.lower() == 'umap':
        if umap_params is None:
            umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean'}
        comp = apply_umap_reduction(matrix, **umap_params)
        print(f"Applied UMAP with parameters: {umap_params}")
    else:  # PCA
        pca = PCA(n_components=2)
        comp = pca.fit_transform(matrix)
        print(f"Applied PCA - explained variance: {pca.explained_variance_ratio_}")
    
    # comp[:n_cells] = gene expression coordinates
    # comp[n_cells:] = morphology coordinates
    
    # Export coordinates to CSV files
    if outname:
        # Extract directory from outname
        import os
        output_dir = os.path.dirname(outname)
        base_name = os.path.basename(outname).replace('.png', '')
        
        # Save gene expression coordinates
        gex_coords_file = os.path.join(output_dir, f"{base_name}_gex_coordinates.csv")
        gex_df = pd.DataFrame(comp[:n_cells], columns=[f'{method.upper()}_1', f'{method.upper()}_2'])
        if labels is not None:
            gex_df['cluster'] = labels[:n_cells]
        gex_df.to_csv(gex_coords_file, index=False)
        print(f"Saved GEX {method.upper()} coordinates to: {gex_coords_file}")
        
        # Save morphology coordinates
        morph_coords_file = os.path.join(output_dir, f"{base_name}_morph_coordinates.csv")
        morph_df = pd.DataFrame(comp[n_cells:], columns=[f'{method.upper()}_1', f'{method.upper()}_2'])
        if labels is not None:
            morph_df['cluster'] = labels[:n_cells]
        morph_df.to_csv(morph_coords_file, index=False)
        print(f"Saved Morphology {method.upper()} coordinates to: {morph_coords_file}")
    
    # Create color mapping for celltypes
    if labels is not None:
        labels_truncated = labels[:n_cells]
        unique_labels = np.unique(labels_truncated)
        n_unique = len(unique_labels)
        
        # Choose color mapping based on data type
        if all(isinstance(label, (int, np.integer)) or str(label).isdigit() for label in unique_labels):
            color_pairs = generate_color_pairs_for_simulation(unique_labels)
            print(f"Using simulation color scheme (Red-Green-Blue)")
            print(f"Clusters detected: {unique_labels}")
        else:
            color_pairs = generate_color_pairs_with_celltype_mapping(unique_labels)
            print(f"Using cell type specific color scheme")
        
        print(f"Found {n_unique} unique labels: {unique_labels}")
    else:
        labels_truncated = None
        print("No labels found, using default colors")

    # Plot Morphology only (lighter colors, triangles)
    if outname1:
        fig, ax = plt.subplots(figsize=(10, 8))
        if labels_truncated is not None:
            for label in unique_labels:
                mask = labels_truncated == label
                if np.any(mask):
                    morph_coords = comp[n_cells:][mask]  # morphology coordinates
                    light_color, _ = color_pairs[label]  # light color for morphology
                    ax.scatter(morph_coords[:, 0], morph_coords[:, 1], 
                             c=light_color, s=100, marker='^',  # Triangles for morphology
                             alpha=0.8, edgecolors='black', linewidth=0.5,
                             label=f'Cluster {label}')
        else:
            ax.scatter(comp[n_cells:, 0], comp[n_cells:, 1], s=100, marker='^',
                     c='lightblue', alpha=0.7, edgecolors='black', linewidth=0.5,
                     label='Morphology')
        
        ax.set_xlabel(f"{method.upper()} 1", fontsize=14)
        ax.set_ylabel(f"{method.upper()} 2", fontsize=14)
        ax.set_title(f"Morphology Embedding ({method.upper()})", fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(outname1, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved morphology plot to: {outname1}")

    # Plot Gene Expression only (darker colors, circles)
    if outname2:
        fig, ax = plt.subplots(figsize=(10, 8))
        if labels_truncated is not None:
            for label in unique_labels:
                mask = labels_truncated == label
                if np.any(mask):
                    gene_coords = comp[:n_cells][mask]  # gene expression coordinates
                    _, dark_color = color_pairs[label]  # dark color for GEX
                    ax.scatter(gene_coords[:, 0], gene_coords[:, 1], 
                             c=dark_color, s=100, marker='o',  # Circles for GEX
                             alpha=0.8, edgecolors='black', linewidth=0.5,
                             label=f'Cluster {label}')
        else:
            ax.scatter(comp[:n_cells, 0], comp[:n_cells, 1], s=100, marker='o',
                     c='darkorange', alpha=0.8, edgecolors='black', linewidth=0.5,
                     label='Gene Expression')
        
        ax.set_xlabel(f"{method.upper()} 1", fontsize=14)
        ax.set_ylabel(f"{method.upper()} 2", fontsize=14)
        ax.set_title(f"Gene Expression ({method.upper()})", fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.savefig(outname2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved gene expression plot to: {outname2}")

    # Plot both modalities together
    if outname:
        fig, ax = plt.subplots(figsize=(14, 10))
        if labels_truncated is not None:
            for label in unique_labels:
                mask = labels_truncated == label
                if np.any(mask):
                    light_color, dark_color = color_pairs[label]
                    
                    # Morphology (lighter colors, triangles)
                    morph_coords = comp[n_cells:][mask]
                    ax.scatter(morph_coords[:, 0], morph_coords[:, 1], 
                             c=light_color, s=100, marker='^',
                             alpha=0.8, edgecolors='black', linewidth=0.5,
                             label=f'Cluster {label} (Morphology)')
                    
                    # Gene Expression (darker colors, circles)
                    gene_coords = comp[:n_cells][mask]
                    ax.scatter(gene_coords[:, 0], gene_coords[:, 1], 
                             c=dark_color, s=100, marker='o',
                             alpha=0.8, edgecolors='black', linewidth=0.5,
                             label=f'Cluster {label} (Gene Expression)')
        else:
            # Fallback without labels
            ax.scatter(comp[n_cells:, 0], comp[n_cells:, 1], s=100, marker='^',
                     c='lightblue', alpha=0.7, edgecolors='black', linewidth=0.5,
                     label='Morphology')
            ax.scatter(comp[:n_cells, 0], comp[:n_cells, 1], s=100, marker='o',
                     c='darkorange', alpha=0.8, edgecolors='black', linewidth=0.5,
                     label='Gene Expression')
        
        ax.set_xlabel(f"{method.upper()} 1", fontsize=14)
        ax.set_ylabel(f"{method.upper()} 2", fontsize=14)  
        ax.set_title(f"Combined Morphology and Gene Expression ({method.upper()})", fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved combined plot to: {outname}")

        

# Update the save_plots function to use the new visualization
def save_plots_with_celltype_colors(trainer, data_a, data_b, directory, suffix, config=None, dataset=None):
    """
    Save plots function with cell type specific colors or simulation colors
    
    Args:
        trainer: Trainer object
        data_a: Morphology data
        data_b: Gene expression data
        directory: Output directory
        suffix: File suffix (e.g., iteration number or 'final')
        config: Configuration dictionary
        dataset: Dataset object (for simulation data with cluster labels)
    """
    latent_a = trainer.gen_a.enc(data_a).data.cpu().numpy()
    latent_b = trainer.gen_b.enc(data_b).data.cpu().numpy()

    # Get visualization method from config
    method = 'pca'  # default
    umap_params = None
    
    if config and 'visualization' in config:
        vis_config = config['visualization']
        method = vis_config.get('method', 'pca')
        
        if method.lower() == 'umap':
            umap_params = {
                'n_neighbors': vis_config.get('umap_n_neighbors', 15),
                'min_dist': vis_config.get('umap_min_dist', 0.1),
                'metric': vis_config.get('umap_metric', 'euclidean')
            }
    
    # Use the visualization function with cell type colors
    plot_latent_visualization_with_celltype_colors(
        latent_a, latent_b,
        outname1=os.path.join(directory, f"_morphology_{suffix}.png"),
        outname2=os.path.join(directory, f"_gene_expression_{suffix}.png"), 
        outname=os.path.join(directory, f"_combined_{suffix}.png"),
        method=method,
        umap_params=umap_params,
        dataset=dataset  # Pass dataset for cluster labels
    )