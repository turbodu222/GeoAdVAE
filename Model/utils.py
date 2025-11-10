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




def knn_label_accuracy(query_latent, reference_latent, labels, k, direction):
    """
    Calculate label matching accuracy using k-nearest neighbors
    
    Args:
        query_latent: Latent representations for query modality (numpy array)
        reference_latent: Latent representations for reference modality (numpy array) 
        labels: RNA type labels for each sample (numpy array)
        k: Number of nearest neighbors to consider
        direction: Direction of matching (e.g., "A→B" or "B→A")
    
    Returns:
        accuracy: Proportion of query samples where at least one of k-NN has matching label
    """
    if labels is None:
        return 0.0
    
    # Build k-NN model on reference modality
    nn = NearestNeighbors(n_neighbors=k, metric="l1")
    nn.fit(reference_latent)
    
    # For each sample in query modality, find k nearest neighbors in reference modality
    knn_indices = nn.kneighbors(query_latent, k, return_distance=False)
    
    match_count = 0
    for i in range(len(query_latent)):
        # Get label of current query sample
        query_label = labels[i]
        
        # Get labels of k nearest neighbors in reference modality
        neighbor_labels = labels[knn_indices[i]]
        
        # Check if any neighbor has the same label
        if query_label in neighbor_labels:
            match_count += 1
    
    # Accuracy = matched query samples / total query samples
    accuracy = match_count / len(query_latent)
    return accuracy


def write_knn(trainer, data_a, data_b, directory, suffix, dataset=None):
    latent_a = trainer.gen_a.enc(data_a).data.cpu().numpy()
    latent_b = trainer.gen_b.enc(data_b).data.cpu().numpy()
    
    if dataset is not None:
        rna_labels = dataset.get_rna_family_labels()
    else:
        rna_labels = load_rna_labels()  # Fallback to old method
    
    output_lines = [f"Iteration: {suffix}"]
    
    # Original k-NN accuracy (position matching)
    for k in [5, 50]:
        acc_ab, _ = knn_accuracy(latent_a, latent_b, k)
        acc_ba, _ = knn_accuracy(latent_b, latent_a, k)
        output_lines.append(f"{k}NN accuracy A→B: {acc_ab:.4f}")
        output_lines.append(f"{k}NN accuracy B→A: {acc_ba:.4f}")
    
    # New label matching accuracy
    if rna_labels is not None:
        output_lines.append("--- RNA Label Matching Accuracy ---")
        
        # A→B label matching (for each A sample, find k-NN in B and check label match)
        for k in [1, 3, 5, 10, 20]:
            label_acc_ab = knn_label_accuracy(latent_a, latent_b, rna_labels, k, "A→B")
            output_lines.append(f"{k}NN label accuracy A→B: {label_acc_ab:.4f}")
        
        # B→A label matching (for each B sample, find k-NN in A and check label match)  
        for k in [1, 3, 5, 10, 20]:
            label_acc_ba = knn_label_accuracy(latent_b, latent_a, rna_labels, k, "B→A")
            output_lines.append(f"{k}NN label accuracy B→A: {label_acc_ba:.4f}")
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


def generate_color_pairs(n_labels):
    """
    Generate pastel-solid color pairs for visualization
    Returns list of (pastel_color, solid_color) tuples
    """
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Base colors for generating pastel/solid pairs
    base_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange  
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#aec7e8',  # light blue
        '#ffbb78',  # light orange
        '#98df8a',  # light green
        '#ff9896',  # light red
        '#c5b0d5',  # light purple
    ]
    
    color_pairs = []
    for i in range(n_labels):
        base_color = base_colors[i % len(base_colors)]
        
        # Convert hex to RGB
        rgb = mcolors.hex2color(base_color)
        
        # Create pastel version (mix with white)
        pastel_rgb = tuple(0.4 * c + 0.6 for c in rgb)  # Mix with 60% white
        pastel_color = mcolors.to_hex(pastel_rgb)
        
        # Solid color is the original
        solid_color = base_color
        
        color_pairs.append((pastel_color, solid_color))
    
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


def plot_latent_visualization(a, b, outname1=None, outname2=None, outname=None, 
                            method='pca', umap_params=None, scale=True):
    """
    Improved plotting function with UMAP support and better color scheme
    
    Args:
        a: Morphology latent representations
        b: Gene expression latent representations  
        outname1: Output filename for morphology only plot
        outname2: Output filename for gene expression only plot
        outname: Output filename for combined plot
        method: 'pca' or 'umap'
        umap_params: dict with UMAP parameters
        scale: Whether to scale data before dimensionality reduction
    """
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
    
    # Create color mapping for celltypes
    if labels is not None:
        labels_truncated = labels[:n_cells]
        unique_labels = np.unique(labels_truncated)
        n_unique = len(unique_labels)
        
        # Generate pastel-solid color pairs
        color_pairs = generate_color_pairs(n_unique)
        color_map_pastel = {label: color_pairs[i][0] for i, label in enumerate(unique_labels)}
        color_map_solid = {label: color_pairs[i][1] for i, label in enumerate(unique_labels)}
        
        print(f"Found {n_unique} unique labels: {unique_labels}")
    else:
        labels_truncated = None
        print("No labels found, using default colors")

    # Plot Morphology only (pastel filled circles)
    if outname1:
        fig, ax = plt.subplots(figsize=(10, 8))
        if labels_truncated is not None:
            for label in unique_labels:
                mask = labels_truncated == label
                if np.any(mask):
                    morph_coords = comp[n_cells:][mask]  # morphology coordinates
                    ax.scatter(morph_coords[:, 0], morph_coords[:, 1], 
                             c=color_map_pastel[label], s=60, marker='o',
                             alpha=0.7, edgecolors='none', 
                             label=f'{label}')
        else:
            ax.scatter(comp[n_cells:, 0], comp[n_cells:, 1], s=60, marker='o',
                     c='lightblue', alpha=0.7, edgecolors='none', 
                     label='Morphology')
        
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_title(f"Morphology Embedding ({method.upper()})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(outname1, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot Gene Expression only (solid empty shapes)
    if outname2:
        fig, ax = plt.subplots(figsize=(10, 8))
        if labels_truncated is not None:
            for label in unique_labels:
                mask = labels_truncated == label
                if np.any(mask):
                    gene_coords = comp[:n_cells][mask]  # gene expression coordinates
                    ax.scatter(gene_coords[:, 0], gene_coords[:, 1], 
                             s=60, marker='^',
                             facecolors='none', edgecolors=color_map_solid[label], 
                             linewidth=2, alpha=0.8, label=f'{label}')
        else:
            ax.scatter(comp[:n_cells, 0], comp[:n_cells, 1], s=60, marker='^',
                     facecolors='none', edgecolors='darkorange', 
                     linewidth=2, alpha=0.8, label='Gene Expression')
        
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_title(f"Gene Expression ({method.upper()})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(outname2, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot both modalities together
    if outname:
        fig, ax = plt.subplots(figsize=(14, 10))
        if labels_truncated is not None:
            for label in unique_labels:
                mask = labels_truncated == label
                if np.any(mask):
                    # Morphology (pastel filled circles)
                    morph_coords = comp[n_cells:][mask]
                    ax.scatter(morph_coords[:, 0], morph_coords[:, 1], 
                             c=color_map_pastel[label], s=60, marker='o',
                             alpha=0.7, edgecolors='none', 
                             label=f'{label} (Morphology)')
                    
                    # Gene Expression (solid empty triangles)
                    gene_coords = comp[:n_cells][mask]
                    ax.scatter(gene_coords[:, 0], gene_coords[:, 1], 
                             s=60, marker='^',
                             facecolors='none', edgecolors=color_map_solid[label], 
                             linewidth=2, alpha=0.8, 
                             label=f'{label} (Gene Expression)')
        else:
            # Fallback without labels
            ax.scatter(comp[n_cells:, 0], comp[n_cells:, 1], s=60, marker='o',
                     c='lightblue', alpha=0.7, edgecolors='none', 
                     label='Morphology')
            ax.scatter(comp[:n_cells, 0], comp[:n_cells, 1], s=60, marker='^',
                     facecolors='none', edgecolors='darkorange', 
                     linewidth=2, alpha=0.8, label='Gene Expression')
        
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")  
        ax.set_title(f"Combined Morphology and Gene Expression ({method.upper()})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(outname, dpi=300, bbox_inches='tight')
        plt.close()
        

# Update the save_plots function to use the new visualization
def save_plots(trainer, data_a, data_b, directory, suffix, config=None):
    """
    Updated save_plots function with configurable visualization method
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
    
    # Use the new visualization function
    plot_latent_visualization(
        latent_a, latent_b,
        outname1=os.path.join(directory, f"_morphology_{suffix}.png"),
        outname2=os.path.join(directory, f"_gene_expression_{suffix}.png"), 
        outname=os.path.join(directory, f"_combined_{suffix}.png"),
        method=method,
        umap_params=umap_params
    )