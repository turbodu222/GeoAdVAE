import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import pandas as pd
from scipy.stats import vonmises, gamma
import random

# Set random seed for reproducibility
np.random.seed(42)

def sigmoid(x):
    """Standard sigmoid function"""
    return 1 / (1 + np.exp(-x))

def generate_gene_expression_data(n_samples=1000, output_dir=None):
    """
    Generate gene expression data with 3 distinct clusters.
    
    Based on the specification:
    - C1 (pyramidal-like): g1~N(2,0.2²), g2~N(0,0.2²), g3~N(1.2,0.2²)
    - C2 (multipolar): g1~N(-2,0.2²), g2~N(2,0.2²), g3~N(-1,0.2²)
    - C3 (bipolar): g1~N(0,0.2²), g2~N(-1,0.2²), g3~N(2,0.2²)
    
    Returns:
        tuple: (gene_expression_matrix, cluster_labels, control_parameters)
    """
    samples_per_cluster = n_samples // 3
    remaining = n_samples % 3
    
    # Initialize arrays
    gene_expression = []
    cluster_labels = []
    control_params = []
    
    # Cluster 1: Pyramidal-like
    n_c1 = samples_per_cluster + (1 if remaining > 0 else 0)
    for i in range(n_c1):
        g1 = np.random.normal(2.0, 0.2)
        g2 = np.random.normal(0.0, 0.2)
        g3 = np.random.normal(1.2, 0.2)
        
        gene_expression.append([g1, g2, g3])
        cluster_labels.append(1)
        
        # Map to control parameters using α=1.5, β=0
        P = sigmoid(1.5 * g1)  # High polarity (should be ~0.95+)
        D = sigmoid(1.5 * g2)  # Medium proximal density (~0.5)
        A = sigmoid(1.5 * g3)  # Medium-high anisotropy (~0.85)
        control_params.append([P, D, A])
    
    # Cluster 2: Multipolar
    n_c2 = samples_per_cluster + (1 if remaining > 1 else 0)
    for i in range(n_c2):
        g1 = np.random.normal(-2.0, 0.2)
        g2 = np.random.normal(2.0, 0.2)
        g3 = np.random.normal(-1.0, 0.2)
        
        gene_expression.append([g1, g2, g3])
        cluster_labels.append(2)
        
        P = sigmoid(1.5 * g1)  # Low polarity (~0.05)
        D = sigmoid(1.5 * g2)  # High proximal density (~0.95)
        A = sigmoid(1.5 * g3)  # Low anisotropy (~0.2)
        control_params.append([P, D, A])
    
    # Cluster 3: Bipolar
    n_c3 = samples_per_cluster
    for i in range(n_c3):
        g1 = np.random.normal(0.0, 0.2)
        g2 = np.random.normal(-1.0, 0.2)
        g3 = np.random.normal(2.0, 0.2)
        
        gene_expression.append([g1, g2, g3])
        cluster_labels.append(3)
        
        P = sigmoid(1.5 * g1)  # Medium polarity (~0.5)
        D = sigmoid(1.5 * g2)  # Low proximal density (~0.2)
        A = sigmoid(1.5 * g3)  # High anisotropy (~0.95)
        control_params.append([P, D, A])
    
    gene_expression = np.array(gene_expression)
    cluster_labels = np.array(cluster_labels)
    control_params = np.array(control_params)
    
    # Save gene expression data
    if output_dir:
        gene_df = pd.DataFrame(gene_expression, columns=['gene1', 'gene2', 'gene3'])
        gene_df.to_csv(os.path.join(output_dir, "gene_expression.csv"), index=False)
        
        # Save cluster assignments
        cluster_df = pd.DataFrame({
            'neuron_id': np.arange(1, len(cluster_labels) + 1),
            'cluster': cluster_labels
        })
        cluster_df.to_csv(os.path.join(output_dir, "cluster_assignments.csv"), index=False)
    
    print(f"Generated gene expression data:")
    print(f"  Cluster 1 (pyramidal): {n_c1} samples")
    print(f"  Cluster 2 (multipolar): {n_c2} samples")
    print(f"  Cluster 3 (bipolar): {n_c3} samples")
    
    return gene_expression, cluster_labels, control_params

class NeuronMorphologyGeneratorV3:
    """Generate neuronal morphology based on detailed specifications"""
    
    def __init__(self, P, D, A, cluster_type):
        """
        Initialize generator with control parameters.
        
        P: Polarity/hierarchy (0-1)
        D: Proximal branching density (0-1)
        A: Anisotropy (0-1)
        cluster_type: 1 (pyramidal), 2 (multipolar), or 3 (bipolar)
        """
        self.P = P
        self.D = D
        self.A = A
        self.cluster_type = cluster_type
        
        # Node tracking
        self.nodes = []
        self.node_id = 1
        
    def _add_node(self, x, y, z, radius, parent_id, node_type):
        """Add a node to the morphology"""
        self.nodes.append([self.node_id, node_type, x, y, z, radius, parent_id])
        current_id = self.node_id
        self.node_id += 1
        return current_id
    
    def _generate_pyramidal(self):
        """Generate pyramidal morphology with strong apical trunk and prominent tuft"""
        # Create soma
        soma_id = self._add_node(0, 0, 0, 3.0, -1, 1)
        
        # Parameters scaled by actual unit
        SCALE = 10.0  # Scale factor for realistic sizes
        
        # APICAL TRUNK - single main trunk going up
        trunk_length = 6.0 * SCALE* random.uniform(0.8, 1.2)  # 60 units total length
        n_trunk_segments = 20  # Number of segments before tuft
        segment_length = trunk_length / n_trunk_segments
        
        # Build main trunk
        trunk_ids = [soma_id]
        current_pos = np.array([0, 0, 0])
        
        for i in range(n_trunk_segments):
            # Slight variation in direction but mainly upward
            direction = np.array([np.random.normal(0, 0.02), 1, np.random.normal(0, 0.02)])
            direction = direction / np.linalg.norm(direction)
            
            current_pos = current_pos + direction * segment_length
            node_id = self._add_node(current_pos[0], current_pos[1], current_pos[2], 
                                    1.5 - 0.5 * i/n_trunk_segments, trunk_ids[-1], 4)
            trunk_ids.append(node_id)
        
        # APICAL TUFT - extensive branching at the end
        tuft_queue = deque([(trunk_ids[-1], current_pos, np.array([0, 1, 0]), 0)])
        max_tuft_depth = 8
        
        while tuft_queue:
            parent_id, parent_pos, parent_dir, depth = tuft_queue.popleft()
            
            if depth >= max_tuft_depth:
                continue
            
            # High branching probability in tuft
            n_branches = 3 if depth < 2 else 2
            
            for j in range(n_branches):
                # Wide angles for tuft spread
                theta = np.random.uniform(-0.8, 0.8)
                phi = np.random.uniform(0, 2*np.pi)
                
                new_dir = np.array([
                    np.sin(theta) * np.cos(phi),
                    0.6 + 0.4 * np.cos(theta),  # Still upward bias
                    np.sin(theta) * np.sin(phi)
                ])
                new_dir = new_dir / np.linalg.norm(new_dir)
                
                # Decreasing segment length in tuft
                seg_length = segment_length * 0.6 * (1 - depth/max_tuft_depth)
                new_pos = parent_pos + new_dir * seg_length
                
                new_id = self._add_node(new_pos[0], new_pos[1], new_pos[2],
                                      0.5, parent_id, 4)
                
                # Continue branching with decreasing probability
                if np.random.random() < 0.8 - 0.1 * depth:
                    tuft_queue.append((new_id, new_pos, new_dir, depth + 1))
        
        # BASAL DENDRITES - spreading horizontally from soma
        n_basal = 6  # Fixed number for consistency
        basal_angles = np.linspace(0, 2*np.pi, n_basal, endpoint=False)
        
        for angle in basal_angles:
            # Initial basal direction - horizontal with slight downward
            direction = np.array([np.cos(angle), -0.2, np.sin(angle)])
            direction = direction / np.linalg.norm(direction)
            
            # Build basal dendrite
            current_pos = np.array([0, 0, 0])
            parent_id = soma_id
            
            for depth in range(8):  # 8 segments per basal
                current_pos = current_pos + direction * 0.8 * SCALE* random.uniform(0.8, 1.2)
                node_id = self._add_node(current_pos[0], current_pos[1], current_pos[2],
                                       1.0 - 0.1 * depth, parent_id, 3)
                parent_id = node_id
                
                # Branch occasionally
                if depth > 2 and np.random.random() < 0.3:
                    # Side branch
                    branch_dir = direction + np.random.normal(0, 0.3, 3)
                    branch_dir = branch_dir / np.linalg.norm(branch_dir)
                    branch_pos = current_pos
                    
                    for b_depth in range(4):
                        branch_pos = branch_pos + branch_dir * 0.5 * SCALE* random.uniform(0.8, 1.2)
                        b_id = self._add_node(branch_pos[0], branch_pos[1], branch_pos[2],
                                            0.5, parent_id if b_depth == 0 else b_id, 3)
                
                # Add some variation to direction
                direction = direction + np.random.normal(0, 0.1, 3)
                direction[1] = min(0, direction[1])  # Keep downward tendency
                direction = direction / np.linalg.norm(direction)
    
    def _generate_multipolar(self):
        """Generate multipolar morphology with dense bush-like structure"""
        # Create soma
        soma_id = self._add_node(0, 0, 0, 3.0, -1, 1)
        
        # Parameters
        SCALE = 5.0  # Smaller scale for compact structure
        
        # MULTIPLE PRIMARY DENDRITES from soma
        n_primary = 10  # 10 primary dendrites for dense structure
        
        for i in range(n_primary):
            # Random 3D directions
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            
            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
            # Build each primary dendrite with dense branching
            current_pos = np.array([0, 0, 0])
            parent_id = soma_id
            
            # Each primary dendrite
            branch_queue = deque([(parent_id, current_pos, direction, 0)])
            
            while branch_queue:
                parent, pos, dir_vec, depth = branch_queue.popleft()
                
                if depth >= 6:  # Limited depth for compact structure
                    continue
                
                # High branching probability near soma
                branch_prob = 0.7 * np.exp(-depth/2)
                
                if np.random.random() < branch_prob and depth > 0:
                    # Branch into 2-3
                    n_branches = np.random.choice([2, 3])
                    
                    for b in range(n_branches):
                        # Wide branching angles
                        angle = np.random.uniform(-0.6, 0.6)
                        rot_axis = np.random.randn(3)
                        rot_axis = rot_axis / np.linalg.norm(rot_axis)
                        
                        # Simple rotation
                        new_dir = dir_vec + angle * rot_axis
                        new_dir = new_dir / np.linalg.norm(new_dir)
                        
                        seg_length = SCALE * random.uniform(0.8, 1.2)* (0.8 - 0.1 * depth)
                        new_pos = pos + new_dir * seg_length
                        
                        new_id = self._add_node(new_pos[0], new_pos[1], new_pos[2],
                                              0.8 - 0.1 * depth, parent, 3)
                        
                        if depth < 5:
                            branch_queue.append((new_id, new_pos, new_dir, depth + 1))
                else:
                    # Continue growth
                    new_dir = dir_vec + np.random.normal(0, 0.2, 3)
                    new_dir = new_dir / np.linalg.norm(new_dir)
                    
                    seg_length = SCALE * random.uniform(0.8, 1.2)* (0.8 - 0.1 * depth)
                    new_pos = pos + new_dir * seg_length
                    
                    new_id = self._add_node(new_pos[0], new_pos[1], new_pos[2],
                                          0.8 - 0.1 * depth, parent, 3)
                    
                    if depth < 5:
                        branch_queue.append((new_id, new_pos, new_dir, depth + 1))
    
    def _generate_bipolar(self):
        """Generate bipolar morphology with two opposite trunks and PROMINENT tufts at BOTH ends"""
        # Create soma
        soma_id = self._add_node(0, 0, 0, 3.0, -1, 1)
        
        # Parameters
        SCALE = 12.0  # Larger scale for more elongated structure
        
        # TWO OPPOSITE MAIN TRUNKS with GUARANTEED TUFTS
        for direction_sign in [1, -1]:
            # Build trunk going up or down
            trunk_length = 6.0 * SCALE* random.uniform(0.8, 1.2)  # 72 units in each direction
            n_trunk_segments = 18  # More segments before tuft
            segment_length = trunk_length / n_trunk_segments
            
            trunk_ids = [soma_id]
            current_pos = np.array([0, 0, 0])
            
            # Build main trunk - VERY STRAIGHT
            for i in range(n_trunk_segments):
                # Almost perfectly straight along Y axis
                direction = np.array([
                    np.random.normal(0, 0.005),  # Even less X deviation
                    direction_sign,  # Pure Y direction
                    np.random.normal(0, 0.005)   # Even less Z deviation
                ])
                direction = direction / np.linalg.norm(direction)
                
                current_pos = current_pos + direction * segment_length
                node_id = self._add_node(current_pos[0], current_pos[1], current_pos[2],
                                    1.5 - 0.2 * i/n_trunk_segments, trunk_ids[-1], 4)
                trunk_ids.append(node_id)
            
            # PROMINENT TUFT AT END - GUARANTEED EXTENSIVE BRANCHING
            # First level: 3-way split for dramatic tuft
            tuft_branches = []
            for angle_offset in [-0.6, 0, 0.6]:  # Three main tuft branches
                # Initial tuft branches spread out
                tuft_dir = np.array([
                    np.sin(angle_offset),
                    direction_sign * 0.7,  # Still going in main direction
                    np.sin(angle_offset) * 0.5
                ])
                tuft_dir = tuft_dir / np.linalg.norm(tuft_dir)
                
                tuft_pos = current_pos + tuft_dir * segment_length * 0.8
                tuft_id = self._add_node(tuft_pos[0], tuft_pos[1], tuft_pos[2],
                                        1.0, trunk_ids[-1], 4)
                tuft_branches.append((tuft_id, tuft_pos, tuft_dir))
            
            # Second and third level branching for fuller tuft
            for depth in range(4):  # 4 more levels of branching
                new_branches = []
                for parent_id, parent_pos, parent_dir in tuft_branches:
                    # Each branch splits into 2
                    for split in [-0.4, 0.4]:
                        # Increasing spread as we go deeper into tuft
                        spread_factor = 0.3 + 0.1 * depth
                        new_dir = parent_dir.copy()
                        new_dir[0] += np.random.uniform(-spread_factor, spread_factor)
                        new_dir[2] += np.random.uniform(-spread_factor, spread_factor)
                        new_dir[1] = direction_sign * (0.5 - 0.1 * depth)  # Gradually less vertical
                        new_dir = new_dir / np.linalg.norm(new_dir)
                        
                        seg_len = segment_length * 0.6 * (1 - depth * 0.2)
                        new_pos = parent_pos + new_dir * seg_len
                        
                        new_id = self._add_node(new_pos[0], new_pos[1], new_pos[2],
                                            0.6 - 0.1 * depth, parent_id, 4)
                        
                        # Only continue branching for first few levels
                        if depth < 3:
                            new_branches.append((new_id, new_pos, new_dir))
                
                tuft_branches = new_branches
            
            # NO lateral branches - pure bipolar structure
            # This ensures clear distinction from pyramidal

    
    def generate(self):
        """Generate morphology based on cluster type"""
        if self.cluster_type == 1:
            self._generate_pyramidal()
        elif self.cluster_type == 2:
            self._generate_multipolar()
        else:  # cluster_type == 3
            self._generate_bipolar()
        
        return self.nodes
    
    def save_swc(self, filepath):
        """Save morphology to SWC file with coordinates rounded to 1 decimal place"""
        with open(filepath, 'w') as f:
            # Write header
            f.write("# SWC format file\n")
            f.write("# Generated morphology with control parameters:\n")
            f.write(f"# P={self.P:.3f}, D={self.D:.3f}, A={self.A:.3f}\n")
            f.write(f"# Cluster type: {self.cluster_type}\n")
            f.write("# id type x y z radius parent\n")
            f.write("# type: 1=soma, 3=basal, 4=apical\n")
            
            # Write nodes with rounded coordinates
            for node in self.nodes:
                node_id, node_type, x, y, z, radius, parent_id = node
                # Round x, y, z coordinates to 1 decimal place
                x_rounded = round(float(x), 1)
                y_rounded = round(float(y), 1)
                z_rounded = round(float(z), 1)
                
                f.write(f'{int(node_id)} {int(node_type)} {x_rounded} {y_rounded} {z_rounded} {radius} {int(parent_id)}\n')
        
        return len(self.nodes)

def generate_all_morphologies(control_params, cluster_labels, output_dir):
    """Generate morphologies for all samples"""
    morphology_dir = os.path.join(output_dir, "swc_files")
    os.makedirs(morphology_dir, exist_ok=True)
    
    morphology_stats = []
    
    for i, (params, cluster) in enumerate(zip(control_params, cluster_labels)):
        P, D, A = params
        
        # Generate morphology
        generator = NeuronMorphologyGeneratorV3(P, D, A, cluster)
        nodes = generator.generate()
        
        # Save SWC file
        filepath = os.path.join(morphology_dir, f"neuron_{i+1}.swc")
        n_nodes = generator.save_swc(filepath)
        
        # Calculate statistics
        if len(nodes) > 0:
            nodes_array = np.array(nodes)
            positions = nodes_array[:, 2:5].astype(float)
            
            # Calculate various statistics
            max_distance = np.max(np.linalg.norm(positions, axis=1))
            
            # Count tips (nodes with no children)
            parent_ids = set(nodes_array[:, 6].astype(int))
            node_ids = set(nodes_array[:, 0].astype(int))
            tips = len(node_ids - parent_ids - {-1})
            
            # Calculate total cable length
            total_length = 0
            for node in nodes:
                if node[6] > 0:  # Has parent
                    parent_idx = int(node[6]) - 1
                    if parent_idx < len(nodes):
                        parent = nodes[parent_idx]
                        length = np.linalg.norm(
                            np.array([node[2], node[3], node[4]]) - 
                            np.array([parent[2], parent[3], parent[4]])
                        )
                        total_length += length
            
            morphology_stats.append({
                'neuron_id': i+1,
                'cluster': cluster,
                'n_nodes': n_nodes,
                'n_tips': tips,
                'max_distance': max_distance,
                'total_length': total_length,
                'P': P,
                'D': D,
                'A': A
            })
        
        if (i+1) % 100 == 0:
            print(f"  Generated {i+1} morphologies...")
    
    # Save statistics
    stats_df = pd.DataFrame(morphology_stats)
    stats_df.to_csv(os.path.join(output_dir, "morphology_stats.csv"), index=False)
    
    print(f"\nGenerated {len(control_params)} morphologies")
    
    # Print cluster-wise statistics
    for c in [1, 2, 3]:
        cluster_stats = stats_df[stats_df['cluster'] == c]
        if len(cluster_stats) > 0:
            print(f"\nCluster {c} statistics:")
            print(f"  Nodes: {cluster_stats['n_nodes'].mean():.1f} ± {cluster_stats['n_nodes'].std():.1f}")
            print(f"  Tips: {cluster_stats['n_tips'].mean():.1f} ± {cluster_stats['n_tips'].std():.1f}")
            print(f"  Max distance: {cluster_stats['max_distance'].mean():.1f} ± {cluster_stats['max_distance'].std():.1f}")
            print(f"  Total length: {cluster_stats['total_length'].mean():.1f} ± {cluster_stats['total_length'].std():.1f}")
    
    return morphology_stats

def visualize_gene_expression(gene_expression, cluster_labels, output_dir):
    """Visualize gene expression data in 3D"""
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    colors = {1: 'red', 2: 'green', 3: 'blue'}
    labels = {1: 'C1 (Pyramidal)', 2: 'C2 (Multipolar)', 3: 'C3 (Bipolar)'}
    
    for cluster_id in [1, 2, 3]:
        mask = cluster_labels == cluster_id
        ax1.scatter(gene_expression[mask, 0], 
                   gene_expression[mask, 1], 
                   gene_expression[mask, 2],
                   c=colors[cluster_id], 
                   s=30, 
                   alpha=0.6,
                   label=labels[cluster_id])
    
    ax1.set_xlabel('Gene 1')
    ax1.set_ylabel('Gene 2')
    ax1.set_zlabel('Gene 3')
    ax1.set_title('Gene Expression Clusters')
    ax1.legend()
    
    # 2D projections
    ax2 = fig.add_subplot(132)
    for cluster_id in [1, 2, 3]:
        mask = cluster_labels == cluster_id
        ax2.scatter(gene_expression[mask, 0], 
                   gene_expression[mask, 1],
                   c=colors[cluster_id], 
                   s=20, 
                   alpha=0.6,
                   label=labels[cluster_id])
    ax2.set_xlabel('Gene 1')
    ax2.set_ylabel('Gene 2')
    ax2.set_title('Gene 1 vs Gene 2')
    ax2.legend()
    
    ax3 = fig.add_subplot(133)
    for cluster_id in [1, 2, 3]:
        mask = cluster_labels == cluster_id
        ax3.scatter(gene_expression[mask, 0], 
                   gene_expression[mask, 2],
                   c=colors[cluster_id], 
                   s=20, 
                   alpha=0.6,
                   label=labels[cluster_id])
    ax3.set_xlabel('Gene 1')
    ax3.set_ylabel('Gene 3')
    ax3.set_title('Gene 1 vs Gene 3')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gene_expression_visualization.png"), dpi=150)
    plt.close()
    print(f"Gene expression visualization saved")

def visualize_sample_morphologies(control_params, cluster_labels, output_dir):
    """Visualize sample morphologies from each cluster with better adaptive scaling"""
    fig = plt.figure(figsize=(15, 5))
    
    for cluster_id in [1, 2, 3]:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 0:
            # Pick a representative sample
            idx = cluster_indices[len(cluster_indices)//2]
            P, D, A = control_params[idx]
            
            # Generate morphology
            generator = NeuronMorphologyGeneratorV3(P, D, A, cluster_id)
            nodes = generator.generate()
            
            # Plot in 2D (x-y projection)
            ax = fig.add_subplot(1, 3, cluster_id)
            
            if len(nodes) > 0:
                nodes_array = np.array(nodes)
                positions = nodes_array[:, 2:5].astype(float)
                
                # Plot by type
                soma_mask = nodes_array[:, 1] == 1
                basal_mask = nodes_array[:, 1] == 3
                apical_mask = nodes_array[:, 1] == 4
                
                # Draw connections first
                for i, node in enumerate(nodes):
                    if node[6] > 0:  # Has parent
                        parent_idx = int(node[6]) - 1
                        if parent_idx < len(nodes):
                            parent = nodes[parent_idx]
                            if node[1] == 3:  # Basal - blue
                                ax.plot([node[2], parent[2]], [node[3], parent[3]], 
                                       'b-', alpha=0.4, linewidth=0.5)
                            elif node[1] == 4:  # Apical - red
                                ax.plot([node[2], parent[2]], [node[3], parent[3]], 
                                       'r-', alpha=0.4, linewidth=0.5)
                
                # Plot soma on top (smaller size)
                if np.any(soma_mask):
                    ax.scatter(positions[soma_mask, 0], positions[soma_mask, 1], 
                              c='black', s=30, marker='o', zorder=3)
                
                # Calculate adaptive axis limits with proper padding
                x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
                y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
                
                # Add 20% padding to see full morphology
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Ensure minimum range for very small structures
                if x_range < 10:
                    x_center = (x_min + x_max) / 2
                    x_min = x_center - 10
                    x_max = x_center + 10
                else:
                    x_padding = x_range * 0.2
                    x_min -= x_padding
                    x_max += x_padding
                
                if y_range < 10:
                    y_center = (y_min + y_max) / 2
                    y_min = y_center - 10
                    y_max = y_center + 10
                else:
                    y_padding = y_range * 0.2
                    y_min -= y_padding
                    y_max += y_padding
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            
            cluster_names = {1: 'C1: Pyramidal', 2: 'C2: Multipolar', 3: 'C3: Bipolar'}
            ax.set_title(f'{cluster_names[cluster_id]}\nP={P:.2f}, D={D:.2f}, A={A:.2f}')
            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Add legend for first plot
            if cluster_id == 1:
                ax.plot([], [], 'r-', label='Apical', alpha=0.6)
                ax.plot([], [], 'b-', label='Basal', alpha=0.6)
                ax.scatter([], [], c='black', s=30, label='Soma')
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_morphologies.png"), dpi=150)
    plt.close()
    print(f"Sample morphologies visualization saved")

def main():
    """Main execution function"""
    # Set base output directory
    base_dir = "/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Simulation/v3"
    os.makedirs(base_dir, exist_ok=True)
    
    print("=" * 60)
    print("NEURON MORPHOLOGY SIMULATION V3 - FINAL")
    print("=" * 60)
    
    # Step 1: Generate gene expression data
    print("\nStep 1: Generating gene expression data...")
    print("-" * 40)
    
    gene_expression, cluster_labels, control_params = generate_gene_expression_data(
        n_samples=1000,
        output_dir=base_dir
    )
    
    # Visualize gene expression
    visualize_gene_expression(gene_expression, cluster_labels, base_dir)
    
    # Step 2: Generate morphologies
    print("\nStep 2: Generating neuronal morphologies...")
    print("-" * 40)
    
    morphology_stats = generate_all_morphologies(
        control_params,
        cluster_labels,
        base_dir
    )
    
    # Step 3: Visualize sample morphologies
    print("\nStep 3: Visualizing sample morphologies...")
    print("-" * 40)
    
    visualize_sample_morphologies(control_params, cluster_labels, base_dir)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    print(f"\nOutput directory: {base_dir}")
    print("\nGenerated files:")
    print("  - gene_expression.csv: Gene expression data (1000 samples)")
    print("  - cluster_assignments.csv: Cluster labels")
    print("  - swc_files/: Directory containing all SWC morphology files")
    print("  - morphology_stats.csv: Morphology statistics")
    print("  - gene_expression_visualization.png: Gene expression clusters")
    print("  - sample_morphologies.png: Representative morphologies")
    
    # Print key characteristics
    print("\n" + "=" * 60)
    print("MORPHOLOGICAL CHARACTERISTICS")
    print("=" * 60)
    
    print("\n🔴 Cluster 1 (Pyramidal):")
    print("  ✓ Single apical trunk extending 60 units upward")
    print("  ✓ PROMINENT APICAL TUFT with 3-way branching")
    print("  ✓ 6 basal dendrites spreading horizontally/downward")
    
    print("\n🟢 Cluster 2 (Multipolar):")
    print("  ✓ 10 primary dendrites from soma")
    print("  ✓ Dense branching near soma")
    print("  ✓ Compact bush-like structure")
    
    print("\n🔵 Cluster 3 (Bipolar):")
    print("  ✓ TWO opposite trunks (±50 units on Y-axis)")
    print("  ✓ PROMINENT TUFTS at BOTH ends")
    print("  ✓ Minimal lateral branching")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()