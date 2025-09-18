import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, RegularPolygon
from scipy.stats import entropy
from scipy.spatial import KDTree
import os
from datetime import datetime
import pandas as pd
import torch

class EnhancedVWPRWrapper:
    def __init__(self):
        # Initialize with separate seeded random generators for different purposes
        self.layout_rng = np.random.RandomState(42)  # For layout generation
        self.sampling_rng = np.random.RandomState(43)  # For node sampling
        self.effects_rng = np.random.RandomState(44)  # For visual effects
        
        # Configurable archetype thresholds
        self.archetype_thresholds = {
            "The Oracle": {"H_min": 0.8, "L_min": 0.3, "L_max": 0.7},
            "The Sentinel": {"ρ_min": 1.5},
            "The Alchemist": {"FD_min": 0.5, "H_min": 0.6},
            "The Archivist": {"H_max": 0.3, "FD_max": 0.2},
            "The Trickster": {"FD_min": 0.7},
            "The Luminary": {"L_min": 0.8}
        }
        
        # Initialize phi storage
        self.phi = {}  # Current phi values by layer
        self.phi_history = []  # List of dictionaries: [{layer_name: tensor}, ...]
        
    def update_phi(self, phi_dict):
        """Update the current phi values."""
        self.phi = phi_dict
        
    def record_epoch(self):
        """Record the current phi values to history."""
        # Convert tensors to numpy arrays for storage
        recorded_phi = {}
        for layer_name, tensor in self.phi.items():
            if hasattr(tensor, 'detach'):
                recorded_phi[layer_name] = tensor.detach().cpu().numpy()
            else:
                recorded_phi[layer_name] = np.array(tensor)
        self.phi_history.append(recorded_phi)
        
    def safe_normalize(self, values):
        """Safely normalize values with numerical stability."""
        if len(values) == 0:
            return values
            
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Handle constant arrays
        if np.isclose(min_val, max_val):
            return np.ones_like(values) * 0.5
            
        return (values - min_val) / (max_val - min_val + 1e-12)
    
    def calculate_metrics(self, phi_vals, layer_name):
        """Calculate standardized metrics for phi values."""
        phi_normalized = self.safe_normalize(phi_vals)
        
        # Luminosity
        L_l = np.mean(phi_normalized)
        
        # Fracture Density
        FD_l = np.std(phi_normalized)
        
        # Entropy
        hist, _ = np.histogram(phi_normalized, bins=min(50, len(phi_vals)//5), density=True)
        H_l = entropy(hist + 1e-12) / np.log(len(hist)) if len(hist) > 1 else 0.0
        
        # Resilience
        ρ_l = self.calculate_resilience(layer_name)
        
        return L_l, FD_l, H_l, ρ_l, phi_normalized
    
    def calculate_resilience(self, layer_name, window_size=10):
        """Calculate resilience based on historical phi values."""
        if len(self.phi_history) <= 1:
            return 1.0
        
        # Get last window_size epochs if available
        start_idx = max(0, len(self.phi_history) - window_size)
        phi_past = []
        
        for epoch_data in self.phi_history[start_idx:]:
            if layer_name in epoch_data:
                past_vals = epoch_data[layer_name].flatten()
                past_normalized = self.safe_normalize(past_vals)
                phi_past.append(np.mean(past_normalized))
        
        if len(phi_past) > 1:
            # Calculate stability as inverse of coefficient of variation
            ρ_l = 1.0 / (np.std(phi_past) / (np.mean(phi_past) + 1e-12) + 1e-12)
            ρ_l = min(ρ_l, 2.0)  # Cap resilience for visualization
            return ρ_l
        
        return 1.0
    
    def determine_archetype(self, L_l, FD_l, H_l, ρ_l):
        """Determine the archetype based on metrics and configurable thresholds."""
        thresholds = self.archetype_thresholds
        
        # Check conditions in priority order
        if (H_l > thresholds["The Oracle"]["H_min"] and 
            thresholds["The Oracle"]["L_min"] < L_l < thresholds["The Oracle"]["L_max"]):
            return "The Oracle"
        elif ρ_l > thresholds["The Sentinel"]["ρ_min"]:
            return "The Sentinel"
        elif (FD_l > thresholds["The Alchemist"]["FD_min"] and 
              H_l > thresholds["The Alchemist"]["H_min"]):
            return "The Alchemist"
        elif (H_l < thresholds["The Archivist"]["H_max"] and 
              FD_l < thresholds["The Archivist"]["FD_max"]):
            return "The Archivist"
        elif FD_l > thresholds["The Trickster"]["FD_min"]:
            return "The Trickster"
        elif L_l > thresholds["The Luminary"]["L_min"]:
            return "The Luminary"
        else:
            return "The Wanderer"
    
    def get_colormap(self, archetype):
        """Get the colormap for the given archetype."""
        colormaps = {
            "The Oracle": LinearSegmentedColormap.from_list('oracle', ['#000022', '#4444AA', '#AAAAFF', '#FFFFFF']),
            "The Sentinel": LinearSegmentedColormap.from_list('sentinel', ['#321407', '#785021', '#BC7D2F', '#FFD700']),
            "The Alchemist": LinearSegmentedColormap.from_list('alchemist', ['#220022', '#660066', '#9933CC', '#FFCCFF']),
            "The Archivist": LinearSegmentedColormap.from_list('archivist', ['#111111', '#555555', '#999999', '#CCCCCC']),
            "The Trickster": LinearSegmentedColormap.from_list('trickster', ['#330000', '#CC0000', '#FF3300', '#FFFF00']),
            "The Luminary": LinearSegmentedColormap.from_list('luminary', ['#333300', '#666600', '#999900', '#FFFF00']),
            "The Wanderer": LinearSegmentedColormap.from_list('wanderer', ['#002233', '#005566', '#008899', '#00CCFF'])
        }
        return colormaps.get(archetype, colormaps["The Wanderer"])
    
    def create_layout(self, archetype, num_nodes, L_l, ρ_l):
        """Create node positions based on archetype."""
        if archetype == "The Sentinel":
            theta = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
            x = np.cos(theta) * (0.5 + L_l)
            y = np.sin(theta) * (0.5 + L_l)
        elif archetype == "The Trickster":
            x = self.layout_rng.normal(0, 0.8, num_nodes)
            y = self.layout_rng.normal(0, 0.8, num_nodes)
        else:
            theta = np.linspace(0, 4*np.pi, num_nodes, endpoint=False)
            r = theta / (4*np.pi) * (1.5 - ρ_l/4)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        
        return x, y
    
    def draw_connections(self, ax, x, y, phi_sampled, cmap, archetype, num_nodes):
        """Draw connections between nodes efficiently using KDTree with connection limit."""
        if archetype not in ["The Oracle", "The Sentinel", "The Archivist"] or num_nodes >= 500:
            return
        
        # Use KDTree for efficient neighbor finding
        points = np.vstack((x, y)).T
        tree = KDTree(points)
        
        # Find all pairs within distance 0.3 using batch query
        indices = tree.query_ball_tree(tree, 0.3)
        
        # Draw connections with limit
        max_connections = 1000
        connection_count = 0
        lines = []
        
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if j <= i:  # Avoid duplicate connections
                    continue
                
                if connection_count >= max_connections:
                    break
                    
                # Check similarity condition
                if abs(phi_sampled[i] - phi_sampled[j]) < 0.2:
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    alpha = 0.7 * (1 - dist/0.3)
                    
                    line = ax.plot([x[i], x[j]], [y[i], y[j]], '-', 
                                  color=cmap((phi_sampled[i] + phi_sampled[j])/2), 
                                  alpha=alpha, linewidth=0.5)
                    lines.append(line[0])
                    connection_count += 1
            
            if connection_count >= max_connections:
                break
        
        return lines
    
    def draw_nodes(self, ax, x, y, phi_sampled, cmap, archetype, FD_l):
        """Draw nodes with appropriate shapes and styles."""
        nodes = []
        for i, phi_val in enumerate(phi_sampled):
            node_size = 30 + (phi_val * 300)
            node_color = cmap(phi_val)
            
            if archetype == "The Sentinel":
                node = RegularPolygon((x[i], y[i]), numVertices=6, radius=node_size/200, 
                                     orientation=np.pi/2, color=node_color, alpha=0.9)
                ax.add_patch(node)
                nodes.append(node)
            elif archetype == "The Archivist":
                node = RegularPolygon((x[i], y[i]), numVertices=4, radius=node_size/200, 
                                     orientation=np.pi/4, color=node_color, alpha=0.9)
                ax.add_patch(node)
                nodes.append(node)
            else:
                node = ax.scatter(x[i], y[i], s=node_size, c=[node_color], alpha=0.9, zorder=3)
                nodes.append(node)
            
            # Add cracks for high fracture density
            if FD_l > 0.2 and archetype not in ["The Archivist", "The Sentinel"]:
                num_cracks = int(phi_val * 3) + 1
                for _ in range(num_cracks):
                    length = 0.2 + (self.effects_rng.rand() * 0.4)
                    crack_angle = self.effects_rng.rand() * 2 * np.pi
                    dx = length * np.cos(crack_angle) / 25
                    dy = length * np.sin(crack_angle) / 25
                    crack = ax.plot([x[i] - dx, x[i] + dx], [y[i] - dy, y[i] + dy], 
                                   'k-', lw=0.8, alpha=0.7)
                    nodes.append(crack[0])
        
        return nodes
    
    def add_archetype_elements(self, ax, cmap, archetype, H_l, ρ_l):
        """Add archetype-specific decorative elements."""
        elements = []
        if archetype == "The Sentinel":
            for i in range(1, int(ρ_l) + 1):
                circle = Circle((0, 0), 0.2 * i, fill=False, 
                               linestyle='-', linewidth=1 + i, 
                               color=cmap(0.8), alpha=0.7)
                ax.add_patch(circle)
                elements.append(circle)
        elif archetype == "The Oracle":
            t = np.linspace(0, 3 * np.pi, 100)
            r = 0.5 * np.exp(0.2 * t)
            x_spiral = r * np.cos(t) * (H_l + 0.5)
            y_spiral = r * np.sin(t) * (H_l + 0.5)
            spiral = ax.plot(x_spiral, y_spiral, color=cmap(0.7), linewidth=2.0, alpha=0.7)
            elements.append(spiral[0])
        elif archetype == "The Alchemist":
            for i in range(3):
                angle = i * 2 * np.pi / 3
                x_symbol = 1.2 * np.cos(angle)
                y_symbol = 1.2 * np.sin(angle)
                ellipse = Ellipse((x_symbol, y_symbol), width=0.2, height=0.1, 
                                 angle=angle*180/np.pi, color=cmap(0.9), alpha=0.7)
                ax.add_patch(ellipse)
                elements.append(ellipse)
        elif archetype == "The Trickster":
            for i in range(20):
                x_chaos = self.effects_rng.uniform(-1.5, 1.5)
                y_chaos = self.effects_rng.uniform(-1.5, 1.5)
                size_chaos = self.effects_rng.uniform(10, 100)
                chaos = ax.scatter(x_chaos, y_chaos, s=size_chaos, 
                                  c=[cmap(self.effects_rng.rand())], alpha=0.3, zorder=1)
                elements.append(chaos)
        
        return elements
    
    def generate_sigil(self, layer_name: str, epoch: int, 
                      save_dir: str = None, dpi: int = 150):
        """
        Generate a ceremonial sigil for a specific layer at a given epoch.
        Returns a matplotlib Figure object and archetype name.
        """
        try:
            # Get the phi values for this layer
            if layer_name not in self.phi:
                raise ValueError(f"Layer {layer_name} not found in phi dictionary")
                
            phi_tensor = self.phi[layer_name]
            if hasattr(phi_tensor, 'detach'):
                phi_vals = phi_tensor.detach().cpu().numpy().flatten()
            else:
                phi_vals = np.array(phi_tensor).flatten()
            
            # Calculate metrics
            L_l, FD_l, H_l, ρ_l, phi_normalized = self.calculate_metrics(phi_vals, layer_name)
            
            # Determine archetype
            archetype = self.determine_archetype(L_l, FD_l, H_l, ρ_l)
            
            # Get appropriate colormap
            cmap = self.get_colormap(archetype)
            
            # Sigil Generation
            fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
            ax.set_facecolor('black')
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Sample nodes if needed
            num_nodes = len(phi_normalized)
            if num_nodes > 1000:
                indices = self.sampling_rng.choice(num_nodes, 1000, replace=False)
                phi_sampled = phi_normalized[indices]
                num_nodes = 1000
            else:
                phi_sampled = phi_normalized
            
            # Create layout
            x, y = self.create_layout(archetype, num_nodes, L_l, ρ_l)
            
            # Draw connections
            connection_lines = self.draw_connections(ax, x, y, phi_sampled, cmap, archetype, num_nodes)
            
            # Draw nodes
            nodes = self.draw_nodes(ax, x, y, phi_sampled, cmap, archetype, FD_l)
            
            # Add archetype-specific elements
            elements = self.add_archetype_elements(ax, cmap, archetype, H_l, ρ_l)
            
            # Add boundary with dynamic sizing
            padding = 0.2
            x_min, x_max = np.min(x) - padding, np.max(x) + padding
            y_min, y_max = np.min(y) - padding, np.max(y) + padding
            max_extent = max(x_max - x_min, y_max - y_min)
            
            boundary_circle = Circle((0, 0), max_extent/2, fill=False, linestyle='-', 
                                    linewidth=3, color='white', alpha=0.5)
            ax.add_patch(boundary_circle)
            
            # Set dynamic limits
            ax.set_xlim(-max_extent/2, max_extent/2)
            ax.set_ylim(-max_extent/2, max_extent/2)
            
            # Add text
            title_color = 'white'
            ax.text(0, -max_extent/2 - 0.1, f"L: {L_l:.2f} | FD: {FD_l:.2f} | H: {H_l:.2f} | ρ: {ρ_l:.2f}", 
                    ha='center', va='top', color=title_color, fontsize=8)
            ax.set_title(f"{layer_name}\n{archetype}\nEpoch: {epoch}", 
                        color=title_color, pad=20, fontsize=14)
            plt.tight_layout()
            
            # Save if requested
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{save_dir}/{layer_name}_{archetype}_epoch_{epoch:04d}_{timestamp}.png"
                fig.savefig(filename, dpi=dpi, facecolor='black', bbox_inches='tight')
            
            # Clear references to prevent memory leaks
            for artist in nodes + elements + (connection_lines if connection_lines else []) + [boundary_circle]:
                try:
                    artist.remove()
                except:
                    pass
            
            return fig, archetype, {"L_l": L_l, "FD_l": FD_l, "H_l": H_l, "ρ_l": ρ_l}
            
        except Exception as e:
            print(f"Error generating sigil for {layer_name}: {e}")
            # Return a minimal error figure
            fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, f"Error: {e}", color='white', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig, "Error", {}
    
    def sigil_gallery(self, total_epochs=100, interval=10, out_dir="sigil_gallery", 
                     selected_layers=None, dpi=150):
        """
        Generate and save a gallery of sigils across epochs and layers.
        """
        try:
            os.makedirs(out_dir, exist_ok=True)
            if selected_layers is None:
                selected_layers = list(self.phi.keys())
            
            # Open summary file
            with open(f"{out_dir}/sigil_summary.csv", "w") as summary_file:
                summary_file.write("Layer,Archetype,Epoch,Luminosity,FractureDensity,Entropy,Resilience\n")
                
                for epoch in range(0, total_epochs, interval):
                    for name in selected_layers:
                        if name in self.phi:
                            fig, archetype, metrics = self.generate_sigil(name, epoch, out_dir, dpi)
                            plt.close(fig)  # Close figure to free memory
                            
                            summary_file.write(f"{name},{archetype},{epoch},{metrics['L_l']:.4f},")
                            summary_file.write(f"{metrics['FD_l']:.4f},{metrics['H_l']:.4f},{metrics['ρ_l']:.4f}\n")
            
            # Create summary visualization
            self.create_summary_visualization(out_dir)
            
        except Exception as e:
            print(f"Error creating sigil gallery: {e}")
    
    def create_summary_visualization(self, out_dir):
        """
        Create a visualization of the sigil summary data.
        """
        try:
            df = pd.read_csv(f"{out_dir}/sigil_summary.csv")
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Sigil Metrics Evolution", fontsize=16)
            
            layers = df['Layer'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
            
            for i, metric in enumerate(['Luminosity', 'FractureDensity', 'Entropy', 'Resilience']):
                ax = axes[i//2, i%2]
                
                for j, layer in enumerate(layers):
                    layer_data = df[df['Layer'] == layer]
                    ax.plot(layer_data['Epoch'], layer_data[metric], 
                           color=colors[j], label=layer, linewidth=2)
                
                ax.set_title(metric)
                ax.set_xlabel('Epoch')
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(f"{out_dir}/metrics_evolution.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Could not create summary visualization: {e}")
    
    def update_archetype_thresholds(self, new_thresholds):
        """Update archetype thresholds with new values."""
        for archetype, thresholds in new_thresholds.items():
            if archetype in self.archetype_thresholds:
                self.archetype_thresholds[archetype].update(thresholds)
    
    def calibrate_thresholds(self, calibration_data):
        """
        Calibrate archetype thresholds based on empirical data.
        calibration_data should be a list of dictionaries with metrics for each archetype.
        """
        # This is a placeholder - implement based on your specific calibration needs
        print("Threshold calibration would be implemented here based on provided data")
