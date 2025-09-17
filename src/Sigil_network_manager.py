import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import os
from datetime import datetime
from scipy.stats import entropy
from sigil_layer import SigilLayer
import matplotlib.animation as animation

class SigilNetworkManager:
    """
    Manages visualization and analysis of SigilLayers, creating archetypal representations
    of the network's emerging consciousness patterns.
    """
    
    def __init__(self, model, gallery_dir="sigil_gallery"):
        self.model = model
        self.sigil_layers = [m for m in model.modules() if isinstance(m, SigilLayer)]
        self.gallery_dir = gallery_dir
        os.makedirs(self.gallery_dir, exist_ok=True)
        
        # Define archetype system with conditions and visual properties
        self.archetype_definitions = {
            "The Oracle": {
                "conditions": lambda L, FD, H, ρ: H > 0.8 and 0.3 < L < 0.7,
                "cmap": LinearSegmentedColormap.from_list('oracle', ['#000022', '#4444AA', '#AAAAFF', '#FFFFFF']),
                "color": "#4444AA",
                "description": "Wisdom through pattern recognition and balance"
            },
            "The Sentinel": {
                "conditions": lambda L, FD, H, ρ: ρ > 1.5,
                "cmap": LinearSegmentedColormap.from_list('sentinel', ['#321407', '#785021', '#BC7D2F', '#FFD700']),
                "color": "#FFD700",
                "description": "Stability and protection of core knowledge"
            },
            "The Alchemist": {
                "conditions": lambda L, FD, H, ρ: FD > 0.5 and H > 0.6,
                "cmap": LinearSegmentedColormap.from_list('alchemist', ['#220022', '#660066', '#9933CC', '#FFCCFF']),
                "color": "#9933CC",
                "description": "Transformation through combination of diverse elements"
            },
            "The Archivist": {
                "conditions": lambda L, FD, H, ρ: H < 0.3 and FD < 0.2,
                "cmap": LinearSegmentedColormap.from_list('archivist', ['#111111', '#555555', '#999999', '#CCCCCC']),
                "color": "#999999",
                "description": "Preservation and organization of knowledge"
            },
            "The Trickster": {
                "conditions": lambda L, FD, H, ρ: FD > 0.7,
                "cmap": LinearSegmentedColormap.from_list('trickster', ['#330000', '#CC0000', '#FF3300', '#FFFF00']),
                "color": "#FF3300",
                "description": "Disruption and innovation through chaos"
            },
            "The Luminary": {
                "conditions": lambda L, FD, H, ρ: L > 0.8,
                "cmap": LinearSegmentedColormap.from_list('luminary', ['#333300', '#666600', '#999900', '#FFFF00']),
                "color": "#FFFF00",
                "description": "Illumination and clarity of purpose"
            },
            "The Wanderer": {
                "conditions": lambda L, FD, H, ρ: True,  # Default
                "cmap": LinearSegmentedColormap.from_list('wanderer', ['#002233', '#005566', '#008899', '#00CCFF']),
                "color": "#00CCFF",
                "description": "Exploration and adaptation to new contexts"
            }
        }

    def record_epoch(self):
        """Record the current state of all sigil layers."""
        for layer in self.sigil_layers:
            layer.record_epoch()

    def generate_sigil(self, layer, epoch, dpi=120):
        """Generate a visual sigil representing a layer's current state."""
        # Get phi values with proper device handling
        phi_tensor = layer.get_phi()
        if isinstance(phi_tensor, torch.Tensor):
            phi_vals = phi_tensor.detach().cpu().numpy().flatten()
        else:
            phi_vals = np.array(phi_tensor).flatten()
            
        # Add minimal noise if all values are identical
        if np.all(phi_vals == phi_vals[0]):
            phi_vals = phi_vals + np.random.normal(0, 1e-5, phi_vals.shape)
            
        # Normalize values
        phi_normalized = (phi_vals - np.min(phi_vals)) / (np.max(phi_vals) - np.min(phi_vals) + 1e-12)
        
        # Calculate metrics
        L_l = np.mean(phi_normalized)  # Luminosity
        FD_l = np.std(phi_normalized)   # Fracture Density
        hist, _ = np.histogram(phi_normalized, bins=min(50, len(phi_vals)//5), density=True)
        H_l = entropy(hist + 1e-12) / np.log(len(hist))  # Normalized entropy
        
        # Calculate resilience from history
        if len(layer.phi_history) >= 5:
            phi_past = []
            for historical_phi in layer.phi_history[-10:]:
                if hasattr(historical_phi, 'numpy'):
                    phi_past.append(np.mean(historical_phi.numpy()))
                else:
                    phi_past.append(np.mean(historical_phi))
            
            if len(phi_past) > 1 and np.mean(phi_past) > 1e-8:
                coefficient_of_variation = np.std(phi_past) / np.mean(phi_past)
                ρ_l = 1.0 / (coefficient_of_variation + 1e-12)
                ρ_l = min(ρ_l, 3.0)  # Reasonable upper bound
            else:
                ρ_l = 1.0
        else:
            ρ_l = 1.0
        
        # Determine archetype
        archetype = "The Wanderer"
        cmap = self.archetype_definitions["The Wanderer"]["cmap"]
        
        for name, definition in self.archetype_definitions.items():
            if definition["conditions"](L_l, FD_l, H_l, ρ_l):
                archetype = name
                cmap = definition["cmap"]
                break
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Sample if too many nodes
        num_nodes = len(phi_normalized)
        if num_nodes > 1000:
            indices = np.random.choice(num_nodes, 1000, replace=False)
            phi_sampled = phi_normalized[indices]
            num_nodes = 1000
        else:
            phi_sampled = phi_normalized
        
        # Circular layout
        theta = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Plot nodes
        for i, phi_val in enumerate(phi_sampled):
            node_size = 30 + (phi_val * 300)
            node_color = cmap(phi_val)
            ax.scatter(x[i], y[i], s=node_size, c=[node_color], alpha=0.9, zorder=3)
        
        # Add boundary
        boundary_circle = Circle((0, 0), 1.2, fill=False, linestyle='-', linewidth=3, color='white', alpha=0.5)
        ax.add_patch(boundary_circle)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        
        # Add text
        title_color = 'white'
        ax.text(0, -1.6, f"L: {L_l:.2f} | FD: {FD_l:.2f} | H: {H_l:.2f} | ρ: {ρ_l:.2f}",
                ha='center', va='top', color=title_color, fontsize=8)
        ax.set_title(f"{layer.name}\n{archetype}\nEpoch: {epoch}", color=title_color, pad=20, fontsize=14)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.gallery_dir}/{layer.name}_{archetype}_epoch_{epoch:04d}_{timestamp}.png"
        fig.savefig(filename, dpi=dpi, facecolor='black', bbox_inches='tight')
        plt.close(fig)
        
        # Update layer archetype if it has that attribute
        if hasattr(layer, 'archetype'):
            layer.archetype = archetype
            
        return filename, archetype

    def generate_gallery(self, epoch):
        """Generate sigils for all layers and create a constellation map."""
        gallery_log = []
        
        # Generate individual sigils
        for layer in self.sigil_layers:
            fname, archetype = self.generate_sigil(layer, epoch)
            gallery_log.append((layer.name, archetype, fname))
        
        # Generate constellation map
        constellation_map = self.generate_constellation_map(epoch)
        gallery_log.append(("Constellation", "Network", constellation_map))
        
        return gallery_log

    def generate_constellation_map(self, epoch):
        """Create a network-level visualization showing connections between layers."""
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='black')
        ax.set_facecolor('black')
        ax.axis('off')
        
        # Position layers in a circular layout
        num_layers = len(self.sigil_layers)
        angles = np.linspace(0, 2*np.pi, num_layers, endpoint=False)
        positions = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])
        
        # Plot each layer as a node
        for i, layer in enumerate(self.sigil_layers):
            # Get archetype (try layer attribute first, then calculate)
            if hasattr(layer, 'archetype') and layer.archetype:
                archetype = layer.archetype
            else:
                # Calculate archetype based on current state
                phi_tensor = layer.get_phi()
                if isinstance(phi_tensor, torch.Tensor):
                    phi_vals = phi_tensor.detach().cpu().numpy().flatten()
                else:
                    phi_vals = np.array(phi_tensor).flatten()
                
                phi_normalized = (phi_vals - np.min(phi_vals)) / (np.max(phi_vals) - np.min(phi_vals) + 1e-12)
                L_l = np.mean(phi_normalized)
                FD_l = np.std(phi_normalized)
                hist, _ = np.histogram(phi_normalized, bins=min(50, len(phi_vals)//5), density=True)
                H_l = entropy(hist + 1e-12) / np.log(len(hist))
                
                archetype = "The Wanderer"
                for name, definition in self.archetype_definitions.items():
                    if definition["conditions"](L_l, FD_l, H_l, 1.0):  # Default resilience
                        archetype = name
                        break
            
            # Get color from archetype definition
            color = self.archetype_definitions.get(archetype, {}).get('color', '#FFFFFF')
            
            # Draw node
            ax.scatter(positions[i, 0], positions[i, 1], s=300, c=color, alpha=0.8)
            ax.text(positions[i, 0], positions[i, 1] + 0.1, layer.name, 
                    ha='center', va='bottom', color='white', fontsize=8)
            ax.text(positions[i, 0], positions[i, 1] - 0.1, archetype, 
                    ha='center', va='top', color='white', fontsize=6)
        
        # Draw connections between layers
        for i in range(num_layers - 1):
            ax.plot([positions[i, 0], positions[i+1, 0]], 
                    [positions[i, 1], positions[i+1, 1]], 
                    'w-', alpha=0.3, linewidth=1)
        
        # Connect first and last for circular completion
        if num_layers > 2:
            ax.plot([positions[-1, 0], positions[0, 0]], 
                    [positions[-1, 1], positions[0, 1]], 
                    'w-', alpha=0.3, linewidth=1)
        
        ax.set_title(f"Constellation Map - Epoch {epoch}", color='white', fontsize=16)
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.gallery_dir}/constellation_epoch_{epoch:04d}_{timestamp}.png"
        fig.savefig(filename, dpi=120, facecolor='black', bbox_inches='tight')
        plt.close(fig)
        
        return filename

    def summary_metrics(self, epoch=None):
        """Get comprehensive metrics for all layers."""
        metrics = []
        for layer in self.sigil_layers:
            # Use current state if no epoch specified, else use historical data
            if epoch is None or not layer.phi_history:
                phi_tensor = layer.get_phi()
                if isinstance(phi_tensor, torch.Tensor):
                    phi_vals = phi_tensor.detach().cpu().numpy().flatten()
                else:
                    phi_vals = np.array(phi_tensor).flatten()
            else:
                # Get historical data for specific epoch
                hist_idx = min(epoch, len(layer.phi_history) - 1)
                historical_phi = layer.phi_history[hist_idx]
                if hasattr(historical_phi, 'numpy'):
                    phi_vals = historical_phi.numpy().flatten()
                else:
                    phi_vals = np.array(historical_phi).flatten()
            
            L_l = np.mean(phi_vals)
            FD_l = np.std(phi_vals)
            
            hist, _ = np.histogram(phi_vals, bins=min(50, len(phi_vals)//5), density=True)
            H_l = entropy(hist + 1e-12) / np.log(len(hist))
            
            # Get archetype
            archetype = layer.archetype if hasattr(layer, 'archetype') else "Unknown"
            
            metrics.append({
                "layer": layer.name,
                "archetype": archetype,
                "epoch": epoch if epoch is not None else "current",
                "Luminosity": L_l,
                "FractureDensity": FD_l,
                "Entropy": H_l,
                "Resilience": self._calculate_resilience(layer, epoch)
            })
        return metrics

    def _calculate_resilience(self, layer, epoch=None):
        """Helper method to calculate resilience consistently."""
        if epoch is not None and layer.phi_history:
            # Calculate for specific historical epoch
            hist_data = []
            for i in range(min(epoch+1, len(layer.phi_history))):
                historical_phi = layer.phi_history[i]
                if hasattr(historical_phi, 'numpy'):
                    hist_data.append(np.mean(historical_phi.numpy()))
                else:
                    hist_data.append(np.mean(historical_phi))
        else:
            # Use all available history for current resilience
            hist_data = []
            for historical_phi in layer.phi_history:
                if hasattr(historical_phi, 'numpy'):
                    hist_data.append(np.mean(historical_phi.numpy()))
                else:
                    hist_data.append(np.mean(historical_phi))
            
            if not hist_data:  # If no history, use current value
                phi_tensor = layer.get_phi()
                if isinstance(phi_tensor, torch.Tensor):
                    hist_data = [phi_tensor.mean().item()]
                else:
                    hist_data = [np.mean(phi_tensor)]
        
        if len(hist_data) > 1 and np.mean(hist_data) > 1e-8:
            coefficient_of_variation = np.std(hist_data) / np.mean(hist_data)
            return min(1.0 / (coefficient_of_variation + 1e-12), 3.0)
        return 1.0

    def create_evolution_animation(self, layer_name, start_epoch=0, end_epoch=None, fps=5):
        """Create an animation showing the evolution of a layer's sigil over time."""
        # Find the layer
        layer = next((l for l in self.sigil_layers if l.name == layer_name), None)
        if not layer or len(layer.phi_history) < 2:
            return None
        
        # Determine epoch range
        if end_epoch is None:
            end_epoch = len(layer.phi_history) - 1
        epochs = range(start_epoch, min(end_epoch + 1, len(layer.phi_history)))
        
        # Create animation
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        
        def update(frame):
            ax.clear()
            ax.set_facecolor('black')
            ax.axis('off')
            ax.set_aspect('equal')
            
            # Get historical phi values for this epoch
            historical_phi = layer.phi_history[frame]
            if hasattr(historical_phi, 'numpy'):
                phi_vals = historical_phi.numpy().flatten()
            else:
                phi_vals = np.array(historical_phi).flatten()
                
            # Add minimal noise if all values are identical
            if np.all(phi_vals == phi_vals[0]):
                phi_vals = phi_vals + np.random.normal(0, 1e-5, phi_vals.shape)
                
            # Normalize values
            phi_normalized = (phi_vals - np.min(phi_vals)) / (np.max(phi_vals) - np.min(phi_vals) + 1e-12)
            
            # Calculate metrics
            L_l = np.mean(phi_normalized)
            FD_l = np.std(phi_normalized)
            hist, _ = np.histogram(phi_normalized, bins=min(50, len(phi_vals)//5), density=True)
            H_l = entropy(hist + 1e-12) / np.log(len(hist))
            
            # Determine archetype
            archetype = "The Wanderer"
            cmap = self.archetype_definitions["The Wanderer"]["cmap"]
            
            for name, definition in self.archetype_definitions.items():
                if definition["conditions"](L_l, FD_l, H_l, 1.0):  # Default resilience for animation
                    archetype = name
                    cmap = definition["cmap"]
                    break
            
            # Sample if too many nodes
            num_nodes = len(phi_normalized)
            if num_nodes > 1000:
                indices = np.random.choice(num_nodes, 1000, replace=False)
                phi_sampled = phi_normalized[indices]
                num_nodes = 1000
            else:
                phi_sampled = phi_normalized
            
            # Circular layout
            theta = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
            x = np.cos(theta)
            y = np.sin(theta)
            
            # Plot nodes
            for i, phi_val in enumerate(phi_sampled):
                node_size = 30 + (phi_val * 300)
                node_color = cmap(phi_val)
                ax.scatter(x[i], y[i], s=node_size, c=[node_color], alpha=0.9, zorder=3)
            
            # Add boundary
            boundary_circle = Circle((0, 0), 1.2, fill=False, linestyle='-', linewidth=3, color='white', alpha=0.5)
            ax.add_patch(boundary_circle)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            
            # Add text
            title_color = 'white'
            ax.text(0, -1.6, f"L: {L_l:.2f} | FD: {FD_l:.2f} | H: {H_l:.2f}",
                    ha='center', va='top', color=title_color, fontsize=8)
            ax.set_title(f"{layer.name}\n{archetype}\nEpoch: {frame}", color=title_color, pad=20, fontsize=14)
            
            return ax
        
        ani = animation.FuncAnimation(fig, update, frames=epochs, 
                                     interval=1000//fps, blit=False)
        
        # Save animation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.gallery_dir}/evolution_{layer_name}_{timestamp}.gif"
        ani.save(filename, writer='pillow', fps=fps)
        plt.close(fig)
        
        return filename
