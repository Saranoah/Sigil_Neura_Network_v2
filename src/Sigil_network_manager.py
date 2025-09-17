import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import os
from datetime import datetime
from scipy.stats import entropy
from sigil_layer import SigilLayer
import matplotlib.animation as animation
from collections import OrderedDict

class SigilNetworkManager:
    """
    Manages visualization and analysis of SigilLayers, creating archetypal representations
    of the network's emerging patterns with improved stability and configurability.
    """
    
    # Fixed sampling seed for reproducible visualizations
    SAMPLING_SEED = 42
    
    def __init__(self, model, gallery_dir="sigil_gallery", archetype_config=None, max_cache_size=1000):
        self.model = model
        self.sigil_layers = [m for m in model.modules() if isinstance(m, SigilLayer)]
        self.gallery_dir = gallery_dir
        os.makedirs(self.gallery_dir, exist_ok=True)
        
        # Metric calculation cache with size limit
        self.metric_cache = OrderedDict()
        self.max_cache_size = max_cache_size
        
        # Define archetype system with configurable conditions
        self.archetype_definitions = archetype_config or self.get_default_archetypes()

    def get_default_archetypes(self):
        """Return default archetype definitions that can be overridden."""
        return {
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

    def safe_normalize(self, values):
        """Safely normalize values with numerical stability."""
        if len(values) == 0:
            return values
            
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Handle constant arrays
        if np.isclose(min_val, max_val):
            return np.ones_like(values) * 0.5
            
        # Normalize with stability term
        return (values - min_val) / (max_val - min_val + 1e-12)

    def sample_nodes(self, phi_vals, max_nodes=1000):
        """Sample nodes deterministically for consistent visualization."""
        if len(phi_vals) <= max_nodes:
            return phi_vals, np.arange(len(phi_vals))
        
        # Use deterministic sampling with fixed seed
        rng = np.random.default_rng(self.SAMPLING_SEED)
        indices = rng.choice(len(phi_vals), max_nodes, replace=False)
        return phi_vals[indices], indices

    def calculate_metrics(self, phi_vals):
        """Calculate standardized metrics for phi values."""
        if len(phi_vals) == 0:
            return {"L": 0.0, "FD": 0.0, "H": 0.0}
        
        # Normalize values
        phi_normalized = self.safe_normalize(phi_vals)
        
        # Calculate metrics
        L = np.mean(phi_normalized)  # Luminosity
        FD = np.std(phi_normalized)   # Fracture Density
        
        # Calculate normalized entropy
        hist, _ = np.histogram(phi_normalized, bins=self._optimal_bins(phi_normalized), density=True)
        H = entropy(hist + 1e-12) / np.log(len(hist)) if len(hist) > 1 else 0.0
        
        return {"L": L, "FD": FD, "H": H}

    def _optimal_bins(self, data):
        """Calculate optimal number of bins for entropy calculation."""
        if len(data) <= 10:
            return 5
        return min(50, max(10, len(data) // 20))

    def calculate_resilience(self, layer, window_size=10):
        """Calculate resilience based on a sliding window of historical values."""
        if not layer.phi_history or len(layer.phi_history) < 2:
            return 1.0
        
        # Extract historical means
        means = []
        for historical_phi in layer.phi_history[-window_size:]:
            if hasattr(historical_phi, 'numpy'):
                means.append(np.mean(historical_phi.numpy()))
            else:
                means.append(np.mean(historical_phi))
        
        if len(means) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        means_array = np.array(means)
        cv = np.std(means_array) / (np.mean(means_array) + 1e-12)
        
        # Bound resilience to reasonable values
        return min(3.0, 1.0 / (cv + 1e-12))

    def get_cached_metrics(self, layer, epoch=None):
        """Get metrics with caching to avoid redundant calculations."""
        cache_key = f"{layer.name}_{epoch if epoch else 'current'}"
        
        if cache_key in self.metric_cache:
            # Move to end to mark as recently used
            self.metric_cache.move_to_end(cache_key)
            return self.metric_cache[cache_key]
        
        # Calculate metrics
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
        
        metrics = self.calculate_metrics(phi_vals)
        
        # Add resilience
        metrics["ρ"] = self.calculate_resilience(layer)
        
        # Cache results with LRU eviction
        self.metric_cache[cache_key] = metrics
        if len(self.metric_cache) > self.max_cache_size:
            self.metric_cache.popitem(last=False)
            
        return metrics

    def clear_cache(self):
        """Clear the metric cache."""
        self.metric_cache.clear()

    def safe_save_figure(self, fig, filename, dpi=120):
        """Safely save figure with error handling."""
        try:
            fig.savefig(filename, dpi=dpi, facecolor='black', bbox_inches='tight')
            return True
        except IOError as e:
            print(f"Error saving figure {filename}: {e}")
            # Try alternate location
            alt_filename = f"/tmp/{os.path.basename(filename)}"
            try:
                fig.savefig(alt_filename, dpi=dpi, facecolor='black', bbox_inches='tight')
                print(f"Figure saved to alternate location: {alt_filename}")
                return True
            except IOError:
                print(f"Could not save figure to alternate location: {alt_filename}")
                return False
        finally:
            plt.close(fig)

    def record_epoch(self):
        """Record the current state of all sigil layers."""
        for layer in self.sigil_layers:
            layer.record_epoch()

    def generate_sigil(self, layer, epoch, dpi=120):
        """Generate a visual sigil representing a layer's current state."""
        # Get metrics
        metrics = self.get_cached_metrics(layer, epoch)
        L_l, FD_l, H_l, ρ_l = metrics["L"], metrics["FD"], metrics["H"], metrics["ρ"]
        
        # Get phi values with proper device handling
        phi_tensor = layer.get_phi()
        if isinstance(phi_tensor, torch.Tensor):
            phi_vals = phi_tensor.detach().cpu().numpy().flatten()
        else:
            phi_vals = np.array(phi_tensor).flatten()
            
        # Normalize values
        phi_normalized = self.safe_normalize(phi_vals)
        
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
        phi_sampled, _ = self.sample_nodes(phi_normalized, 1000)
        num_nodes = len(phi_sampled)
        
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
        self.safe_save_figure(fig, filename, dpi)
        
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
                metrics = self.get_cached_metrics(layer)
                L_l, FD_l, H_l, ρ_l = metrics["L"], metrics["FD"], metrics["H"], metrics["ρ"]
                
                archetype = "The Wanderer"
                for name, definition in self.archetype_definitions.items():
                    if definition["conditions"](L_l, FD_l, H_l, ρ_l):
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
        self.safe_save_figure(fig, filename)
        
        return filename

    def summary_metrics(self, epoch=None):
        """Get comprehensive metrics for all layers."""
        metrics = []
        for layer in self.sigil_layers:
            layer_metrics = self.get_cached_metrics(layer, epoch)
            
            # Get archetype
            archetype = layer.archetype if hasattr(layer, 'archetype') else "Unknown"
            
            metrics.append({
                "layer": layer.name,
                "archetype": archetype,
                "epoch": epoch if epoch is not None else "current",
                "Luminosity": layer_metrics["L"],
                "FractureDensity": layer_metrics["FD"],
                "Entropy": layer_metrics["H"],
                "Resilience": layer_metrics["ρ"]
            })
        return metrics

    def create_evolution_animation(self, layer_name, start_epoch=0, end_epoch=None, fps=5):
        """Create an animation showing the evolution of a layer's sigil over time."""
        try:
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
                    
                # Normalize values
                phi_normalized = self.safe_normalize(phi_vals)
                
                # Calculate metrics
                metrics = self.calculate_metrics(phi_normalized)
                L_l, FD_l, H_l = metrics["L"], metrics["FD"], metrics["H"]
                
                # Calculate resilience for this specific epoch
                ρ_l = self.calculate_resilience_for_epoch(layer, frame)
                
                # Determine archetype
                archetype = "The Wanderer"
                cmap = self.archetype_definitions["The Wanderer"]["cmap"]
                
                for name, definition in self.archetype_definitions.items():
                    if definition["conditions"](L_l, FD_l, H_l, ρ_l):
                        archetype = name
                        cmap = definition["cmap"]
                        break
                
                # Sample if too many nodes
                phi_sampled, _ = self.sample_nodes(phi_normalized, 1000)
                num_nodes = len(phi_sampled)
                
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
                ax.set_title(f"{layer.name}\n{archetype}\nEpoch: {frame}", color=title_color, pad=20, fontsize=14)
                
                return ax
            
            ani = animation.FuncAnimation(fig, update, frames=epochs, 
                                         interval=1000//fps, blit=False)
            
            # Save animation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.gallery_dir}/evolution_{layer_name}_{timestamp}.gif"
            ani.save(filename, writer='pillow', fps=fps)
            
            return filename
        except Exception as e:
            print(f"Error creating animation: {e}")
            return None
        finally:
            plt.close('all')  # Ensure all figures are closed

    def calculate_resilience_for_epoch(self, layer, epoch, window_size=5):
        """Calculate resilience for a specific epoch using a window of previous epochs."""
        if not layer.phi_history or epoch < window_size:
            return 1.0
        
        # Extract historical means from the window before this epoch
        means = []
        for i in range(max(0, epoch - window_size), epoch):
            historical_phi = layer.phi_history[i]
            if hasattr(historical_phi, 'numpy'):
                means.append(np.mean(historical_phi.numpy()))
            else:
                means.append(np.mean(historical_phi))
        
        if len(means) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        means_array = np.array(means)
        cv = np.std(means_array) / (np.mean(means_array) + 1e-12)
        
        # Bound resilience to reasonable values
        return min(3.0, 1.0 / (cv + 1e-12))

    def validate_archetypes(self, validation_data):
        """Validate archetype classifications against known network states."""
        results = {}
        
        for archetype_name, definition in self.archetype_definitions.items():
            # Test archetype conditions on validation data
            matches = []
            for data_point in validation_data:
                # Extract metrics from validation data point
                L, FD, H, ρ = data_point.get('metrics', (0, 0, 0, 1))
                matches.append(definition["conditions"](L, FD, H, ρ))
            
            # Calculate precision and support
            precision = np.mean(matches) if matches else 0
            results[archetype_name] = {
                "precision": precision,
                "support": len(matches),
                "description": definition["description"]
            }
        
        return results

    def generate_topology_aware_sigil(self, layer, epoch):
        """Generate sigil that respects network topology."""
        # Get the actual network topology if available
        connectivity = None
        if hasattr(layer, 'get_connectivity'):
            try:
                connectivity = layer.get_connectivity()
            except Exception as e:
                print(f"Error getting connectivity for layer {layer.name}: {e}")
                connectivity = None
        
        # Use appropriate layout based on available connectivity
        if connectivity is not None:
            try:
                import networkx as nx
                G = nx.from_numpy_array(connectivity)
                pos = nx.spring_layout(G, seed=self.SAMPLING_SEED)
            except ImportError:
                print("NetworkX not available, using circular layout")
                num_nodes = len(layer.get_phi())
                theta = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
                pos = {i: (np.cos(theta[i]), np.sin(theta[i])) for i in range(num_nodes)}
        else:
            # Fall back to circular layout
            num_nodes = len(layer.get_phi())
            theta = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
            pos = {i: (np.cos(theta[i]), np.sin(theta[i])) for i in range(num_nodes)}
        
        # Get metrics
        metrics = self.get_cached_metrics(layer, epoch)
        L_l, FD_l, H_l, ρ_l = metrics["L"], metrics["FD"], metrics["H"], metrics["ρ"]
        
        # Get phi values
        phi_tensor = layer.get_phi()
        if isinstance(phi_tensor, torch.Tensor):
            phi_vals = phi_tensor.detach().cpu().numpy().flatten()
        else:
            phi_vals = np.array(phi_tensor).flatten()
            
        # Normalize values
        phi_normalized = self.safe_normalize(phi_vals)
        
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
        if len(phi_normalized) > 1000:
            indices = np.random.default_rng(self.SAMPLING_SEED).choice(
                len(phi_normalized), 1000, replace=False)
            phi_sampled = phi_normalized[indices]
            pos_sampled = {i: pos[idx] for i, idx in enumerate(indices)}
        else:
            phi_sampled = phi_normalized
            pos_sampled = {i: pos[i] for i in range(len(phi_normalized))}
        
        # Plot nodes using topology-aware positions
        for i, phi_val in enumerate(phi_sampled):
            node_size = 30 + (phi_val * 300)
            node_color = cmap(phi_val)
            ax.scatter(pos_sampled[i][0], pos_sampled[i][1], s=node_size, 
                      c=[node_color], alpha=0.9, zorder=3)
        
        # Add text
        title_color = 'white'
        ax.text(0, -1.6, f"L: {L_l:.2f} | FD: {FD_l:.2f} | H: {H_l:.2f} | ρ: {ρ_l:.2f}",
                ha='center', va='top', color=title_color, fontsize=8)
        ax.set_title(f"{layer.name}\n{archetype}\nEpoch: {epoch}", color=title_color, pad=20, fontsize=14)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.gallery_dir}/topology_{layer.name}_{archetype}_epoch_{epoch:04d}_{timestamp}.png"
        self.safe_save_figure(fig, filename)
        
        return filename, archetype

    def calibrate_archetype_thresholds(self, calibration_data):
        """
        Calibrate archetype thresholds based on empirical data.
        This helps address the arbitrariness of the initial thresholds.
        """
        # Collect metrics from calibration data
        all_metrics = []
        for data_point in calibration_data:
            if hasattr(data_point, 'metrics'):
                all_metrics.append(data_point.metrics)
        
        if not all_metrics:
            print("No calibration data available")
            return
        
        # Calculate percentiles for each metric
        L_vals = [m[0] for m in all_metrics]
        FD_vals = [m[1] for m in all_metrics]
        H_vals = [m[2] for m in all_metrics]
        ρ_vals = [m[3] for m in all_metrics] if len(all_metrics[0]) > 3 else [1.0] * len(all_metrics)
        
        # Update archetype thresholds based on percentiles
        self.archetype_definitions["The Oracle"]["conditions"] = lambda L, FD, H, ρ: (
            H > np.percentile(H_vals, 80) and 
            np.percentile(L_vals, 30) < L < np.percentile(L_vals, 70)
        )
        
        self.archetype_definitions["The Sentinel"]["conditions"] = lambda L, FD, H, ρ: (
            ρ > np.percentile(ρ_vals, 85)
        )
        
        self.archetype_definitions["The Alchemist"]["conditions"] = lambda L, FD, H, ρ: (
            FD > np.percentile(FD_vals, 70) and H > np.percentile(H_vals, 60)
        )
        
        self.archetype_definitions["The Archivist"]["conditions"] = lambda L, FD, H, ρ: (
            H < np.percentile(H_vals, 20) and FD < np.percentile(FD_vals, 20)
        )
        
        self.archetype_definitions["The Trickster"]["conditions"] = lambda L, FD, H, ρ: (
            FD > np.percentile(FD_vals, 90)
        )
        
        self.archetype_definitions["The Luminary"]["conditions"] = lambda L, FD, H, ρ: (
            L > np.percentile(L_vals, 90)
        )
        
        print("Archetype thresholds calibrated based on empirical data")

