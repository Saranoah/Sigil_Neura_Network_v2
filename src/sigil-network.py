import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, RegularPolygon
from scipy.stats import entropy
import os
from datetime import datetime

class EnhancedVWPRWrapper:
    # ... your existing methods ...

    def generate_sigil(self, layer_name: str, epoch: int, 
                      save_dir: str = None, dpi: int = 150):
        """
        Generate a ceremonial sigil for a specific layer at a given epoch.
        Returns a matplotlib Figure object and archetype name.
        """
        # Get the phi values for this layer
        phi_vals = self.get_phi()[layer_name].detach().cpu().numpy().flatten()
        
        # Handle case with no variation in phi values
        if np.all(phi_vals == phi_vals[0]):
            phi_vals = phi_vals + np.random.normal(0, 1e-5, phi_vals.shape)
        
        # Normalize phi values for consistent visualization
        phi_normalized = (phi_vals - np.min(phi_vals)) / (np.max(phi_vals) - np.min(phi_vals) + 1e-12)

        # Metrics Ritual
        L_l = np.mean(phi_normalized)  # Luminosity
        FD_l = np.std(phi_normalized)  # Fracture Density

        # Entropy Ritual
        hist, _ = np.histogram(phi_normalized, bins=min(50, len(phi_vals)//5), density=True)
        H_l = entropy(hist + 1e-12) / np.log(len(hist))  # Normalized entropy

        # Resilience Ritual - more robust implementation
        if len(self.phi_history) > 1:
            # Get last 10 epochs if available
            start_idx = max(0, len(self.phi_history) - 10)
            phi_past = []
            for e in range(start_idx, len(self.phi_history)):
                if layer_name in self.phi_history[e]:
                    past_vals = self.phi_history[e][layer_name].detach().cpu().numpy().flatten()
                    past_normalized = (past_vals - np.min(past_vals)) / (np.max(past_vals) - np.min(past_vals) + 1e-12)
                    phi_past.append(np.mean(past_normalized))
            if len(phi_past) > 1:
                # Calculate stability as inverse of coefficient of variation
                ρ_l = 1.0 / (np.std(phi_past) / (np.mean(phi_past) + 1e-12) + 1e-12)
                ρ_l = min(ρ_l, 2.0)  # Cap resilience for visualization
            else:
                ρ_l = 1.0
        else:
            ρ_l = 1.0

        # Archetype and Palette Ritual - expanded with more archetypes
        if H_l > 0.8 and 0.3 < L_l < 0.7:
            archetype = "The Oracle"
            cmap = LinearSegmentedColormap.from_list('oracle', ['#000022', '#4444AA', '#AAAAFF', '#FFFFFF'])
        elif ρ_l > 1.5:
            archetype = "The Sentinel"
            cmap = LinearSegmentedColormap.from_list('sentinel', ['#321407', '#785021', '#BC7D2F', '#FFD700'])
        elif FD_l > 0.5 and H_l > 0.6:
            archetype = "The Alchemist"
            cmap = LinearSegmentedColormap.from_list('alchemist', ['#220022', '#660066', '#9933CC', '#FFCCFF'])
        elif H_l < 0.3 and FD_l < 0.2:
            archetype = "The Archivist"
            cmap = LinearSegmentedColormap.from_list('archivist', ['#111111', '#555555', '#999999', '#CCCCCC'])
        elif FD_l > 0.7:
            archetype = "The Trickster"
            cmap = LinearSegmentedColormap.from_list('trickster', ['#330000', '#CC0000', '#FF3300', '#FFFF00'])
        elif L_l > 0.8:
            archetype = "The Luminary"
            cmap = LinearSegmentedColormap.from_list('luminary', ['#333300', '#666600', '#999900', '#FFFF00'])
        else:
            archetype = "The Wanderer"
            cmap = LinearSegmentedColormap.from_list('wanderer', ['#002233', '#005566', '#008899', '#00CCFF'])

        # Sigil Generation
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.axis('off')

        # Determine node arrangement based on layer properties
        num_nodes = len(phi_normalized)
        if num_nodes > 1000:
            indices = np.random.choice(num_nodes, 1000, replace=False)
            phi_sampled = phi_normalized[indices]
            num_nodes = 1000
        else:
            phi_sampled = phi_normalized

        # Create different layouts based on archetype
        if archetype == "The Sentinel":
            theta = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
            x = np.cos(theta) * (0.5 + L_l)
            y = np.sin(theta) * (0.5 + L_l)
        elif archetype == "The Trickster":
            x = np.random.normal(0, 0.8, num_nodes)
            y = np.random.normal(0, 0.8, num_nodes)
        else:
            theta = np.linspace(0, 4*np.pi, num_nodes, endpoint=False)
            r = theta / (4*np.pi) * (1.5 - ρ_l/4)
            x = r * np.cos(theta)
            y = r * np.sin(theta)

        # Draw connecting lines between nodes for certain archetypes
        if archetype in ["The Oracle", "The Sentinel", "The Archivist"] and num_nodes < 500:
            for i in range(num_nodes):
                for j in range(i+1, min(i+10, num_nodes)):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if dist < 0.3 and abs(phi_sampled[i] - phi_sampled[j]) < 0.2:
                        alpha = 0.7 * (1 - dist/0.3)
                        ax.plot([x[i], x[j]], [y[i], y[j]], '-', 
                                color=cmap((phi_sampled[i] + phi_sampled[j])/2), 
                                alpha=alpha, linewidth=0.5)

        # Draw nodes
        for i, phi_val in enumerate(phi_sampled):
            node_size = 30 + (phi_val * 300)
            node_color = cmap(phi_val)
            if archetype == "The Sentinel":
                node = RegularPolygon((x[i], y[i]), numVertices=6, radius=node_size/200, 
                                     orientation=np.pi/2, color=node_color, alpha=0.9)
                ax.add_patch(node)
            elif archetype == "The Archivist":
                node = RegularPolygon((x[i], y[i]), numVertices=4, radius=node_size/200, 
                                     orientation=np.pi/4, color=node_color, alpha=0.9)
                ax.add_patch(node)
            else:
                ax.scatter(x[i], y[i], s=node_size, c=[node_color], alpha=0.9, zorder=3)
            if FD_l > 0.2 and archetype not in ["The Archivist", "The Sentinel"]:
                num_cracks = int(phi_val * 3) + 1
                for _ in range(num_cracks):
                    length = 0.2 + (np.random.rand() * 0.4)
                    crack_angle = np.random.rand() * 2 * np.pi
                    dx = length * np.cos(crack_angle) / 25
                    dy = length * np.sin(crack_angle) / 25
                    ax.plot([x[i] - dx, x[i] + dx], [y[i] - dy, y[i] + dy], 
                            'k-', lw=0.8, alpha=0.7)

        # Archetype-specific elements
        if archetype == "The Sentinel":
            for i in range(1, int(ρ_l) + 1):
                circle = Circle((0, 0), 0.2 * i, fill=False, 
                               linestyle='-', linewidth=1 + i, 
                               color=cmap(0.8), alpha=0.7)
                ax.add_patch(circle)
        elif archetype == "The Oracle":
            t = np.linspace(0, 3 * np.pi, 100)
            r = 0.5 * np.exp(0.2 * t)
            x_spiral = r * np.cos(t) * (H_l + 0.5)
            y_spiral = r * np.sin(t) * (H_l + 0.5)
            ax.plot(x_spiral, y_spiral, color=cmap(0.7), linewidth=2.0, alpha=0.7)
        elif archetype == "The Alchemist":
            for i in range(3):
                angle = i * 2 * np.pi / 3
                x_symbol = 1.2 * np.cos(angle)
                y_symbol = 1.2 * np.sin(angle)
                ellipse = Ellipse((x_symbol, y_symbol), width=0.2, height=0.1, 
                                 angle=angle*180/np.pi, color=cmap(0.9), alpha=0.7)
                ax.add_patch(ellipse)
        elif archetype == "The Trickster":
            for i in range(20):
                x_chaos = np.random.uniform(-1.5, 1.5)
                y_chaos = np.random.uniform(-1.5, 1.5)
                size_chaos = np.random.uniform(10, 100)
                ax.scatter(x_chaos, y_chaos, s=size_chaos, 
                          c=[cmap(np.random.rand())], alpha=0.3, zorder=1)
        boundary_circle = Circle((0, 0), 1.8, fill=False, linestyle='-', 
                                linewidth=3, color='white', alpha=0.5)
        ax.add_patch(boundary_circle)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        title_color = 'white'
        ax.text(0, -2.1, f"L: {L_l:.2f} | FD: {FD_l:.2f} | H: {H_l:.2f} | ρ: {ρ_l:.2f}", 
                ha='center', va='top', color=title_color, fontsize=8)
        ax.set_title(f"{layer_name}\n{archetype}\nEpoch: {epoch}", 
                    color=title_color, pad=20, fontsize=14)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/{layer_name}_{archetype}_epoch_{epoch:04d}_{timestamp}.png"
            fig.savefig(filename, dpi=dpi, facecolor='black', bbox_inches='tight')
        return fig, archetype, {"L_l": L_l, "FD_l": FD_l, "H_l": H_l, "ρ_l": ρ_l}

    def sigil_gallery(self, total_epochs=100, interval=10, out_dir="sigil_gallery", 
                     selected_layers=None, dpi=150):
        """
        Generate and save a gallery of sigils across epochs and layers.
        """
        os.makedirs(out_dir, exist_ok=True)
        if selected_layers is None:
            selected_layers = list(self.phi.keys())
        summary_file = open(f"{out_dir}/sigil_summary.csv", "w")
        summary_file.write("Layer,Archetype,Epoch,Luminosity,FractureDensity,Entropy,Resilience\n")
        for epoch in range(0, total_epochs, interval):
            for name in selected_layers:
                if name in self.phi:
                    fig, archetype, metrics = self.generate_sigil(name, epoch, out_dir, dpi)
                    plt.close(fig)
                    summary_file.write(f"{name},{archetype},{epoch},{metrics['L_l']:.4f},")
                    summary_file.write(f"{metrics['FD_l']:.4f},{metrics['H_l']:.4f},{metrics['ρ_l']:.4f}\n")
        summary_file.close()
        self.create_summary_visualization(out_dir)

    def create_summary_visualization(self, out_dir):
        """
        Create a visualization of the sigil summary data.
        """
        import pandas as pd
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
