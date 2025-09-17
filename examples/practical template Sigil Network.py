import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, RegularPolygon
from scipy.stats import entropy
import os
from datetime import datetime

class SigilLayer(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        out_features = getattr(base_layer, 'out_features', 64)  # fallback
        self.phi = nn.Parameter(torch.ones(out_features))
        self.phi_history = []
        self.name = getattr(base_layer, 'name', base_layer.__class__.__name__)

    def forward(self, x):
        return self.base_layer(x)

    def update_phi(self, loss_map=None):
        # Placeholder: in practice, update phi according to your Kintsugi logic
        # For demo, random drift
        self.phi.data += torch.randn_like(self.phi) * 1e-3

    def record_epoch(self):
        self.phi_history.append(self.phi.detach().cpu().clone())

    def get_phi(self):
        return self.phi

class SigilNetworkManager:
    def __init__(self, model, gallery_dir="sigil_gallery"):
        self.model = model
        self.sigil_layers = [m for m in model.modules() if isinstance(m, SigilLayer)]
        self.gallery_dir = gallery_dir
        os.makedirs(self.gallery_dir, exist_ok=True)

    def record_epoch(self):
        for layer in self.sigil_layers:
            layer.record_epoch()

    def generate_sigil(self, layer, epoch, dpi=120):
        phi_vals = layer.get_phi().detach().cpu().numpy().flatten()
        if np.all(phi_vals == phi_vals[0]):
            phi_vals = phi_vals + np.random.normal(0, 1e-5, phi_vals.shape)
        phi_normalized = (phi_vals - np.min(phi_vals)) / (np.max(phi_vals) - np.min(phi_vals) + 1e-12)
        L_l = np.mean(phi_normalized)
        FD_l = np.std(phi_normalized)
        hist, _ = np.histogram(phi_normalized, bins=min(50, len(phi_vals)//5), density=True)
        H_l = entropy(hist + 1e-12) / np.log(len(hist))
        phi_past = [torch.mean(p).item() for p in layer.phi_history[-10:]] if len(layer.phi_history) > 1 else [L_l]
        if len(phi_past) > 1:
            ρ_l = 1.0 / (np.std(phi_past) / (np.mean(phi_past) + 1e-12) + 1e-12)
            ρ_l = min(ρ_l, 2.0)
        else:
            ρ_l = 1.0
        # Archetype logic
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
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.axis('off')
        num_nodes = len(phi_normalized)
        # Node layout
        if num_nodes > 1000:
            indices = np.random.choice(num_nodes, 1000, replace=False)
            phi_sampled = phi_normalized[indices]
            num_nodes = 1000
        else:
            phi_sampled = phi_normalized
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
        # Nodes
        for i, phi_val in enumerate(phi_sampled):
            node_size = 30 + (phi_val * 300)
            node_color = cmap(phi_val)
            ax.scatter(x[i], y[i], s=node_size, c=[node_color], alpha=0.9, zorder=3)
        boundary_circle = Circle((0, 0), 1.8, fill=False, linestyle='-', linewidth=3, color='white', alpha=0.5)
        ax.add_patch(boundary_circle)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        title_color = 'white'
        ax.text(0, -2.1, f"L: {L_l:.2f} | FD: {FD_l:.2f} | H: {H_l:.2f} | ρ: {ρ_l:.2f}",
                ha='center', va='top', color=title_color, fontsize=8)
        ax.set_title(f"{layer.name}\n{archetype}\nEpoch: {epoch}", color=title_color, pad=20, fontsize=14)
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.gallery_dir}/{layer.name}_{archetype}_epoch_{epoch:04d}_{timestamp}.png"
        fig.savefig(filename, dpi=dpi, facecolor='black', bbox_inches='tight')
        plt.close(fig)
        return filename, archetype

    def generate_gallery(self, epoch):
        gallery_log = []
        for layer in self.sigil_layers:
            fname, archetype = self.generate_sigil(layer, epoch)
            gallery_log.append((layer.name, archetype, fname))
        return gallery_log

    def summary_metrics(self):
        # Optionally, collect and return all metrics for dashboard/README
        metrics = []
        for layer in self.sigil_layers:
            if layer.phi_history:
                phi_vals = layer.phi_history[-1].numpy().flatten()
                L_l = np.mean(phi_vals)
                FD_l = np.std(phi_vals)
                hist, _ = np.histogram(phi_vals, bins=min(50, len(phi_vals)//5), density=True)
                H_l = entropy(hist + 1e-12) / np.log(len(hist))
                metrics.append({
                    "layer": layer.name,
                    "Luminosity": L_l,
                    "FractureDensity": FD_l,
                    "Entropy": H_l
                })
        return metrics

# Example usage
if __name__ == "__main__":
    # Build your network with SigilLayer wrappers
    model = nn.Sequential(
        SigilLayer(nn.Linear(128, 64)),
        SigilLayer(nn.ReLU()),
        SigilLayer(nn.Linear(64, 32)),
        SigilLayer(nn.ReLU()),
        SigilLayer(nn.Linear(32, 10))
    )
    manager = SigilNetworkManager(model)
    for epoch in range(50):
        # ... training logic ...
        for layer in manager.sigil_layers:
            layer.update_phi()  # update phi after each epoch
        manager.record_epoch()
        if epoch % 10 == 0:
            manager.generate_gallery(epoch)

    # Optionally, print summary metrics for README
    print(manager.summary_metrics())
