import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os

def load_sigil_image(path, zoom=0.18):
    # Load an image as a thumbnail for a node
    img = plt.imread(path)
    return OffsetImage(img, zoom=zoom)

def sigil_constellation_map(layer_infos, out_path="sigil_constellation.png", figsize=(14, 12)):
    """
    Draws a constellation map of the sigil network, with images as node glyphs
    layer_infos: list of dicts, each dict contains:
        {
            'layer_name': str,
            'archetype': str,
            'sigil_img_path': str,
            'metrics': dict,
            'neighbors': list of connected layer_names (optional)
        }
    """
    # Build graph
    G = nx.Graph()
    for info in layer_infos:
        G.add_node(info['layer_name'], archetype=info['archetype'],
                   sigil_img_path=info['sigil_img_path'], metrics=info['metrics'])
        # Add edges if neighbors are specified
        for nbr in info.get('neighbors', []):
            G.add_edge(info['layer_name'], nbr)

    # Layout: spring by default, circular for <7 nodes
    if len(G.nodes) < 7:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.2 / np.sqrt(len(G.nodes)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("black")

    # Draw edges
    edge_colors = []
    for u, v in G.edges:
        edge_colors.append("white")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=2, alpha=0.25)

    # Draw nodes as sigil images
    for node in G.nodes:
        info = G.nodes[node]
        imagebox = load_sigil_image(info['sigil_img_path'], zoom=0.16)
        ab = AnnotationBbox(imagebox, pos[node], frameon=False, bboxprops=None)
        ax.add_artist(ab)
        # Add label below node
        ax.text(pos[node][0], pos[node][1] - 0.15, f"{node}\n{info['archetype']}",
                color="white", fontsize=11, ha="center", va="top", fontweight="bold")

    # Optionally, add metric tooltips or overlay
    for node in G.nodes:
        m = G.nodes[node]["metrics"]
        metrics_str = f"L:{m['L_l']:.2f} FD:{m['FD_l']:.2f}\nH:{m['H_l']:.2f} ρ:{m['ρ_l']:.2f}"
        ax.text(pos[node][0], pos[node][1] + 0.13, metrics_str,
                color="cyan", fontsize=8, ha="center", va="bottom", fontweight="normal")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_title("Sigil Network Constellation Map", color="gold", fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

# Example usage:
if __name__ == "__main__":
    # Example layer_infos, replace with real layer names, archetypes, sigil image paths
    layer_infos = [
        {
            "layer_name": "Encoder",
            "archetype": "The Oracle",
            "sigil_img_path": "sigil_gallery/Encoder_TheOracle_epoch_0020_20250911_230215.png",
            "metrics": {"L_l": 0.34, "FD_l": 0.22, "H_l": 0.92, "ρ_l": 1.23},
            "neighbors": ["Midlayer"]
        },
        {
            "layer_name": "Midlayer",
            "archetype": "The Alchemist",
            "sigil_img_path": "sigil_gallery/Midlayer_TheAlchemist_epoch_0020_20250911_230215.png",
            "metrics": {"L_l": 0.51, "FD_l": 0.60, "H_l": 0.80, "ρ_l": 1.01},
            "neighbors": ["Encoder", "Classifier"]
        },
        {
            "layer_name": "Classifier",
            "archetype": "The Sentinel",
            "sigil_img_path": "sigil_gallery/Classifier_TheSentinel_epoch_0020_20250911_230215.png",
            "metrics": {"L_l": 0.77, "FD_l": 0.10, "H_l": 0.35, "ρ_l": 1.75},
            "neighbors": ["Midlayer"]
        }
    ]
    sigil_constellation_map(layer_infos)
