import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import svgwrite

def oracle_sigil(mean_phi, entropy, resilience, outliers, svg_path=None):
    # Matplotlib version (for quick prototyping)
    fig, ax = plt.subplots(figsize=(4,4))
    arms = int(5 + entropy*10)
    theta = np.linspace(0, 2*np.pi, arms)
    for t in theta:
        r = np.linspace(0.3, 1.2, 100)
        ax.plot(r*np.cos(t + r*entropy), r*np.sin(t + r*entropy), color='indigo', alpha=0.7, lw=2)
    for val in outliers:
        ax.scatter(np.cos(val)*1.2, np.sin(val)*1.2, s=80, c='gold', alpha=0.8)
    circle = Circle((0,0), 1.5, color='violet', alpha=resilience*0.2)
    ax.add_artist(circle)
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.axis('off')
    plt.savefig('oracle_sigil.png')
    plt.close()
    # SVG version for branding
    if svg_path:
        dwg = svgwrite.Drawing(svg_path, size=("400px", "400px"))
        # Outer glow
        dwg.add(dwg.circle(center=(200,200), r=150, fill="violet", fill_opacity=resilience*0.2))
        # Spiral arms
        for t in theta:
            points = [(200 + (30 + r*80)*np.cos(t + r*entropy), 200 + (30 + r*80)*np.sin(t + r*entropy)) for r in np.linspace(0.3, 1.2, 20)]
            dwg.add(dwg.polyline(points=points, stroke="indigo", fill="none", stroke_width=3, opacity=0.7))
        # Outlier nodes
        for val in outliers:
            x, y = 200 + 120*np.cos(val), 200 + 120*np.sin(val)
            dwg.add(dwg.circle(center=(x, y), r=12, fill="gold", fill_opacity=0.8))
        dwg.save()

def alchemist_sigil(mean_phi, variance, epochs, svg_path=None):
    # Matplotlib version
    fig, ax = plt.subplots(figsize=(4,4))
    for i in range(epochs):
        circle = Circle((0,0), 0.5 + i*0.2, color='gold', alpha=0.15 + 0.1*i)
        ax.add_artist(circle)
    spokes = int(8 + variance*12)
    theta = np.linspace(0, 2*np.pi, spokes, endpoint=False)
    for t in theta:
        ax.plot([0, np.cos(t)], [0, np.sin(t)], color='crimson', lw=2)
    ax.scatter(0,0, s=100, c='lime', marker='o')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.axis('off')
    plt.savefig('alchemist_sigil.png')
    plt.close()
    # SVG version for branding
    if svg_path:
        dwg = svgwrite.Drawing(svg_path, size=("400px", "400px"))
        # Nested rings
        for i in range(epochs):
            dwg.add(dwg.circle(center=(200,200), r=50 + i*20, fill="gold", fill_opacity=0.15 + 0.1*i))
        # Radial spokes
        for t in theta:
            x = 200 + 90*np.cos(t)
            y = 200 + 90*np.sin(t)
            dwg.add(dwg.line(start=(200,200), end=(x,y), stroke="crimson", stroke_width=4))
        # Central rune
        dwg.add(dwg.circle(center=(200,200), r=18, fill="lime"))
        dwg.save()
