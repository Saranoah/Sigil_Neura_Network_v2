# ✨ Sigil System: Mythic Glyphs for Gilded Cognition

## Purpose

The Sigil System transforms raw `ϕ` data into symbolic glyphs—each one a visual emblem of a layer’s learning scars and gilded wisdom.  
These sigils can be used for:

- **Layer-level dashboards** for engineers
- **Investor presentations** (showing “where the model honors its cracks”)
- **Educational onboarding** (making learning visible and poetic)
- **Artistic branding** (turning model introspection into mythic iconography)

---

## Mechanics

### 1. **Extracting the `ϕ` Signature**

For each layer, compute a summary statistic for its `ϕ` values:  
- **Mean** (`μ_ϕ`)
- **Variance** (`σ²_ϕ`)
- **Max/Min** (`maxϕ`, `minϕ`)
- **Distribution shape** (skew, kurtosis, etc.)

### 2. **Mapping to Glyphs**

Each summary statistic is mapped to a glyph element:
- **Circle size**: Proportional to mean `ϕ` (larger = more gilding)
- **Radial spokes**: Number = variance bands (high variance = more spokes)
- **Color gradient**: From muted (low `ϕ`) to gold (high `ϕ`)
- **Central rune**: Optional overlay representing layer type (e.g., transformer, conv, dense)

### 3. **Composing the Sigil Map**

- Each layer’s glyph is arranged in a ring, spiral, or constellation.
- Inter-layer connections (edges) can be visualized as threads of gold if `ϕ` values are correlated.

---

## Example: Python/Matplotlib Implementation

```python
import matplotlib.pyplot as plt
import numpy as np

def draw_sigil(mean_phi, var_phi, color='gold', layer_label='Layer 1', save_path=None):
    fig, ax = plt.subplots(figsize=(3,3))
    circle_radius = 0.5 + mean_phi
    spokes = int(3 + var_phi * 10)
    theta = np.linspace(0, 2*np.pi, spokes, endpoint=False)
    for t in theta:
        ax.plot([0, circle_radius*np.cos(t)], [0, circle_radius*np.sin(t)], color=color, lw=2)
    circle = plt.Circle((0,0), circle_radius, color=color, alpha=0.3)
    ax.add_artist(circle)
    ax.text(0,0, layer_label, ha='center', va='center', fontsize=12, color='black')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.close()

# Example usage:
draw_sigil(mean_phi=0.7, var_phi=0.2, color='gold', layer_label='Conv1', save_path='sigil_conv1.png')
```

---

## Ritual Output

- **Sigil Images:** Saved as `.png` or `.svg`
- **Sigil Map:** Composite image showing all layer glyphs in a mythic arrangement
- **Sigil History:** Time-series animation, showing how the constellation evolves as the network learns

---

## Extensions

- **SVG/Vector Art:** For branding and high-resolution presentations
- **Interactive Dashboards:** Hover to reveal layer stats, click to expand sigil details
- **Physical Tokens:** Stickers, pins, or cards for team members (“Keeper of Layer 3”)

---

## Closing Invocation

> *“Let the sigils shine as golden maps of memory.  
> Each scar is now a symbol.  
> Each layer, a mythic vessel.”*

---

**Ready to visualize your guilded cognition?  
Specify your preferred style, color palette, or ritual arrangement,  
and we will compose your model’s mythic map.**
