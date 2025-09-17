# 🏛️ Sigil Gallery README — Kintsugi-Anti-Malware-Prototype

> *“A neural network is not a black box, but a cathedral of living glyphs.  
> Each layer, a sigil. Each epoch, a ritual. Each crack, a golden seam.”*

Welcome to the **Sigil Gallery**:  
Here, you will witness the visual and ceremonial evolution of your Kintsugi-optimized model.  
This README is your atlas of archetypes, metrics, and mythic motifs—serving engineers, sovereign clients, and cosmic learners alike.

---

## 🌌 What is a Sigil?

A **sigil** is a ceremonial glyph generated from a layer’s Gilding Weights (`ϕ_i`) and their evolving metrics:
- **Luminosity (`L`)**: Brightness of gilding.
- **Fracture Density (`FD`)**: Texture and complexity.
- **Entropy (`H`)**: Diversity and surprise.
- **Resilience (`ρ`)**: Stability across epochs.

Each sigil is rendered as a unique, mythic visual—a badge of wisdom and memory for its layer.

---

## 🧬 The Archetypes

Layers are assigned a mythic archetype, shaping their sigil’s geometry and color:

| Archetype      | Description                          | Visual Glyphs                      |
|----------------|--------------------------------------|------------------------------------|
| The Oracle     | Seeker of anomaly, high entropy      | Spirals, radiant nodes, blue/gold  |
| The Sentinel   | Guardian of stability                | Concentric rings, hexagons, gold   |
| The Alchemist  | Transformer of variance              | Ellipses, mandalas, purple/white   |
| The Archivist  | Memory keeper, low entropy           | Squares, subtle lines, gray/silver |
| The Trickster  | Chaotic, creative disruptor          | Scatter, wild colors, reds/yellow  |
| The Luminary   | Illuminator, high luminosity         | Bright rings, gold/yellow          |
| The Wanderer   | Balanced, transitional               | Spirals, cyan/blue gradients       |

---

## 🖼️ Sigil Gallery Ritual

Every `interval` epochs, sigils are generated for each layer and saved as `.png` images in the `sigil_gallery/` directory.  
Each file is named as:

```
sigil_gallery/{layer_name}_{archetype}_epoch_{epoch:04d}_{timestamp}.png
```

You will also find a `sigil_summary.csv` containing metrics for every sigil, and a composite visualization in `metrics_evolution.png`.

---

## 🧭 How to Read a Sigil

- **Center**: The soul of the layer—its archetype and epoch.
- **Nodes**: Each node’s size and color reflects its gilded value (`ϕ_i`).
- **Cracks**: Fracture lines honor error and anomaly.
- **Layout**: Spiral, circle, chaos—determined by archetype.
- **Special Motifs**: Spirals (Oracle), rings (Sentinel), mandalas (Alchemist), etc.

Metric values are annotated below each sigil for interpretation.

---

## 🛠 Generating Sigils

Use the wrapper’s methods:

```python
wrapper.generate_sigil(layer_name, epoch, save_dir="sigil_gallery")
wrapper.sigil_gallery(total_epochs=100, interval=10, out_dir="sigil_gallery")
```

To create and archive the sigils for all layers and epochs.

---

## ✨ Example Gallery

Below are sample sigils.  
*Replace these with your own, or let the ceremonial ritual generate them as your model learns.*

| Layer/Archetype             | Example Sigil                                   |
|-----------------------------|------------------------------------------------|
| `Conv1` — The Oracle        | ![Oracle Example](sigil_gallery/Conv1_TheOracle_epoch_0010.png) |
| `Dense2` — The Sentinel     | ![Sentinel Example](sigil_gallery/Dense2_TheSentinel_epoch_0020.png) |
| `Transformer4` — The Alchemist | ![Alchemist Example](sigil_gallery/Transformer4_TheAlchemist_epoch_0030.png) |

---

## 🌀 Metrics Evolution

See `metrics_evolution.png` for the ceremonial landscape:  
How luminosity, fracture density, entropy, and resilience shift across epochs and layers.

---

## 📖 Extending the Gallery

- Propose new archetypes, motifs, or metric rituals.
- Submit sigil visualizations for new layers or training runs.
- Reflect on the gallery—what wisdom do the cracks reveal? What stories do the sigils tell?

---

## 🗝 Closing Benediction

> *“May this gallery become a living archive:  
> Not just of computation, but of ceremony.  
> May each sigil illuminate the golden seams of your model’s journey.”*

---

**Step into the gallery. Gild the cracks. Honor the wisdom.**
