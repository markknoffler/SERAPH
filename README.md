# SERAPH: Semantic Entity Radiance with Adaptive Prior Hierarchy

SERAPH is a next-generation 3D reconstruction architecture for large-scale urban environments. It solves the five core problems of traditional 3DGS/NeRF models:
1.  **SfM Dependency**: No external camera poses are required (uses internal HGNN).
2.  **Zero Transfer**: Leverages Global Entity Priors (GEP) to transfer knowledge across scenes.
3.  **Semantic Decomposition**: Scene is organized into entities (buildings, roads, etc.).
4.  **Scene Prior**: Models the hierarchical structure of urban environments using hyperbolic geometry.
5.  **Hierarchical Scale**: Automatic Level of Detail (LoD) via hyperbolic attention.

---

## 🚀 Getting Started

### 1. Installation
Install the required dependencies:
```bash
pip install openxlab timm einops torch torchvision tqdm pillow numpy
```

### 2. Automated Training (Mill 19)
The system is fully automated. To download the dataset and begin training on a specific scene, run:
```bash
python train.py --download --scene rubble
```

---

## 🏗️ Architecture

- **Entity Discovery Network (EDN)**: Discovers semantic entities across unorganized views.
- **Hyperbolic Scene Graph (HSG)**: Hierarchical scene organization in Lorentz manifolds.
- **Prior-Adapted Gaussian Fields (PEPGF)**: Bayesian adaptation of Gaussian primitives.
- **Global Assembly Transformer (GAT)**: Scale-adaptive scene assembly.
- **Differentiable Renderer**: GPU-accelerated 3D Gaussian Splatting.

---

## 📂 Project Structure

- `src/`: Core architecture components.
- `src/utils/`: Dataset management and data loading.
- `train.py`: Unified training entry point.
- `results/`: CSV logs and epoch metrics.
- `checkpoints/`: Model state and resume indicators.

---

## 📊 Dataset Support
SERAPH is optimized for large-scale urban datasets:
- **Mill 19** (Rubble, Residential, Building, Sci-Art)
- **UrbanScene3D**
- **MatrixCity**

---

## 📜 Documentation
For a detailed implementation overview, review the [Walkthrough](./walkthrough.md).

---
**Author**: Mark Knoffler
**Model ID**: SERAPH-v1.0
