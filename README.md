# 🛰️ Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Hackathon](https://img.shields.io/badge/Bharat%20Antariksh%20Hackathon-ISRO%20%7C%20MeitY-orange.svg)](https://isro.gov.in)

> **Autonomous Road Network Extraction, Topological Gap Healing, and Urban Disaster Resilience Analysis from High-Resolution Earth Observation Imagery.**

---

## 📖 Overview

During severe monsoon flooding and urban infrastructure failures, road networks in Indian metropolises (Bengaluru, Mumbai, Chennai, Delhi) fragment, hindering disaster management authorities (NDRF / SDMA) and emergency response vehicles.

This project delivers an end-to-end framework that:
1. **Extracts Occlusion-Robust Road Networks** from satellite imagery using **MLPRoadNet** (DSD Encoder + ASPP + MLP-Mixer Bottleneck + Dual Decoder).
2. **Heals Tree Canopy & Shadow Occlusions** using **Directional Angular Gap Healing** and Minimum Spanning Tree (MST) graph reconstruction.
3. **Identifies Critical Urban Chokepoints (Gatekeeper Nodes)** to calculate a city-wide **Resilience Index ($R$)** under simulated disaster collapse scenarios.
4. **Provides an Interactive Web-GIS Command Center** for real-time disaster simulation and dynamic ambulance detour routing.

---

## 🏆 Current Verified Benchmark Results (DeepGlobe Test Set)

| Metric | Score | Performance Assessment |
|---|:---:|---|
| **Intersection over Union (IoU)** | **`0.6256`** | **State-of-the-Art** on complex satellite terrain |
| **Precision** | **`74.51%`** | Sharp road boundaries with minimal rooftop/soil false positives |
| **Recall (Road Detection)** | **`79.60%`** | Captures over 31.8 Million ground-truth road pixels |
| **Specificity** | **`98.39%`** | Near-zero false alarm rate on background forests/buildings |
| **Topological Connectivity (APLS)** | **`91.54%`** | Continuous, routable road graphs |
| **ROC AUC** | **`0.9828`** | High discriminative classification capability |
| **Canopy Occlusions Repaired** | **`1,874+ Gaps`** | Successfully bridged tree canopy and shadow gaps |

---

## 🏗️ Architecture Flow

```
Input Satellite Tile (3 x 512 x 512)
        │
   [DSD Encoder (Dilated Squeeze-and-Excitation)]
        │
   [ASPP Bottleneck (Multi-Scale Receptive Fields)]
        │
   [4-Depth MLP-Mixer (Spatial Token & Channel Mixing)]
        │
   [Dual-Branch Decoder (Road Mask + Centerline Heads)]
        │
   [Sigmoid Fusion Gate] ──► Binary Road Mask & Centerline Probability
        │
   [Phase II: Directional Angular Gap Healing]
        │
   Routable Planar NetworkX Graph G = (V, E)
```

---

## 🚀 Quickstart & Setup Guide for Teammates

### 1. Clone & Install Dependencies

```bash
# Clone repository
git clone https://github.com/OmGawde/Route-Resilience-MLPRoadNet.git
cd Route-Resilience-MLPRoadNet

# Create and activate virtual environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Run Interactive Gap Healing Demo

To visualize how the model extracts roads and heals tree canopy occlusions on test satellite images:

```bash
python scripts/demo_healing.py --num-samples 8
```
*Outputs a 5-column comparison grid to `output/deepglobe_finetuned/results/viz_10_finetuned_comparison.png`.*

---

### 3. Run Full Test Set Evaluation

```bash
python scripts/evaluate_refined.py --checkpoint trained_models/best_model_finetuned.pth
```

---

### 4. Fetch Satellite & Road Vectors for Indian Cities (Track 2)

Auto-harvest ESRI World Imagery satellite tiles and OpenStreetMap road vector masks without manual labeling:

```bash
python scripts/fetch_indian_city.py --city Bengaluru --zoom 17 --num-tiles 10
```
*(Supports `Bengaluru`, `Delhi`, `Mumbai`, and `Hyderabad`)*

---

### 5. Model Training & Fine-Tuning CLI

```bash
# Train from scratch with PyTorch Automatic Mixed Precision (AMP):
python scripts/train.py --data-root dataset/deepgloble_raw --output-dir output/deepglobe_model --epochs 100 --batch-size 8 --num-workers 4

# Run targeted fine-tuning with boosted Topology Loss penalty:
python scripts/finetune.py --checkpoint trained_models/best_model_finetuned.pth --epochs 10 --batch-size 8 --topo-weight 1.5
```

---

## 📂 Project Structure

```
Route-Resilience-MLPRoadNet/
├── configs/
│   └── default.yaml                    # Base training & model hyperparameters
│
├── trained_models/                     # Packaged Production Model Checkpoints
│   ├── best_model_finetuned.pth        # Peak fine-tuned PyTorch checkpoint
│   ├── MLPRoadNet_best_weights.pth     # Packaged weights + dataset normalization metadata
│   ├── MLPRoadNet_finetuned_scripted.pt# Standalone TorchScript compiled model
│   └── README.md                       # Model usage instructions
│
├── src/
│   ├── config.py                       # Configuration & device detection
│   │
│   ├── data/                           # Data pipelines
│   │   ├── dataset.py                  # PyTorch SpaceNet / DeepGlobe dataset loader
│   │   ├── patches.py                  # Fast OpenCV resumable patch extractor
│   │   ├── augmentation.py             # Albumentations high-speed geometric transforms
│   │   └── osm_ingest.py               # Track 2: OpenStreetMap & ESRI tile harvester
│   │
│   ├── models/                         # Neural network architecture
│   │   └── mlproadnet.py               # DSD-Encoder + ASPP + MLP-Mixer + Dual Decoder
│   │
│   ├── losses/                         # Composite multi-task loss
│   │   └── losses.py                   # BCE + Dice + Topology + Consistency Loss
│   │
│   ├── training/                       # Training engine
│   │   └── trainer.py                  # PyTorch AMP trainer with live ETA & early stopping
│   │
│   ├── evaluation/                     # Metric evaluation & TTA
│   │   ├── evaluator.py                # IoU, Precision, Recall, F1, APLS metrics
│   │   ├── tta.py                      # Test-Time Augmentation multi-view ensembling
│   │   └── refinement.py               # Centerline-guided morphological refinement
│   │
│   ├── graph/                          # Graph vectorization & gap healing
│   │   ├── vectorize.py                # Centerline to NetworkX planar graph extractor
│   │   └── healing.py                  # Directional Angular Gap Healing algorithm
│   │
│   └── visualization/                  # Plotting suite
│       └── visualize.py                # Confusion matrix, ROC, and training curve plots
│
├── scripts/                            # Executable CLI tools
│   ├── train.py                        # Training CLI
│   ├── finetune.py                     # Targeted fine-tuning CLI
│   ├── evaluate.py                     # Baseline evaluation CLI
│   ├── evaluate_refined.py             # Full TTA + Gap Healing evaluation CLI
│   ├── demo_healing.py                 # Visual 5-column comparative benchmark CLI
│   ├── infer.py                        # Single-image inference CLI
│   └── fetch_indian_city.py            # Automated Indian city data harvester
│
├── HACKATHON_IMPLEMENTATION_PLAN.md    # Master Hackathon implementation roadmap
├── DASHBOARD_IMPLEMENTATION_PLAN.md    # Standalone Web-GIS dashboard specification
├── requirements.txt                    # Project Python dependencies
└── README.md                           # This document
```

---

## 🗺️ Project Roadmap

- [x] **Phase I**: MLPRoadNet Core AI Segmentation & Multi-Stage Fine-Tuning (**Completed ✅**)
- [x] **Phase II**: Topological Vectorization & Directional Angular Gap Healing (**Completed ✅**)
- [x] **Track 2**: Automated Indian City Satellite & OSM Ingestion (**Completed ✅**)
- [ ] **Phase III**: Graph-Theoretic Criticality Analysis, Gatekeeper Chokepoints & Resilience Index ($R$) (**In Progress ⏳**)
- [ ] **Phase IV**: Interactive Web-GIS Decision Support Dashboard (Streamlit + Folium) (**Queued ⏳**)
- [ ] **Phase V**: GeoJSON / ESRI Shapefile Export for QGIS & Disaster Simulation Overlay

---

## 👥 Authors & Acknowledgments

- **Team MLPRoadNet** — Bharat Antariksh Hackathon
- Architecture based on custom multi-scale MLP-Mixer spatial reasoning for occluded remote sensing.
