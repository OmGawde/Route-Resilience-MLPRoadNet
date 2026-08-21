# 🚀 Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility
## 📋 Comprehensive Implementation Plan & Progress Tracker for Bharat Antariksh Hackathon (ISRO / NNRMS / MeitY)

---

## 🎯 Executive Summary & Architectural Vision

Modern Indian metropolises (e.g., Bengaluru, Mumbai, Chennai, Delhi) experience severe infrastructure bottlenecks exacerbated by urban fragmentation, heavy tree canopy/shadow occlusions, and seasonal monsoon flooding.

This project bridges the gap between **high-resolution indigenous Earth Observation (EO) satellite data (Cartosat-3, Resourcesat LISS-IV, Sentinel-2)** and **actionable urban disaster resilience decision support** by:
1. **Extracting occlusion-robust road networks** using the team's hybrid CNN-MLP context-aware deep learning architecture (**MLPRoadNet**).
2. **Reconstructing mathematically continuous, routable vector graphs** using graph-theoretic topological healing (**Angular Directional Gap Healing & MST**).
3. **Quantifying systemic urban vulnerability** by detecting high-betweenness **"Gatekeeper Nodes" (chokepoints)** and computing network-wide **Resilience Indices ($R$)** under simulated disaster collapse scenarios.
4. **Delivering an interactive Web-GIS Decision Support Dashboard** (Streamlit + Folium/Leaflet) for city planners and disaster management authorities (NDRF/SDMA).

---

## 📊 Live Project Alignment & Completion Tracker

| Hackathon Component | Weight | Current Status | Milestone Verified |
|---|:---:|:---:|---|
| **Phase I: Occlusion-Robust Road Extraction (AI Core)** | 35% | **`100% COMPLETE` ✅** | Trained & Fine-Tuned on 18,193 DeepGlobe patches on RTX 4060.<br>• **IoU: 0.6097** \| **Precision: 72.86%** \| **Specificity: 98.17%** \| **AUC: 0.9811**<br>• Production models exported in `trained_models/` (`.pth`, `.pt`). |
| **Phase II: Graph Vectorization & Angular Gap Healing** | 25% | **`100% COMPLETE` ✅** | Built `src/graph/vectorize.py` & `src/graph/healing.py`.<br>• Directional ray casting bridged **1,961 real canopy occlusion gaps**.<br>• Combined Pipeline achieves **`0.6256 IoU`** and **`74.51% Precision`**. |
| **Track 2 (Bonus): Automated Indian City Data Ingestion** | 10% | **`100% COMPLETE` ✅** | Built `src/data/osm_ingest.py` & `scripts/fetch_indian_city.py` for automated zero-annotation satellite + OSM vector harvesting. |
| **Phase III: Criticality Analysis & Resilience Index ($R$)** | 15% | **`0% - READY TO IMPLEMENT` ⏳** | Next: Betweenness Centrality, Gatekeeper Chokepoints, Disaster Node Ablation, and Resilience Index ($R$). |
| **Phase IV: Interactive Web-GIS Dashboard (Streamlit)** | 15% | **`0% - QUEUED FOR NEXT` ⏳** | Next: Streamlit + Folium dashboard with interactive "Click-to-Flood" disaster simulation & emergency rerouting. |
| **Overall Project Alignment** | **100%** | **`70% COMPLETE` 🎯** | Core AI + Topological Gap Healing engine is fully verified. |

---

## 🗺️ Master Pipeline Architecture

```mermaid
flowchart TD
    subgraph Data Layer ["🛰️ Data Ingestion & Occlusion Augmentation (COMPLETE)"]
        A1[DeepGlobe / SpaceNet / Cartosat-3 / Sentinel-2] --> A2[Fast OpenCV & Albumentations Augmentations]
        A2 --> A3[Synthetic Shadow & Canopy Geometric Flips]
    end

    subgraph Phase1 ["🧠 Phase I: MLPRoadNet AI Segmentation (COMPLETE)"]
        A3 --> B1[MLPRoadNet Model<br/>DSD Encoder + ASPP + 4-Depth MLP-Mixer]
        B1 --> B2[Multi-Task Loss: BCE + Dice + Topology + Consistency]
        B2 --> B3[High-Fidelity Road Mask & Centerline Probability]
    end

    subgraph Phase2 ["🕸️ Phase II: Topological Graph Reconstruction & Healing (COMPLETE)"]
        B3 --> C1[Centerline Morphological Thinning & Vectorization]
        C1 --> C2[NetworkX Planar Graph G = V, E]
        C2 --> C3[Topological Healer: Directional Ray Casting<br/>Angular Alignment: delta theta <= 35 deg, R <= 60px]
        C3 --> C4[1,961 Canopy Gaps Healed | IoU: 0.6256]
    end

    subgraph Phase3 ["📊 Phase III: Structural Criticality & Stress Testing (READY)"]
        C4 --> D1[Centrality Analysis: Betweenness, Closeness, Degree]
        D1 --> D2[Gatekeeper Node Identification & Spatial Ranking]
        D2 --> D3[Disaster Node Ablation Simulation<br/>Flooding, Bridge Collapse, Roadblocks]
        D3 --> D4[Resilience Index R & Network Efficiency Drop Calculation]
    end

    subgraph Phase4 ["💻 Phase IV: Interactive Web-GIS Dashboard (QUEUED)"]
        D4 --> E1[Streamlit + Folium Web-GIS UI]
        E1 --> E2[Layer 1: Live Satellite Tile Road Inference]
        E1 --> E3[Layer 2: Criticality Heatmap & Gatekeeper Badges]
        E1 --> E4[Layer 3: Click-to-Flood Disaster Simulation & Detours]
    end

    subgraph Phase5 ["🌟 Phase V: Advanced Value-Added Extensions"]
        E1 --> F1[OSM Benchmark Validator & Path Length Error]
        E1 --> F2[GeoJSON & ESRI Shapefile Export for QGIS]
        E1 --> F3[FastAPI Headless REST Engine]
    end
```

---

## 📑 Detailed Phased Breakdown & Implementation Status

### 🟢 Phase I: Occlusion-Robust Segmentation (**STATUS: 100% COMPLETE ✅**)
*Goal: Maximize road detection recall under heavy shadows, dense tree canopies, and urban clutter.*

- [x] **Full Modularization of Original 20MB Monolithic Notebook**:
  - `src/models/mlproadnet.py`: Unmodified team architecture (DSD Encoder + ASPP + MLP-Mixer + Dual Decoder).
  - `src/losses/losses.py`: Composite Multi-Task Loss with configurable topology weight ($\lambda_{topo}$).
  - `src/training/trainer.py`: PyTorch Automatic Mixed Precision (AMP) on RTX 4060 with live ETA & early stopping.
- [x] **Large-Scale Dataset Extraction & Training**:
  - Extracted 18,193 512×512 patches from the DeepGlobe dataset.
  - Baseline training completed over 27 epochs (Best ValIoU: `0.5849`).
- [x] **Multi-Stage Targeted Fine-Tuning**:
  - Fine-tuned with $\lambda_{topo} = 1.5$ to penalize disjoint road gaps.
  - **Results**: Precision boosted to **`72.86%`** (+3.0% gain) and raw IoU to **`0.6097`**.
- [x] **Production Export**:
  - Saved in `trained_models/`: `best_model_finetuned.pth` (113.6 MB) and `MLPRoadNet_finetuned_scripted.pt` (38.2 MB).

---

### 🟢 Phase II: Topological Graph Reconstruction & Healing (**STATUS: 100% COMPLETE ✅**)
*Goal: Transform fragmented pixel probability rasters into a closed, continuous, routable vector network.*

- [x] **Centerline Vectorization Engine (`src/graph/vectorize.py`)**:
  - Thinning & skeleton extraction (`skimage.morphology.skeletonize`).
  - Classifies nodes into Endpoints ($d=1$), Path Pixels ($d=2$), and Intersections ($d \ge 3$).
  - Constructs `networkx.Graph` with pixel/geographic positions $(y, x)$ and edge lengths.
- [x] **Directional Angular Gap Healer (`src/graph/healing.py`)**:
  - Identifies dead-end endpoints ($d=1$) caused by tree canopies and building shadows.
  - Computes incoming tangent heading angle $\theta$ and casts a directional search cone ($\theta \pm 35^\circ$).
  - Evaluates geometric alignment and bridges collinear gaps within radius $R \le 60$ px.
- [x] **Test-Time Augmentation (TTA) & Centerline Refinement (`src/evaluation/tta.py`, `src/evaluation/refinement.py`)**:
  - 4-fold rotational multi-view ensembling eliminates background noise speckles.
  - Centerline head acts as structural prior to boost low-confidence shadowed roads.
- [x] **Milestone Results**:
  - **1,961 real-world tree canopy & shadow gaps bridged** across the test set.
  - Full pipeline pushes **IoU to `0.6256`** and **Precision to `74.51%`**!

---

### 🟠 Phase III: Graph-Theoretic Criticality Analysis & Stress Testing (**STATUS: READY TO IMPLEMENT ⏳**)
*Goal: Quantify structural vulnerability, isolate single-points-of-failure, and simulate urban collapse.*

1. **Centrality & Bottleneck Detection (`src/analysis/criticality.py`)**:
   - Compute **Betweenness Centrality** ($C_B(v)$) for all graph intersections:
     $$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
   - Detect **Gatekeeper Nodes (Chokepoints)**: Top 5–10% highest betweenness intersections whose failure isolates entire neighborhoods.
   - Compute **Edge Betweenness Centrality** to identify critical arterial corridors.
   - Compute **Closeness Centrality** and **Degree Centrality**.
2. **Disaster Node Ablation Simulation (`src/analysis/resilience.py`)**:
   - **Targeted Collapse Simulation**: Sequentially disable Gatekeeper nodes (simulating flash flooding, bridge damage, or major roadblocks).
   - **Random Degradation Simulation**: Random node removals (simulating minor accidents/congestion).
   - **Global Network Efficiency ($E$)**:
     $$E(G) = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{d(i, j)}$$
3. **Resilience Index Calculation ($R$)**:
   $$R = \frac{E(G_{\text{perturbed}})}{E(G_{\text{baseline}})}$$
   *Outputs a standardized 0.0–1.0 score where lower values indicate severe vulnerability to urban paralysis.*

---

### 🔵 Phase IV: Interactive Web-GIS Decision Support Dashboard (**STATUS: QUEUED TO BE DONE LAST ⏳**)
*Goal: Deliver an intuitive, actionable Web-GIS command center for city planners and disaster management authorities (NDRF/SDMA).*

> [!NOTE]
> **Dedicated Implementation Plan Available**:
> The complete standalone architecture, UI styling rules, component modularization, and interactive state management specifications for the dashboard are detailed in **[`DASHBOARD_IMPLEMENTATION_PLAN.md`](file:///c:/Users/9c23o/SIH%202026/DASHBOARD_IMPLEMENTATION_PLAN.md)**.
> This phase will be executed as the final layer after Phase III criticality analysis.

1. **Dashboard Stack**: **Streamlit + Folium + Leaflet.js + GeoPandas + Plotly**.
2. **Key Capabilities**:
   - **Tab 1 (AI Satellite Road Extraction)**: Live MLPRoadNet inference on uploaded imagery with toggleable TTA and Angular Gap Healing.
   - **Tab 2 (Criticality Heatmap)**: Leaflet map color-coded by betweenness centrality with pulsing Gatekeeper chokepoint badges.
   - **Tab 3 ("What-If" Disaster Simulation & Emergency Detour)**: Click-to-flood simulation, real-time Resilience Index ($R$) needle gauge, and dynamic A* emergency detour rerouting.
   - **Tab 4 (GIS Export)**: Instant GeoJSON, ESRI Shapefile, and CSV report download for QGIS.

---

### 🟣 Phase V: Advanced Hackathon-Winning Features
*Goal: End-to-end integration with ISRO/MeitY mandates and global spatial benchmarks.*

1. **GeoTIFF & Indian Satellite Ingestion (`src/geospatial/raster_handler.py`)**:
   - Native coordinate reference system (CRS) preservation with `rasterio`.
   - Export extracted road vectors directly as **GeoJSON** and **ESRI Shapefiles** for QGIS.
2. **OpenStreetMap Benchmark Validation (`src/evaluation/osm_benchmark.py`)**:
   - Automated Average Path Length Error (APLE) calculation against OSM ground truth.
3. **Multi-Hazard Inundation Overlay (`src/analysis/flood_sim.py`)**:
   - Intersect Digital Elevation Models (DEM) or satellite flood masks with the road graph.
4. **FastAPI Headless REST Engine (`api/main.py`)**:
   - REST API endpoints for seamless integration into government GIS portals.

---

## 📂 Master Codebase Architecture

```
SIH 2026/
├── configs/
│   ├── default.yaml                    # Base training & model config
│   └── hackathon_demo.yaml             # Demo dataset & simulation config
│
├── trained_models/                     # Packaged Production Weights
│   ├── best_model_finetuned.pth        # Peak fine-tuned PyTorch weights (113.6 MB)
│   ├── MLPRoadNet_best_weights.pth     # Packaged weights + metadata (37.9 MB)
│   ├── MLPRoadNet_finetuned_scripted.pt# Standalone TorchScript model (38.2 MB)
│   └── best_model.pth                  # Baseline model backup (113.6 MB)
│
├── src/
│   ├── config.py                       # Config loader & system setup
│   │
│   ├── data/                           # Data layer (COMPLETE)
│   │   ├── dataset.py                  # PyTorch SpaceNet / DeepGlobe dataset
│   │   ├── patches.py                  # Fast OpenCV resumable patch extractor
│   │   ├── augmentation.py             # High-throughput Albumentations transforms
│   │   └── osm_ingest.py               # Track 2: OpenStreetMap & ESRI tile fetcher
│   │
│   ├── models/                         # Model layer (COMPLETE)
│   │   └── mlproadnet.py               # DSD-Encoder + ASPP + MLP-Mixer + Dual Decoder
│   │
│   ├── losses/                         # Loss layer (COMPLETE)
│   │   └── losses.py                   # Multi-task BCE/Dice/Topology/Consistency loss
│   │
│   ├── training/                       # Training layer (COMPLETE)
│   │   └── trainer.py                  # PyTorch AMP trainer with live ETA & early stopping
│   │
│   ├── evaluation/                     # Evaluation layer (COMPLETE)
│   │   ├── evaluator.py                # IoU, Precision, Recall, F1, AUC, APLS evaluation
│   │   ├── tta.py                      # Test-Time Augmentation multi-view ensembling
│   │   └── refinement.py               # Centerline-guided morphological refinement
│   │
│   ├── graph/                          # Graph Healing Layer (COMPLETE)
│   │   ├── vectorize.py                # Centerline to NetworkX planar graph extractor
│   │   └── healing.py                  # Directional Angular Gap Healing algorithm
│   │
│   ├── analysis/                       # [PHASE III - NEXT] Criticality Layer
│   │   ├── criticality.py              # Betweenness centrality & Gatekeeper detector
│   │   └── resilience.py               # Node ablation & Resilience Index (R) engine
│   │
│   ├── visualization/                  # Plotting layer (COMPLETE)
│   │   └── visualize.py                # Publication-grade chart generation
│   │
│   └── geospatial/                     # [PHASE V] GIS Layer
│       ├── raster_handler.py           # GeoTIFF/Cartosat reader & GeoJSON exporter
│       └── osm_benchmark.py            # Automated OSM path length error benchmark
│
├── dashboard/                          # [PHASE IV - UPCOMING] Web Application
│   ├── app.py                          # Streamlit interactive Web-GIS dashboard
│   └── components/
│       ├── map_view.py                 # Folium interactive map component
│       ├── metrics_card.py             # Resilience index & delay KPI cards
│       └── simulation_panel.py         # What-If node ablation controls
│
├── scripts/                            # CLI Commands
│   ├── train.py                        # Training CLI with live progress
│   ├── finetune.py                     # Targeted fine-tuning CLI
│   ├── evaluate.py                     # Standalone evaluation CLI
│   ├── evaluate_refined.py             # Calibrated TTA + Gap Healing evaluation CLI
│   ├── demo_healing.py                 # Comparative visualization grid CLI
│   ├── fetch_indian_city.py            # Indian city automated data ingestor
│   └── run_dashboard.bat               # Single-click launcher for Streamlit UI
│
├── README.md                           # Comprehensive documentation
├── requirements.txt                    # Project dependencies
└── HACKATHON_IMPLEMENTATION_PLAN.md    # This document
```

---

## 🏆 Key Differentiation Factors for Hackathon Judges

| Evaluation Dimension | Standard Hackathon Submission | **Our MLPRoadNet + Route Resilience Solution** |
|---|---|---|
| **Segmentation Output** | Fragmented, broken pixel mask | **Continuous, topologically healed vector graph (1,961 gaps healed)** |
| **Occlusion Handling** | Standard CNN fails under shadows/trees | **MLP-Mixer global context + Angular Gap Healing ($0.6256$ IoU, $74.5\%$ Precision)** |
| **Real-World Value** | Just another computer vision model | **Actionable decision support for ISRO / NNRMS / Disaster Relief (NDRF)** |
| **Resilience Analytics** | None (only pixel IoU reported) | **Betweenness Centrality, Gatekeeper Nodes, and Resilience Index ($R$)** |
| **User Experience** | Static Jupyter notebook plots | **Interactive Streamlit + Leaflet map with live "Click-to-Flood" simulation** |
| **Geospatial Readiness** | Standard PNG only | **Full GeoTIFF, GeoJSON, Shapefile, and OpenStreetMap (OSM) validation** |

---

## 📌 Next Implementation Steps (Ready on Your Command)

1. **Step 1:** Implement `src/analysis/criticality.py` & `src/analysis/resilience.py` (Phase III: Betweenness Centrality, Gatekeeper chokepoints, Disaster Node Ablation, and Resilience Index $R$).
2. **Step 2:** Implement `dashboard/app.py` (Phase IV: Streamlit + Folium interactive Web-GIS map with live click-to-disable flood simulation & emergency rerouting).
3. **Step 3:** Implement `src/geospatial/raster_handler.py` (Phase V: GeoJSON / Shapefile export for QGIS & OSM validation).