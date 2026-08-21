# 🖥️ Route Resilience: Web-GIS Interactive Decision Support Dashboard
## 📋 Standalone Implementation Plan & Architecture Specification (Phase IV)

---

## 🎯 Dashboard Overview & Purpose

The **Route Resilience Dashboard** is a state-of-the-art Web-GIS decision support application engineered for urban planners, disaster response teams (NDRF / SDMA), and municipal corporations. 

It translates raw satellite Earth Observation (EO) imagery and deep learning road extractions into an **actionable, interactive command center** where users can:
1. **Upload or Select Satellite Imagery** (Cartosat-3, Sentinel-2, DeepGlobe, Indian City tiles) and view live AI road extraction with toggleable topological healing.
2. **Visualize Structural Vulnerability Heatmaps** color-coded by graph centrality to instantly pinpoint single-points-of-failure (**Gatekeeper Nodes**).
3. **Run Live "What-If" Disaster Simulations**: Click any intersection or road to simulate a bridge collapse, waterlogging, or roadblock, and immediately calculate the drop in **Network Resilience ($R$)** and travel delays.
4. **Compute Dynamic Emergency Rerouting**: Generate alternate detour routes for emergency service vehicles (ambulances, fire engines) around disaster-collapsed sectors.
5. **Export GIS-Ready Deliverables**: Download continuous road networks as **GeoJSON**, **ESRI Shapefiles**, or **CSV Vulnerability Reports** for direct QGIS integration.

---

## 🎨 UI Design System & Aesthetic Guidelines

| Component | Design Standard | Implementation Details |
|---|---|---|
| **Theme / Color Palette** | Sleek Dark Mode (`#0e1117` background, `#1a1c24` cards) | High-contrast neon accents (Emerald `#00E676` for safe roads, Amber `#FFD600` for moderate traffic, Ruby `#FF1744` for critical gatekeepers, Cyan `#00E5FF` for healed bridges). |
| **Typography** | Modern Sans-Serif | Clean Inter / Outfit font family for maximum readability in emergency command centers. |
| **Interactive Maps** | Folium / Leaflet.js with Dark Basemaps | CartoDB Dark Matter tiles, full vector layer toggles, zoom/pan controls, click-to-select node markers. |
| **Micro-Interactions** | Dynamic KPI Metric Cards & Gauges | Plotly animated gauge charts for Resilience Index $R$, delta indicators (+% delay), and real-time computation spinners. |

---

## 🏗️ Technical Stack & Dependencies

- **Frontend / Application Framework**: `streamlit>=1.32.0`
- **Mapping & Spatial Visualization**: `folium>=0.16.0`, `streamlit-folium>=0.18.0`
- **Vector & Raster Spatial Handling**: `geopandas>=0.14.0`, `shapely>=2.0.0`, `rasterio>=1.3.9`
- **Graph & Routing Engine**: `networkx>=3.2.0`, `scipy>=1.11.0`
- **Chart & Gauge Analytics**: `plotly>=5.19.0`, `matplotlib>=3.8.0`
- **Backend Model Inference**: PyTorch (`trained_models/best_model_finetuned.pth` or `MLPRoadNet_finetuned_scripted.pt`)

---

## 📑 Detailed Tab-by-Tab Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 🛰️ ROUTE RESILIENCE: URBAN CRITICALITY & DISASTER RESPONSE COMMAND CENTER              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [ Tab 1: AI Road Extraction ]  [ Tab 2: Criticality Heatmap ]  [ Tab 3: Disaster Sim ] │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ ┌───────────────────────────────────────┐ ┌──────────────────────────────────────────┐ │
│ │ 🛰️ SATELLITE & GRAPH VIEWER           │ │ 📊 LIVE RESILIENCE & METRIC TELEMETRY    │ │
│ │                                       │ │                                          │ │
│ │  [ Interactive Folium Leaflet Map ]   │ │  ┌───────────────┐ ┌──────────────────┐  │ │
│ │   - Raw Imagery / Overlay             │ │  │ Resilience R  │ │ Gatekeeper Nodes │  │ │
│ │   - Green: Operational Roads          │ │  │     0.84      │ │    12 Critical   │  │ │
│ │   - Cyan: Healed Canopy Gaps          │ │  └───────────────┘ └──────────────────┘  │ │
│ │   - Red: Disabled Disaster Sector     │ │                                          │ │
│ │   - Blue: Emergency Detour Route      │ │  📈 Efficiency Drop: -16.2%              │ │
│ │                                       │ │  ⏱️ Avg Travel Delay: +24.8%            │ │
│ └───────────────────────────────────────┘ └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 🔹 Tab 1: AI Satellite Road Extraction & Occlusion Healing

1. **Input Selector**:
   - Option A: Upload custom Satellite Image (GeoTIFF / PNG / JPG).
   - Option B: Quick-select pre-loaded city presets (**Bengaluru Central**, **Delhi Outer Ring**, **Mumbai Western Corridor**, **DeepGlobe Test Tile**).
2. **Live Extraction Controls**:
   - Run **MLPRoadNet** inference live on GPU/CPU.
   - **TTA Multi-View Ensembling Toggle**: ON / OFF.
   - **Angular Gap Healing Toggle**: ON / OFF (adjust search cone $\theta$ and max gap distance sliders).
3. **Visual Output**:
   - Multi-column display: `[Satellite RGB]` | `[AI Predicted Mask]` | `[Healed Vector Skeleton Overlay]`.
   - Inspection metric card showing number of tree canopy gaps bridged and total road length in kilometers.

---

### 🔹 Tab 2: Structural Criticality & Gatekeeper Heatmap

1. **Graph Metric Engine**:
   - Converts the extracted road network into an active topological graph.
   - Calculates **Betweenness Centrality ($C_B$)**, **Closeness Centrality**, and **Edge Criticality**.
2. **Interactive Folium Map**:
   - Road segments color-coded by centrality rank:
     - 🟢 **Green (Low Centrality)**: Local residential alleys with high redundancy.
     - 🟡 **Yellow (Medium Centrality)**: Secondary connecting avenues.
     - 🔴 **Red (High Centrality)**: **Critical Arterials / Bridges** with zero alternative bypasses.
   - **Gatekeeper Intersections**: Pulsing red icon markers on top 5% most critical junctions.
3. **Vulnerability Data Table**:
   - Sortable table listing each Gatekeeper node ID, geographic coordinate, betweenness score, and connected neighborhoods.

---

### 🔹 Tab 3: Live "What-If" Disaster Simulation & Emergency Detour Routing

1. **Simulation Controls**:
   - **Interactive Node Selection**: Click any intersection on the map or select from a dropdown to simulate disaster closure (e.g., "Underpass Flooded", "Flyover Collapsed").
   - **Scenario Presets**:
     - *Preset 1: Heavy Monsoon Inundation (disables low-elevation roads)*.
     - *Preset 2: Key Arterial Bridge Failure*.
     - *Preset 3: Multiple Random Roadblocks*.
2. **Dynamic Impact Telemetry**:
   - **Network Resilience Index ($R$) Gauge**: Real-time needle gauge showing systemic structural integrity ($1.00 \rightarrow 0.62$).
   - **Network Efficiency Drop**: Percentage decrease in overall urban traversability.
   - **Isolated Neighborhoods Alert**: Visual notification highlighting disconnected clusters unable to reach hospitals/highways.
3. **Emergency Service Vehicle (ESV) Rerouting**:
   - Set Origin $(S)$ (e.g. Fire Station / Depot) and Destination $(T)$ (Disaster Zone).
   - Draws the **Original Optimal Route** (dashed red, now blocked) vs. the **New Live Healed Detour** (solid electric blue) with turn-by-turn distance/time penalty.

---

### 🔹 Tab 4: GIS Export & Benchmark Reports

1. **One-Click Data Export**:
   - Download **GeoJSON** of healed road vector lines.
   - Download **ESRI Shapefile (.zip)** ready for drag-and-drop into QGIS / ArcGIS.
   - Download **CSV Criticality Assessment Report** formatted for government disaster planning documentation.
2. **Automated Benchmark Card**:
   - Side-by-side comparison against OpenStreetMap (OSM) ground truth with Average Path Length Error (APLE) score.

---

## 📂 Dashboard Code Directory Structure

```
dashboard/
├── app.py                          # Main Streamlit application entry point
├── config.py                       # Dashboard themes, colors, and preset paths
│
├── components/                     # Modular UI Components
│   ├── header.py                   # App navigation bar & title banner
│   ├── map_viewer.py               # Folium / Leaflet map renderer & layer manager
│   ├── extraction_panel.py         # AI inference & gap healing controls
│   ├── criticality_panel.py        # Centrality heatmap & gatekeeper ranking UI
│   ├── simulation_panel.py         # "What-If" disaster collapse & rerouting UI
│   ├── metrics_cards.py            # Resilience gauge & delay telemetry widgets
│   └── export_panel.py             # GeoJSON / Shapefile / CSV download triggers
│
├── utils/                          # Helper Utilities
│   ├── inference_engine.py         # Fast PyTorch model loader & TTA runner
│   ├── graph_processor.py          # NetworkX conversion & centrality computer
│   ├── routing_engine.py           # Dijkstra / A* emergency detour router
│   └── geo_exporter.py             # GeoPandas Shapefile & GeoJSON converter
│
└── assets/                         # UI Styling & Presets
    ├── style.css                   # Custom modern dark-mode CSS stylesheet
    ├── isro_logo.png               # Hackathon branding assets
    └── sample_tiles/               # Pre-loaded satellite test imagery
```

---

## 🚀 Execution & Deployment Workflow

### 1. Requirements Installation:
```bash
pip install streamlit folium streamlit-folium geopandas shapely plotly rasterio
```

### 2. Single-Click Launcher (`scripts/run_dashboard.bat`):
```bat
@echo off
echo Starting Route Resilience Web-GIS Dashboard...
streamlit run dashboard/app.py --server.port 8501 --theme.base "dark"
pause
```

### 3. Direct Terminal Command:
```bash
streamlit run dashboard/app.py
```

---

## 📌 Implementation Status:
- **Phase Status**: `QUEUED (To be implemented after Phase III Criticality Analysis)` ⏳
- **Estimated Build Time**: ~1–2 hours once Phase III analytics engine is in place.
