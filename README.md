# MLPRoadNet-SpaceNet-Vegas
<div align="center">

> 🛰️ **MLPRoadNet** leverages hybrid CNN–MLP intelligence, multi-scale context aggregation, and topology-aware decoding to extract accurate and connected road networks from high-resolution satellite imagery—even under severe occlusions.

</div>

<div align="center">

### From Pixels to Pathways

**Topology-Aware Hybrid CNN–MLP Framework for Occlusion-Robust Road Extraction from Satellite Imagery**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Dataset-SpaceNet_Vegas-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge">
</p>

<p align="center">
  <img src="assets/architecture.png" width="90%">
</p>

</div>

---

## 🌍 Overview

Urban road extraction remains challenging due to:

- 🌳 Vegetation Occlusions
- 🏢 Building Shadows
- ☁️ Annotation Noise
- 🛣️ Fragmented Road Structures
- 🌐 Complex Urban Topologies

**MLPRoadNet** combines multi-scale convolutional feature extraction with global MLP-based reasoning to generate accurate, connected, and topology-preserving road networks from high-resolution satellite imagery.

> Designed for smart cities, urban planning, navigation systems, disaster response, and geospatial intelligence applications.

---

# 🏗 Architecture

```text
Satellite Image
       │
       ▼
Depthwise-Separable Encoder
       │
       ▼
 ASPP Context Module
       │
       ▼
MLP-Mixer Bottleneck
       │
       ▼
Dual-Branch Decoder
 ┌──────────────┬──────────────┐
 │              │              │
 ▼              ▼              ▼
Road Mask   Centerline Map  Connectivity
       │
       ▼
Connected Road Network
```

---

# ✨ Key Features

| Feature | Description |
|----------|-------------|
| 🧠 MLP-Mixer Bottleneck | Captures global spatial dependencies |
| 🌐 ASPP Module | Multi-scale context aggregation |
| 🛣️ Dual-Branch Decoder | Joint road & centerline prediction |
| 🔗 Topology Preservation | Improved network connectivity |
| ⚡ Lightweight Design | Efficient training and inference |
| 🛰️ Satellite Optimized | Designed for SpaceNet imagery |

---

# 📊 Highlights

| Capability | Benefit |
|------------|----------|
| Occlusion Robustness | Handles trees, shadows, and clutter |
| Multi-Scale Learning | Captures both local and global context |
| Topology Awareness | Preserves road connectivity |
| Lightweight Architecture | Faster deployment |
| End-to-End Training | Simplified workflow |

---

# 🚀 Quick Start

## Clone Repository

```bash
git clone https://github.com/yourusername/MLPRoadNet.git
cd MLPRoadNet
```

## Create Environment

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

## Evaluate

```bash
python evaluate.py
```

## Inference

```bash
python predict.py --image sample.png
```

---

# 💻 Usage

```python
from mlproadnet import MLPRoadNet

model = MLPRoadNet()

model.load_weights("best_model.pth")

prediction = model.predict("satellite_image.png")

prediction.save("road_mask.png")
```

---

# 📂 Repository Structure

```text
MLPRoadNet
│
├── assets/
│   ├── architecture.png
│   ├── predictions.png
│
├── datasets/
├── checkpoints/
├── notebooks/
├── models/
├── utils/
│
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# 🛣 Roadmap

- [x] Hybrid CNN–MLP Architecture
- [x] Multi-Scale ASPP Module
- [x] Topology-Aware Decoder
- [x] SpaceNet Vegas Benchmarking
- [ ] Graph Neural Network Refinement
- [ ] TensorRT Deployment
- [ ] ONNX Export Support
- [ ] Interactive GIS Integration
- [ ] Multi-City Domain Adaptation

---

# 📈 Results

| Input Image | Ground Truth | Prediction |
|-------------|-------------|------------|
| ![](assets/input.png) | ![](assets/gt.png) | ![](assets/prediction.png) |

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to open an issue or submit a pull request.

---

# 📜 License

Distributed under the MIT License.

See `LICENSE` for more information.

---

<div align="center">

### 🌍 Mapping Tomorrow's Roads, Today.

Built for next-generation geospatial intelligence and urban mobility analysis.

⭐ Star this repository if you find it useful.

</div>
