# 🏆 Trained Models Directory

This folder contains the saved checkpoints and production exports of **MLPRoadNet**:

| File | Format | Description |
|---|---|---|
| `best_model.pth` | PyTorch State Dict | Full model checkpoint with the highest validation IoU. |
| `MLPRoadNet_best_weights.pth` | PyTorch Package | Model weights packaged with configuration, dataset mean/std, and evaluation metrics. |
| `MLPRoadNet_scripted.pt` | TorchScript | Standalone, compiled PyTorch model (runs without Python class definitions). |
| `MLPRoadNet.onnx` | ONNX | Open Neural Network Exchange format for TensorRT / C++ / Web deployment. |

---

### 🚀 How to use in Python:
```python
import torch
from src.models import MLPRoadNet

model = MLPRoadNet(in_ch=3, base_ch=32, mlp_depth=4, patch_size=512)
checkpoint = torch.load("trained_models/best_model.pth")
model.load_state_dict(checkpoint["model"])
model.eval()
```
