import argparse
from pathlib import Path
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_device
from src.models.mlproadnet import MLPRoadNet
from src.training.trainer import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Export MLPRoadNet model to PyTorch, TorchScript, and ONNX")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save exported model formats")
    args = parser.parse_args()

    cfg = load_config(args.config)
    export_dir = Path(args.output_dir or (Path(cfg["output"]["output_dir"]) / "exports"))
    export_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint or (Path(cfg["output"]["output_dir"]) / "checkpoints" / "best_model.pth"))
    patch_size = cfg["patches"]["patch_size"]
    device = get_device()

    model = MLPRoadNet(
        in_ch=3,
        base_ch=cfg["model"]["base_channels"],
        mlp_depth=cfg["model"]["mlp_depth"],
        patch_size=patch_size,
    ).to(device)

    if ckpt_path.exists():
        print(f"Loading weights from {ckpt_path}")
        load_checkpoint(ckpt_path, model, device=device)
    else:
        print("⚠️ Checkpoint not found. Exporting model with initialized weights.")

    model.eval()

    # 1. State dict export
    weights_path = export_dir / "MLPRoadNet_best_weights.pth"
    torch.save({
        "model_state": model.state_dict(),
        "architecture": "MLPRoadNet",
        "config": cfg,
    }, weights_path)
    print(f"✅ PyTorch weights saved: {weights_path}")

    dummy_input = torch.randn(1, 3, patch_size, patch_size).to(device)

    # 2. TorchScript export
    try:
        scripted = torch.jit.trace(model, dummy_input)
        script_path = export_dir / "MLPRoadNet_scripted.pt"
        scripted.save(str(script_path))
        print(f"✅ TorchScript export saved: {script_path}")
    except Exception as e:
        print(f"⚠️ TorchScript export failed: {e}")

    # 3. ONNX export
    try:
        onnx_path = export_dir / "MLPRoadNet.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=13,
            input_names=["image"],
            output_names=["final_mask", "mask_logit", "centerline_logit"],
            dynamic_axes={"image": {0: "batch_size"}},
        )
        print(f"✅ ONNX export saved: {onnx_path}")
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")

    print(f"\n📦 All available exports saved in: {export_dir}")


if __name__ == "__main__":
    main()
