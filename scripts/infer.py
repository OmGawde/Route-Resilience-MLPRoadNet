import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_device
from src.data.augmentation import get_inference_augmentation
from src.models.mlproadnet import MLPRoadNet
from src.training.trainer import load_checkpoint
from src.evaluation.evaluator import extract_centerline, make_overlay


def main():
    parser = argparse.ArgumentParser(description="Run single-image road extraction inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input satellite image (.tif, .png, .jpg)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default="prediction_overlay.png", help="Path to save prediction output image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary classification probability threshold")
    args = parser.parse_args()

    cfg = load_config(args.config)
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found at: {img_path}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    device = get_device()
    patch_size = cfg["patches"]["patch_size"]

    model = MLPRoadNet(
        in_ch=3,
        base_ch=cfg["model"]["base_channels"],
        mlp_depth=cfg["model"]["mlp_depth"],
        patch_size=patch_size,
    ).to(device)

    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    # Inference transformation
    mean = [0.31, 0.29, 0.29]
    std = [0.25, 0.24, 0.23]
    infer_aug = get_inference_augmentation(patch_size, mean, std)

    raw_img = np.array(Image.open(img_path).convert("RGB"))
    aug = infer_aug(image=raw_img)
    input_tensor = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        out_final, out_mask, out_cline = model(input_tensor)
        prob_map = torch.sigmoid(out_final).squeeze().cpu().numpy()
        pred_bin = (prob_map > args.threshold).astype(np.uint8)

    # Save output mask & overlay
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overlay
    resized_raw = np.array(Image.fromarray(raw_img).resize((patch_size, patch_size)))
    overlay = resized_raw.copy().astype(np.float32)
    overlay[pred_bin == 1] = overlay[pred_bin == 1] * 0.35 + np.array([255, 180, 0]) * 0.65
    overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    overlay_img.save(args.output)

    # Save binary mask
    stem = Path(args.output).stem
    mask_path = Path(args.output).with_name(f"{stem}_mask.png")
    Image.fromarray(pred_bin * 255).save(mask_path)

    print(f"✅ Prediction completed:")
    print(f"   Overlay: {args.output}")
    print(f"   Mask:    {mask_path}")


if __name__ == "__main__":
    main()
