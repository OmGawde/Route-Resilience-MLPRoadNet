"""
Interactive Demo & Evaluation: TTA + Morphological Refinement + Angular Gap Healing.
Demonstrates end-to-end post-processing pipeline and produces side-by-side comparative visual grids.
"""

import argparse
import json
from pathlib import Path
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_device
from src.data.augmentation import get_test_augmentation
from src.models.mlproadnet import MLPRoadNet
from src.training.trainer import load_checkpoint
from src.evaluation.tta import predict_tta
from src.evaluation.refinement import refine_road_mask
from src.graph.vectorize import mask_to_road_graph
from src.graph.healing import heal_road_network, rasterize_graph


def compute_binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    inter = np.logical_and(pred > 0, target > 0).sum()
    union = np.logical_or(pred > 0, target > 0).sum()
    return float(inter / (union + 1e-8))


def main():
    parser = argparse.ArgumentParser(description="Test TTA + Refinement + Angular Gap Healing")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="trained_models/best_model.pth", help="Model checkpoint")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of test samples to visualize")
    parser.add_argument("--output-file", type=str, default="output/deepglobe_model/results/viz_10_tta_healing_comparison.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print(f"--> Running Category 1 Enhancements on {device}...")

    # Load Model
    model = MLPRoadNet(
        in_ch=3,
        base_ch=cfg["model"]["base_channels"],
        mlp_depth=cfg["model"]["mlp_depth"],
        patch_size=cfg["patches"]["patch_size"],
    ).to(device)

    load_checkpoint(Path(args.checkpoint), model, device=device)
    model.eval()

    # Load test patch index
    index_file = Path("output/deepglobe_model/patches/patch_index.pkl")
    if not index_file.exists():
        index_file = Path(cfg["output"]["output_dir"]) / "patches" / "patch_index.pkl"

    with open(index_file, "rb") as f:
        patch_index = pickle.load(f)

    test_split = cfg["split"]["test_split"]
    split_idx = int(len(patch_index) * (1 - test_split))
    test_patches = patch_index[split_idx:]

    mean = [0.4094, 0.3791, 0.2814]
    std = [0.1225, 0.0972, 0.0880]
    test_aug = get_test_augmentation(mean, std)

    # Evaluate across sample test patches
    sample_patches = test_patches[:args.num_samples]
    raw_ious, refined_ious, healed_ious = [], [], []

    fig, axes = plt.subplots(len(sample_patches), 5, figsize=(20, len(sample_patches) * 4))
    if len(sample_patches) == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "1. Satellite RGB",
        "2. Ground Truth",
        "3. Raw MLPRoadNet",
        "4. TTA + Refined Mask",
        "5. Phase II Healed Graph (Cyan = Healed Gaps)",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=10)

    total_healed_gaps = 0

    for row, (img_p, mask_p, _) in enumerate(sample_patches):
        rgb_raw = np.array(Image.open(img_p).convert("RGB"))
        gt_raw = (np.array(Image.open(mask_p).convert("L")) > 127).astype(np.uint8)

        # 1. Normal single-forward inference (Raw)
        aug = test_aug(image=rgb_raw, mask=gt_raw)
        inp_tensor = aug["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            out_f, out_m, out_c = model(inp_tensor)
            raw_prob = torch.sigmoid(out_f).squeeze().cpu().numpy()
            raw_binary = (raw_prob > 0.5).astype(np.uint8)

        # 2. Test-Time Augmentation (TTA)
        tta_fused, tta_mask, tta_cline = predict_tta(model, inp_tensor, device, use_8_fold=True)
        prob_f_np = tta_fused.squeeze().cpu().numpy()
        prob_c_np = tta_cline.squeeze().cpu().numpy()

        # 3. Morphological & Centerline Refinement
        refined_binary = refine_road_mask(prob_f_np, prob_c_np, mask_thresh=0.45, min_area_pixels=80)

        # 4. Phase II Graph Vectorization & Angular Gap Healing
        G, skel = mask_to_road_graph(refined_binary, min_branch_len=8)
        G_healed, n_healed = heal_road_network(G, max_gap_distance=60.0, max_angle_diff_deg=35.0)
        total_healed_gaps += n_healed

        healed_full, healed_bridges = rasterize_graph(G_healed, height=512, width=512, road_width=8)
        combined_healed = np.maximum(refined_binary, healed_full)

        # Compute IoUs
        iou_raw = compute_binary_iou(raw_binary, gt_raw)
        iou_ref = compute_binary_iou(refined_binary, gt_raw)
        iou_heal = compute_binary_iou(combined_healed, gt_raw)

        raw_ious.append(iou_raw)
        refined_ious.append(iou_ref)
        healed_ious.append(iou_heal)

        # Visualization Overlays
        # Col 0: RGB
        axes[row, 0].imshow(rgb_raw)
        axes[row, 0].axis("off")

        # Col 1: Ground Truth
        axes[row, 1].imshow(gt_raw, cmap="hot")
        axes[row, 1].axis("off")

        # Col 2: Raw
        axes[row, 2].imshow(raw_binary, cmap="hot")
        axes[row, 2].set_xlabel(f"IoU: {iou_raw:.3f}", fontsize=10, fontweight="bold")
        axes[row, 2].axis("off")

        # Col 3: Refined
        axes[row, 3].imshow(refined_binary, cmap="hot")
        axes[row, 3].set_xlabel(f"IoU: {iou_ref:.3f} (+{(iou_ref-iou_raw)*100:+.1f}%)", fontsize=10, fontweight="bold")
        axes[row, 3].axis("off")

        # Col 4: Graph Overlay on RGB with Cyan Healed Bridges
        overlay = rgb_raw.copy()
        # Draw road in green
        overlay[combined_healed > 0] = [0, 230, 80]
        # Highlight healed canopy gaps in bright cyan/blue
        if healed_bridges.sum() > 0:
            overlay[healed_bridges > 0] = [0, 220, 255]

        axes[row, 4].imshow(overlay)
        axes[row, 4].set_xlabel(f"Healed Gaps: {n_healed} | IoU: {iou_heal:.3f}", fontsize=10, fontweight="bold")
        axes[row, 4].axis("off")

    plt.tight_layout()
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close()

    print("\n" + "=" * 55)
    print("  Category 1 Enhancement Benchmark Results")
    print("=" * 55)
    print(f"  Mean Raw IoU       : {np.mean(raw_ious):.4f}")
    print(f"  Mean Refined IoU   : {np.mean(refined_ious):.4f} (+{(np.mean(refined_ious)-np.mean(raw_ious))*100:+.2f}%)")
    print(f"  Mean Healed IoU    : {np.mean(healed_ious):.4f} (+{(np.mean(healed_ious)-np.mean(raw_ious))*100:+.2f}%)")
    print(f"  Total Canopy Gaps Healed: {total_healed_gaps} gaps bridged!")
    print(f"  Saved comparison grid to: {out_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
