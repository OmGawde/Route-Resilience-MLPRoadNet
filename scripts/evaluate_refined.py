"""
Full Test-Set Evaluation with Category 1 Enhancements (TTA + Centerline Refinement + Graph Healing).
"""

import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_device
from src.data.augmentation import get_test_augmentation
from src.models.mlproadnet import MLPRoadNet
from src.training.trainer import load_checkpoint
from src.evaluation.evaluator import compute_apls
from src.evaluation.tta import predict_tta
from src.evaluation.refinement import refine_road_mask
from src.graph.vectorize import mask_to_road_graph
from src.graph.healing import heal_road_network, rasterize_graph
from src.visualization.visualize import plot_metrics_and_cm, plot_roc_curve


def main():
    parser = argparse.ArgumentParser(description="Evaluate Category 1 Enhanced MLPRoadNet")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="trained_models/best_model.pth", help="Model checkpoint")
    parser.add_argument("--output-dir", type=str, default="output/deepglobe_model/results_refined", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--> Running Category 1 Enhanced Evaluation on {device}...")

    model = MLPRoadNet(
        in_ch=3,
        base_ch=cfg["model"]["base_channels"],
        mlp_depth=cfg["model"]["mlp_depth"],
        patch_size=cfg["patches"]["patch_size"],
    ).to(device)

    load_checkpoint(Path(args.checkpoint), model, device=device)
    model.eval()

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

    tp, tn, fp, fn = 0, 0, 0, 0
    sampled_probs, sampled_tgts = [], []
    apls_scores = []
    total_gaps_healed = 0

    print(f"Evaluating {len(test_patches)} test patches with TTA + Refinement + Angular Gap Healing...")

    for i, (img_p, mask_p, _) in enumerate(tqdm(test_patches, desc="Enhanced Evaluation")):
        rgb_raw = np.array(Image.open(img_p).convert("RGB"))
        gt_raw = (np.array(Image.open(mask_p).convert("L")) > 127).astype(np.uint8)

        aug = test_aug(image=rgb_raw, mask=gt_raw)
        inp_tensor = aug["image"].unsqueeze(0).to(device)

        # 1. 4-fold TTA
        tta_fused, _, tta_cline = predict_tta(model, inp_tensor, device, use_8_fold=False)
        prob_f_np = tta_fused.squeeze().cpu().numpy()
        prob_c_np = tta_cline.squeeze().cpu().numpy()

        # 2. Refinement & Speckle Removal
        refined_binary = refine_road_mask(prob_f_np, prob_c_np, mask_thresh=0.50, min_area_pixels=100, close_kernel_size=3)

        # 3. Phase II Graph Healing
        G, _ = mask_to_road_graph(refined_binary, min_branch_len=8)
        G_healed, n_healed = heal_road_network(G, max_gap_distance=60.0, max_angle_diff_deg=35.0)
        total_gaps_healed += n_healed

        healed_full, _ = rasterize_graph(G_healed, height=512, width=512, road_width=4)
        final_binary = np.maximum(refined_binary, healed_full)

        preds = (final_binary > 0)
        tgts = (gt_raw > 0)

        tp += int((preds & tgts).sum())
        tn += int((~preds & ~tgts).sum())
        fp += int((preds & ~tgts).sum())
        fn += int((~preds & tgts).sum())

        if len(sampled_probs) < 100000:
            p_flat = prob_f_np.flatten()
            t_flat = tgts.flatten().astype(np.uint8)
            sub_n = min(500, len(p_flat))
            idx_sub = np.random.choice(len(p_flat), sub_n, replace=False)
            sampled_probs.extend(p_flat[idx_sub].tolist())
            sampled_tgts.extend(t_flat[idx_sub].tolist())

        if i < 100:
            apls_scores.append(compute_apls(final_binary, gt_raw))

    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    sampled_probs = np.array(sampled_probs, dtype=np.float32)
    sampled_tgts = np.array(sampled_tgts, dtype=np.uint8)
    try:
        auc_val = float(roc_auc_score(sampled_tgts, sampled_probs))
    except Exception:
        auc_val = 0.5

    apls_mean = float(np.mean(apls_scores)) if apls_scores else 0.0

    metrics = {
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Specificity": float(specificity),
        "Accuracy": float(accuracy),
        "AUC": float(auc_val),
        "APLS": float(apls_mean),
        "Total_Gaps_Healed": int(total_gaps_healed),
    }

    cm = np.array([[tn, fp], [fn, tp]])

    print("\n" + "=" * 55)
    print("  Category 1 Enhanced Evaluation Results (Test Set)")
    print("=" * 55)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")
    print("=" * 55)

    with open(output_dir / "enhanced_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_metrics_and_cm(metrics, cm, output_dir / "enhanced_metrics.png")
    plot_roc_curve(sampled_tgts, sampled_probs, metrics["AUC"], output_dir / "enhanced_roc.png")

    print(f"\n[Done] Enhanced evaluation artifacts saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
