import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
try:
    import seaborn as sns
except ImportError:
    sns = None
from ..data.dataset import load_image, load_mask
from ..evaluation.evaluator import extract_centerline, make_overlay


ROAD_CMAP = LinearSegmentedColormap.from_list("road", ["#0a0a0a", "#f7c948"])


def plot_sample_grid(pairs: List[Tuple[Path, Path]], save_path: Path, n: int = 8):
    """Plot sample RGB, ground truth mask, and overlay grid."""
    n = min(n, len(pairs))
    idx = random.sample(range(len(pairs)), n)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 8))
    fig.suptitle("Sample Images & Masks", fontsize=14, fontweight="bold", y=1.01)

    for col, i in enumerate(idx):
        img = load_image(pairs[i][0])
        mask = load_mask(pairs[i][1])
        overlay = img.copy()
        overlay[mask == 1] = [255, 200, 0]

        axes[0, col].imshow(img)
        axes[0, col].set_title(pairs[i][0].stem[:12], fontsize=7)
        axes[0, col].axis("off")
        axes[1, col].imshow(mask, cmap=ROAD_CMAP, vmin=0, vmax=1)
        axes[1, col].axis("off")
        axes[2, col].imshow(overlay)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("RGB", fontsize=9, rotation=0, labelpad=30)
    axes[1, 0].set_ylabel("Mask", fontsize=9, rotation=0, labelpad=30)
    axes[2, 0].set_ylabel("Overlay", fontsize=9, rotation=0, labelpad=30)

    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close()


def plot_training_curves(history: Dict[str, list], best_iou: float, save_path: Path):
    """Plot Loss, Validation IoU, and Learning Rate curves."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("MLPRoadNet Training Curves", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], color="#e74c3c", linewidth=1.8, label="Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_iou"], color="#2ecc71", linewidth=1.8, label="Val IoU")
    axes[1].axhline(best_iou, linestyle="--", color="#f7c948", label=f"Best={best_iou:.4f}")
    axes[1].set_title("Validation IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["lr"], color="#9b59b6", linewidth=1.8)
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close()


def plot_metrics_and_cm(metrics: Dict[str, Any], cm: np.ndarray, save_path: Path):
    """Plot evaluation metrics bar chart and confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("MLPRoadNet Evaluation Metrics", fontsize=14, fontweight="bold")

    # Only include [0, 1] evaluation metric scores on the bar chart
    standard_metrics = ["IoU", "Precision", "Recall", "F1", "Specificity", "Accuracy", "AUC", "APLS"]
    names = [k for k in standard_metrics if k in metrics]
    vals = [float(metrics[k]) for k in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))

    bars = axes[0].bar(names, vals, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    axes[0].set_title("Standard Segmentation Metrics", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30, labelsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    
    for bar, val in zip(bars, vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Format Confusion Matrix with comma thousand separators
    if sns is not None:
        cm_df = pd.DataFrame(cm, index=["Actual Non-Road", "Actual Road"], columns=["Pred Non-Road", "Pred Road"])
        sns.heatmap(cm_df, annot=True, fmt=",d", cmap="Blues", ax=axes[1], linewidths=0.5, cbar_kws={"shrink": 0.8})
    else:
        im = axes[1].imshow(cm, cmap="Blues")
        axes[1].figure.colorbar(im, ax=axes[1], shrink=0.8)
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(["Pred Non-Road", "Pred Road"])
        axes[1].set_yticklabels(["Actual Non-Road", "Actual Road"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1].text(j, i, f"{cm[i, j]:,d}", ha="center", va="center", color="black", fontweight="bold")
    axes[1].set_title("Confusion Matrix (Pixels)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close()


def plot_roc_curve(all_tgts: np.ndarray, all_probs: np.ndarray, auc_val: float, save_path: Path):
    """Plot ROC / AUC curve."""
    sub = min(100000, len(all_tgts))
    idx = np.random.choice(len(all_tgts), sub, replace=False)
    fpr, tpr, _ = roc_curve(all_tgts[idx], all_probs[idx])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1a6b8a", linewidth=2.0, label=f"MLPRoadNet (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1a6b8a")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Road Detection", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close()


@torch.no_grad()
def plot_test_predictions(
    model: nn.Module,
    test_patches: List[Tuple[str, str, float]],
    transform,
    device: torch.device,
    save_path: Path,
    n: int = 10,
):
    """Plot 10 qualitative sample panels (RGB, GT, Pred, Centerline, Overlay)."""
    n = min(n, len(test_patches))
    idxs = random.sample(range(len(test_patches)), n)

    fig, axes = plt.subplots(n, 6, figsize=(22, n * 3.5))
    if n == 1:
        axes = [axes]
    col_labels = ["Filename", "RGB Image", "Ground Truth", "Predicted Mask", "Centerline", "Overlay (TP/FP/FN)"]

    for col, lbl in enumerate(col_labels):
        if col > 0:
            axes[0][col].set_title(lbl, fontsize=9, fontweight="bold")

    model.eval()
    for row, idx in enumerate(idxs):
        img_p, mask_p, _ = test_patches[idx]
        fname = Path(img_p).stem

        rgb = np.array(Image.open(img_p))
        gt = (np.array(Image.open(mask_p).convert("L")) > 127).astype(np.uint8)

        aug = transform(image=rgb, mask=gt)
        inp = aug["image"].unsqueeze(0).to(device)
        out, _, _ = model(inp)
        pred_prob = torch.sigmoid(out).squeeze().cpu().numpy()
        pred_bin = (pred_prob > 0.5).astype(np.uint8)

        cline = extract_centerline(pred_bin)
        overlay = make_overlay(rgb, pred_bin.astype(bool), gt.astype(bool))

        axes[row][0].text(0.5, 0.5, fname[:20], ha="center", va="center", fontsize=7, wrap=True, transform=axes[row][0].transAxes)
        axes[row][0].axis("off")
        axes[row][1].imshow(rgb)
        axes[row][1].axis("off")
        axes[row][2].imshow(gt, cmap=ROAD_CMAP, vmin=0, vmax=1)
        axes[row][2].axis("off")
        axes[row][3].imshow(pred_bin, cmap=ROAD_CMAP, vmin=0, vmax=1)
        axes[row][3].axis("off")
        axes[row][4].imshow(cline, cmap="hot")
        axes[row][4].axis("off")
        axes[row][5].imshow(overlay)
        axes[row][5].axis("off")

        inter = np.logical_and(pred_bin, gt).sum()
        union = np.logical_or(pred_bin, gt).sum()
        iou_r = inter / (union + 1e-8)
        axes[row][1].set_ylabel(f"IoU={iou_r:.3f}", fontsize=7, rotation=0, labelpad=45)

    legend_elements = [
        Patch(facecolor="#00dc00", label="TP (correct road)"),
        Patch(facecolor="#dc0000", label="FP (false road)"),
        Patch(facecolor="#0000dc", label="FN (missed road)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    plt.suptitle("MLPRoadNet — Test Set Predictions", fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close()
