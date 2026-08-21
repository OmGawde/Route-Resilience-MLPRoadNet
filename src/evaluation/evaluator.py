from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..data.dataset import SpaceNetDataset


def extract_centerline(mask_np: np.ndarray) -> np.ndarray:
    """Compute topological centerline skeleton of a binary mask."""
    return skeletonize(mask_np > 0).astype(np.uint8) * 255


def make_overlay(rgb: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Create RGB overlay: Green=True Positive, Red=False Positive, Blue=False Negative."""
    overlay = rgb.copy().astype(np.float32)
    tp = (pred_mask & gt_mask).astype(bool)
    fp = (pred_mask & ~gt_mask).astype(bool)
    fn = (~pred_mask & gt_mask).astype(bool)

    overlay[tp] = overlay[tp] * 0.3 + np.array([0, 220, 0]) * 0.7
    overlay[fp] = overlay[fp] * 0.3 + np.array([220, 0, 0]) * 0.7
    overlay[fn] = overlay[fn] * 0.3 + np.array([0, 0, 220]) * 0.7

    return np.clip(overlay, 0, 255).astype(np.uint8)


def compute_apls(pred_mask: np.ndarray, gt_mask: np.ndarray, num_samples: int = 50) -> float:
    """
    Approximate Average Path Length Similarity (APLS) via skeleton connectivity sampling.
    """
    def skel_pts(mask):
        s = skeletonize(mask > 0)
        ys, xs = np.where(s)
        return list(zip(ys.tolist(), xs.tolist()))

    pred_pts = skel_pts(pred_mask)
    gt_pts = skel_pts(gt_mask)

    if not pred_pts or not gt_pts:
        return 0.0

    pred_arr = np.array(pred_pts, dtype=np.float32)
    gt_arr = np.array(gt_pts, dtype=np.float32)
    num_samples = min(num_samples, len(gt_pts))
    idx1 = np.random.choice(len(gt_pts), num_samples, replace=False)

    scores = []
    for i in idx1:
        g = gt_arr[i]
        dists = np.linalg.norm(pred_arr - g, axis=1)
        min_d = dists.min()
        scores.append(np.exp(-min_d / (pred_mask.shape[0] * 0.1)))

    return float(np.mean(scores))


@torch.no_grad()
def full_evaluation(
    model: nn.Module,
    test_patches: List[Tuple[str, str, float]],
    transform,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs evaluation over the entire test split and computes standard metrics:
    IoU, Precision, Recall, F1, Specificity, Accuracy, AUC, APLS.
    """
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    sampled_probs, sampled_tgts = [], []
    apls_scores = []

    ds = SpaceNetDataset(test_patches, transform=transform, label_smooth=0.0)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    for imgs, masks in tqdm(dl, desc="Evaluating test set"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
            out_f, _, _ = model(imgs)
            probs = torch.sigmoid(out_f)
            preds = (probs > threshold)
            tgts = (masks.to(device) > 0.5)

            # Fast GPU binary tensor statistics
            tp += int((preds & tgts).sum().item())
            tn += int((~preds & ~tgts).sum().item())
            fp += int((preds & ~tgts).sum().item())
            fn += int((~preds & tgts).sum().item())

            # Reservoir sampling for ROC AUC (keep memory low)
            if len(sampled_probs) < 100000:
                p_flat = probs.view(-1).cpu().numpy()
                t_flat = tgts.view(-1).cpu().numpy().astype(np.uint8)
                sub_n = min(1000, len(p_flat))
                idx_sub = np.random.choice(len(p_flat), sub_n, replace=False)
                sampled_probs.extend(p_flat[idx_sub].tolist())
                sampled_tgts.extend(t_flat[idx_sub].tolist())

    # Compute APLS on a sample of test patches
    for img_p, mask_p, _ in tqdm(test_patches[:100], desc="Computing APLS sample", leave=False):
        rgb = np.array(Image.open(img_p))
        gt = (np.array(Image.open(mask_p).convert("L")) > 127).astype(np.uint8)

        aug = transform(image=rgb, mask=gt)
        inp = aug["image"].unsqueeze(0).to(device)
        with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
            out, _, _ = model(inp)
            pred_np = (torch.sigmoid(out).squeeze().cpu().numpy() > threshold).astype(np.uint8)
        apls_scores.append(compute_apls(pred_np, gt))

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
    }

    cm = np.array([[tn, fp], [fn, tp]])
    return metrics, cm, sampled_probs, sampled_tgts
