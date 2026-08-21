"""
Test-Time Augmentation (TTA) Module for MLPRoadNet.
Ensembles predictions across geometric flips and rotations to suppress noise and smooth boundaries.
"""

from typing import Tuple
import torch
import torch.nn as nn


def predict_tta(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    use_8_fold: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs Test-Time Augmentation (TTA) over input tensor.
    image_tensor: (B, 3, H, W) normalized tensor
    Returns: (fused_prob, mask_prob, cline_prob) tensors of shape (B, 1, H, W) in [0, 1].
    """
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # 4-fold or 8-fold transformations
    transforms = [
        # (rot_k, flip_dim)
        (0, None),          # Identity
        (1, None),          # 90 deg
        (2, None),          # 180 deg
        (3, None),          # 270 deg
    ]

    if use_8_fold:
        transforms.extend([
            (0, 2),         # Horizontal flip
            (0, 3),         # Vertical flip
            (1, 2),         # 90 deg + HFlip
            (1, 3),         # 90 deg + VFlip
        ])

    accum_fused = torch.zeros(image_tensor.shape[0], 1, image_tensor.shape[2], image_tensor.shape[3], device=device)
    accum_mask = torch.zeros_like(accum_fused)
    accum_cline = torch.zeros_like(accum_fused)

    with torch.no_grad():
        for k, flip_d in transforms:
            aug_img = image_tensor
            if flip_d is not None:
                aug_img = torch.flip(aug_img, dims=[flip_d])
            if k > 0:
                aug_img = torch.rot90(aug_img, k=k, dims=[2, 3])

            out_f, out_m, out_c = model(aug_img)
            prob_f = torch.sigmoid(out_f)
            prob_m = torch.sigmoid(out_m)
            prob_c = torch.sigmoid(out_c)

            # Invert transformations
            if k > 0:
                prob_f = torch.rot90(prob_f, k=-k, dims=[2, 3])
                prob_m = torch.rot90(prob_m, k=-k, dims=[2, 3])
                prob_c = torch.rot90(prob_c, k=-k, dims=[2, 3])
            if flip_d is not None:
                prob_f = torch.flip(prob_f, dims=[flip_d])
                prob_m = torch.flip(prob_m, dims=[flip_d])
                prob_c = torch.flip(prob_c, dims=[flip_d])

            accum_fused += prob_f
            accum_mask += prob_m
            accum_cline += prob_c

    n = float(len(transforms))
    return accum_fused / n, accum_mask / n, accum_cline / n
