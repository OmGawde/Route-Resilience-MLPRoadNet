from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Smooth Dice Loss for binary segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        inter = (pred * target).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class TopologyLoss(nn.Module):
    """
    Topology continuity loss: penalizes disconnected road segments
    using morphological dilation.
    """
    def __init__(self):
        super().__init__()
        k = torch.ones(1, 1, 5, 5)
        self.register_buffer("kernel", k)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(pred)
        kernel = self.kernel.to(p.device)
        p_dil = (F.conv2d(p, kernel, padding=2) > 0).float()
        gap = target * (1.0 - p_dil)
        return gap.mean()


class MLPRoadNetLoss(nn.Module):
    """
    Composite Multi-Task Loss:
      L = (BCE + Dice)[final]
        + aux_weight * (BCE + Dice)[mask]
        + aux_weight * (BCE + Dice)[centerline]
        + noise_reg * Consistency(mask, cline)
        + topo_weight * Topology(final)
    """
    def __init__(
        self,
        noise_reg_weight: float = 0.1,
        topo_weight: float = 0.1,
        aux_weight: float = 0.5,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.topo = TopologyLoss()
        self.nrw = noise_reg_weight
        self.topo_w = topo_weight
        self.aux_w = aux_weight

    def forward(
        self,
        out_final: torch.Tensor,
        out_mask: torch.Tensor,
        out_cline: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Approximate centerline skeleton via 3x3 morphological erosion
        with torch.no_grad():
            t_bin = (target > 0.5).float()
            k3 = torch.ones(1, 1, 3, 3, device=target.device)
            eroded = F.conv2d(t_bin, k3, padding=1) == 9
            cline_t = (t_bin - eroded.float()).clamp(0.0, 1.0)

        # Primary and auxiliary losses
        l_final = self.bce(out_final, target) + self.dice(out_final, target)
        l_mask = self.bce(out_mask, target) + self.dice(out_mask, target)
        l_cline = self.bce(out_cline, cline_t) + self.dice(out_cline, cline_t)

        # Consistency loss between mask and centerline branch logits
        l_cons = F.mse_loss(torch.sigmoid(out_mask), torch.sigmoid(out_cline))

        # Topology loss
        l_topo = self.topo(out_final, t_bin)

        total = l_final + self.aux_w * l_mask + self.aux_w * l_cline + self.nrw * l_cons + self.topo_w * l_topo

        return total, {
            "total": total.item(),
            "final": l_final.item(),
            "mask": l_mask.item(),
            "cline": l_cline.item(),
            "cons": l_cons.item(),
            "topo": l_topo.item(),
        }
