import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_iou_batch(preds_bin: torch.Tensor, targets_bin: torch.Tensor) -> float:
    """Batch-level Intersection over Union (IoU)."""
    inter = (preds_bin & targets_bin).float().sum((1, 2, 3))
    union = (preds_bin | targets_bin).float().sum((1, 2, 3))
    return float((inter / (union + 1e-8)).mean().item())


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    history: Dict[str, list],
    path: Path,
    extra: Dict[str, Any] = None,
):
    """Save training state checkpoint."""
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "history": history,
    }
    if extra:
        state.update(extra)
    torch.save(state, str(path))


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, Dict[str, list], float]:
    """Load model state and training progress from checkpoint."""
    state = torch.load(str(path), map_location=device)
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and "scheduler" in state and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state.get("epoch", 0), state.get("history", {}), state.get("best_iou", 0.0)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable HH:MM:SS or MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler = None,
    grad_clip: float = 1.0,
    epoch_idx: int = 1,
    total_epochs: int = 100,
) -> Tuple[float, Dict[str, float]]:
    """Train model for one full epoch with live batch speed & loss tracking."""
    model.train()
    total_loss = 0.0
    loss_components = {"final": 0.0, "mask": 0.0, "cline": 0.0, "cons": 0.0, "topo": 0.0}
    use_amp = (device.type == "cuda" and scaler is not None)

    pbar = tqdm(loader, desc=f"Train Ep {epoch_idx:03d}/{total_epochs}", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                out_final, out_mask, out_cline = model(imgs)
                loss, comp = criterion(out_final, out_mask, out_cline, masks)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out_final, out_mask, out_cline = model(imgs)
            loss, comp = criterion(out_final, out_mask, out_cline, masks)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += comp["total"]
        for k in loss_components:
            loss_components[k] += comp[k]

        pbar.set_postfix({
            "Loss": f"{comp['total']:.4f}",
            "Dice": f"{comp['final']:.3f}",
        })

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model and compute validation loss & mean IoU with live progress."""
    model.eval()
    total_loss, total_iou = 0.0, 0.0
    use_amp = (device.type == "cuda")

    pbar = tqdm(loader, desc="Validating", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                out_final, out_mask, out_cline = model(imgs)
                loss, _ = criterion(out_final, out_mask, out_cline, masks)
        else:
            out_final, out_mask, out_cline = model(imgs)
            loss, _ = criterion(out_final, out_mask, out_cline, masks)

        total_loss += loss.item()
        preds_bin = (torch.sigmoid(out_final) > 0.5).bool()
        targets_bin = (masks > 0.5).bool()
        batch_iou = compute_iou_batch(preds_bin, targets_bin)
        total_iou += batch_iou

        pbar.set_postfix({"ValIoU": f"{batch_iou:.4f}"})

    n = len(loader)
    return total_loss / n, total_iou / n


class Trainer:
    """Orchestrates end-to-end model training, validation, checkpointing, and early stopping."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        output_dirs: Dict[str, Path],
        epochs: int = 120,
        patience: int = 8,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dirs = output_dirs
        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip

        self.ckpt_latest = output_dirs["checkpoints"] / "latest.pth"
        self.ckpt_best = output_dirs["checkpoints"] / "best_model.pth"
        self.history_file = output_dirs["logs"] / "train_history.json"
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    def train(self, resume: bool = True) -> Dict[str, list]:
        start_epoch = 0
        best_iou = 0.0
        history = {"train_loss": [], "val_iou": [], "lr": []}

        if resume and self.ckpt_latest.exists():
            print(f"Resuming training from {self.ckpt_latest}")
            start_epoch, history, best_iou = load_checkpoint(
                self.ckpt_latest, self.model, self.optimizer, self.scheduler, self.device
            )
            start_epoch += 1
            print(f"   Resuming at epoch {start_epoch + 1} | Best IoU so far: {best_iou:.4f}")

        no_improve = 0
        total_start_time = time.time()
        epoch_durations = []

        print(f"Training MLPRoadNet | Max Epochs: {self.epochs} | Early-Stop Patience: {self.patience}")
        print(f"Batches per Epoch: Train={len(self.train_loader)} | Val={len(self.val_loader)}")
        print(f"Mixed Precision (AMP): {'ENABLED (Tensor Cores active)' if self.device.type == 'cuda' else 'Disabled'}")
        print("=" * 80)

        for epoch in range(start_epoch, self.epochs):
            t0 = time.time()
            print(f"\n--> [Epoch {epoch + 1:03d}/{self.epochs}] Training {len(self.train_loader)} batches on GPU ({self.device})...")
            train_loss, train_comps = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                scaler=self.scaler,
                grad_clip=self.grad_clip,
                epoch_idx=epoch + 1,
                total_epochs=self.epochs,
            )
            val_loss, val_iou = validate(self.model, self.val_loader, self.criterion, self.device)

            if self.scheduler is not None:
                self.scheduler.step()

            cur_lr = self.optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_iou"].append(val_iou)
            history["lr"].append(cur_lr)

            dt = time.time() - t0
            epoch_durations.append(dt)

            avg_epoch_time = float(np.mean(epoch_durations))
            remaining_epochs = self.epochs - (epoch + 1)
            est_remaining_time = avg_epoch_time * remaining_epochs
            total_elapsed = time.time() - total_start_time

            improved_flag = " [BEST]" if val_iou > best_iou else ""
            print(
                f"Ep {epoch + 1:03d}/{self.epochs} | "
                f"TrLoss: {train_loss:.4f} | "
                f"ValIoU: {val_iou:.4f}{improved_flag} | "
                f"Speed: {dt:.1f}s/ep | "
                f"Elapsed: {format_time(total_elapsed)} | "
                f"ETA: {format_time(est_remaining_time)}"
            )

            # Save latest checkpoint
            save_checkpoint(
                epoch, self.model, self.optimizer, self.scheduler, history,
                self.ckpt_latest, extra={"best_iou": best_iou}
            )

            # Save best checkpoint
            if val_iou > best_iou:
                best_iou = val_iou
                no_improve = 0
                save_checkpoint(
                    epoch, self.model, self.optimizer, self.scheduler, history,
                    self.ckpt_best, extra={"best_iou": best_iou}
                )
                print(f"   --> Saved new best checkpoint (IoU: {best_iou:.4f})")
            else:
                no_improve += 1

            # Save training history json
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if no_improve >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1} (patience={self.patience} exceeded).")
                break

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        total_time_str = format_time(time.time() - total_start_time)
        print(f"\n[Done] Training completed in {total_time_str} | Best Validation IoU: {best_iou:.4f}")
        return history
