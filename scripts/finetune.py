"""
Fine-Tuning Script for MLPRoadNet with High-Weight Topology Loss and Hard-Negative Focal Loss.
Resumes from best_model.pth to polish boundary sharpness and enforce graph connectivity.
"""

import argparse
import json
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_seed, get_device, setup_directories
from src.data.dataset import SpaceNetDataset
from src.data.augmentation import get_train_augmentation, get_test_augmentation
from src.models.mlproadnet import MLPRoadNet
from src.losses.losses import MLPRoadNetLoss
from src.training.trainer import Trainer, load_checkpoint
from src.evaluation.evaluator import full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MLPRoadNet on Satellite Imagery")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="trained_models/best_model.pth", help="Starting checkpoint")
    parser.add_argument("--output-dir", type=str, default="output/deepglobe_finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Fine-tuning learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--topo-weight", type=float, default=1.5, help="Weight for topological gap penalty")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples per epoch for fast turnaround (e.g. 4000)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"Compute Device: {device}")

    output_dirs = setup_directories(Path(args.output_dir))
    
    # Load patch index
    import pickle
    index_file = Path("output/deepglobe_model/patches/patch_index.pkl")
    if not index_file.exists():
        index_file = Path(cfg["output"]["output_dir"]) / "patches" / "patch_index.pkl"

    with open(index_file, "rb") as f:
        patch_index = pickle.load(f)

    patch_index = [p for p in patch_index if Path(p[0]).exists() and Path(p[1]).exists()]

    random.shuffle(patch_index)
    train_patches, test_patches = train_test_split(
        patch_index,
        test_size=cfg["split"]["test_split"],
        random_state=cfg.get("seed", 42),
    )

    if args.max_train_samples and args.max_train_samples < len(train_patches):
        active_train_patches = train_patches[:args.max_train_samples]
        print(f"Active fine-tuning samples per epoch: {len(active_train_patches)} / {len(train_patches)} (fast mode)")
    else:
        active_train_patches = train_patches

    mean = [0.4094, 0.3791, 0.2814]
    std = [0.1225, 0.0972, 0.0880]

    train_aug = get_train_augmentation(mean, std)
    test_aug = get_test_augmentation(mean, std)

    train_ds = SpaceNetDataset(active_train_patches, transform=train_aug, label_smooth=0.02)
    test_ds = SpaceNetDataset(test_patches, transform=test_aug, label_smooth=0.0)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Initialize model and load best checkpoint
    model = MLPRoadNet(
        in_ch=3,
        base_ch=cfg["model"]["base_channels"],
        mlp_depth=cfg["model"]["mlp_depth"],
        patch_size=cfg["patches"]["patch_size"],
    ).to(device)

    print(f"Loading base checkpoint: {args.checkpoint}")
    load_checkpoint(Path(args.checkpoint), model, device=device)

    # Multi-task loss with boosted topology penalty
    criterion = MLPRoadNetLoss(
        noise_reg_weight=0.2,
        topo_weight=args.topo_weight,
        aux_weight=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_dl,
        val_loader=test_dl,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dirs=output_dirs,
        epochs=args.epochs,
        patience=6,
        grad_clip=1.0,
    )

    print(f"\n--> Starting targeted fine-tuning for {args.epochs} epochs (Topo Weight = {args.topo_weight})...")
    history = trainer.train(resume=False)

    ckpt_best = output_dirs["checkpoints"] / "best_model.pth"
    if ckpt_best.exists():
        import shutil
        shutil.copy2(str(ckpt_best), "trained_models/best_model_finetuned.pth")
        print(f"\n[SAVED] Best fine-tuned model saved to: trained_models/best_model_finetuned.pth")

    print("\nRunning full evaluation on test set...")
    metrics, cm, all_probs, all_tgts = full_evaluation(model, test_patches, test_aug, device)

    print("\n" + "=" * 50)
    print("  Fine-Tuned MLPRoadNet Evaluation Metrics")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
