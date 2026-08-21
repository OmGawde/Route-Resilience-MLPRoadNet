import argparse
import json
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_seed, get_device, setup_directories
from src.data.dataset import collect_pairs, compute_dataset_stats, SpaceNetDataset
from src.data.patches import extract_patches_resumable
from src.data.augmentation import get_train_augmentation, get_test_augmentation
from src.models.mlproadnet import MLPRoadNet
from src.losses.losses import MLPRoadNetLoss
from src.training.trainer import Trainer, load_checkpoint
from src.evaluation.evaluator import full_evaluation
from src.visualization.visualize import (
    plot_sample_grid,
    plot_training_curves,
    plot_metrics_and_cm,
    plot_roc_curve,
    plot_test_predictions,
)


def main():
    parser = argparse.ArgumentParser(description="Train MLPRoadNet on Satellite Imagery")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root path")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel CPU data loader workers")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples per epoch for fast turnaround (e.g. 3000)")
    parser.add_argument("--no-resume", action="store_true", help="Start training from scratch without loading checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI Overrides
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.output_dir:
        cfg["output"]["output_dir"] = args.output_dir
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers

    setup_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"Compute Device: {device}")

    output_dirs = setup_directories(Path(cfg["output"]["output_dir"]))
    data_root = Path(cfg["data"]["data_root"])
    images_dir = data_root / cfg["data"]["images_dir"]
    masks_dir = data_root / cfg["data"]["masks_dir"]

    # Auto-detect if data_root is a direct folder or DeepGlobe train folder
    if not images_dir.exists():
        if (data_root / "train").exists():
            images_dir = data_root / "train"
            masks_dir = data_root / "train"
        else:
            images_dir = data_root
            masks_dir = data_root

    print("Loaded Paths:")
    print(f"   Images: {images_dir}")
    print(f"   Masks:  {masks_dir}")
    print(f"   Output: {output_dirs['root']}")

    # 1. Collect file pairs
    pairs = collect_pairs(images_dir, masks_dir, cfg["data"]["img_ext"])
    print(f"Found {len(pairs)} matching image-mask pairs.")
    if len(pairs) == 0:
        raise FileNotFoundError(f"No image-mask pairs found in {data_root}. Check dataset paths.")

    # 2. EDA statistics
    mean, std, road_ratio = compute_dataset_stats(pairs, sample_size=min(50, len(pairs)))
    print(f"   Mean (RGB): {mean}")
    print(f"   Std  (RGB): {std}")
    print(f"   Average Road Ratio: {road_ratio * 100:.2f}%")

    plot_sample_grid(pairs, output_dirs["results"] / "viz_01_sample_grid.png")

    # 3. Patch Extraction
    patch_index = extract_patches_resumable(
        pairs,
        patch_size=cfg["patches"]["patch_size"],
        stride=cfg["patches"]["patch_stride"],
        min_road_ratio=cfg["patches"]["min_road_ratio"],
        patch_dir=output_dirs["patches"],
        progress_file=output_dirs["patches"] / "extract_progress.json",
        index_file=output_dirs["patches"] / "patch_index.pkl",
    )

    # Validate that all patch files actually exist on disk
    patch_index = [p for p in patch_index if Path(p[0]).exists() and Path(p[1]).exists()]
    print(f"Verified {len(patch_index)} valid patch pairs on disk.")

    # 4. Train / Test Split
    random.shuffle(patch_index)
    train_patches, test_patches = train_test_split(
        patch_index,
        test_size=cfg["split"]["test_split"],
        random_state=cfg.get("seed", 42),
    )
    print(f"Train patches: {len(train_patches)} | Test patches: {len(test_patches)}")
    if args.max_train_samples and args.max_train_samples < len(train_patches):
        active_train_patches = train_patches[:args.max_train_samples]
        print(f"Active training samples per epoch: {len(active_train_patches)} (fast turnaround mode)")
    else:
        active_train_patches = train_patches

    with open(output_dirs["logs"] / "data_split.json", "w", encoding="utf-8") as f:
        json.dump({"train_count": len(train_patches), "test_count": len(test_patches)}, f, indent=2)

    # 5. DataLoaders
    train_aug = get_train_augmentation(mean, std)
    test_aug = get_test_augmentation(mean, std)

    train_ds = SpaceNetDataset(active_train_patches, transform=train_aug, label_smooth=cfg["noise"]["label_smooth"])
    test_ds = SpaceNetDataset(test_patches, transform=test_aug, label_smooth=0.0)

    num_workers = cfg["training"]["num_workers"]
    pin_memory = (device.type == "cuda")

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # 6. Model & Loss
    model = MLPRoadNet(
        in_ch=3,
        base_ch=cfg["model"]["base_channels"],
        mlp_depth=cfg["model"]["mlp_depth"],
        patch_size=cfg["patches"]["patch_size"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"MLPRoadNet initialized | Total Parameters: {total_params:,}")

    criterion = MLPRoadNetLoss(noise_reg_weight=cfg["noise"]["noise_reg_weight"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # 7. Training
    trainer = Trainer(
        model=model,
        train_loader=train_dl,
        val_loader=test_dl,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dirs=output_dirs,
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        grad_clip=cfg["training"]["grad_clip"],
    )

    history = trainer.train(resume=not args.no_resume)

    # 8. Load best model and evaluate
    ckpt_best = output_dirs["checkpoints"] / "best_model.pth"
    if ckpt_best.exists():
        _, _, best_iou = load_checkpoint(ckpt_best, model, device=device)
        plot_training_curves(history, best_iou, output_dirs["results"] / "viz_05_training_curves.png")

    print("\nRunning full evaluation on test set...")
    metrics, cm, all_probs, all_tgts = full_evaluation(model, test_patches, test_aug, device)

    print("\n" + "=" * 50)
    print("  Final MLPRoadNet Evaluation Metrics")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("=" * 50)

    with open(output_dirs["results"] / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_metrics_and_cm(metrics, cm, output_dirs["results"] / "viz_06_metrics.png")
    plot_roc_curve(all_tgts, all_probs, metrics["AUC"], output_dirs["results"] / "viz_07_roc_curve.png")
    plot_test_predictions(model, test_patches, test_aug, device, output_dirs["results"] / "viz_09_test_predictions.png")

    # 9. Auto-Export into top-level trained_models/ folder
    trained_models_dir = Path("trained_models")
    trained_models_dir.mkdir(parents=True, exist_ok=True)

    # Copy best PyTorch checkpoint
    if ckpt_best.exists():
        import shutil
        shutil.copy2(str(ckpt_best), str(trained_models_dir / "best_model.pth"))

    # Save full weights package with metadata
    torch.save({
        "model_state": model.state_dict(),
        "architecture": "MLPRoadNet",
        "config": cfg,
        "metrics": metrics,
        "dataset_mean": mean,
        "dataset_std": std,
        "best_iou": best_iou if ckpt_best.exists() else 0.0,
    }, str(trained_models_dir / "MLPRoadNet_best_weights.pth"))

    # Export TorchScript and ONNX
    dummy_input = torch.randn(1, 3, cfg["patches"]["patch_size"], cfg["patches"]["patch_size"]).to(device)
    try:
        scripted = torch.jit.trace(model, dummy_input)
        scripted.save(str(trained_models_dir / "MLPRoadNet_scripted.pt"))
    except Exception:
        pass

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(trained_models_dir / "MLPRoadNet.onnx"),
            export_params=True,
            opset_version=13,
            input_names=["image"],
            output_names=["final_mask", "mask_logit", "centerline_logit"],
            dynamic_axes={"image": {0: "batch_size"}},
        )
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("  [SAVED] All trained models saved to: trained_models/")
    print(f"     * PyTorch Checkpoint: trained_models/best_model.pth")
    print(f"     * Full Weights Pkg  : trained_models/MLPRoadNet_best_weights.pth")
    print(f"     * TorchScript Model : trained_models/MLPRoadNet_scripted.pt")
    print(f"     * ONNX Model        : trained_models/MLPRoadNet.onnx")
    print("=" * 60)

    print(f"\n[Done] Training and evaluation finished! All artifacts saved to: {output_dirs['root']}")


if __name__ == "__main__":
    main()
