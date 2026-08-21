import argparse
import json
from pathlib import Path
import pickle
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_device
from src.data.augmentation import get_test_augmentation
from src.models.mlproadnet import MLPRoadNet
from src.training.trainer import load_checkpoint
from src.evaluation.evaluator import full_evaluation
from src.visualization.visualize import plot_metrics_and_cm, plot_roc_curve, plot_test_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MLPRoadNet model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to save evaluation artifacts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir or cfg["output"]["output_dir"])
    ckpt_path = Path(args.checkpoint or (output_dir / "checkpoints" / "best_model.pth"))

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    index_file = output_dir / "patches" / "patch_index.pkl"
    if not index_file.exists():
        raise FileNotFoundError(f"Patch index not found at: {index_file}. Run training first.")

    with open(index_file, "rb") as f:
        patch_index = pickle.load(f)

    test_split = cfg["split"]["test_split"]
    split_idx = int(len(patch_index) * (1 - test_split))
    test_patches = patch_index[split_idx:]

    device = get_device()
    print(f"--> Evaluating on {device} using checkpoint: {ckpt_path}")

    model = MLPRoadNet(
        in_ch=3,
        base_ch=cfg["model"]["base_channels"],
        mlp_depth=cfg["model"]["mlp_depth"],
        patch_size=cfg["patches"]["patch_size"],
    ).to(device)

    load_checkpoint(ckpt_path, model, device=device)

    # DeepGlobe dataset mean/std
    mean = [0.4094, 0.3791, 0.2814]
    std = [0.1225, 0.0972, 0.0880]
    test_aug = get_test_augmentation(mean, std)

    metrics, cm, all_probs, all_tgts = full_evaluation(model, test_patches, test_aug, device)

    print("\n" + "=" * 50)
    print("  Evaluation Metrics")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("=" * 50)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_metrics_and_cm(metrics, cm, results_dir / "eval_metrics.png")
    plot_roc_curve(all_tgts, all_probs, metrics["AUC"], results_dir / "eval_roc.png")
    plot_test_predictions(model, test_patches, test_aug, device, results_dir / "eval_predictions.png")

    # Export to trained_models/
    trained_models_dir = Path("trained_models")
    trained_models_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(ckpt_path), str(trained_models_dir / "best_model.pth"))

    torch.save({
        "model_state": model.state_dict(),
        "architecture": "MLPRoadNet",
        "config": cfg,
        "metrics": metrics,
        "dataset_mean": mean,
        "dataset_std": std,
    }, str(trained_models_dir / "MLPRoadNet_best_weights.pth"))

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

    print(f"\n[Done] Evaluation complete. Artifacts saved to: {results_dir}")
    print(f"[Export] Production models packaged to: trained_models/\n")


if __name__ == "__main__":
    main()
