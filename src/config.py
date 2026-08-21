import os
import random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import yaml


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the preferred compute device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def setup_directories(output_dir: Path) -> Dict[str, Path]:
    """Create and return standard output directories."""
    output_dir = Path(output_dir)
    dirs = {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "patches": output_dir / "patches",
        "results": output_dir / "results",
        "exports": output_dir / "exports",
        "logs": output_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
