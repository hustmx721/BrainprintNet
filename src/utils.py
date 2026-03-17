"""Shared utility helpers used across the BrainprintNet package."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def ensure_directory(path: Path) -> Path:
    """Create a directory when it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Configure Python, NumPy, and PyTorch for reproducible experiments."""

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_index: int) -> torch.device:
    """Resolve the target torch device from a CUDA index."""

    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def save_frame(frame: pd.DataFrame, output_path: Path) -> None:
    """Persist a pandas dataframe as a CSV file."""

    ensure_directory(output_path.parent)
    frame.to_csv(output_path, index=True)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Persist a JSON-serialisable dictionary to disk."""

    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
