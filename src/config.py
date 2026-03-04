from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Optional

import numpy as np
import torch


@dataclass
class TrainConfig:
    """
    Centralized experiment configuration.
    """
    # Data
    data_root: str = "./data"
    num_classes: int = 100
    num_workers: int = 4

    # Training
    epochs: int = 300
    batch_size: int = 128
    lr: float = 0.1
    temperature: float = 3.0
    seed: int = 0

    # Experiment
    arch: str = "hetero"  # hetero | homo
    mode: str = "dynamic_v17_11"
    save_dir: str = "./experiments"

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic settings (may reduce speed).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
