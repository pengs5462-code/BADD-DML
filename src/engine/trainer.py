from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch

from ..config import TrainConfig
from ..distill.loss import distill_loss


def train_one_epoch(
    net1: torch.nn.Module,
    net2: torch.nn.Module,
    loader,
    opt1: torch.optim.Optimizer,
    opt2: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train one epoch for online DML with symmetric distillation.
    Returns aggregated statistics for logging.
    """
    net1.train()
    net2.train()

    loss1_list, loss2_list = [], []
    w1_list, w2_list = [], []
    kl1_list, kl2_list = [], []
    alpha_list = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        out1 = net1(inputs)
        out2 = net2(inputs)

        l1, s1 = distill_loss(out1, out2, labels, epoch, cfg.mode, T=cfg.temperature, epochs_total=cfg.epochs, device=device)
        l2, s2 = distill_loss(out2, out1, labels, epoch, cfg.mode, T=cfg.temperature, epochs_total=cfg.epochs, device=device)

        opt1.zero_grad()
        opt2.zero_grad()
        l1.backward(retain_graph=True)
        l2.backward()
        opt1.step()
        opt2.step()

        loss1_list.append(l1.item())
        loss2_list.append(l2.item())
        w1_list.append(s1.get("mean", 1.0))
        w2_list.append(s2.get("mean", 1.0))
        kl1_list.append(s1.get("kl", 0.0))
        kl2_list.append(s2.get("kl", 0.0))

        if "alpha" in s1:
            alpha_list.append(s1["alpha"])

    stats = {
        "train_loss_1": float(np.mean(loss1_list)),
        "train_loss_2": float(np.mean(loss2_list)),
        "weight_1_mean": float(np.mean(w1_list)),
        "weight_2_mean": float(np.mean(w2_list)),
        "kl_1": float(np.mean(kl1_list)),
        "kl_2": float(np.mean(kl2_list)),
    }
    if len(alpha_list) > 0:
        stats["alpha_mean"] = float(np.mean(alpha_list))
    return stats
