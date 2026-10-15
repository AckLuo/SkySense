"""M3：调制识别训练流程.

对应周计划：第 18-19 周
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .models import CNNModRec


def train_modrec(train_set, val_set=None, n_classes: int = 4,
                 epochs: int = 20, batch_size: int = 64, lr: float = 1e-3,
                 device: str = "cpu", verbose: bool = True) -> dict:
    """训练调制识别模型.

    Returns:
        dict: 包含训练好的模型、训练历史.
    """
    device = torch.device(device)
    model = CNNModRec(n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = (DataLoader(val_set, batch_size=batch_size) if val_set is not None else None)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        train_loss = total_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        msg = f"[Epoch {epoch:3d}/{epochs}] loss={train_loss:.4f} acc={train_acc:.4f}"

        if val_loader is not None:
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    v_correct += (model(xb).argmax(1) == yb).sum().item()
                    v_total += len(yb)
            val_acc = v_correct / v_total
            history["val_acc"].append(val_acc)
            msg += f" val_acc={val_acc:.4f}"

        if verbose:
            print(msg)

    return {"model": model, "history": history}
