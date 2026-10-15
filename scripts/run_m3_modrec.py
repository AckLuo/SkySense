"""SkySense M3 一键运行脚本：调制识别（CNN）.

使用合成数据集快速跑通深度学习流程。
若有 RadioML 数据集，可切换为真实数据。

运行::

    python scripts/run_m3_modrec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split

from common import load_config, setup_plot_style, save_figure, set_seed, get_device
from m3_deeplearning import SyntheticModDataset, train_modrec


def plot_data_samples(dataset) -> None:
    """可视化各类调制的 I/Q 波形."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    per = len(dataset) // len(SyntheticModDataset.SCHEMES)
    for ax, (i, name) in zip(axes.ravel(), enumerate(SyntheticModDataset.SCHEMES)):
        iq, _ = dataset[i * per]
        iq = iq.numpy()
        ax.plot(iq[0], label="I")
        ax.plot(iq[1], label="Q")
        ax.set_title(f"{name.upper()}  I/Q 波形")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("M3 · 合成调制数据集样本", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "m3_data_samples.png")


def plot_training(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("训练损失"); axes[0].set_xlabel("Epoch")

    axes[1].plot(history["train_acc"], label="训练准确率")
    if history["val_acc"]:
        axes[1].plot(history["val_acc"], label="验证准确率")
    axes[1].set_title("准确率"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("M3 · CNN 调制识别训练曲线", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "m3_training_curve.png")


def main() -> None:
    set_seed(42)
    setup_plot_style()
    cfg = load_config("configs/default.yaml")

    print("=" * 60)
    print("SkySense · M3 深度学习模块（调制识别）")
    print("=" * 60)

    # 生成合成数据
    print("[数据] 生成合成调制数据集 ...")
    dataset = SyntheticModDataset(n_samples=4000, seq_len=128, seed=42)
    plot_data_samples(dataset)
    print(f"[数据] 样本数={len(dataset)}, 类别数={len(SyntheticModDataset.SCHEMES)}")

    # 划分训练/验证
    n_val = len(dataset) // 5
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])

    device = get_device(cfg.m3_dl.device)
    print(f"[设备] {device}")

    # 训练
    result = train_modrec(
        train_set, val_set,
        n_classes=len(SyntheticModDataset.SCHEMES),
        epochs=15, batch_size=cfg.m3_dl.batch_size,
        lr=cfg.m3_dl.lr, device=str(device),
    )
    plot_training(result["history"])

    # 保存模型
    model_path = Path(cfg.output.model_dir); model_path.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(result["model"].state_dict(), model_path / "m3_modrec_cnn.pt")
    print(f"[保存] 模型 -> {model_path / 'm3_modrec_cnn.pt'}")

    print("=" * 60)
    print("M3 完成！请查看 outputs/figures/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
