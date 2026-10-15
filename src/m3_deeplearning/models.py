"""M3：深度学习模型定义（CNN 调制识别 / 信道估计）.

核心亮点：这是简历上的技术核心。

对应周计划：第 15-23 周
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNModRec(nn.Module):
    """用于调制识别的 CNN.

    输入: (B, 2, L) —— I/Q 两路信号，长度 L
    输出: (B, n_classes) —— 各调制类别的 logits
    """

    def __init__(self, n_classes: int = 11, in_ch: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNChannelEstimator(nn.Module):
    """用于信道估计的 CNN.

    输入: (B, 2, L) —— 接收到的导频信号（实部/虚部）
    输出: (B, 2, L) —— 估计的信道实部/虚部
    """

    def __init__(self, in_ch: int = 2, out_ch: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model(name: str, **kwargs) -> nn.Module:
    """模型工厂."""
    name = name.lower()
    if name == "cnn":
        if kwargs.get("task") == "chanest":
            return CNNChannelEstimator()
        return CNNModRec(**{k: v for k, v in kwargs.items() if k in ("n_classes", "in_ch")})
    raise ValueError(f"未知模型: {name}")
