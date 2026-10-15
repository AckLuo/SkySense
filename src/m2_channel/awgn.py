"""M2：加性高斯白噪声（AWGN）.

对应周计划：第 9 周
"""

from __future__ import annotations

import numpy as np


def add_awgn(x: np.ndarray, noise_power: float = 1.0) -> np.ndarray:
    """添加指定功率的高斯白噪声."""
    x = np.asarray(x, dtype=complex)
    n = np.sqrt(noise_power / 2) * (
        np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
    )
    return x + n


def add_awgn_snr(x: np.ndarray, snr_db: float) -> np.ndarray:
    """按目标 SNR(dB) 添加高斯白噪声.

    噪声功率由信号功率与目标 SNR 反推。
    """
    x = np.asarray(x, dtype=complex)
    signal_power = np.mean(np.abs(x) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return add_awgn(x, noise_power)
