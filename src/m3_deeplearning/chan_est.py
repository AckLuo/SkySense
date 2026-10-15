"""M3：传统信道估计算法（LS / MMSE）——作为深度学习的 baseline.

对应周计划：第 20 周
"""

from __future__ import annotations

import numpy as np


def ls_estimate(rx_pilot: np.ndarray, tx_pilot: np.ndarray) -> np.ndarray:
    """最小二乘（Least Squares）信道估计.

    H_ls = Y / X  （逐元素相除）
    """
    rx_pilot = np.asarray(rx_pilot, dtype=complex)
    tx_pilot = np.asarray(tx_pilot, dtype=complex)
    return rx_pilot / (tx_pilot + 1e-12)


def mmse_estimate(rx_pilot: np.ndarray, tx_pilot: np.ndarray,
                  snr_db: float = 10.0, channel_var: float = 1.0) -> np.ndarray:
    """最小均方误差（MMSE）信道估计（简化对角模型）.

    H_mmse = R_hh (R_hh + σ²I)^{-1} H_ls
    对角近似下：H_mmse = H_ls · channel_var / (channel_var + σ²)
    """
    h_ls = ls_estimate(rx_pilot, tx_pilot)
    noise_var = channel_var / (10 ** (snr_db / 10))
    return h_ls * channel_var / (channel_var + noise_var)
