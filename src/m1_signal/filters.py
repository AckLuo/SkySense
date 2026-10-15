"""M1-练习8：FIR 滤波器设计与应用.

用窗函数法设计低通 FIR 滤波器，并用卷积实现滤波。

对应周计划：第 8 周
"""

from __future__ import annotations

import numpy as np


def design_fir_lowpass(cutoff: float, fs: float, numtaps: int = 101) -> np.ndarray:
    """窗函数法设计低通 FIR 滤波器.

    Args:
        cutoff: 截止频率 (Hz).
        fs: 采样率 (Hz).
        numtaps: 滤波器阶数+1（建议奇数）.

    Returns:
        滤波器系数 h.
    """
    try:
        from scipy.signal import firwin

        return firwin(numtaps, cutoff, fs=fs)
    except ImportError:
        # 无 scipy 时用理想低通 + 汉明窗近似
        m = np.arange(numtaps) - (numtaps - 1) / 2
        fc = cutoff / fs  # 归一化截止频率
        h = 2 * fc * np.sinc(2 * fc * m)
        h *= np.hamming(numtaps)
        h /= np.sum(h)
        return h


def apply_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """对信号做 FIR 滤波（卷积）."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    return np.convolve(x, h, mode="same")
