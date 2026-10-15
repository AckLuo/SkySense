"""M1-练习2/3/4：变换工具（DFT / FFT / 傅里叶级数）.

关键思想：
    DFT:      X[k] = Σ_{n=0}^{N-1} x[n] · exp(-j2πkn/N)
    IDFT:     x[n] = (1/N) Σ_{k=0}^{N-1} X[k] · exp(j2πkn/N)

对应周计划：第 3-5 周
"""

from __future__ import annotations

import numpy as np


def dft(x: np.ndarray) -> np.ndarray:
    """手写 DFT（O(N²)），用于理解定义.

    Args:
        x: 输入实/复序列.

    Returns:
        频域复数序列，长度与输入相同.
    """
    x = np.asarray(x, dtype=complex)
    n = len(x)
    k = np.arange(n).reshape(-1, 1)
    nn = np.arange(n).reshape(1, -1)
    w = np.exp(-1j * 2 * np.pi * k * nn / n)
    return w @ x


def idft(x: np.ndarray) -> np.ndarray:
    """手写 IDFT（逆变换）."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    k = np.arange(n).reshape(-1, 1)
    nn = np.arange(n).reshape(1, -1)
    w = np.exp(1j * 2 * np.pi * k * nn / n)
    return (w @ x) / n


def fft_spectrum(x: np.ndarray, fs: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算单边幅度谱.

    Args:
        x: 时域信号.
        fs: 采样率 (Hz).

    Returns:
        (freqs, mag, phase)：频率轴、幅度谱、相位谱.
    """
    x = np.asarray(x)
    n = len(x)
    xf = np.fft.fft(x)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    # 只取非负频率部分
    half = n // 2
    freqs_pos = freqs[:half]
    mag = np.abs(xf[:half]) / n * 2  # 归一化
    phase = np.angle(xf[:half])
    return freqs_pos, mag, phase


def fourier_series_square(freq: float, fs: float, duration: float,
                          n_harmonics: int, amplitude: float = 1.0) -> np.ndarray:
    """用前 N 项奇次谐波重构方波.

    方波的傅里叶级数（奇对称）：
        x(t) = (4A/π) · Σ_{k=1,3,5,...} (1/k)·sin(2πkf₀t)

    Args:
        n_harmonics: 使用的谐波项数（只取奇次）.

    Returns:
        重构后的信号.
    """
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    x = np.zeros_like(t)
    for k in range(1, 2 * n_harmonics, 2):  # 1, 3, 5, ...
        x += (1.0 / k) * np.sin(2 * np.pi * k * freq * t)
    return x * (4 * amplitude / np.pi)
