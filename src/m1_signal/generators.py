"""M1-练习1：基本信号生成.

本模块实现信号处理中最基础的几类信号，是后续卷积、傅里叶、采样的原料。

对应周计划：第 1 周
"""

from __future__ import annotations

import numpy as np


def _time_axis(fs: float, duration: float) -> np.ndarray:
    """生成时间轴 t = [0, dt, 2dt, ...]，共 duration 秒."""
    n = int(round(fs * duration))
    return np.arange(n) / fs


def sine_wave(freq: float, fs: float = 1000.0, duration: float = 1.0,
              amplitude: float = 1.0, phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """正弦信号 x(t) = A·sin(2πft + φ).

    Returns:
        (t, x) 时间轴与信号.
    """
    t = _time_axis(fs, duration)
    x = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return t, x


def cosine_wave(freq: float, fs: float = 1000.0, duration: float = 1.0,
                amplitude: float = 1.0, phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """余弦信号 x(t) = A·cos(2πft + φ)."""
    t = _time_axis(fs, duration)
    x = amplitude * np.cos(2 * np.pi * freq * t + phase)
    return t, x


def square_wave(freq: float, fs: float = 1000.0, duration: float = 1.0,
                amplitude: float = 1.0, duty: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """方波信号（占空比可调），由相位累积生成，避免边界不连续.

    Args:
        duty: 占空比，0.5 为标准方波.
    """
    t = _time_axis(fs, duration)
    phase = (t * freq) % 1.0
    x = np.where(phase < duty, amplitude, -amplitude).astype(float)
    return t, x


def triangle_wave(freq: float, fs: float = 1000.0, duration: float = 1.0,
                  amplitude: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """三角波信号（幅值范围 [-A, A]）."""
    t = _time_axis(fs, duration)
    phase = (t * freq) % 1.0
    # 相位 0->0.25 上升至 +1，0.25->0.75 下降至 -1，0.75->1 上升至 0
    x = 4.0 * np.abs(phase - 0.5) - 1.0
    x = amplitude * x
    return t, x


def unit_impulse(n: int = 100, index: int = 0, amplitude: float = 1.0) -> np.ndarray:
    """离散单位冲激 δ[n - index]."""
    x = np.zeros(n)
    if 0 <= index < n:
        x[index] = amplitude
    return x


def unit_step(n: int = 100, index: int = 0) -> np.ndarray:
    """离散单位阶跃 u[n - index]."""
    x = np.zeros(n)
    if 0 <= index < n:
        x[index:] = 1.0
    return x


def complex_exponential(freq: float, fs: float = 1000.0, duration: float = 1.0) -> np.ndarray:
    """复指数信号 x(t) = exp(j2πft)."""
    t = _time_axis(fs, duration)
    return np.exp(1j * 2 * np.pi * freq * t)
