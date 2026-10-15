"""M1-练习5：采样与混叠.

采样定理：fs ≥ 2·fmax 才能无失真恢复信号，否则发生**混叠（aliasing）**。

对应周计划：第 6 周
"""

from __future__ import annotations

import numpy as np


def sample_signal(x: np.ndarray, fs_orig: float, fs_new: float) -> tuple[np.ndarray, np.ndarray]:
    """对信号重采样（抽取），返回新时间轴与采样值."""
    x = np.asarray(x)
    step = int(round(fs_orig / fs_new))
    if step < 1:
        raise ValueError("fs_new 必须小于 fs_orig")
    xs = x[::step]
    t_new = np.arange(len(xs)) / fs_new
    return t_new, xs


def demonstrate_aliasing(freq: float, fs: float, fs_sample: float,
                         duration: float = 1.0) -> dict:
    """演示欠采样导致的混叠.

    Args:
        freq: 原始信号频率.
        fs: 原始（高）采样率.
        fs_sample: 目标（低）采样率，若 < 2*freq 则出现混叠.

    Returns:
        包含原始信号、采样点、混叠频率等信息的字典.
    """
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * freq * t)

    t_s, x_s = sample_signal(x, fs, fs_sample)

    # 计算混叠后的表观频率
    f_alias = abs(freq - round(freq / fs_sample) * fs_sample)
    nyquist = fs_sample / 2
    aliased = freq > nyquist

    return {
        "t": t,
        "x": x,
        "t_sampled": t_s,
        "x_sampled": x_s,
        "freq": freq,
        "fs_sample": fs_sample,
        "nyquist": nyquist,
        "alias_freq": f_alias,
        "aliased": aliased,
    }
