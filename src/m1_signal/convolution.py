"""M1-练习7：卷积.

卷积的物理意义：系统对输入信号每一个时刻冲激响应的加权叠加。
    (x * h)[n] = Σ_k x[k]·h[n-k]

对应周计划：第 2 周
"""

from __future__ import annotations

import numpy as np


def conv_manual(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """手写一维离散卷积（双重循环版，用于理解定义）.

    输出长度 = len(x) + len(h) - 1
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    n_out = len(x) + len(h) - 1
    y = np.zeros(n_out)
    for n in range(n_out):
        acc = 0.0
        for k in range(len(x)):
            if 0 <= n - k < len(h):
                acc += x[k] * h[n - k]
        y[n] = acc
    return y


def conv_compare(x: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对比手写卷积与 scipy/numpy 卷积结果.

    Returns:
        (y_manual, y_numpy)
    """
    y_manual = conv_manual(x, h)
    y_numpy = np.convolve(x, h, mode="full")
    max_err = np.max(np.abs(y_manual - y_numpy))
    print(f"[卷积对比] 最大误差 = {max_err:.2e}")
    return y_manual, y_numpy
