"""M2：多径信道模型（含时延与多普勒）.

多径效应：信号经不同路径到达接收端，产生时延扩展 → 符号间干扰（ISI）。
无人机场景下还存在多普勒频移（因高速运动）。

对应周计划：第 12-13 周
"""

from __future__ import annotations

import numpy as np


class MultipathChannel:
    """时变多径信道.

    模型：
        y(t) = Σ_p a_p · x(t - τ_p) · exp(j2π f_d,p t)
    """

    def __init__(self, n_paths: int = 4, max_delay: float = 5e-3,
                 fs: float = 1000.0, doppler: float = 0.0, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.fs = fs
        self.n_paths = n_paths
        self.delays = self.rng.uniform(0, max_delay, n_paths)
        self.delays[0] = 0.0  # 首径时延为 0
        # 瑞利衰落幅度（指数功率时延谱）
        power = np.exp(-self.delays / (max_delay / 3 + 1e-12))
        self.gains = (self.rng.standard_normal(n_paths)
                      + 1j * self.rng.standard_normal(n_paths)) * np.sqrt(power / 2)
        self.dopplers = self.rng.uniform(-doppler, doppler, n_paths)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """对输入信号施加多径 + 多普勒效应."""
        x = np.asarray(x, dtype=complex)
        n = len(x)
        t = np.arange(n) / self.fs
        y = np.zeros(n, dtype=complex)

        for delay, gain, fd in zip(self.delays, self.gains, self.dopplers):
            shift = int(round(delay * self.fs))
            delayed = np.zeros(n, dtype=complex)
            if shift < n:
                delayed[shift:] = x[: n - shift]
            doppler_phase = np.exp(1j * 2 * np.pi * fd * t)
            y += gain * delayed * doppler_phase

        return y

    def info(self) -> dict:
        return {
            "n_paths": self.n_paths,
            "delays_ms": (self.delays * 1e3).tolist(),
            "doppler_hz": self.dopplers.tolist(),
        }
