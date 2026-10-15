"""M4：无人机信道（含多普勒与时变特性）.

无人机高速运动 → 显著多普勒频移 → 信道时变
    f_d = (v / c) · f_c · cos(θ)

对应周计划：第 26 周
"""

from __future__ import annotations

import numpy as np

C_LIGHT = 3e8  # 光速 (m/s)


class UAVChannel:
    """无人机时变信道模型."""

    def __init__(self, carrier_freq: float = 2.4e9, fs: float = 1000.0,
                 n_paths: int = 3, seed: int = 0):
        self.fc = carrier_freq
        self.fs = fs
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def doppler_shift(self, velocity: float, angle_deg: float = 0.0) -> float:
        """计算多普勒频移.

        Args:
            velocity: 相对速度 (m/s).
            angle_deg: 运动方向与接收方向的夹角（度）.
        """
        theta = np.deg2rad(angle_deg)
        return (velocity / C_LIGHT) * self.fc * np.cos(theta)

    def apply(self, x: np.ndarray, trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """对信号施加随轨迹变化的时变信道."""
        x = np.asarray(x, dtype=complex)
        n = len(x)
        y = np.zeros(n, dtype=complex)

        # 由轨迹估计逐点速度
        if len(trajectory) > 1:
            disp = np.diff(trajectory, axis=0)
            speed = np.linalg.norm(disp, axis=1) / dt
            speed = np.concatenate([[speed[0]], speed])
        else:
            speed = np.zeros(n)

        speed = np.resize(speed, n)

        for p in range(self.n_paths):
            gain = (self.rng.standard_normal() + 1j * self.rng.standard_normal()) / np.sqrt(2 * self.n_paths)
            angle = self.rng.uniform(-60, 60)
            fd = self.doppler_shift(np.mean(speed), angle)
            t = np.arange(n) / self.fs
            y += gain * x * np.exp(1j * 2 * np.pi * fd * t)

        return y
