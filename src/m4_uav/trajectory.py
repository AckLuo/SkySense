"""M4：无人机轨迹生成.

对应周计划：第 27 周
"""

from __future__ import annotations

import numpy as np


def generate_trajectory(kind: str = "circle", n_steps: int = 200,
                        velocity: float = 20.0, dt: float = 0.1,
                        radius: float = 50.0) -> np.ndarray:
    """生成无人机 3D 轨迹.

    Args:
        kind: circle / line / random.
        n_steps: 轨迹点数.
        velocity: 速度 (m/s).
        dt: 时间步长 (s).
        radius: 圆周半径 (m).

    Returns:
        (n_steps, 3) 的 (x, y, z) 轨迹.
    """
    t = np.arange(n_steps) * dt
    if kind == "circle":
        omega = velocity / max(radius, 1e-6)
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        z = np.full_like(t, 100.0)
    elif kind == "line":
        x = velocity * t
        y = np.zeros_like(t)
        z = np.full_like(t, 100.0)
    elif kind == "random":
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.normal(0, velocity * dt, n_steps))
        y = np.cumsum(rng.normal(0, velocity * dt, n_steps))
        z = np.full_like(t, 100.0)
    else:
        raise ValueError(f"未知轨迹类型: {kind}")

    return np.stack([x, y, z], axis=1)
