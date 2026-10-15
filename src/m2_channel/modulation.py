"""M2：数字调制与解调（BPSK / QPSK / 16QAM）.

对应周计划：第 10-11 周
"""

from __future__ import annotations

import numpy as np

_CONSTELLATIONS = {
    "bpsk": np.array([1, -1], dtype=complex),
    "qpsk": np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2),
    "16qam": None,  # 运行时生成
}


def _build_16qam() -> np.ndarray:
    levels = np.array([-3, -1, 1, 3])
    pts = [l_i + 1j * l_q for l_q in levels for l_i in levels]
    return np.array(pts, dtype=complex) / np.sqrt(10)  # 归一化平均功率为 1


def get_constellation(scheme: str) -> np.ndarray:
    """返回调制星座点."""
    scheme = scheme.lower()
    if scheme == "16qam":
        return _build_16qam()
    if scheme not in _CONSTELLATIONS:
        raise ValueError(f"不支持的调制方式: {scheme}")
    return _CONSTELLATIONS[scheme]


def modulate(bits: np.ndarray, scheme: str = "qpsk") -> np.ndarray:
    """将比特映射为星座符号.

    Args:
        bits: 0/1 序列.
        scheme: bpsk / qpsk / 16qam.

    Returns:
        复符号序列.
    """
    bits = np.asarray(bits, dtype=int)
    scheme = scheme.lower()
    k = {"bpsk": 1, "qpsk": 2, "16qam": 4}[scheme]

    # 补齐到 k 的整数倍
    pad = (-len(bits)) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    symbols_idx = bits.reshape(-1, k).dot(1 << np.arange(k - 1, -1, -1))

    constellation = get_constellation(scheme)
    return constellation[symbols_idx]


def demodulate(symbols: np.ndarray, scheme: str = "qpsk") -> np.ndarray:
    """最小距离判决解调（硬判决）."""
    constellation = get_constellation(scheme)
    symbols = np.asarray(symbols, dtype=complex)
    dist = np.abs(symbols[:, None] - constellation[None, :])
    idx = np.argmin(dist, axis=1)

    k = {"bpsk": 1, "qpsk": 2, "16qam": 4}[scheme.lower()]
    bits = ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1).reshape(-1)
    return bits
