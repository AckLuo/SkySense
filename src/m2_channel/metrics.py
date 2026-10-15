"""M2：性能指标（BER / EVM）.

对应周计划：第 13-14 周
"""

from __future__ import annotations

import numpy as np


def ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    """误比特率 = 错误比特数 / 总比特数."""
    bits_tx = np.asarray(bits_tx)
    bits_rx = np.asarray(bits_rx)
    n = min(len(bits_tx), len(bits_rx))
    return float(np.mean(bits_tx[:n] != bits_rx[:n]))


def evm(rx_symbols: np.ndarray, tx_symbols: np.ndarray) -> float:
    """误差矢量幅度（Error Vector Magnitude, %）."""
    rx = np.asarray(rx_symbols, dtype=complex)
    tx = np.asarray(tx_symbols, dtype=complex)
    n = min(len(rx), len(tx))
    err = rx[:n] - tx[:n]
    return float(np.sqrt(np.mean(np.abs(err) ** 2) / np.mean(np.abs(tx[:n]) ** 2)) * 100)
