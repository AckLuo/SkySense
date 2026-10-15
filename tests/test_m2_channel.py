"""M2 信道模块单元测试."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m2_channel import modulate, demodulate, add_awgn_snr, ber


@pytest.mark.parametrize("scheme", ["bpsk", "qpsk", "16qam"])
def test_modulate_demodulate_noiseless(scheme):
    """无噪声下调制解调应完全恢复."""
    bits = np.random.randint(0, 2, 200)
    tx = modulate(bits, scheme)
    rx = demodulate(tx, scheme)
    n = min(len(bits), len(rx))
    assert ber(bits[:n], rx[:n]) == 0.0


def test_ber_decreases_with_snr():
    """SNR 越高，BER 应越低（统计趋势）."""
    np.random.seed(0)
    bits = np.random.randint(0, 2, 40000)
    tx = modulate(bits, "qpsk")

    ber_low = ber(bits, demodulate(add_awgn_snr(tx, 0), "qpsk"))
    ber_high = ber(bits, demodulate(add_awgn_snr(tx, 15), "qpsk"))

    assert ber_high < ber_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
