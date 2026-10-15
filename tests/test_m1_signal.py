"""M1 信号基础模块单元测试.

运行::
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m1_signal import dft, idft, conv_manual, sine_wave, fourier_series_square


def test_sine_shape():
    t, x = sine_wave(freq=10, fs=1000, duration=1.0)
    assert len(t) == 1000
    assert len(x) == 1000
    assert np.isclose(np.max(np.abs(x)), 1.0, atol=1e-6)


def test_dft_matches_numpy():
    x = np.random.randn(32)
    assert np.allclose(dft(x), np.fft.fft(x), atol=1e-8)


def test_dft_idft_roundtrip():
    x = np.random.randn(16) + 1j * np.random.randn(16)
    assert np.allclose(idft(dft(x)), x, atol=1e-8)


def test_conv_matches_numpy():
    x = np.array([1.0, 2.0, 3.0])
    h = np.array([0.0, 1.0, 0.5])
    assert np.allclose(conv_manual(x, h), np.convolve(x, h))


def test_fourier_series_convergence():
    """谐波项越多，重构方波误差越小."""
    fs, dur, f0 = 1000, 1.0, 5.0
    t = np.arange(int(fs * dur)) / fs
    ideal = np.where((t * f0) % 1.0 < 0.5, 1.0, -1.0)

    err_low = np.mean((fourier_series_square(f0, fs, dur, 1) - ideal) ** 2)
    err_high = np.mean((fourier_series_square(f0, fs, dur, 20) - ideal) ** 2)
    assert err_high < err_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
