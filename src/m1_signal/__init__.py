"""M1 信号基础模块：信号生成、卷积、傅里叶、采样、滤波."""

from .generators import (
    sine_wave,
    cosine_wave,
    square_wave,
    triangle_wave,
    unit_impulse,
    unit_step,
    complex_exponential,
)
from .transforms import dft, idft, fft_spectrum, fourier_series_square
from .convolution import conv_manual, conv_compare
from .sampling import sample_signal, demonstrate_aliasing
from .filters import design_fir_lowpass, apply_filter

__all__ = [
    "sine_wave",
    "cosine_wave",
    "square_wave",
    "triangle_wave",
    "unit_impulse",
    "unit_step",
    "complex_exponential",
    "dft",
    "idft",
    "fft_spectrum",
    "fourier_series_square",
    "conv_manual",
    "conv_compare",
    "sample_signal",
    "demonstrate_aliasing",
    "design_fir_lowpass",
    "apply_filter",
]
