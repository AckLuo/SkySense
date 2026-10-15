"""SkySense M1 一键运行脚本：信号基础模块演示.

运行::

    python scripts/run_m1_signal.py

将生成以下图像到 outputs/figures/：
    - m1_basic_signals.png   基本信号
    - m1_fft_spectrum.png    频谱分析（含混叠演示）
    - m1_fourier_series.png  傅里叶级数重构方波
    - m1_convolution.png     卷积演示
    - m1_filtering.png       滤波演示
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 允许直接运行（无需安装）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from common import load_config, setup_plot_style, save_figure, set_seed
from m1_signal import (
    sine_wave, cosine_wave, square_wave, triangle_wave,
    unit_impulse, unit_step,
    dft, fft_spectrum, fourier_series_square,
    conv_compare, demonstrate_aliasing,
    design_fir_lowpass, apply_filter,
)


def plot_basic_signals(cfg) -> None:
    """练习1：绘制基本信号."""
    fs, dur, f0 = cfg.m1_signal.fs, cfg.m1_signal.duration, cfg.m1_signal.f0

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    t, x = sine_wave(f0, fs, dur)
    axes[0, 0].plot(t, x)
    axes[0, 0].set_title(f"正弦波  {f0} Hz")

    t, x = cosine_wave(f0, fs, dur)
    axes[0, 1].plot(t, x, color="orange")
    axes[0, 1].set_title(f"余弦波  {f0} Hz")

    t, x = square_wave(f0, fs, dur)
    axes[1, 0].plot(t, x, color="green")
    axes[1, 0].set_title(f"方波  {f0} Hz")

    t, x = triangle_wave(f0, fs, dur)
    axes[1, 1].plot(t, x, color="red")
    axes[1, 1].set_title(f"三角波  {f0} Hz")

    axes[2, 0].stem(unit_impulse(50, index=10), basefmt=" ")
    axes[2, 0].set_title("单位冲激 δ[n-10]")

    axes[2, 1].stem(unit_step(50, index=10), basefmt=" ")
    axes[2, 1].set_title("单位阶跃 u[n-10]")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)

    fig.suptitle("M1 · 基本信号", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "m1_basic_signals.png")


def plot_fft_spectrum(cfg) -> None:
    """练习2/4/5：FFT 频谱 + 手写DFT验证 + 混叠演示."""
    fs, dur, f0 = cfg.m1_signal.fs, cfg.m1_signal.duration, cfg.m1_signal.f0

    # 构造多频叠加信号
    t, x = sine_wave(f0, fs, dur, amplitude=1.0)
    _, x2 = sine_wave(3 * f0, fs, dur, amplitude=0.5)
    _, x3 = sine_wave(7 * f0, fs, dur, amplitude=0.3)
    x_sum = x + x2 + x3

    freqs, mag, _ = fft_spectrum(x_sum, fs)

    # 手写 DFT 验证（用小样本）
    n_small = 64
    x_small = x_sum[:n_small]
    err = np.max(np.abs(dft(x_small) - np.fft.fft(x_small)))
    print(f"[DFT 验证] 手写DFT 与 np.fft 最大误差 = {err:.2e}")

    # 混叠演示
    alias = demonstrate_aliasing(freq=30, fs=fs, fs_sample=40, duration=dur)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, x_sum)
    axes[0, 0].set_title(f"时域信号（{f0}, {3*f0}, {7*f0} Hz 叠加）")
    axes[0, 0].set_xlabel("时间 (s)")

    axes[0, 1].plot(freqs, mag)
    axes[0, 1].set_title("单边幅度谱")
    axes[0, 1].set_xlabel("频率 (Hz)")
    axes[0, 1].set_xlim(0, 200)

    axes[1, 0].plot(alias["t"], alias["x"], label="原始 30 Hz")
    axes[1, 0].plot(alias["t_sampled"], alias["x_sampled"], "ro-",
                    markersize=4, label=f"采样 @{alias['fs_sample']} Hz")
    axes[1, 0].set_title(f"混叠演示（Nyquist={alias['nyquist']} Hz < 30 Hz）")
    axes[1, 0].legend()
    print(f"[混叠] 表观频率 = {alias['alias_freq']:.1f} Hz，发生混叠 = {alias['aliased']}")

    axes[1, 1].plot(alias["t"], np.sin(2 * np.pi * alias["alias_freq"] * alias["t"]),
                    "--", label=f"混叠后表观 {alias['alias_freq']:.1f} Hz")
    axes[1, 1].plot(alias["t"], alias["x"], alpha=0.4, label="原始 30 Hz")
    axes[1, 1].set_title("频域解释视角")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)

    fig.suptitle("M1 · 频谱分析与采样定理", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "m1_fft_spectrum.png")


def plot_fourier_series(cfg) -> None:
    """练习3：傅里叶级数重构方波."""
    fs, dur, f0 = cfg.m1_signal.fs, cfg.m1_signal.duration, cfg.m1_signal.f0
    n_harm = cfg.m1_signal.n_harmonics

    t, square = square_wave(f0, fs, dur, amplitude=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, n in zip(axes.ravel(), [1, 3, 7, n_harm]):
        recon = fourier_series_square(f0, fs, dur, n_harmonics=n, amplitude=1.0)
        ax.plot(t, square, alpha=0.4, label="理想方波")
        ax.plot(t, recon, color="red", label=f"{n} 项谐波(奇次)")
        ax.set_title(f"前 {n} 项奇次谐波重构")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("M1 · 傅里叶级数逼近方波（吉布斯现象）", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "m1_fourier_series.png")


def plot_convolution(cfg) -> None:
    """练习7：卷积与系统响应."""
    fs = cfg.m1_signal.fs

    x = np.zeros(40)
    x[5:15] = 1.0                       # 矩形脉冲输入
    h = np.exp(-np.arange(15) / 3.0)    # 指数衰减冲激响应
    h /= np.sum(h)

    y_manual, y_numpy = conv_compare(x, h)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].stem(x, basefmt=" ")
    axes[0].set_title("输入信号 x[n]（矩形脉冲）")

    axes[1].stem(h, basefmt=" ", linefmt="orange", markerfmt="or")
    axes[1].set_title("系统冲激响应 h[n]（指数衰减）")

    axes[2].plot(y_manual, "o-", label="手写卷积")
    axes[2].plot(y_numpy, "x--", label="np.convolve")
    axes[2].set_title("输出 y[n] = x * h（卷积结果）")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle("M1 · 卷积：系统响应的本质", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "m1_convolution.png")


def plot_filtering(cfg) -> None:
    """练习8：FIR 低通滤波."""
    fs, dur = cfg.m1_signal.fs, cfg.m1_signal.duration

    # 信号 = 低频(10Hz) + 高频(150Hz) + 噪声
    t, low = sine_wave(10, fs, dur, amplitude=1.0)
    _, high = sine_wave(150, fs, dur, amplitude=0.5)
    noise = 0.3 * np.random.randn(len(t))
    x = low + high + noise

    h = design_fir_lowpass(cutoff=30, fs=fs, numtaps=101)
    y = apply_filter(x, h)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(t, x, alpha=0.6)
    axes[0].set_title("原始信号 = 10Hz + 150Hz + 噪声")
    axes[1].plot(t, y, color="red")
    axes[1].set_title("低通滤波后（截止 30 Hz，高频与噪声被抑制）")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("时间 (s)")

    fig.suptitle("M1 · FIR 低通滤波", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "m1_filtering.png")


def main() -> None:
    set_seed(42)
    setup_plot_style()
    cfg = load_config("configs/default.yaml")

    print("=" * 60)
    print("SkySense · M1 信号基础模块")
    print("=" * 60)

    plot_basic_signals(cfg)
    plot_fft_spectrum(cfg)
    plot_fourier_series(cfg)
    plot_convolution(cfg)
    plot_filtering(cfg)

    print("=" * 60)
    print("M1 完成！请查看 outputs/figures/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
