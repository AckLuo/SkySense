"""SkySense M2 一键运行脚本：完整链路仿真.

链路：随机比特 → 调制 → AWGN → 解调 → 计算 BER

运行::

    python scripts/run_m2_channel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from common import load_config, setup_plot_style, save_figure, set_seed
from m2_channel import modulate, demodulate, add_awgn_snr, ber, get_constellation
from m2_channel.multipath import MultipathChannel


def plot_constellations(cfg) -> None:
    """绘制 BPSK / QPSK / 16QAM 星座图."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, scheme in zip(axes, ["bpsk", "qpsk", "16qam"]):
        c = get_constellation(scheme)
        ax.scatter(c.real, c.imag, s=80)
        ax.set_title(f"{scheme.upper()} 星座图")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.grid(True, alpha=0.3); ax.axis("equal")
    fig.suptitle("M2 · 数字调制星座图", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "m2_constellations.png")


def plot_ber_curve(cfg) -> None:
    """绘制 BER vs SNR 曲线（对比理论值）."""
    scheme = cfg.m2_channel.modulation
    n_bits = 200_000
    snr_range = np.arange(0, 15, 1)

    ber_sim = []
    for snr in snr_range:
        bits = np.random.randint(0, 2, n_bits)
        tx = modulate(bits, scheme)
        rx = add_awgn_snr(tx, snr)
        bits_hat = demodulate(rx, scheme)
        ber_sim.append(ber(bits, bits_hat))

    # 理论 BER（仅 BPSK/QPSK 有闭式解）
    from scipy.special import erfc
    snr_lin = 10 ** (snr_range / 10)
    if scheme == "bpsk":
        ber_theory = 0.5 * erfc(np.sqrt(snr_lin))
    elif scheme == "qpsk":
        ber_theory = 0.5 * erfc(np.sqrt(snr_lin / 2))
    else:
        ber_theory = None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(snr_range, ber_sim, "o-", label=f"{scheme.upper()} 仿真")
    if ber_theory is not None:
        ax.semilogy(snr_range, ber_theory, "--", label="理论值")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_title(f"M2 · {scheme.upper()} 误比特率曲线")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    save_figure(fig, "m2_ber_curve.png")


def plot_multipath(cfg) -> None:
    """多径信道对信号的影响."""
    fs = cfg.m2_channel.fs
    ch = MultipathChannel(n_paths=cfg.m2_channel.n_paths,
                          max_delay=cfg.m2_channel.max_delay,
                          fs=fs, doppler=cfg.m2_channel.doppler, seed=0)

    # 发送单脉冲，观察信道冲激响应
    x = np.zeros(200, dtype=complex); x[0] = 1.0
    y = ch.apply(x)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].stem(np.abs(y), basefmt=" ")
    axes[0].set_title("信道冲激响应 |h[n]|（多径导致多个抽头）")
    axes[1].stem(np.angle(y + 1e-12), basefmt=" ", linefmt="orange", markerfmt="or")
    axes[1].set_title("信道相位响应（多普勒引入相位旋转）")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("M2 · 多径 + 多普勒信道", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "m2_multipath.png")
    print("[多径信道]", ch.info())


def main() -> None:
    set_seed(42)
    setup_plot_style()
    cfg = load_config("configs/default.yaml")

    print("=" * 60)
    print("SkySense · M2 信道建模模块")
    print("=" * 60)

    plot_constellations(cfg)
    plot_ber_curve(cfg)
    plot_multipath(cfg)

    print("=" * 60)
    print("M2 完成！请查看 outputs/figures/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
