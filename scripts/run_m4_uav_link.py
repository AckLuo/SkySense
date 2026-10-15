"""SkySense M4 一键运行脚本：无人机链路仿真.

运行::

    python scripts/run_m4_uav_link.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from common import load_config, setup_plot_style, save_figure, set_seed
from m4_uav import generate_trajectory, UAVChannel, UAVLinkSimulator


def plot_trajectory(cfg) -> None:
    traj = generate_trajectory(cfg.m4_uav.trajectory, cfg.m4_uav.n_steps,
                               cfg.m4_uav.velocity)
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(traj[:, 0], traj[:, 1], "-o", markersize=2)
    ax1.scatter(traj[0, 0], traj[0, 1], c="green", s=80, label="起点")
    ax1.scatter(traj[-1, 0], traj[-1, 1], c="red", s=80, label="终点")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.set_title(f"无人机轨迹（{cfg.m4_uav.trajectory}）")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.axis("equal")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2])
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.set_title("3D 轨迹")

    fig.suptitle("M4 · 无人机轨迹", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "m4_trajectory.png")


def plot_doppler(cfg) -> None:
    ch = UAVChannel(cfg.m4_uav.carrier_freq)
    velocities = np.arange(0, 101, 5)
    shifts = [ch.doppler_shift(v, 0) for v in velocities]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(velocities, shifts, "o-")
    ax.set_xlabel("相对速度 (m/s)"); ax.set_ylabel("多普勒频移 (Hz)")
    ax.set_title(f"M4 · 多普勒频移 vs 速度（载频 {cfg.m4_uav.carrier_freq/1e9:.1f} GHz）")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "m4_doppler.png")


def plot_link_performance(cfg) -> None:
    sim = UAVLinkSimulator(cfg)
    snr_range = np.arange(0, 16, 2)
    bers = [sim.run(snr_db=s, scheme="qpsk", n_bits=20000)["ber"] for s in snr_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snr_range, bers, "s-", color="red", label="无人机链路 BER")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_title("M4 · 无人机通信链路性能")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    save_figure(fig, "m4_link_performance.png")

    for s, b in zip(snr_range, bers):
        print(f"  SNR={s:2d} dB -> BER={b:.2e}")


def main() -> None:
    set_seed(42)
    setup_plot_style()
    cfg = load_config("configs/default.yaml")

    print("=" * 60)
    print("SkySense · M4 无人机链路仿真模块")
    print("=" * 60)

    plot_trajectory(cfg)
    plot_doppler(cfg)
    plot_link_performance(cfg)

    print("=" * 60)
    print("M4 完成！请查看 outputs/figures/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
