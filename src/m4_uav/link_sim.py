"""M4：无人机链路级仿真器（端到端）.

链路：比特 → 调制 → 无人机时变信道 → AWGN → 解调 → BER

对应周计划：第 28-30 周
"""

from __future__ import annotations

import numpy as np

from m2_channel.modulation import modulate, demodulate
from m2_channel.awgn import add_awgn_snr
from m2_channel.metrics import ber
from m4_uav.trajectory import generate_trajectory
from m4_uav.uav_channel import UAVChannel


class UAVLinkSimulator:
    """端到端无人机通信链路仿真器."""

    def __init__(self, cfg, channel_estimator=None):
        self.cfg = cfg
        self.channel = UAVChannel(
            carrier_freq=cfg.m4_uav.carrier_freq,
            fs=cfg.m2_channel.fs,
            n_paths=cfg.m2_channel.n_paths,
        )
        self.estimator = channel_estimator  # 可注入 M3 的 DL 信道估计器

    def run(self, snr_db: float = 10.0, scheme: str = "qpsk",
            n_bits: int = 4000) -> dict:
        """跑一次端到端仿真.

        Returns:
            dict: ber, snr, n_bits, trajectory 等.
        """
        traj = generate_trajectory(
            kind=self.cfg.m4_uav.trajectory,
            n_steps=self.cfg.m4_uav.n_steps,
            velocity=self.cfg.m4_uav.velocity,
        )

        bits = np.random.randint(0, 2, n_bits)
        tx = modulate(bits, scheme)

        # 无人机时变信道
        rx_ch = self.channel.apply(tx, traj, dt=0.1)
        # 加噪
        rx = add_awgn_snr(rx_ch, snr_db)

        bits_hat = demodulate(rx, scheme)
        ber_val = ber(bits, bits_hat)

        return {
            "ber": ber_val,
            "snr_db": snr_db,
            "n_bits": n_bits,
            "trajectory": traj,
            "scheme": scheme,
        }
