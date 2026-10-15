"""M3：数据集（RadioML 加载器 + 合成数据生成）.

RadioML 2018 是调制识别的标准数据集，包含 24 种调制方式。
若本地没有数据集，可用 SyntheticModDataset 自举。

对应周计划：第 17-18 周
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticModDataset(Dataset):
    """合成调制识别数据集（无需外部数据，快速跑通流程）.

    生成 BPSK / QPSK / 8PSK / 16QAM 四种调制在不同 SNR 下的 I/Q 采样。
    """

    SCHEMES = ["bpsk", "qpsk", "8psk", "16qam"]

    def __init__(self, n_samples: int = 4000, seq_len: int = 128,
                 snr_range: tuple = (-10, 20), seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seq_len = seq_len
        self.data = []
        self.labels = []

        per_class = n_samples // len(self.SCHEMES)
        for label, scheme in enumerate(self.SCHEMES):
            for _ in range(per_class):
                snr = self.rng.uniform(*snr_range)
                iq = self._gen_one(scheme, seq_len, snr)
                self.data.append(iq)
                self.labels.append(label)

        self.data = np.stack(self.data).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def _gen_one(self, scheme: str, n: int, snr_db: float) -> np.ndarray:
        n_sym = n // 2
        bits = self.rng.integers(0, 2, n_sym * 4)
        symbols = self._map_bits(bits, scheme, n_sym)
        iq = np.concatenate([symbols.real, symbols.imag])[: n]
        iq = self._add_noise(iq, snr_db)
        return np.stack([iq.real, iq.imag], axis=0) if np.iscomplexobj(iq) else \
               np.stack([iq[:n // 2], np.zeros(n // 2)], axis=0)

    def _map_bits(self, bits: np.ndarray, scheme: str, n_sym: int) -> np.ndarray:
        if scheme == "bpsk":
            s = 1 - 2 * bits[:n_sym]
        elif scheme == "qpsk":
            b = bits[: 2 * n_sym].reshape(-1, 2)
            s = (1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])
        elif scheme == "8psk":
            b = bits[: 3 * n_sym].reshape(-1, 3)
            idx = b.dot(1 << np.arange(2, -1, -1))
            s = np.exp(1j * 2 * np.pi * idx / 8)
        else:  # 16qam
            b = bits[: 4 * n_sym].reshape(-1, 4)
            idx = b.dot(1 << np.arange(3, -1, -1))
            levels = np.array([-3, -1, 1, 3])
            s = levels[idx >> 2] + 1j * levels[idx & 3]
        return s / np.sqrt(np.mean(np.abs(s) ** 2))

    def _add_noise(self, x: np.ndarray, snr_db: float) -> np.ndarray:
        p = np.mean(np.abs(x) ** 2)
        npow = p / (10 ** (snr_db / 10))
        return x + np.sqrt(npow / 2) * self.rng.standard_normal(x.shape)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.data[idx]), int(self.labels[idx])


class RadioMLDataset(Dataset):
    """RadioML 2018 数据集加载器（.hdf5 或 .pkl）.

    数据集下载: https://www.deepsig.ai/datasets
    """

    def __init__(self, path: str, snr_filter: int | None = None):
        self.path = path
        self.data, self.labels = self._load(path, snr_filter)

    def _load(self, path: str, snr_filter):
        try:
            import h5py
            with h5py.File(path, "r") as f:
                X = f["X"][:]
                Y = f["Y"][:]
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(
                f"无法加载 RadioML 数据集: {path}\n"
                f"请先从 https://www.deepsig.ai/datasets 下载。错误: {exc}"
            ) from exc

        labels = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        if snr_filter is not None and "Z" in Y.dtype.names if hasattr(Y, "dtype") else False:
            pass
        data = np.transpose(X, (0, 2, 1)).astype(np.float32)  # (N, 2, L)
        return data, labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.data[idx]), int(self.labels[idx])
