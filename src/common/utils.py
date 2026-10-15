"""通用工具：随机种子、设备选择、计时器."""

from __future__ import annotations

import random
import time
from contextlib import contextmanager

import numpy as np


def set_seed(seed: int = 42) -> None:
    """固定所有随机种子，保证实验可复现."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "cuda"):
    """返回可用设备（cuda 优先，不可用则回退 cpu）."""
    import torch

    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@contextmanager
def timer(name: str = "task"):
    """简易计时上下文管理器.

    用法::

        with timer("加载数据"):
            ...
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[计时] {name}: {elapsed:.3f} s")
