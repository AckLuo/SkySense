"""配置加载工具：读取 YAML 配置并以属性方式访问."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """支持点号访问的配置字典.

    用法::

        cfg = load_config("configs/default.yaml")
        print(cfg.m1_signal.fs)   # -> 1000
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(f"配置中不存在字段: {name}") from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | os.PathLike = "configs/default.yaml") -> Config:
    """加载 YAML 配置文件.

    Args:
        path: 配置文件路径.

    Returns:
        Config 对象.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw or {})
