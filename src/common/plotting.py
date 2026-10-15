"""绘图工具：统一风格与图片保存."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能出图
import matplotlib.pyplot as plt


# 中文字体兼容（按可用性回退）
plt.rcParams["axes.unicode_minus"] = False
for _font in ["SimHei", "Microsoft YaHei", "DejaVu Sans"]:
    try:
        plt.rcParams["font.sans-serif"] = [_font]
        break
    except Exception:
        continue


def setup_plot_style(style: str = "seaborn-v0_8-darkgrid") -> None:
    """设置统一绘图风格."""
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("ggplot")


def save_figure(fig, filename: str, out_dir: str = "outputs/figures", dpi: int = 150) -> Path:
    """保存图像到指定目录.

    Args:
        fig: matplotlib Figure 对象.
        filename: 文件名（含扩展名，如 ``signals.png``）.
        out_dir: 输出目录.
        dpi: 分辨率.

    Returns:
        保存后的完整路径.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    full = out_path / filename
    fig.savefig(full, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] 图像 -> {full}")
    return full
