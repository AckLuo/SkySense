# SkySense 🚁📡

> **无人机信道智能感知与通信链路仿真平台**
> UAV Channel Intelligent Sensing & Communication Link Simulation Platform

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 项目简介

SkySense 是一个面向 **无人机（UAV）通信链路** 的仿真与智能感知平台，融合
**数字信号处理（DSP）** 与 **深度学习（DL）**，用于：

- 无人机通信链路的**信道建模与仿真**（AWGN / 多径 / 瑞利衰落 / 多普勒）
- 基于深度学习的**信道估计**与**调制识别**
- 无人机移动场景下的**链路性能评估**（BER / 吞吐量 / 频谱效率）

本项目同时是一个**系统化学习载体**——通过 4 个渐进式模块（M1→M4），
从信号基础一路走到无人机链路级仿真。

---

## 目录结构

```
SkySense/
├── configs/                    # 配置文件（YAML）
│   └── default.yaml
├── src/
│   ├── common/                 # 公共工具（配置、绘图、日志、种子）
│   ├── m1_signal/              # M1：信号基础模块
│   ├── m2_channel/             # M2：信道建模模块
│   ├── m3_deeplearning/        # M3：深度学习模块（核心）
│   └── m4_uav/                 # M4：无人机链路仿真
├── scripts/                    # 可执行脚本（一键跑通各模块）
├── notebooks/                  # 探索性分析与可视化
├── tests/                      # 单元测试
├── data/                       # 数据（raw 原始 / processed 处理后）
├── outputs/                    # 输出（figures 图 / models 模型）
└── docs/                       # 文档与实验报告
```

---

## 快速开始

### 1. 环境安装

```bash
# 建议使用 conda 创建虚拟环境
conda create -n skysense python=3.10 -y
conda activate skysense

# 安装依赖
pip install -r requirements.txt

# 以可编辑模式安装本项目（使得 src 可被导入）
pip install -e .
```

### 2. 跑通 M1 信号基础模块

```bash
python scripts/run_m1_signal.py
```

生成的图像会保存到 `outputs/figures/`。

### 3. 逐个模块推进

```bash
python scripts/run_m2_channel.py       # M2：信道仿真
python scripts/run_m3_modrec.py        # M3：调制识别（需先准备数据）
python scripts/run_m4_uav_link.py      # M4：无人机链路仿真
```

---

## 四大模块

| 模块 | 名称 | 内容 | 核心产出 |
|:---:|---|---|---|
| **M1** | 信号基础 | 信号生成、卷积、傅里叶、采样、滤波 | 信号处理工具箱 |
| **M2** | 信道建模 | AWGN、多径、瑞利衰落、调制解调、BER | 链路级仿真器 |
| **M3** | 深度学习 | CNN 调制识别、DL 信道估计 | 智能感知模型 |
| **M4** | 无人机链路 | 轨迹、多普勒、时变信道、端到端评估 | 完整仿真平台 |

---

## 学习路径 & 周计划

详见 [`docs/ROADMAP.md`](docs/ROADMAP.md)，包含从 2026-10 到 2027-07 的
38 周详细任务清单。

---

## 技术栈

- **语言**：Python 3.10+
- **科学计算**：NumPy, SciPy, Matplotlib
- **深度学习**：PyTorch
- **配置**：PyYAML
- **测试**：pytest
- **数据集**：RadioML 2018, DeepMIMO, 自生成

---

## 参考资源

- [Think DSP](https://greenteapress.com/wp/think-dsp/) — 代码驱动的信号处理入门
- [RadioML 2018 数据集](https://www.deepsig.ai/datasets)
- [DeepMIMO 数据集](https://www.deepmimo.net/)
- [D2L 动手学深度学习](https://zh.d2l.ai/)

---

## License

MIT License — 详见 [LICENSE](LICENSE)
