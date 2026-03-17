# BrainprintNet

BrainprintNet 是一个经过工程化重构的 EEG 用户识别与跨任务评测项目。当前代码以标准 Python 包结构组织，训练入口、数据加载、模型定义、评估逻辑和工程文档已经拆分到清晰的模块中，便于维护、复现和继续扩展。

## Features

- 标准包结构，支持 `python -m brainprintnet` 与 `python run.py`
- 训练入口与业务逻辑分离，主程序只负责参数解析和调度
- 数据处理、增强、模型、训练、评估按职责分层
- 所有新增函数均带类型提示和 docstring
- 内置 logging、版本号、输出目录管理
- 保留旧实验代码，便于迁移和回溯

## Project Structure

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── run.py
├── brainprintnet
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── logging_config.py
│   ├── outputs
│   │   ├── checkpoints
│   │   ├── legacy_artifacts
│   │   ├── logs
│   │   └── reports
│   ├── utils.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── alignment.py
│   │   ├── augmentation.py
│   │   ├── datasets.py
│   │   ├── loaders.py
│   │   └── preprocessing.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── brainprint.py
│   │   ├── cnn.py
│   │   ├── common.py
│   │   ├── filter_bank.py
│   │   ├── registry.py
│   │   └── transformer.py
│   └── training
│       ├── __init__.py
│       ├── engine.py
│       ├── evaluation.py
│       └── losses.py
├── legacy
│   ├── README.md
│   └── source
│       ├── DeepTransferEEG
│       ├── HandiFea
│       ├── baseline
│       ├── dataset
│       ├── models
│       └── utils
└── ...
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

查看支持的数据集与模型：

```bash
python -m brainprintnet --list-datasets
python -m brainprintnet --list-models
```

运行单数据集实验：

```bash
python -m brainprintnet \
  --mode baseline \
  --dataset 001 \
  --model ResEEGNet \
  --data-root /path/to/data
```

运行跨任务实验：

```bash
python run.py \
  --mode cross-task \
  --cross-tasks MI ERP \
  --model BrainprintNet \
  --session-num 1 \
  --data-root /path/to/data
```

启用 center loss：

```bash
python -m brainprintnet \
  --mode baseline \
  --dataset BCI85 \
  --model ResEEGNet \
  --center-loss \
  --alpha 0.001
```

## Output Conventions

- 默认日志输出到 `brainprintnet/outputs/logs/`
- 默认模型权重输出到 `brainprintnet/outputs/checkpoints/`
- 默认实验 CSV 与 JSON 摘要输出到 `brainprintnet/outputs/reports/`
- 如需改位置，可显式传入 `--output-root /your/path`

## Directory Guide

- `brainprintnet/cli.py`: 命令行入口，负责参数解析和任务调度
- `brainprintnet/config.py`: 数据集元信息与运行时配置
- `brainprintnet/outputs/`: 默认输出目录，和源码主目录放在一起
- `brainprintnet/data/`: 数据读取、预处理、增强与 DataLoader
- `brainprintnet/models/`: 全部模型定义与模型工厂
- `brainprintnet/training/`: 训练循环、损失函数和评估逻辑
- `brainprintnet/logging_config.py`: 全局日志配置
- `legacy/source/`: 原始实验脚本和历史代码归档区

## Supported Datasets

`001`, `004`, `BCI85`, `Rest85`, `LJ30`, `Rest`, `MI`, `ERP`, `SSVEP`, `CatERP`, `M3CV_Rest`, `M3CV_Transient`, `M3CV_Steady`, `M3CV_P300`, `M3CV_Motor`, `M3CV_SSVEP_SA`, `SEED`

## Supported Models

`EEGNet`, `DeepConvNet`, `ShallowConvNet`, `Conformer`, `FBCNet`, `FBMSNet`, `IFNet`, `GWNet`, `ResEEGNet`, `1D_LSTM`, `CNN_LSTM`, `BrainprintNet`, `MSNet`, `CBFNet`

## Development Notes

- 代码风格按 PEP 8 与 Black 的 88 列约定编写
- 根目录旧的脚本式实现已从新包入口中解耦
- 如果需要继续迁移历史实验方法，建议新增模块后接入 `brainprintnet/models/registry.py`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
