"""Model factory and registry."""

from __future__ import annotations

import torch.nn as nn

from ..config import ExperimentConfig
from .BrainprintNet import BrainprintNet, CBFNet, MSNet
from .CNN import CNNLSTM, DeepConvNet, EEGNet, GWNet, ShallowConvNet
from .filter_bank import FBCNet, FBMSNet, IFNet
from .Conformer import Conformer


def create_model(config: ExperimentConfig) -> nn.Module:
    """Instantiate one supported model from configuration."""

    sample_points = config.sample_points
    if config.model == "EEGNet":
        return EEGNet(classes_num=config.n_classes, chans=config.channels, samples=sample_points)
    if config.model == "DeepConvNet":
        return DeepConvNet(chans=config.channels, samples=sample_points, classes_num=config.n_classes)
    if config.model == "ShallowConvNet":
        return ShallowConvNet(classes_num=config.n_classes, chans=config.channels, samples=sample_points)
    if config.model == "Conformer":
        return Conformer(chans=config.channels, samples=sample_points, n_classes=config.n_classes)
    if config.model == "FBCNet":
        return FBCNet(n_channels=config.channels, sampling_rate=config.sampling_rate, n_classes=config.n_classes, n_bands=12)
    if config.model == "FBMSNet":
        return FBMSNet(sampling_rate=config.sampling_rate, n_channels=config.channels, n_time=sample_points, n_classes=config.n_classes)
    if config.model == "IFNet":
        return IFNet(
            in_planes=config.channels,
            out_planes=64,
            kernel_size=63,
            radix=2,
            patch_size=int(config.sampling_rate // 2),
            time_points=sample_points,
            num_classes=config.n_classes,
            sampling_rate=config.sampling_rate,
        )
    if config.model == "GWNet":
        return GWNet(
            kernels=[11, 21, 31, 41, 51],
            samples=sample_points,
            res_blocks=config.res_blocks,
            in_channels=config.channels,
            num_classes=config.n_classes,
        )
    if config.model in {"1D_LSTM", "CNN_LSTM"}:
        return CNNLSTM(
            channels=config.channels,
            time_points=sample_points,
            hidden_size=128,
            n_classes=config.n_classes,
            num_layers=2,
        )
    if config.model == "BrainprintNet":
        return BrainprintNet(
            kernels=[7, 15, 31, 63, 127],
            sampling_rate=config.sampling_rate,
            num_classes=config.n_classes,
            in_channels=config.channels,
            n_bands=12,
            stride_factor=config.stride_factor,
        )
    if config.model == "MSNet":
        return MSNet(
            kernels=[11, 21, 31, 41, 51],
            num_classes=config.n_classes,
            in_channels=config.channels,
            hidden_chans=64,
            stride_factor=config.stride_factor,
        )
    if config.model == "CBFNet":
        return CBFNet(
            kernels=[31],
            sampling_rate=config.sampling_rate,
            num_classes=config.n_classes,
            in_channels=config.channels,
            n_bands=12,
            stride_factor=config.stride_factor,
        )
    raise ValueError(f"Unsupported model: {config.model}")
