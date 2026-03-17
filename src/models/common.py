"""Common model helpers shared by multiple BrainprintNet architectures."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import scipy.signal
import torch
import torch.nn as nn


def calculate_output_size(model: nn.Module, channels: int, samples: int) -> int:
    """Return the flattened feature dimension produced by a module."""

    device = next(model.parameters()).device
    input_tensor = torch.rand(1, 1, channels, samples, device=device)
    output = model(input_tensor)
    if isinstance(output, tuple):
        output = output[0]
    output = output.reshape(output.size(0), -1)
    return int(output.shape[-1])


def subband_split(
    data: np.ndarray,
    freq_start: int,
    freq_end: int,
    bandwidth: int,
    sampling_rate: int,
) -> np.ndarray:
    """Split EEG trials into fixed-width filter-bank sub-bands."""

    @lru_cache(maxsize=32)
    def get_sos_coeffs(low_hz: int, high_hz: int, fs: int) -> np.ndarray:
        return scipy.signal.butter(6, [2.0 * low_hz / fs, 2.0 * high_hz / fs], "bandpass", output="sos")

    def process_single_band(args: tuple[np.ndarray, int, int]) -> np.ndarray:
        array, low_hz, high_hz = args
        sos = get_sos_coeffs(low_hz, high_hz, sampling_rate)
        return scipy.signal.sosfilt(sos, array, axis=-1)

    bands = np.arange(freq_start, freq_end + 1, bandwidth)
    band_args = [(data, low_hz, high_hz) for low_hz, high_hz in zip(bands[:-1], bands[1:])]
    with ThreadPoolExecutor() as executor:
        subbands = list(executor.map(process_single_band, band_args))
    return np.stack(subbands, axis=1).astype(np.float32)


def subband_split_edges(
    data: np.ndarray,
    edges: list[tuple[int, int]],
    sampling_rate: int,
) -> np.ndarray:
    """Split EEG trials according to explicit band edges."""

    parts = [
        subband_split(data=data, freq_start=low_hz, freq_end=high_hz, bandwidth=high_hz - low_hz, sampling_rate=sampling_rate)[
            :, 0
        ]
        for low_hz, high_hz in edges
    ]
    return np.stack(parts, axis=1).astype(np.float32)


class Conv1dWithConstraint(nn.Conv1d):
    """One-dimensional convolution with optional max-norm weight renormalisation."""

    def __init__(self, *args: object, do_weight_norm: bool = True, max_norm: float = 1.0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.do_weight_norm = do_weight_norm
        self.max_norm = max_norm

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the constrained convolution."""

        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(input_tensor)


class Conv2dWithConstraint(nn.Conv2d):
    """Two-dimensional convolution with optional max-norm weight renormalisation."""

    def __init__(self, *args: object, do_weight_norm: bool = True, max_norm: float = 1.0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.do_weight_norm = do_weight_norm
        self.max_norm = max_norm

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the constrained convolution."""

        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(input_tensor)


class LinearWithConstraint(nn.Linear):
    """Linear layer with optional max-norm weight renormalisation."""

    def __init__(self, *args: object, do_weight_norm: bool = True, max_norm: float = 1.0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.do_weight_norm = do_weight_norm
        self.max_norm = max_norm

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the constrained linear projection."""

        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(input_tensor)


class VarLayer(nn.Module):
    """Reduce a tensor by variance along one dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the variance along the configured dimension."""

        return input_tensor.var(dim=self.dim, keepdim=True)


class StdLayer(nn.Module):
    """Reduce a tensor by standard deviation along one dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the standard deviation along the configured dimension."""

        return input_tensor.std(dim=self.dim, keepdim=True)


class LogVarLayer(nn.Module):
    """Reduce a tensor by log-variance along one dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the clamped log-variance along the configured dimension."""

        return torch.log(torch.clamp(input_tensor.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


class MeanLayer(nn.Module):
    """Reduce a tensor by mean along one dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the mean along the configured dimension."""

        return input_tensor.mean(dim=self.dim, keepdim=True)


class MaxLayer(nn.Module):
    """Reduce a tensor by max along one dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the maximum along the configured dimension."""

        values, _ = input_tensor.max(dim=self.dim, keepdim=True)
        return values


TEMPORAL_LAYERS: dict[str, type[nn.Module]] = {
    "VarLayer": VarLayer,
    "StdLayer": StdLayer,
    "LogVarLayer": LogVarLayer,
    "MeanLayer": MeanLayer,
    "MaxLayer": MaxLayer,
}


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply element-wise swish activation."""

        return input_tensor * torch.sigmoid(input_tensor)


def squeeze_to_channels_first(input_tensor: torch.Tensor) -> torch.Tensor:
    """Convert model inputs to ``(batch, channels, time)`` when needed."""

    if input_tensor.dim() == 4:
        return input_tensor.squeeze(1)
    return input_tensor


def get_feature_shape(model: nn.Module, input_tensor: torch.Tensor) -> torch.Size:
    """Return the feature shape emitted by a model."""

    with torch.no_grad():
        features, _ = model(input_tensor)
    return features.shape
