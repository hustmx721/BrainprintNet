"""Transformer-based EEG backbones."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor

from .common import calculate_output_size


class PatchEmbedding(nn.Module):
    """Convolutional patch embedding used by the EEG Conformer."""

    def __init__(self, chans: int, emb_size: int = 40) -> None:
        """Initialise the shallow convolutional patch extractor."""

        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (chans, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Extract patch tokens from the input EEG trial."""

        output = self.shallownet(input_tensor)
        return self.projection(output)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        """Initialise projection matrices for self-attention."""

        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, input_tensor: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply multi-head self-attention."""

        queries = rearrange(self.queries(input_tensor), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(input_tensor), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(input_tensor), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(torch.float32).min)
        attention = F.softmax(energy / math.sqrt(self.emb_size), dim=-1)
        attention = self.dropout(attention)
        output = torch.einsum("bhal, bhlv -> bhav", attention, values)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.projection(output)


class ResidualAdd(nn.Module):
    """Wrap a module with a residual shortcut."""

    def __init__(self, function: nn.Module) -> None:
        """Store the wrapped function."""

        super().__init__()
        self.function = function

    def forward(self, input_tensor: Tensor, **kwargs: Tensor) -> Tensor:
        """Apply the wrapped function and add the residual input."""

        return input_tensor + self.function(input_tensor, **kwargs)


class FeedForwardBlock(nn.Sequential):
    """Position-wise feed-forward block."""

    def __init__(self, emb_size: int, expansion: int, dropout: float) -> None:
        """Create the two-layer feed-forward network."""

        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    """Transformer encoder block tailored for EEG sequences."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int = 10,
        dropout: float = 0.5,
        forward_expansion: int = 4,
        forward_dropout: float = 0.5,
    ) -> None:
        """Create a transformer encoder block."""

        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, dropout),
                    nn.Dropout(dropout),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, dropout=forward_dropout),
                    nn.Dropout(dropout),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    """Stacked transformer encoder blocks."""

    def __init__(self, depth: int, emb_size: int) -> None:
        """Create the requested number of encoder blocks."""

        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Module):
    """Classifier head for the EEG Conformer."""

    def __init__(self, chans: int, samples: int, depth: int, emb_size: int, n_classes: int) -> None:
        """Initialise the final feed-forward classifier."""

        super().__init__()
        feature_size = calculate_output_size(
            nn.Sequential(PatchEmbedding(chans, emb_size), TransformerEncoder(depth, emb_size)),
            chans,
            samples,
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )
        self.pool = Reduce("b n e -> b e", reduction="mean")

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor]:
        """Flatten transformer tokens and return logits."""

        features = input_tensor.contiguous().view(input_tensor.size(0), -1)
        return features, self.classifier(features)


class Conformer(nn.Sequential):
    """EEG Conformer architecture."""

    def __init__(self, chans: int, samples: int, emb_size: int = 40, depth: int = 6, n_classes: int = 4) -> None:
        """Create the EEG Conformer model."""

        super().__init__(
            PatchEmbedding(chans, emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(chans, samples, depth, emb_size, n_classes),
        )
