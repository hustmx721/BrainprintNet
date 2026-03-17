"""Dataset containers and PyTorch dataloader helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

ArrayFloat = NDArray[np.floating]
ArrayInt = NDArray[np.integer]


class EEGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple tensor-ready EEG dataset."""

    def __init__(self, data: ArrayFloat, labels: ArrayInt) -> None:
        """Store EEG trials and labels."""

        self._data = data.astype(np.float32, copy=False)
        self._labels = labels.astype(np.int64, copy=False)

    def __len__(self) -> int:
        """Return the number of trials in the dataset."""

        return int(self._data.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one EEG trial and its class label."""

        data = torch.from_numpy(self._data[index])
        label = torch.tensor(self._labels[index], dtype=torch.long)
        return data, label


@dataclass
class LoaderBundle:
    """Grouped train/validation/test dataloaders."""

    train: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    validation: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test: DataLoader[tuple[torch.Tensor, torch.Tensor]]


def create_dataloader(
    data: ArrayFloat,
    labels: ArrayInt,
    split: Literal["train", "eval", "test"],
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a dataloader with a consistent split policy."""

    dataset = EEGDataset(data, labels)
    shuffle = split == "train"
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
