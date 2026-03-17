"""EEG preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import torch
from numpy.linalg import eig
from numpy.typing import NDArray
from scipy import signal
from scipy.linalg import fractional_matrix_power

from .alignment import CentroidAlign

ArrayFloat = NDArray[np.floating]


def bandpass_filter(
    data: ArrayFloat,
    sampling_rate: int,
    low_cut_hz: float = 8.0,
    high_cut_hz: float = 32.0,
) -> ArrayFloat:
    """Apply a Butterworth band-pass filter to EEG trials."""

    wn1 = 2 * low_cut_hz / sampling_rate
    wn2 = 2 * high_cut_hz / sampling_rate
    detrended = signal.detrend(np.asarray(data), axis=-1, type="linear")
    b, a = signal.butter(6, [wn1, wn2], "bandpass")
    return signal.filtfilt(b, a, detrended, axis=-1)


@dataclass
class EEGPreprocessor:
    """Reusable EEG preprocessing pipeline."""

    sampling_rate: int

    def resample(self, data: ArrayFloat, factor: int) -> ArrayFloat:
        """Downsample EEG trials with MNE."""

        return mne.filter.resample(data, down=factor, axis=-1)

    def detrend(self, data: ArrayFloat) -> ArrayFloat:
        """Remove linear trends along the time axis."""

        return signal.detrend(data, axis=-1, type="linear")

    def common_reference(self, data: ArrayFloat) -> ArrayFloat:
        """Apply common average referencing across channels."""

        return data - data.mean(axis=-2, keepdims=True)

    def bandpass_filter(
        self,
        data: ArrayFloat,
        low_cut_hz: float = 8.0,
        high_cut_hz: float = 32.0,
    ) -> ArrayFloat:
        """Apply a Butterworth band-pass filter."""

        b, a = signal.butter(
            6,
            [2.0 * low_cut_hz / self.sampling_rate, 2.0 * high_cut_hz / self.sampling_rate],
            "bandpass",
        )
        return signal.filtfilt(b, a, data, axis=-1)

    def notch_filter(self, data: ArrayFloat, notch_hz: float = 50.0) -> ArrayFloat:
        """Apply a notch filter to remove power-line noise."""

        w0 = 2.0 * notch_hz / self.sampling_rate
        b, a = signal.iirnotch(w0, self.sampling_rate)
        return signal.filtfilt(b, a, data, axis=-1)

    def normalize(self, data: ArrayFloat) -> ArrayFloat:
        """Normalize trials to zero mean and unit variance."""

        return (data - data.mean(axis=(1, 2), keepdims=True)) / data.std(axis=(1, 2), keepdims=True)

    def slide_window_augment(
        self,
        data: ArrayFloat,
        labels: NDArray[np.integer],
        window_length: int,
        stride: int,
    ) -> tuple[ArrayFloat, NDArray[np.integer]]:
        """Create augmented windows from longer EEG recordings."""

        segments: list[ArrayFloat] = []
        repeats = 0
        index = window_length
        while index <= data.shape[-1]:
            segments.append(data[:, :, index - window_length : index])
            index += stride
            repeats += 1
        augmented = np.concatenate(segments, axis=0)
        augmented_labels = np.tile(labels, repeats)
        return augmented, augmented_labels

    def euclidean_alignment(
        self,
        data: ArrayFloat,
        return_reference: bool = False,
    ) -> ArrayFloat | tuple[ArrayFloat, ArrayFloat]:
        """Apply Euclidean alignment to EEG trials."""

        covariance = np.stack([np.cov(trial) for trial in data], axis=0)
        reference = np.mean(covariance, axis=0)
        whitening = fractional_matrix_power(reference, -0.5) + 1e-8 * np.eye(data.shape[1])
        aligned = np.stack([whitening @ trial for trial in data], axis=0)
        if return_reference:
            return aligned, whitening
        return aligned

    def torch_euclidean_alignment(
        self,
        data: ArrayFloat,
        device: str = "cuda:0",
        return_reference: bool = False,
    ) -> ArrayFloat | tuple[ArrayFloat, ArrayFloat]:
        """Apply Euclidean alignment using torch tensor operations."""

        tensor = torch.from_numpy(data.copy()).to(torch.float32).to(device)
        batch_size, channels = tensor.shape[0], tensor.shape[1]
        covariance = torch.zeros((batch_size, channels, channels), device=tensor.device)
        for index in range(batch_size):
            covariance[index] = torch.cov(tensor[index])

        reference = torch.mean(covariance, dim=0).cpu().numpy()
        whitening = fractional_matrix_power(reference, -0.5) + 1e-8 * np.eye(channels)
        whitening_tensor = torch.from_numpy(whitening).to(tensor.device).to(torch.float32)
        aligned = torch.zeros_like(tensor)
        for index in range(batch_size):
            aligned[index] = whitening_tensor @ tensor[index]
        aligned_array = aligned.cpu().numpy()
        reference_array = whitening_tensor.cpu().numpy()
        if return_reference:
            return aligned_array, reference_array
        return aligned_array

    def centroid_alignment(
        self,
        data: ArrayFloat,
        return_covariance: bool = False,
    ) -> ArrayFloat | tuple[ArrayFloat, ArrayFloat]:
        """Apply centroid alignment to EEG trials."""

        aligner = CentroidAlign(center_type="euclid", covariance_type="lwf")
        covariance, aligned = aligner.fit_transform(data)
        if return_covariance:
            return aligned, covariance
        return aligned

    def pipeline(self, data: ArrayFloat) -> ArrayFloat:
        """Run the default preprocessing pipeline used by the project."""

        return self.bandpass_filter(data, 8.0, 32.0)


def compute_csp(data_train: ArrayFloat, label_train: NDArray[np.integer]) -> ArrayFloat:
    """Compute Common Spatial Pattern filters for three-class EEG data."""

    channel_num = data_train.shape[2]
    indices_0 = np.squeeze(np.where(label_train == 0))
    indices_1 = np.squeeze(np.where(label_train == 1))
    indices_2 = np.squeeze(np.where(label_train == 2))

    filters: list[ArrayFloat] = []
    for class_index in range(3):
        if class_index == 0:
            positive_indices = indices_0
            negative_indices = np.concatenate((indices_1, indices_2))
        elif class_index == 1:
            positive_indices = indices_1
            negative_indices = np.concatenate((indices_0, indices_2))
        else:
            positive_indices = indices_2
            negative_indices = np.concatenate((indices_0, indices_1))

        negative_indices = np.sort(negative_indices)
        covariance_positive = np.zeros((channel_num, channel_num, len(positive_indices)))
        covariance_negative = np.zeros((channel_num, channel_num, len(negative_indices)))

        for index, positive_index in enumerate(positive_indices):
            eeg = data_train[positive_index]
            covariance_positive[:, :, index] = (eeg.T @ eeg) / np.trace(eeg.T @ eeg)
        for index, negative_index in enumerate(negative_indices):
            eeg = data_train[negative_index]
            covariance_negative[:, :, index] = (eeg.T @ eeg) / np.trace(eeg.T @ eeg)

        covariance_positive_mean = np.mean(covariance_positive, axis=2)
        covariance_negative_mean = np.mean(covariance_negative, axis=2)
        covariance_total = covariance_positive_mean + covariance_negative_mean

        eigen_values, eigen_vectors = eig(covariance_total)
        order = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[order]
        eigen_vectors = eigen_vectors[:, order]

        whitening = np.sqrt(np.diag(np.power(eigen_values, -1))) @ eigen_vectors.T
        transformed_negative = whitening @ covariance_negative_mean @ whitening.T
        _, basis = eig(transformed_negative)
        basis = basis[:, np.argsort(eig(transformed_negative)[0])]
        filters.append(whitening.T @ basis)

    return np.concatenate((filters[0][:, 0:4], filters[1][:, 0:4], filters[2][:, 0:4]), axis=1)
