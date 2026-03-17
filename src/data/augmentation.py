"""EEG data augmentation strategies."""

from __future__ import annotations

import random
from itertools import combinations
from typing import Callable

import numpy as np
import pywt
from numpy.typing import NDArray

ArrayFloat = NDArray[np.floating]
ArrayInt = NDArray[np.integer]


def channel_mixure(
    data: ArrayFloat,
    labels: ArrayInt,
    channel_num: int = 5,
    multiplier: int = 4,
) -> tuple[ArrayFloat, ArrayInt]:
    """Create same-class samples by mixing channels from different trials."""

    augmented_data: list[ArrayFloat] = []
    augmented_labels: list[int] = []
    max_channels = min(channel_num, int(data.shape[1] // 4))
    for label in np.unique(labels):
        class_data = data[labels == label]
        num_trials, num_channels, _ = class_data.shape
        for _ in range(int(num_trials * multiplier)):
            selected_channels = np.random.choice(num_channels, size=max_channels, replace=False)
            selected_trials = np.random.choice(num_trials, size=max_channels, replace=False)
            trial_index = int(np.random.randint(num_trials, size=1)[0])
            new_trial = class_data[trial_index].copy()
            for channel_index, source_index in zip(selected_channels, selected_trials):
                new_trial[channel_index] = class_data[source_index, channel_index]
            augmented_data.append(new_trial)
            augmented_labels.append(int(label))
    return np.asarray(augmented_data), np.asarray(augmented_labels)


def channel_mixup(
    data: ArrayFloat,
    labels: ArrayInt,
    multiplier: int = 1,
) -> tuple[ArrayFloat, ArrayInt]:
    """Create same-class samples by convexly mixing two trials."""

    augmented_data: list[ArrayFloat] = []
    augmented_labels: list[int] = []
    for label in np.unique(labels):
        class_data = data[labels == label]
        num_trials = class_data.shape[0]
        for _ in range(int(num_trials * multiplier)):
            first_index, second_index = np.random.choice(num_trials, size=2, replace=False)
            first_sample = class_data[first_index]
            second_sample = class_data[second_index]
            mix_ratio = float(np.random.uniform(0.3, 0.7))
            augmented_data.append(mix_ratio * first_sample + (1 - mix_ratio) * second_sample)
            augmented_labels.append(int(label))
    return np.asarray(augmented_data), np.asarray(augmented_labels)


def trial_mixup(
    data: ArrayFloat,
    labels: ArrayInt,
    multiplier: int = 1,
) -> tuple[ArrayFloat, ArrayInt]:
    """Swap the first half of two same-class trials."""

    augmented_data: list[ArrayFloat] = []
    augmented_labels: list[int] = []
    for label in np.unique(labels):
        class_data = data[labels == label]
        num_trials, _, num_points = class_data.shape
        half = num_points // 2
        for _ in range(int(num_trials * multiplier)):
            first_index, second_index = np.random.choice(num_trials, size=2, replace=False)
            first_sample = class_data[first_index].copy()
            second_sample = class_data[second_index].copy()
            first_half = first_sample[:, :half].copy()
            first_sample[:, :half] = second_sample[:, :half]
            second_sample[:, :half] = first_half
            augmented_data.extend([first_sample, second_sample])
            augmented_labels.extend([int(label), int(label)])
    return np.asarray(augmented_data), np.asarray(augmented_labels)


def channel_reverse(
    data: ArrayFloat,
    labels: ArrayInt,
    multiplier: int = 1,
) -> tuple[ArrayFloat, ArrayInt]:
    """Reverse EEG signals along the time dimension."""

    reversed_trials = np.flip(data, axis=-1)
    selection = np.random.choice(data.shape[0], size=int(data.shape[0] * multiplier), replace=False)
    return reversed_trials[selection], labels[selection]


def channel_noise(
    data: ArrayFloat,
    labels: ArrayInt,
    multiplier: int = 1,
    noise_type: str = "pink",
) -> tuple[ArrayFloat, ArrayInt]:
    """Add synthetic noise to EEG trials."""

    def add_gaussian_noise(trials: ArrayFloat, mean: float = 0.0, std: float = 0.1) -> ArrayFloat:
        return trials + np.random.normal(mean, std, trials.shape)

    def add_salt_and_pepper_noise(
        trials: ArrayFloat,
        salt_prob: float = 0.01,
        pepper_prob: float = 0.01,
    ) -> ArrayFloat:
        noisy = trials.copy()
        num_salt = int(np.ceil(salt_prob * trials.size))
        num_pepper = int(np.ceil(pepper_prob * trials.size))
        salt_coordinates = [np.random.randint(0, value, num_salt) for value in trials.shape]
        pepper_coordinates = [np.random.randint(0, value, num_pepper) for value in trials.shape]
        noisy[tuple(salt_coordinates)] = np.max(trials)
        noisy[tuple(pepper_coordinates)] = np.min(trials)
        return noisy

    def add_poisson_noise(trials: ArrayFloat) -> ArrayFloat:
        return trials + np.random.poisson(size=trials.shape)

    def add_pink_noise(trials: ArrayFloat, alpha: float = 1.0) -> ArrayFloat:
        num_samples = trials.shape[-1]
        num_columns = int(np.ceil(np.log2(num_samples)))
        noise_shape = (trials.shape[0], trials.shape[1], 2**num_columns)
        noise = np.zeros(noise_shape)
        base = np.random.randn(*noise_shape)
        for column in range(1, num_columns):
            noise[:, :, :: 2**column] += base[:, :, :: 2**column]
        noise = noise[:, :, :num_samples]
        noise *= (np.arange(num_samples) + 1) ** (-alpha / 2.0)
        return trials + noise

    noise_builders: dict[str, Callable[..., ArrayFloat]] = {
        "gaussian": add_gaussian_noise,
        "salt_and_pepper": add_salt_and_pepper_noise,
        "poisson": add_poisson_noise,
        "pink": add_pink_noise,
    }
    if noise_type not in noise_builders:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    augmented = [noise_builders[noise_type](data) for _ in range(multiplier)]
    noisy_data = np.asarray(augmented).reshape(multiplier * data.shape[0], data.shape[1], data.shape[2])
    return noisy_data, labels.repeat(multiplier)


def augment_with_cr(data: ArrayFloat, labels: ArrayInt) -> tuple[ArrayFloat, ArrayInt]:
    """Mirror channels for 59-channel motor-imagery EEG data."""

    left_channels = np.array([2, 4, 6, 9, 11, 13, 15, 18, 20, 22, 24, 27, 29, 31, 33, 35, 37, 39, 41, 44, 46, 48, 51, 53, 55, 58]) - 1
    right_channels = np.array([3, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34, 36, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59]) - 1
    augmented_data = data.copy()
    augmented_labels = labels.copy()
    augmented_data[:, left_channels, :] = data[:, right_channels, :]
    augmented_data[:, right_channels, :] = data[:, left_channels, :]
    augmented_labels[labels == 0] = 1
    augmented_labels[labels == 1] = 0
    return augmented_data, augmented_labels


def dwta(source: ArrayFloat, target: ArrayFloat) -> tuple[ArrayFloat, ArrayFloat]:
    """Wavelet-based dual-domain transfer augmentation."""

    wavelet_name = "db5"
    target_approx, target_detail = pywt.dwt(target, wavelet_name)
    source_approx, source_detail = pywt.dwt(source, wavelet_name)
    source_augmented = pywt.idwt(source_approx, target_detail, wavelet_name, "smooth")
    target_augmented = pywt.idwt(target_approx, source_detail, wavelet_name, "smooth")
    return source_augmented, target_augmented


def use_dwta(
    data: ArrayFloat,
    labels: ArrayInt,
    multiplier: int,
) -> tuple[ArrayFloat, ArrayInt]:
    """Augment same-class samples with wavelet-domain transfer."""

    if multiplier < 1:
        raise ValueError("multiplier must be at least 1.")

    augmented_data: list[ArrayFloat] = list(data)
    augmented_labels: list[int] = list(labels)
    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        class_data = data[class_indices]
        target_count = len(class_data) * (multiplier - 1)
        if target_count == 0:
            continue
        all_pairs = list(combinations(range(len(class_data)), 2))
        if len(all_pairs) < target_count:
            repeats = target_count // len(all_pairs) + 1
            all_pairs = all_pairs * repeats
        selected_pairs = random.sample(all_pairs, target_count)
        for first_index, second_index in selected_pairs:
            source_augmented, target_augmented = dwta(class_data[first_index], class_data[second_index])
            augmented_data.append(source_augmented if random.choice([True, False]) else target_augmented)
            augmented_labels.append(int(label))
    return np.asarray(augmented_data), np.asarray(augmented_labels)


def apply_augmentation(
    data: ArrayFloat,
    labels: ArrayInt,
    strategy: str,
    multiplier: int = 1,
) -> tuple[ArrayFloat, ArrayInt]:
    """Dispatch a named augmentation strategy."""

    strategies: dict[str, Callable[..., tuple[ArrayFloat, ArrayInt]]] = {
        "channel_mixup": channel_mixup,
        "trial_mixup": trial_mixup,
        "channel_reverse": channel_reverse,
        "channel_noise": channel_noise,
        "channel_mixure": channel_mixure,
        "use_DWTA": use_dwta,
        "augment_with_CR": augment_with_cr,
    }
    if strategy not in strategies:
        raise ValueError(f"Unsupported augmentation strategy: {strategy}")
    if strategy == "channel_noise":
        return strategies[strategy](data, labels, multiplier=multiplier, noise_type="pink")
    if strategy == "augment_with_CR":
        return strategies[strategy](data, labels)
    return strategies[strategy](data, labels, multiplier=multiplier)
