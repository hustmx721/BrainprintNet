"""Dataset loading entry points for BrainprintNet experiments."""

from __future__ import annotations

import gc
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mne
import numpy as np
import psutil
import scipy.io as scio
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from ..config import M3CV_DATASETS, OPENBMI_DATASETS, ExperimentConfig
from .augmentation import apply_augmentation
from .datasets import LoaderBundle, create_dataloader
from .preprocessing import EEGPreprocessor

ArrayFloat = NDArray[np.floating]
ArrayInt = NDArray[np.integer]


def load_experiment_data(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load dataloaders for one experiment run."""

    if config.mode == "baseline":
        if config.dataset is None:
            raise ValueError("`dataset` is required for baseline experiments.")
        return _load_baseline_dataset(config, seed)
    if config.mode == "within-session":
        if config.dataset is None:
            raise ValueError("`dataset` is required for within-session experiments.")
        return _load_within_session_m3cv(config, seed)
    if config.mode == "cross-task":
        if not config.cross_tasks or len(config.cross_tasks) != 2:
            raise ValueError("`cross_tasks` must contain exactly two dataset names.")
        return _load_cross_task_dataset(config, seed)
    raise ValueError(f"Unsupported run mode: {config.mode}")


def _make_loader_bundle(
    train_x: ArrayFloat,
    train_y: ArrayInt,
    test_x: ArrayFloat,
    test_y: ArrayInt,
    seed: int,
    batch_size: int,
) -> LoaderBundle:
    """Create train/validation/test loaders from prepared arrays."""

    train_split_x, validation_x, train_split_y, validation_y = train_test_split(
        train_x,
        train_y,
        test_size=0.2,
        random_state=seed,
        stratify=train_y,
    )
    train_split_x, validation_x, test_x = [np.expand_dims(array, axis=1) for array in [train_split_x, validation_x, test_x]]
    return LoaderBundle(
        train=create_dataloader(train_split_x, train_split_y, "train", batch_size),
        validation=create_dataloader(validation_x, validation_y, "eval", batch_size),
        test=create_dataloader(test_x, test_y, "test", batch_size),
    )


def _make_within_session_bundle(
    data: ArrayFloat,
    labels: ArrayInt,
    seed: int,
    batch_size: int,
) -> LoaderBundle:
    """Create train/validation/test loaders from one merged pool of trials."""

    train_x, test_x, train_y, test_y = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_x,
        train_y,
        test_size=0.2,
        random_state=seed,
        stratify=train_y,
    )
    train_x, validation_x, test_x = [np.expand_dims(array, axis=1) for array in [train_x, validation_x, test_x]]
    return LoaderBundle(
        train=create_dataloader(train_x, train_y, "train", batch_size),
        validation=create_dataloader(validation_x, validation_y, "eval", batch_size),
        test=create_dataloader(test_x, test_y, "test", batch_size),
    )


def _load_pickle(path: Path) -> dict[str, np.ndarray]:
    """Load a pickle file containing EEG arrays."""

    with path.open("rb") as file_pointer:
        return pickle.load(file_pointer)


def _parallel_process(data: ArrayFloat, processor: callable) -> ArrayFloat:
    """Process large EEG arrays in chunks to reduce peak memory."""

    processed = np.empty_like(data)
    num_cores = psutil.cpu_count(logical=False) or 1
    available_memory = psutil.virtual_memory().available
    max_chunk_size = int((available_memory // (4 * num_cores)) // 16) or data.shape[0]
    chunk_size = min(data.shape[0], max_chunk_size)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures: list[tuple[int, object]] = []
        for start in range(0, data.shape[0], chunk_size):
            chunk = data[start : start + chunk_size]
            futures.append((start, executor.submit(processor, chunk)))
        for start, future in futures:
            processed[start : start + chunk_size] = future.result()
    return processed


def _apply_optional_augmentation(
    train_x: ArrayFloat,
    train_y: ArrayInt,
    seed: int,
    aug_type: str | None,
) -> tuple[ArrayFloat, ArrayInt]:
    """Apply one optional augmentation strategy to half of the training set."""

    if aug_type is None:
        return train_x, train_y
    subset_x, _, subset_y, _ = train_test_split(
        train_x,
        train_y,
        test_size=0.5,
        random_state=seed,
        stratify=train_y,
    )
    augmented_x, augmented_y = apply_augmentation(subset_x, subset_y, aug_type, multiplier=1)
    return np.concatenate([train_x, augmented_x], axis=0), np.concatenate([train_y, augmented_y], axis=0)


def _load_baseline_dataset(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load one baseline dataset based on the experiment configuration."""

    dataset_name = config.dataset
    if dataset_name in {"001", "004"}:
        return _load_bnci_14xxx(config, seed)
    if dataset_name == "BCI85":
        return _load_bci85(config, seed)
    if dataset_name == "Rest85":
        return _load_rest85(config, seed)
    if dataset_name == "LJ30":
        return _load_lj30(config, seed)
    if dataset_name in OPENBMI_DATASETS:
        return _load_openbmi(config, seed)
    if dataset_name in M3CV_DATASETS:
        return _load_m3cv(config, seed)
    if dataset_name == "SEED":
        return _load_seed(config, seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_bnci_14xxx(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load BNCI 2014-001 or 2014-004 EEG trials."""

    dataset_path = config.paths.data_root / "mydata" / f"ori_{config.dataset}.mat"
    data = scio.loadmat(dataset_path)
    train_x, train_y = data["ori_train_x"], data["ori_train_s"].squeeze()
    test_x, test_y = data["ori_test_x"], data["ori_test_s"].squeeze()
    preprocessor = EEGPreprocessor(sampling_rate=250)
    train_x, test_x = [preprocessor.pipeline(array) for array in [train_x, test_x]]
    bundle = _make_loader_bundle(train_x, train_y, test_x, test_y, seed, config.batch_size)
    gc.collect()
    return bundle


def _load_bci85(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load the BCI85 benchmark dataset."""

    dataset_path = config.paths.data_root / "mydata" / "ori_BCI85.pkl"
    dataset = np.load(dataset_path, allow_pickle=True, mmap_mode="r")
    train_x, test_x = dataset["ori_train_x"], dataset["ori_test_x"]
    train_x, test_x = [array.reshape((-1, array.shape[-2], array.shape[-1])) for array in [train_x, test_x]]
    train_y, test_y = dataset["ori_train_s"].reshape(-1), dataset["ori_test_s"].reshape(-1)

    preprocessor = EEGPreprocessor(sampling_rate=512)
    processor = preprocessor.pipeline
    train_x, test_x = [_parallel_process(array, processor) for array in [train_x, test_x]]
    train_x, test_x = [mne.filter.resample(array, 250, 512, npad="auto") for array in [train_x, test_x]]
    return _make_loader_bundle(train_x, train_y, test_x, test_y, seed, config.batch_size)


def _load_rest85(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load the resting-state BCI85 dataset."""

    dataset_path = config.paths.data_root / "mydata" / "rest_BCI85.pkl"
    dataset = np.load(dataset_path, allow_pickle=True, mmap_mode="r")
    train_x, test_x = dataset["ori_train_x"] * 1e5, dataset["ori_test_x"] * 1e5
    train_x, test_x = [array.reshape((-1, array.shape[-2], array.shape[-1])) for array in [train_x, test_x]]
    train_y, test_y = dataset["ori_train_s"].reshape(-1), dataset["ori_test_s"].reshape(-1)
    preprocessor = EEGPreprocessor(sampling_rate=512)
    train_x, test_x = [preprocessor.pipeline(array) for array in [train_x, test_x]]
    train_x, test_x = [mne.filter.resample(array, 250, 512, npad="auto") for array in [train_x, test_x]]
    return _make_loader_bundle(train_x, train_y, test_x, test_y, seed, config.batch_size)


def _load_lj30(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load the LJ30 benchmark dataset."""

    dataset_path = config.paths.data_root / "mydata" / "ori_LingJiu30.mat"
    data = scio.loadmat(dataset_path)
    train_x, test_x = data["ori_train_x"], data["ori_test_x"]
    train_x, test_x = [array.reshape((-1, array.shape[-2], array.shape[-1])) for array in [train_x, test_x]]
    train_y, test_y = data["ori_train_s"].reshape(-1), data["ori_test_s"].reshape(-1)
    preprocessor = EEGPreprocessor(sampling_rate=300)
    train_x, test_x = [preprocessor.pipeline(array) for array in [train_x, test_x]]
    train_x, train_y = _apply_optional_augmentation(train_x, train_y, seed, config.aug_type)
    return _make_loader_bundle(train_x, train_y, test_x, test_y, seed, config.batch_size)


def _load_openbmi(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load one OpenBMI paradigm."""

    dataset_name = config.dataset
    if dataset_name is None:
        raise ValueError("OpenBMI loading requires a dataset name.")
    train_path = config.paths.data_root / "OpenBMI" / "processed" / dataset_name / "train.pkl"
    test_path = config.paths.data_root / "OpenBMI" / "processed" / dataset_name / "test.pkl"
    train_data = _load_pickle(train_path)
    test_data = _load_pickle(test_path)
    train_x = train_data["ori_train_x"].astype(np.float32).reshape((-1, train_data["ori_train_x"].shape[-2], train_data["ori_train_x"].shape[-1]))
    test_x = test_data["ori_test_x"].astype(np.float32).reshape((-1, test_data["ori_test_x"].shape[-2], test_data["ori_test_x"].shape[-1]))
    train_y = (train_data["ori_train_s"] - 1).astype(np.int16).reshape(-1)
    test_y = (test_data["ori_test_s"] - 1).astype(np.int16).reshape(-1)
    sampling_rate = 1000 if dataset_name == "ERP" else 250
    preprocessor = EEGPreprocessor(sampling_rate=sampling_rate)
    processor = preprocessor.pipeline
    train_x, test_x = [_parallel_process(array, processor) for array in [train_x, test_x]]
    train_x, train_y = _apply_optional_augmentation(train_x, train_y, seed, config.aug_type)
    return _make_loader_bundle(train_x, train_y, test_x, test_y, seed, config.batch_size)


def _load_m3cv(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load one M3CV paradigm."""

    if config.dataset is None:
        raise ValueError("M3CV loading requires a dataset name.")
    split_name = config.dataset[5:]
    train_path = config.paths.data_root / "M3CV" / "Train" / f"T_{split_name}.pkl"
    test_path = config.paths.data_root / "M3CV" / "Test" / f"{split_name}.pkl"
    train_data = _load_pickle(train_path)
    test_data = _load_pickle(test_path)
    train_x = train_data["data"][:, :-1, :].astype(np.float32)
    test_x = test_data["data"][:, :-1, :].astype(np.float32)
    train_y = train_data["label"].astype(np.int16)
    test_y = test_data["label"].astype(np.int16)
    preprocessor = EEGPreprocessor(sampling_rate=250)
    processor = preprocessor.pipeline
    train_x, test_x = [_parallel_process(array, processor) for array in [train_x, test_x]]
    train_x, train_y = _apply_optional_augmentation(train_x, train_y, seed, config.aug_type)
    bundle = _make_loader_bundle(train_x, train_y, test_x, test_y, seed, config.batch_size)
    gc.collect()
    return bundle


def _load_seed(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load the SEED dataset."""

    train_path = config.paths.data_root / "SEED" / "processed" / "train.pkl"
    test_path = config.paths.data_root / "SEED" / "processed" / "test.pkl"
    train_data = _load_pickle(train_path)
    test_data = _load_pickle(test_path)
    train_x = train_data["data"].astype(np.float32)
    test_x = test_data["data"].astype(np.float32)
    train_y = train_data["label"].astype(np.int16) - 1
    test_y = test_data["label"].astype(np.int16) - 1
    preprocessor = EEGPreprocessor(sampling_rate=200)
    processor = preprocessor.pipeline
    train_x, test_x = [_parallel_process(array, processor) for array in [train_x, test_x]]
    return _make_loader_bundle(train_x, train_y, test_x, test_y, seed, config.batch_size)


def _load_within_session_m3cv(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load and re-split M3CV sessions for within-session experiments."""

    if config.dataset not in M3CV_DATASETS:
        raise ValueError("Within-session experiments currently support only M3CV datasets.")
    split_name = config.dataset[5:]
    train_path = config.paths.data_root / "M3CV" / "Train" / f"T_{split_name}.pkl"
    test_path = config.paths.data_root / "M3CV" / "Test" / f"{split_name}.pkl"
    train_data = _load_pickle(train_path)
    test_data = _load_pickle(test_path)
    train_x = train_data["data"][:, :-1, :].astype(np.float32)
    test_x = test_data["data"][:, :-1, :].astype(np.float32)
    train_y = train_data["label"].astype(np.int16)
    test_y = test_data["label"].astype(np.int16)
    preprocessor = EEGPreprocessor(sampling_rate=250)
    processor = preprocessor.pipeline
    train_x, test_x = [_parallel_process(array, processor) for array in [train_x, test_x]]
    merged_x = np.concatenate([train_x, test_x], axis=0)
    merged_y = np.concatenate([train_y, test_y], axis=0)
    return _make_within_session_bundle(merged_x, merged_y, seed, config.batch_size)


def _load_cross_task_dataset(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load one cross-task experiment."""

    if config.cross_tasks is None:
        raise ValueError("Cross-task loading requires `cross_tasks`.")
    if set(config.cross_tasks).issubset(M3CV_DATASETS):
        return _load_m3cv_cross_task(config, seed)
    if set(config.cross_tasks).issubset(OPENBMI_DATASETS):
        return _load_openbmi_cross_task(config, seed)
    raise ValueError("Cross-task datasets must come from the same family.")


def _load_m3cv_cross_task(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load a two-task M3CV cross-task experiment."""

    if config.cross_tasks is None:
        raise ValueError("Cross-task loading requires `cross_tasks`.")
    first_task, second_task = [task[5:] for task in config.cross_tasks]
    first_train = _load_pickle(config.paths.data_root / "M3CV" / "Train" / f"T_{first_task}.pkl")
    first_test = _load_pickle(config.paths.data_root / "M3CV" / "Test" / f"{first_task}.pkl")
    second_train = _load_pickle(config.paths.data_root / "M3CV" / "Train" / f"T_{second_task}.pkl")
    second_test = _load_pickle(config.paths.data_root / "M3CV" / "Test" / f"{second_task}.pkl")

    first_x_train = first_train["data"][:, :-1, :].astype(np.float32)
    first_x_test = first_test["data"][:, :-1, :].astype(np.float32)
    second_x_train = second_train["data"][:, :-1, :].astype(np.float32)
    second_x_test = second_test["data"][:, :-1, :].astype(np.float32)
    first_y_train = first_train["label"].astype(np.int16)
    first_y_test = first_test["label"].astype(np.int16)
    second_y_train = second_train["label"].astype(np.int16)
    second_y_test = second_test["label"].astype(np.int16)

    if config.session_num is None:
        train_x = np.concatenate([first_x_train, first_x_test], axis=0)
        train_y = np.concatenate([first_y_train, first_y_test], axis=0)
        test_x = np.concatenate([second_x_train, second_x_test], axis=0)
        test_y = np.concatenate([second_y_train, second_y_test], axis=0)
    else:
        session_map = {
            1: (first_x_train, first_y_train, second_x_train, second_y_train),
            2: (first_x_test, first_y_test, second_x_test, second_y_test),
            12: (first_x_train, first_y_train, second_x_test, second_y_test),
            21: (first_x_test, first_y_test, second_x_train, second_y_train),
        }
        if config.session_num not in session_map:
            raise ValueError("session_num must be one of: 1, 2, 12, 21.")
        train_x, train_y, test_x, test_y = session_map[config.session_num]

    preprocessor = EEGPreprocessor(sampling_rate=250)
    train_x, test_x = [preprocessor.pipeline(array) for array in [train_x, test_x]]
    return _make_loader_bundle(train_x, train_y, test_x, test_y, seed, config.batch_size)


def _load_openbmi_cross_task(config: ExperimentConfig, seed: int) -> LoaderBundle:
    """Load a two-task OpenBMI cross-task experiment."""

    if config.cross_tasks is None:
        raise ValueError("Cross-task loading requires `cross_tasks`.")
    first_task, second_task = config.cross_tasks

    def _read_split(task: str, split: str) -> dict[str, np.ndarray]:
        return _load_pickle(config.paths.data_root / "OpenBMI" / "processed" / task / f"{split}.pkl")

    first_train, first_test = _read_split(first_task, "train"), _read_split(first_task, "test")
    second_train, second_test = _read_split(second_task, "train"), _read_split(second_task, "test")

    first_x_train = first_train["ori_train_x"].astype(np.float32).reshape((-1, first_train["ori_train_x"].shape[-2], first_train["ori_train_x"].shape[-1]))
    first_x_test = first_test["ori_test_x"].astype(np.float32).reshape((-1, first_test["ori_test_x"].shape[-2], first_test["ori_test_x"].shape[-1]))
    second_x_train = second_train["ori_train_x"].astype(np.float32).reshape((-1, second_train["ori_train_x"].shape[-2], second_train["ori_train_x"].shape[-1]))
    second_x_test = second_test["ori_test_x"].astype(np.float32).reshape((-1, second_test["ori_test_x"].shape[-2], second_test["ori_test_x"].shape[-1]))
    first_y_train = (first_train["ori_train_s"] - 1).astype(np.int16).reshape(-1)
    first_y_test = (first_test["ori_test_s"] - 1).astype(np.int16).reshape(-1)
    second_y_train = (second_train["ori_train_s"] - 1).astype(np.int16).reshape(-1)
    second_y_test = (second_test["ori_test_s"] - 1).astype(np.int16).reshape(-1)

    if "ERP" in {first_task, second_task}:
        if first_task == "ERP":
            second_x_train = second_x_train[:, :, 100:-100]
            second_x_test = second_x_test[:, :, 100:-100]
        if second_task == "ERP":
            first_x_train = first_x_train[:, :, 100:-100]
            first_x_test = first_x_test[:, :, 100:-100]

    if config.session_num is None:
        train_x = np.concatenate([first_x_train, first_x_test], axis=0)
        train_y = np.concatenate([first_y_train, first_y_test], axis=0)
        test_x = np.concatenate([second_x_train, second_x_test], axis=0)
        test_y = np.concatenate([second_y_train, second_y_test], axis=0)
    else:
        session_map = {
            1: (first_x_train, first_y_train, second_x_train, second_y_train),
            2: (first_x_test, first_y_test, second_x_test, second_y_test),
            12: (first_x_train, first_y_train, second_x_test, second_y_test),
            21: (first_x_test, first_y_test, second_x_train, second_y_train),
        }
        if config.session_num not in session_map:
            raise ValueError("session_num must be one of: 1, 2, 12, 21.")
        train_x, train_y, test_x, test_y = session_map[config.session_num]

    default_processor = EEGPreprocessor(sampling_rate=250)
    erp_processor = EEGPreprocessor(sampling_rate=1000)
    if first_task == "ERP":
        train_x = erp_processor.pipeline(train_x)
        test_x = default_processor.pipeline(test_x)
    elif second_task == "ERP":
        train_x = default_processor.pipeline(train_x)
        test_x = erp_processor.pipeline(test_x)
    else:
        train_x, test_x = [default_processor.pipeline(array) for array in [train_x, test_x]]
    return _make_loader_bundle(train_x, train_y, test_x, test_y, seed, config.batch_size)
