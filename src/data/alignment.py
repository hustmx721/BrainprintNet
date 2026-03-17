"""EEG data alignment methods."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from scipy.linalg import fractional_matrix_power
from sklearn.base import BaseEstimator, TransformerMixin

ArrayFloat = NDArray[np.floating]


def euclidean_alignment(
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


class CentroidAlign(BaseEstimator, TransformerMixin):
    """Align EEG trials with respect to a covariance centroid."""

    def __init__(self, center_type: str = "euclid", covariance_type: str = "cov") -> None:
        """Store centroid and covariance estimation strategies."""

        self.center_type = center_type
        self.covariance_type = covariance_type
        self.reference_matrix: ArrayFloat | None = None

    def fit(self, data: ArrayFloat, y: None = None) -> "CentroidAlign":
        """Estimate the whitening matrix from training data."""

        covariance = covariances(data, estimator=self.covariance_type)
        center_covariance = self._compute_center(covariance)
        self.reference_matrix = fractional_matrix_power(center_covariance, -0.5)
        return self

    def transform(self, data: ArrayFloat) -> tuple[ArrayFloat, ArrayFloat]:
        """Apply the fitted whitening matrix to EEG trials."""

        if self.reference_matrix is None:
            raise RuntimeError("CentroidAlign must be fitted before calling transform().")

        covariance = covariances(data, estimator=self.covariance_type)
        aligned_covariance = np.zeros_like(covariance)
        aligned_trials = np.zeros_like(data)
        for index, (trial_cov, trial_data) in enumerate(zip(covariance, data)):
            aligned_covariance[index] = self.reference_matrix @ trial_cov @ self.reference_matrix
            aligned_trials[index] = self.reference_matrix @ trial_data
        return aligned_covariance, aligned_trials

    def fit_transform(self, data: ArrayFloat, y: None = None) -> tuple[ArrayFloat, ArrayFloat]:
        """Fit and transform in a single call."""

        return self.fit(data, y).transform(data)

    def _compute_center(self, covariance: ArrayFloat) -> ArrayFloat:
        """Compute the covariance centroid according to the selected metric."""

        if self.center_type == "riemann":
            return mean_covariance(covariance, metric="riemann")
        if self.center_type == "logeuclid":
            return mean_covariance(covariance, metric="logeuclid")
        if self.center_type == "euclid":
            return np.mean(covariance, axis=0)
        raise ValueError(f"Unsupported center type: {self.center_type}")
