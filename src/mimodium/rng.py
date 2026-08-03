from typing import Any

import numpy as np
from dagreon import task

#: Reproducibility seed for stochastic scenario tasks.
type Seed = int


def factor_correlation_matrix(correlation: np.ndarray) -> np.ndarray:
    r"""Factor one or more positive-semidefinite correlation matrices.

    The matrices reside in the final two dimensions. The returned array ``A`` satisfies
    ``A @ A.swapaxes(-1, -2).conj() == correlation``. The usual positive-definite case
    takes the fast batched Cholesky path. If any matrix is singular, all matrices fall
    back to a batched Hermitian eigendecomposition.
    """
    try:
        return np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        correlation = (correlation + correlation.swapaxes(-1, -2).conj()) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        maximum_eigenvalues = np.maximum(
            np.max(np.abs(eigenvalues), axis=-1),
            1.0,
        )
        tolerance = (
            100
            * np.finfo(eigenvalues.dtype).eps
            * correlation.shape[-1]
            * maximum_eigenvalues
        )
        if np.any(eigenvalues[..., 0] < -tolerance):
            raise ValueError("correlation matrix must be positive semidefinite")
        return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[..., np.newaxis, :]


@task
class CfgSeed:
    """Provide a reproducibility seed."""

    seed: int

    def __call__(self) -> Seed:
        return self.seed


def get_rng(seed: int, task: Any) -> np.random.Generator:
    """
    Create a random number generator that is uniquely seeded given a seed and a dagreon
    task. The rng is reproducible when using the same seed and object, but different if
    either one changes.
    """
    return np.random.default_rng(
        [
            seed,
            int.from_bytes(task.__task_fingerprint__, byteorder="big"),
        ]
    )
