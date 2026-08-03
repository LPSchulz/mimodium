import numpy as np


def concatenate_all_channels_of_aps(ch: np.ndarray, ap_set: np.ndarray) -> np.ndarray:
    # ch may be ChannelRealizations or ChannelEstimates
    K, L, N, O = ch.shape
    assert len(ap_set) <= L
    return ch[:, ap_set].reshape((K, len(ap_set) * N, O))


def concatenate_all_correlations_of_aps(
    corr: np.ndarray, ap_set: np.ndarray
) -> np.ndarray:
    # corr may be EstimationErrorCorrelation or SpatialCorrelationMatrices
    K, L, N, N2 = corr.shape
    assert N == N2
    assert len(ap_set) <= L
    return as_block_diag(corr[:, ap_set])


def as_block_diag(arr: np.ndarray) -> np.ndarray:
    # scipy.linalg.block_diag can only handle a sequence of 2D arrays
    # here we handle arrays with an arbitrary number of dimensions, the matrices reside
    # in the last two dimensions
    shape = arr.shape
    if len(shape) < 3:
        raise ValueError("Array must have at least 3 dimensions")
    N, M = arr.shape[-2], arr.shape[-1]
    num_matrices = arr.shape[-3]
    keep_shapes = arr.shape[:-3]
    out = np.zeros(keep_shapes + (N * num_matrices, M * num_matrices), dtype=arr.dtype)
    for num in range(num_matrices):
        out[..., num * N : (num + 1) * N, num * M : (num + 1) * M] = arr[..., num, :, :]
    return out


def outer_product_with_self(vec: np.ndarray, axis=-2) -> np.ndarray:
    first = np.expand_dims(vec, axis=axis + 1)
    second = np.conj(np.expand_dims(vec, axis=axis))
    return first * second
