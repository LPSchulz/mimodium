import numpy as np
import pytest
from mimodium import array_ops


def test_concatenate_all_channels_of_aps_stacks_selected_ap_antennas():
    channels = np.arange(2 * 3 * 2 * 1).reshape(2, 3, 2, 1)

    concatenated = array_ops.concatenate_all_channels_of_aps(channels, np.array([0, 2]))

    np.testing.assert_array_equal(concatenated, channels[:, [0, 2]].reshape(2, 4, 1))


def test_as_block_diag_places_matrix_axis_on_block_diagonal():
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    out = array_ops.as_block_diag(arr)

    np.testing.assert_array_equal(
        out,
        np.array(
            [
                [1, 2, 0, 0],
                [3, 4, 0, 0],
                [0, 0, 5, 6],
                [0, 0, 7, 8],
            ]
        ),
    )


def test_as_block_diag_rejects_arrays_without_matrix_collection_axis():
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        array_ops.as_block_diag(np.eye(2))


def test_concatenate_all_correlations_of_aps_builds_per_user_block_diagonal():
    corr = np.zeros((1, 2, 2, 2), dtype=float)
    corr[0, 0] = np.eye(2)
    corr[0, 1] = 2 * np.eye(2)

    out = array_ops.concatenate_all_correlations_of_aps(corr, np.array([0, 1]))

    np.testing.assert_array_equal(out[0], np.diag([1.0, 1.0, 2.0, 2.0]))


def test_outer_product_with_self_uses_conjugate_second_factor():
    vec = np.array([[1.0 + 1.0j, 2.0]])

    outer = array_ops.outer_product_with_self(vec, axis=1)

    expected = np.array([[[2.0 + 0.0j, 2.0 + 2.0j], [2.0 - 2.0j, 4.0 + 0.0j]]])
    np.testing.assert_allclose(outer, expected)
