import contextlib

import numpy as np


@contextlib.contextmanager
def without_numpy_in1d():
    sentinel = object()
    original = getattr(np, "in1d", sentinel)
    if original is not sentinel:
        delattr(np, "in1d")
    try:
        yield
    finally:
        if original is not sentinel:
            np.in1d = original


def isin_compat(element, test_elements, assume_unique=False, invert=False):
    return np.isin(
        element,
        test_elements,
        assume_unique=assume_unique,
        invert=invert,
    )


def in1d_or_expected(expected, element, test_elements, assume_unique=False, invert=False):
    if hasattr(np, "in1d"):
        return np.in1d(
            element,
            test_elements,
            assume_unique=assume_unique,
            invert=invert,
        )
    return expected


def test_isin_matches_in1d_for_grid_indexes_out_region():
    mesh_size = 12
    idx_inside = np.array([1, 2, 5, 8, 11])

    expected_mask = in1d_or_expected(
        np.array([False, True, True, False, False, True, False, False, True, False, False, True]),
        np.arange(mesh_size),
        idx_inside,
    )
    expected = np.flatnonzero(~expected_mask)
    actual = np.flatnonzero(~isin_compat(np.arange(mesh_size), idx_inside))

    np.testing.assert_array_equal(actual, expected)


def test_isin_matches_in1d_for_missing_frequency_picks():
    freqax_picks = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    picks = [(0.04, 3.2), (0.05, 3.15), (0.08, 3.0)]
    idx0 = 0
    idxpick = 5

    expected = in1d_or_expected(
        np.array([False, False, True, True, False]),
        freqax_picks[idx0:idxpick],
        np.array(picks)[:, 0],
        assume_unique=True,
        invert=True,
    )
    actual = isin_compat(
        freqax_picks[idx0:idxpick],
        np.array(picks)[:, 0],
        assume_unique=True,
        invert=True,
    )

    np.testing.assert_array_equal(actual, expected)


def test_isin_matches_in1d_for_common_noise_correlation_times():
    times1 = np.array([10, 20, 30, 40, 50])
    times2 = np.array([20, 30, 50, 70])
    common_times = np.intersect1d(np.intersect1d(times1, times2), [20, 50, 90])

    expected_idx1 = in1d_or_expected(
        np.array([False, True, False, False, True]),
        times1,
        common_times,
    ).nonzero()[0]
    expected_idx2 = in1d_or_expected(
        np.array([True, False, True, False]),
        times2,
        common_times,
    ).nonzero()[0]
    actual_idx1 = isin_compat(times1, common_times).nonzero()[0]
    actual_idx2 = isin_compat(times2, common_times).nonzero()[0]

    np.testing.assert_array_equal(actual_idx1, expected_idx1)
    np.testing.assert_array_equal(actual_idx2, expected_idx2)


def test_isin_compat_works_when_numpy_in1d_is_missing():
    with without_numpy_in1d():
        assert not hasattr(np, "in1d")

        mesh_size = 12
        idx_inside = np.array([1, 2, 5, 8, 11])
        idx_outside = np.flatnonzero(~isin_compat(np.arange(mesh_size), idx_inside))
        np.testing.assert_array_equal(idx_outside, np.array([0, 3, 4, 6, 7, 9, 10]))

        freqax_picks = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
        picks = [(0.04, 3.2), (0.05, 3.15), (0.08, 3.0)]
        missing = isin_compat(
            freqax_picks[:5],
            np.array(picks)[:, 0],
            assume_unique=True,
            invert=True,
        )
        np.testing.assert_array_equal(missing, np.array([False, False, True, True, False]))

        times1 = np.array([10, 20, 30, 40, 50])
        times2 = np.array([20, 30, 50, 70])
        common_times = np.intersect1d(np.intersect1d(times1, times2), [20, 50, 90])
        np.testing.assert_array_equal(isin_compat(times1, common_times).nonzero()[0], np.array([1, 4]))
        np.testing.assert_array_equal(isin_compat(times2, common_times).nonzero()[0], np.array([0, 2]))


if __name__ == "__main__":
    test_isin_matches_in1d_for_grid_indexes_out_region()
    test_isin_matches_in1d_for_missing_frequency_picks()
    test_isin_matches_in1d_for_common_noise_correlation_times()
    test_isin_compat_works_when_numpy_in1d_is_missing()
