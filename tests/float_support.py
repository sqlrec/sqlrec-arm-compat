"""Shared tolerances for float32 feature outputs."""

import numpy as np


FLOAT_RTOL = 1e-5
FLOAT_ATOL = 1e-7


def assert_float_output(actual, expected, *, err_msg=""):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, err_msg
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), err_msg
    np.testing.assert_allclose(actual, expected, rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
                               err_msg=err_msg)
    zeros = (actual == 0) & (expected == 0)
    np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]),
                                  err_msg=err_msg)
