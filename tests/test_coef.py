"""The coefficient statistic (tests/testthat/test-coef.R of the R package dflimit).

Independent references: the Julia values stored in matlab/ltratio_test.m of dflimit-matlab (computed with the Julia
engine) and the 5% critical values printed in the paper (its replication script ltratio_paper.m, and matlab/Contents.m).
"""
import math

import numpy as np
import pytest

from dflimit import coef_cdf, coef_icdf, ltratio_cdf
from helpers import expect_equal


def test_stored_julia_values():
    assert abs(coef_cdf(-14, "DU", -5) - 0.1938495145682877) < 1e-10
    assert abs(coef_cdf(-14, "SU", -10) - 0.45401150754561903) < 1e-10
    assert abs(coef_cdf(-16.59, "DG", -13.5) - 0.51132404681639121) < 1e-10
    assert abs(coef_icdf(0.05, "DU") + 14.093590831267115) < 1e-7


def test_published_critical_values():
    assert round(coef_icdf(0.05, "DU"), 4) == -14.0936          # matlab/Contents.m of dflimit-matlab
    assert round(coef_icdf(0.05, "DK"), 3) == -8.039            # the paper: -8.039 -14.094 -21.711
    assert round(coef_icdf(0.05, "DT"), 3) == -21.711
    assert round(coef_icdf(0.05, "DG"), 2) == -16.59            # the paper: -16.59


# self-consistency only (no stored value): the size at the critical value, round trips, and the definition
def test_coef_icdf_inverts_coef_cdf():
    expect_equal(coef_cdf(-14.0936, "DU"), 0.05, 1e-5)
    expect_equal(coef_cdf(coef_icdf(0.1, "DK"), "DK"), 0.1, 1e-8)
    expect_equal(coef_cdf(coef_icdf(0.9, "DU"), "DU"), 0.9, 1e-8)
    expect_equal(coef_cdf(coef_icdf(0.05, "DT", -5), "DT", -5), 0.05, 1e-8)
    expect_equal(coef_cdf(coef_icdf(0.1, "SU", -5), "SU", -5), 0.1, 1e-8)


def test_coef_cdf_is_the_ltratio_cdf_at_0_with_the_shifted_null():
    # equal to rounding, not identical: with threads the order of summation in the engine can differ by one ulp
    # (method="engine", the computation of before; the default, the single inversion of Bao and Kan, to 1e-11)
    expect_equal(coef_cdf(-5, "DT", -2, c0=1, method="engine"), ltratio_cdf(0, "DT", -2, c0=-4), 1e-13)
    expect_equal(coef_cdf(-3, "DK", 0, g0=1, method="engine"), ltratio_cdf(0, "DK", 0, c0=-3, g0=1), 1e-13)
    assert abs(coef_cdf(-5, "DT", -2, c0=1) - ltratio_cdf(0, "DT", -2, c0=-4)) < 1e-11
    assert abs(coef_cdf(-3, "DK", 0, g0=1) - ltratio_cdf(0, "DK", 0, c0=-3, g0=1)) < 1e-11


def test_shape_and_argument_checks():
    y = np.array([[-math.inf, math.nan], [-5, math.inf]])
    F = coef_cdf(y, "DK")
    assert F.shape == y.shape
    assert F[0, 0] == 0 and math.isnan(F[0, 1]) and F[1, 1] == 1
    assert isinstance(coef_cdf(-1, "DU"), float)
    assert coef_cdf([], "DU").shape == (0,)
    with pytest.raises(ValueError, match="the case must be"):
        coef_cdf(0, "XX")
    with pytest.raises(ValueError, match="method must be"):
        coef_cdf(0, "DU", method="fast")
    with pytest.raises(ValueError, match=r"requires c <= -1e-4"):
        coef_cdf(-1, "SU", 1)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        coef_icdf(0, "DU")
    with pytest.raises(ValueError, match="single number"):
        coef_icdf(0.5, "DU", c=math.nan)
    with pytest.raises(TypeError, match="single number"):
        coef_icdf(0.5, "DU", c=None)
    assert coef_icdf([[0.05, 0.1]], "DK").shape == (1, 2)
