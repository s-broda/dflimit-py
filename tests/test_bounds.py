"""The repairs of 0.5.0 (tests/testthat/test-bounds.R of the R package dflimit): the points where the line integral
exceeds the bounds of the engine and the fixed start whose path climbed.  The values: the Julia engine DFLimit.jl 0.5.0,
which has the same repairs, and the contour integral of a newer engine (the values of the Julia tests)."""
import numpy as np
import pytest

from dflimit import ltratio_cdf, ltratio_pdf


def test_large_x_above_the_unit_root():
    # the engine crashed here (a segmentation fault): the automatic method now takes the conditional integral
    assert abs(ltratio_cdf(751, "DT", 7) - 0.990823749760809) < 1e-12
    assert abs(ltratio_pdf(751, "DT", 7) / 9.2924954e-5 - 1) < 1e-6
    with pytest.raises(ValueError, match="at large x above the unit root"):
        ltratio_cdf(751, "DT", 7, method="single")


def test_negative_c0_near_zero():
    # more than 2 GB before: the conditional integral, with a warning where its values are flagged (the classical engine;
    # the default takes the general engine with a stationary start and c0 != 0 for c <= -3e-3)
    with pytest.warns(RuntimeWarning, match="flags some of them"):
        F = ltratio_cdf(0.05, "SK", -1e-3, c0=-1e-3, engine="classical")
    assert abs(F - 0.54749158678338294) < 1e-10
    with pytest.warns(RuntimeWarning, match="flags some of them"):
        F = ltratio_cdf(1, "SK", -1e-2, c0=-1e-2, engine="classical")
    assert abs(F - 0.87075930276798486) < 1e-10
    with pytest.warns(RuntimeWarning, match="exceed the bounds of the engine"):
        F = ltratio_cdf([-1, 0.05, 0.5, 1, 1.5], "SK", -1e-4, c0=-1e-4)
    np.testing.assert_allclose(F, [0.1666135587511895, 0.531233538152, 0.700742367308, 0.847290600876, 0.936188673511],
                               rtol=0, atol=1e-6)


def test_the_line_integral_requested_is_an_error_that_names_the_arguments():
    with pytest.raises(ValueError, match="the line integral needs more nodes than the engine allows") as e:
        ltratio_cdf(1, "SK", -1e-4, c0=-1e-4, method="single")
    assert "c0 = -0.0001, x = 1" in str(e.value)
    with pytest.raises(ValueError, match="case DG has the line integral representation only"):
        ltratio_cdf(1, "DG", -1, c0=-1e-4)


def test_fixed_start_with_c0_near_zero():
    # 6.0e74 before
    assert abs(ltratio_cdf(1.5, "DK", 0, c0=-0.01, g0=1) - 0.960991862318) < 1e-10
    assert abs(ltratio_pdf(1.5, "DK", 0, c0=-0.01, g0=1) - 0.079242719297) < 1e-10
    assert abs(ltratio_cdf(1.5, "DK", 2, c0=-0.01, g0=1) - 0.086764388544) < 1e-10


def test_the_upper_tail_above_the_unit_root():
    # case DK with c0 != c beyond the line integral's bound, for c > 4 at x > 300 and with c0 > c at x < -20: the
    # conditional integral with its checks (before, its one evaluation where the checked value was flagged: 0.8415,
    # -1.3674, 1.0479 here); above c = 14 an error; the general engine's line cut to the rate with c0 (the C engine of
    # dflimit-matlab, November 23, 2026).  The values: an independent computation (the law of V given J(1) by Girsanov's
    # theorem, inverted and integrated over J(1), DFLimit.jl).
    assert abs(ltratio_cdf(1e4, "DK", 10, c0=2) - 0.743623945123) < 1e-10
    assert abs(ltratio_cdf(3e4, "DK", 10, c0=2) - 0.999338346257) < 1e-10
    assert abs(ltratio_cdf(3e3, "DK", 10, c0=9) - 0.993550212928) < 1e-10
    assert abs(ltratio_cdf(-200, "DK", 10, c0=10.5) - 0.716459029660) < 1e-10
    np.testing.assert_allclose(ltratio_cdf([1e3, 3e3], "DK", 10, c0=9, engine="general"), [0.636120922790, 0.993550212928],
                               rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match="is computed for c <= 14 only"):
        ltratio_cdf(1e7, "DK", 16, c0=2)
