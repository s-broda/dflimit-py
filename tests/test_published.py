"""Published values and consistency (tests/testthat/test-published.R of the R package dflimit)."""
import numpy as np

from dflimit import (adfmax_cdf, adfmax_icdf, ltratio_cdf, ltratio_icdf, ltratio_lincdf, ltratio_mom, ltratio_region,
                     power_envelope)
from helpers import expect_equal


def test_published_critical_values_and_moments():
    expect_equal(ltratio_icdf(0.05, "DU"), -2.86157, 1e-5)        # 5% critical value of tau_mu
    expect_equal(ltratio_icdf(0.05, "DK"), -1.94087, 1e-5)        # 5% critical value of tau
    expect_equal(ltratio_mom(2, "DU")[1], 3.0559953450, 1e-10)    # Nabeya (1999)
    expect_equal(adfmax_cdf(-2.43542, "DU"), 0.05, 1e-5)          # the quantiles of the ADFmax statistic in the paper
    expect_equal(ltratio_region({"type": "wap", "k": 3.6658, "omega": 10}), 0.05, 1e-4)   # the weighted-average-power test of the paper
    expect_equal(ltratio_lincdf(-0.2639, -3.5, "DK"), 0.05, 1e-4)  # the point-optimal test at cbar = -7
    expect_equal(power_envelope(-7), 0.5, 1e-2)                     # power one half at c = -7


def test_power_is_the_cdf_at_the_critical_value():
    q = ltratio_icdf(0.05, "DU")
    expect_equal(ltratio_cdf(q, "DU"), 0.05, 1e-10)
    expect_equal(ltratio_cdf(-2.86157, "DU", -5), 0.116513562093, 1e-9)


def test_quantiles_invert_the_cdf():
    r = ltratio_icdf([0.1, 0.9], "ST", -10, info=True)
    assert np.all(r["conv"])
    expect_equal(r["F"], [0.1, 0.9], 1e-9)
    expect_equal(ltratio_cdf(r["q"], "ST", -10), [0.1, 0.9], 1e-9)
    expect_equal(adfmax_cdf(adfmax_icdf(0.05, "DU"), "DU"), 0.05, 1e-9)


def test_the_two_representations_agree():
    x = [-3, -1.5, 0.5]
    expect_equal(ltratio_cdf(x, "DU", -2, method="line"), ltratio_cdf(x, "DU", -2, method="conditional"), 1e-10)
    expect_equal(ltratio_cdf(-1, "DK", 2, c0=2), ltratio_region({"type": "parabola", "x": -1, "c0": 2}, "DK", 2),
                 1e-10)
    # the line of the point-optimal test by the conditional integral and by the inversion of U - aV
    expect_equal(ltratio_region({"type": "line", "a": -3.5, "b": -0.2639}, "DK", -5),
                 ltratio_lincdf(-0.2639, -3.5, "DK", -5), 1e-10)


def test_the_linear_form_near_a_zero_and_outside_its_support():
    import math
    import pytest
    # a = 0: P(U <= b) = erf(sqrt(kappa (b + 1/2))), kappa = 2c/(e^{2c} - 1) (1 at c = 0), zero below the support
    for c in (-50.0, -5.0, 0.0, 2.0):
        kap = 1.0 if c == 0 else 2 * c / math.expm1(2 * c)
        b = [-0.5 + m / kap for m in (-0.1, 0.0, 1e-6, 1e-3, 0.05, 0.5, 2.0)]
        expect_equal(ltratio_lincdf(b, 0, "DK", c), [math.erf(math.sqrt(max(kap * (x + 0.5), 0.0))) for x in b], 1e-13)
    assert ltratio_lincdf(-0.6, 0, "DK", -5) == 0.0
    # small |a| and c > 0, against the Julia values (the conditional integral given U in case DG)
    expect_equal(ltratio_lincdf(-0.51, 0.3, "DG", -5), 0.291315235195, 1e-11)
    expect_equal(ltratio_lincdf(-0.51, 0.3, "DG", 0), 0.287795093100, 1e-11)
    P = np.asarray(ltratio_lincdf(np.arange(-0.7, 1.05, 0.1), 1e-3, "DG", 2))
    assert np.all((P >= 0) & (P <= 1)) and np.all(np.diff(P) >= 0)
    # the power envelope near c = 0: alpha + O(c^2)
    expect_equal(power_envelope(-0.01), 0.050249919491, 1e-10)
    e = power_envelope(-1e-3, "DG")
    assert 0.05 < e < 0.05 + 1e-8
    # the ends of the range
    with pytest.raises(ValueError, match="at least -550"):
        ltratio_lincdf(0, 0, "DK", -600)
    with pytest.raises(ValueError, match="case DG is available for c up to about 3.5"):
        ltratio_lincdf(0, 0, "DG", 6)
