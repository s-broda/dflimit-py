"""The devices of Bao and Kan (tests/testthat/test-kan.R of the R package dflimit): the single inversion of the
half-plane statistics, the moments from jets, the single inversion of the general engine's point-optimal statistic and
coef_cdf_gp, against the values of the Julia engine DFLimit.jl (ref/kan_ref.txt, the user-level lines of
c/test/kan_ref.txt of dflimit-matlab), with their options, error estimates, warnings and refusals.  The method is that of
Yong Bao and Raymond Kan, "Exact Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order
Autoregressions: O(1) Algorithms"."""
import math
import os
import warnings

import numpy as np
import pytest

from dflimit import (coef_cdf, coef_cdf_gp, coef_icdf, coef_mom, coef_pdf, ltratio_cdf, ltratio_lincdf, ltratio_mom, perron,
                     power_envelope, pt_cdf, pt_envelope, pt_power, pt_quantile, trend)

REF = os.path.join(os.path.dirname(__file__), "ref", "kan_ref.txt")
# the tolerances of the C engine's test (dfl_test_kan.c): absolute for probabilities, relative to max(1, |value|) for
# densities and quantiles, relative to the size of the moments (max(|m_j|, |m_2|^(j/2)))
TOL = {"rcoef": 1e-11, "rcoefpdf": 1e-11, "rcoefq": 1e-8, "rcoefmom": 1e-10, "rmom": 1e-10, "rlin": 1e-11, "renv": 1e-11,
       "ptcdf": 1e-11, "ptpdf": 1e-11, "ptq": 1e-10, "ptpow": 1e-10, "ptenv": 1e-11, "gp": 1e-11}
START = ("zero", "fixed", "stationary")


def _lines():
    with open(REF) as fh:
        return [ln.split() for ln in fh if ln.strip() and not ln.startswith("#")]


def _case(model, start):
    return "DG" if model == 4 else ("S" if start == 2 else "D") + "KUT"[int(model) - 1]


def _design(v):
    kind, npar = int(v[0]), int(v[1])
    par = v[2:2 + npar]
    D = trend(int(par[0])) if kind == 0 else perron("ABC"[kind - 1], par if npar > 1 else par[0])
    return D, v[2 + npar:]


def _same(g, r):
    g, r = np.atleast_1d(np.asarray(g, dtype=float)), np.atleast_1d(np.asarray(r, dtype=float))
    return np.where(np.isnan(g) & np.isnan(r), 0.0, np.abs(g - r))


def _rel(g, r):
    g, r = np.atleast_1d(np.asarray(g, dtype=float)), np.atleast_1d(np.asarray(r, dtype=float))
    return np.abs(g - r) / np.maximum(1, np.abs(r))


def _dev(w):
    tag = w[0]
    v = [float(t) for t in (w[2:] if tag.startswith("pt") or tag == "gp" else w[1:])]
    if tag in ("rcoef", "rcoefpdf", "rcoefq", "rcoefmom", "rmom"):
        cas = _case(v[0], v[1])
        g0 = v[2] if v[1] == 1 else 0.0
        c, c0, n = v[3], v[4], int(v[5])
        if tag in ("rcoefmom", "rmom"):
            r = np.array(v[6:6 + n])
            g = ltratio_mom(n, cas, c, c0, g0) if tag == "rmom" else coef_mom(n, cas, c, c0, g0)
            return np.abs(g - r) / np.maximum(np.abs(r), np.abs(r[min(1, n - 1)]) ** (np.arange(1, n + 1) / 2))
        x, r = np.array(v[6:6 + n]), np.array(v[6 + n:6 + 2 * n])
        if tag == "rcoef":
            return _same(coef_cdf(x, cas, c, c0, g0), r)
        if tag == "rcoefpdf":
            return _rel(coef_pdf(cas, x, c, c0=c0, gamma0=g0), r)
        return _rel(coef_icdf(x, cas, c, c0, g0, tol=1e-9), r)
    if tag == "rlin":
        n = int(v[5])
        b, r = np.array(v[6:6 + n]), np.array(v[6 + n:6 + 2 * n])
        st = int(v[1])
        g = ltratio_lincdf(b, v[4], trend(int(v[0])), v[3], start=START[st], g0=v[2] if st == 1 else 0.0)
        return np.abs(g - r)
    if tag == "renv":
        n = int(v[1])
        cs, r = np.array(v[2:2 + n]), np.array(v[2 + n:2 + 2 * n])
        return _same(power_envelope(cs, "DK", alpha=v[0]), r)
    D, a = _design(v)
    if tag in ("ptcdf", "ptpdf"):
        n = int(a[2])
        q, r = np.array(a[3:3 + n]), np.array(a[3 + n:3 + 2 * n])
        if tag == "ptcdf":
            return np.abs(pt_cdf(D, q, a[0], a[1]) - r)
        return _rel(pt_pdf_(D, q, a[0], a[1]), r)
    if tag == "ptq":
        return _rel(pt_quantile(D, a[1], a[0]), a[2])
    if tag == "ptpow":
        b, P = pt_power(D, a[0], a[1], level=a[2])
        return np.array([max(_rel(b, a[3])[0], abs(P - a[4]))])
    if tag == "ptenv":
        n = int(a[1])
        cs, r = np.array(a[2:2 + n]), np.array(a[2 + n:2 + 2 * n])
        return _same(pt_envelope(D, cs, level=a[0]), r)
    if tag == "gp":
        n = int(a[4])
        y, r = np.array(a[5:5 + n]), np.array(a[5 + n:5 + 2 * n])
        return _same(coef_cdf_gp(D, y, a[2], c0=a[3], start=START[int(a[0])], gls=None if math.isnan(a[1]) else a[1]), r)
    raise AssertionError(tag)


def pt_pdf_(D, q, cbar, c):
    from dflimit import pt_pdf
    return pt_pdf(D, q, cbar, c)


@pytest.mark.parametrize("w", _lines(), ids=lambda w: " ".join(w)[:60])
def test_julia_values(w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = _dev(w)
    assert np.all(d <= TOL[w[0]]), f"deviation {np.max(d):.3g} above {TOL[w[0]]:.0g}"


def test_method_engine_reproduces_the_engines():
    y = np.array([-12.0, -5.0, 0.5])
    Fe = coef_cdf(y, "DU", -5, method="engine")
    np.testing.assert_array_equal(Fe, [ltratio_cdf(0.0, "DU", -5, c0=x) for x in y])
    assert np.max(np.abs(coef_cdf(y, "DU", -5) - Fe)) < 1e-9
    np.testing.assert_array_equal(coef_cdf(y, "DU", -5, engine="classical"), coef_cdf(y, "DU", -5, method="engine",
                                                                                         engine="classical"))
    b = np.array([-0.2639, 0.3])
    Pe = ltratio_lincdf(b, -3.5, "DK", -7, method="engine")
    np.testing.assert_array_equal(Pe, ltratio_lincdf(b, -3.5, "DK", -7, engine="classical"))
    assert np.max(np.abs(ltratio_lincdf(b, -3.5, "DK", -7) - Pe)) < 1e-9
    Ee = power_envelope([-5.0, -13.5], method="engine")
    np.testing.assert_array_equal(Ee, power_envelope([-5.0, -13.5], engine="classical"))
    assert np.max(np.abs(power_envelope([-5.0, -13.5]) - Ee)) < 1e-9
    me = ltratio_mom(3, "DT", -5, method="engine")
    np.testing.assert_allclose(me, ltratio_mom(3, "DT", -5, engine="classical"), rtol=1e-12)
    assert np.max(np.abs(ltratio_mom(3, "DT", -5) - me) / np.abs(me)) < 1e-9
    fe = coef_pdf("DU", [-10.0, -3.0], -5, method="engine")
    np.testing.assert_array_equal(fe, coef_pdf(trend(0), [-10.0, -3.0], -5, engine="general"))
    assert np.max(np.abs(coef_pdf("DU", [-10.0, -3.0], -5) - fe)) < 1e-9
    D = trend(1)
    assert np.max(np.abs(pt_cdf(D, [4.0, 7.0], -13.5, -5) - pt_cdf(D, [4.0, 7.0], -13.5, -5, method="grid"))) < 1e-8
    np.testing.assert_array_equal(pt_cdf(D, [4.0, 7.0], -13.5, -5, method="grid"), pt_cdf(D, [4.0, 7.0], -13.5, -5, N=60000))
    assert abs(coef_cdf_gp(trend(0), -12.0, -5) - coef_cdf_gp(trend(0), -12.0, -5, method="grid")) < 1e-9
    assert abs(coef_cdf_gp(trend(0), -12.0, -5) - coef_cdf(-12.0, "DU", -5)) < 1e-11


def test_error_estimates_warnings_verified():
    v, e = coef_cdf([-9.0, -3.0], "DK", -5, err=True)
    assert np.all((e >= 0) & (e <= 1e-12))
    assert np.all(np.isnan(coef_cdf([-9.0, -3.0], "DK", -5, method="engine", err=True)[1]))
    assert np.all(np.isnan(coef_cdf(-9.0, "DG", -5, err=True)[1]))
    with pytest.warns(RuntimeWarning, match="exceeds"):
        coef_cdf(-9.0, "DK", -5, tol=1e-30)
    with pytest.warns(RuntimeWarning, match="exceeds"):
        ltratio_lincdf(-0.2639, -3.5, "DK", -7, tol=1e-30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert coef_cdf(-9.0, "DK", -5, tol=1e-30, verified=True) == coef_cdf(-9.0, "DK", -5, method="engine", verified=True)
    assert ltratio_lincdf(-0.2639, -3.5, "DK", -7, tol=1e-30, verified=True) == ltratio_lincdf(-0.2639, -3.5, "DK", -7,
                                                                                               method="engine")
    assert power_envelope(-7.0, err=True)[1] < 1e-11
    m = ltratio_mom(2, "DU", info=True)
    assert np.all((m.err >= 0) & (m.err < 1e-9))
    assert np.all(np.isnan(ltratio_mom(2, "DU", info=True, engine="general").err))
    p, e = pt_cdf(trend(1), [4.0, 7.0], -13.5, -5, err=True)
    assert np.all(e <= 1e-12)
    g, e = coef_cdf_gp(trend(0), [-12.0, math.inf], -5, err=True)
    assert e[0] <= 1e-12 and e[1] == 0
    assert math.isnan(coef_cdf_gp(trend(0), -12.0, -5, method="grid", err=True)[1])


def test_the_rules_of_the_defaults():
    assert coef_cdf(-2.0, "DU", 25) == coef_cdf(-2.0, "DU", 25, method="engine")
    assert abs(coef_cdf(-2.0, "DU", 25, method="inversion") - coef_cdf(-2.0, "DU", 25)) < 1e-8
    assert ltratio_lincdf(1.0, -1, "DK", 5) == ltratio_lincdf(1.0, -1, "DK", 5, method="engine")
    assert power_envelope(-1e-5) == power_envelope(-1e-5, method="engine")
    D = trend(0)
    for f in (lambda m: pt_cdf(D, 30.0, -7, 4.2, method=m), lambda m: coef_cdf_gp(D, -2.0, 5.5, method=m)):
        msgs = []
        for m in ("auto", "grid"):
            try:
                msgs.append(f(m))
            except ValueError as e:
                msgs.append(str(e))
        assert msgs[0] == msgs[1]
    with pytest.raises(ValueError, match="does not resolve"):
        pt_cdf(D, 30.0, -7, 4.2)
    assert coef_cdf_gp(D, -12.0, -5, start="fixed", gamma0=1) == coef_cdf_gp(D, -12.0, -5, start="fixed", gamma0=1,
                                                                             method="grid")
    assert coef_pdf(trend(-1), -8.0) == coef_pdf("DK", -8.0)
    assert ltratio_lincdf(-0.26, -3.5, trend(0), -5) == ltratio_lincdf(-0.26, -3.5, trend(0), -5, method="inversion")
    assert coef_pdf(perron("A", 0.5), -8.0) == coef_pdf(perron("A", 0.5), -8.0, method="engine")
    assert coef_mom(1, "DK")[0] == pytest.approx(-1.7814, abs=1e-4)


@pytest.mark.parametrize("f, msg", [
    (lambda: coef_cdf(-9.0, "DK", method="inversion", engine="general"), "does not apply with engine"),
    (lambda: coef_cdf(-9.0, "DG", method="inversion"), "applies to the cases DK, DU, DT, SK, SU, ST"),
    (lambda: coef_cdf(-9.0, "DK", method="fast"), "method must be"),
    (lambda: coef_cdf(-9.0, "DK", tol=-1.0), "tol must be"),
    (lambda: coef_pdf(perron("A", 0.5), -8.0, method="inversion"), "applies to the cases"),
    (lambda: coef_mom(2, "DG"), "available for the cases"),
    (lambda: coef_mom(2, "DK", engine="general"), "no moments of the coefficient statistic"),
    (lambda: coef_mom(2, "DK", 25), "at most 20"),
    (lambda: coef_mom(2, "DK", method="engine"), "method must be"),
    (lambda: ltratio_mom(2, "DK", method="jets", engine="general"), "does not apply with engine"),
    (lambda: ltratio_lincdf(0.0, -3.5, "DG", method="inversion"), "not to case DG"),
    (lambda: ltratio_lincdf(0.0, math.nan, "DK", method="inversion"), "finite a"),
    (lambda: power_envelope(-5.0, "DG", method="inversion"), "not to case DG"),
    (lambda: power_envelope(-5.0, method="inversion", engine="general"), "does not apply with engine"),
    (lambda: pt_quantile(trend(1), 0.05, -13.5, method="inversion", umax=10), "N and umax belong"),
    (lambda: coef_cdf_gp(trend(0), -12.0, -5, start="fixed", gamma0=1, method="inversion"), "zero or stationary start"),
])
def test_refusals(f, msg):
    with pytest.raises(ValueError, match=msg):
        f()
