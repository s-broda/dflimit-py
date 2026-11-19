"""The stored Julia reference values (tests/testthat/test-reference.R of the R package dflimit)."""
import math
import warnings

import numpy as np
import pytest

from dflimit import (adfmax_cdf, adfmax_pdf, ltratio_cdf, ltratio_icdf, ltratio_lincdf, ltratio_mom, ltratio_pdf, ltratio_region,
                     power_envelope)
from helpers import case_code, check_Ff, read_app_ref, read_ref

SINGLE = read_ref("single_ref.txt")       # m c st x F f
GEN = read_ref("gen_ref.txt")             # m c c0 st x F f
FIXED = read_ref("fixed_ref.txt")         # m c c0 g0 st x F f
DBL = read_ref("dbl_ref.txt")             # m c c0 g0 st x F f
CONTOUR = read_ref("contour_ref.txt")     # m c c0 g0 x F f errF errf
MOM = read_ref("mom_ref.txt")             # m c c0 g0 st m1 m2 m3 m4
APP = read_app_ref()


@pytest.mark.parametrize("r", SINGLE, ids=lambda r: f"{case_code(r[0], r[2])} c={r[1]:g} x={r[3]:g}")
def test_single_c0(r):
    """the line integral at c0 = 0 reproduces the Julia values"""
    m, c, st, x, F, f = r
    cas = case_code(m, st)
    check_Ff(ltratio_cdf(x, cas, c, method="single"), ltratio_pdf(x, cas, c, method="single"),
             F, f, 1e-11, 1e-9, 1e-11, f"{cas} c = {c:g} x = {x:g}")


@pytest.mark.parametrize("r", GEN, ids=lambda r: f"{case_code(r[0], r[3])} c={r[1]:g} c0={r[2]:g} x={r[4]:g}")
def test_single_general(r):
    """the line integral with c0 != 0 reproduces the Julia values"""
    m, c, c0, st, x, F, f = r
    cas = case_code(m, st)
    check_Ff(ltratio_cdf(x, cas, c, c0, method="single"), ltratio_pdf(x, cas, c, c0, method="single"),
             F, f, 1e-11, 1e-9, 1e-11, f"{cas} c = {c:g} c0 = {c0:g} x = {x:g}")


@pytest.mark.parametrize("r", FIXED,
                         ids=lambda r: f"{case_code(r[0], r[4])} c={r[1]:g} c0={r[2]:g} g0={r[3]:g} x={r[5]:g}")
def test_single_fixed_start(r):
    """the line integral with a fixed start reproduces the Julia values"""
    m, c, c0, g0, st, x, F, f = r
    cas = case_code(m, st)
    check_Ff(ltratio_cdf(x, cas, c, c0, g0, method="single"), ltratio_pdf(x, cas, c, c0, g0, method="single"),
             F, f, 1e-11, 1e-9, 1e-11, f"{cas} c = {c:g} c0 = {c0:g} g0 = {g0:g} x = {x:g}")


@pytest.mark.parametrize("r", DBL,
                         ids=lambda r: f"{case_code(r[0], r[4])} c={r[1]:g} c0={r[2]:g} g0={r[3]:g} x={r[5]:g}")
def test_conditional(r):
    """the conditional integral reproduces the Julia values (without a warning)"""
    m, c, c0, g0, st, x, F, f = r
    cas = case_code(m, st)
    lab = f"{cas} c = {c:g} c0 = {c0:g} g0 = {g0:g} x = {x:g}"
    # verified: the checked computation at the tolerances of the C engine's tests, without a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Fc = ltratio_cdf(x, cas, c, c0, g0, method="double", verified=True)
        fc = ltratio_pdf(x, cas, c, c0, g0, method="double", verified=True)
    check_Ff(Fc, fc, F, f, 1e-10, 1e-8, 1e-10, lab)
    # the default, one evaluation: at the tolerance it meets, unless in a brittle regime (which it warns of)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        F1 = ltratio_cdf(x, cas, c, c0, g0, method="double")
        f1 = ltratio_pdf(x, cas, c, c0, g0, method="double")
    if not w:
        check_Ff(F1, f1, F, f, 5e-7, 1e-6, 1e-5, lab + " (one evaluation)")


@pytest.mark.parametrize("r", MOM, ids=lambda r: f"{case_code(r[0], r[4])} c={r[1]:g} c0={r[2]:g} g0={r[3]:g}")
def test_moments(r):
    """the moments reproduce the Julia values"""
    m, c, c0, g0, st = r[:5]
    mr = r[5:9]
    assert np.max(np.abs(ltratio_mom(4, case_code(m, st), c, c0, g0) / mr - 1)) < 1e-10


@pytest.mark.parametrize("row", APP, ids=lambda row: row[0] + " " + " ".join(f"{v:g}" for v in row[1][:5]))
def test_applications(row):
    """ADFmax, the regions, the linear form and the power envelope reproduce the Julia values"""
    tag, v = row

    def pairs(start):               # the pairs (point, value) of a row from position start on
        return np.array(v[start::2]), np.array(v[start + 1::2])

    if tag == "lincdf":             # model a c cbar k b_1 P_1 ... b_k P_k
        b, P = pairs(5)
        cas = {1: "DK", 4: "DG"}[int(v[0])]
        assert np.max(np.abs(ltratio_lincdf(b, v[1], cas, v[2], v[3]) - P)) < 1e-10
    elif tag == "vcdf":             # c k b_1 P_1 ...
        b, P = pairs(2)
        assert np.max(np.abs(ltratio_lincdf(b, math.nan, "DK", v[0]) - P)) < 1e-10
    elif tag == "envelope":         # model alpha k c_1 P_1 ...
        c, P = pairs(3)
        cas = {1: "DK", 4: "DG"}[int(v[0])]
        assert np.max(np.abs(power_envelope(c, cas, v[1]) - P)) < 1e-10
    elif tag == "adfmax":
        model, stat, c, x, F, f = v
        cas = ("S" if stat == 1 else "D") + "KUT"[int(model) - 1]
        check_Ff(adfmax_cdf(x, cas, c), adfmax_pdf(x, cas, c), F, f, 1e-10, 0, 1e-10,
                 f"ADFmax {cas} c = {c:g} x = {x:g}")
    else:
        typ, p1, p2, v0, c, P = v
        bnd = [{"type": "line", "a": p1, "b": p2}, {"type": "parabola", "x": p1, "c0": p2},
               {"type": "wap", "k": p1, "omega": p2}][int(typ) - 1]
        bnd["v0"] = v0
        assert abs(ltratio_region(bnd, "DK", c) - P) < 1e-10


@pytest.mark.parametrize("r", CONTOUR, ids=lambda r: f"D{'KUT'[int(r[0]) - 1]} c={r[1]:g} c0={r[2]:g} g0={r[3]:g} x={r[4]:g}")
def test_contour_fixed_start(r):
    """the contour integral with a fixed start in cases DU, DT reproduces the Julia values; method "auto" takes it"""
    m, c, c0, g0, x, F, f, eF, ef = r
    cas = case_code(m, 0)
    lab = f"{cas} c = {c:g} c0 = {c0:g} g0 = {g0:g} x = {x:g}"
    # verified: the stored values, without a warning; method "auto" takes the contour integral there
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Fc = ltratio_cdf(x, cas, c, c0, g0, method="contour", verified=True)
        fc = ltratio_pdf(x, cas, c, c0, g0, method="contour", verified=True)
        if Fc >= 1e-3 and fc >= 1e-3:
            assert ltratio_cdf(x, cas, c, c0, g0, verified=True) == Fc
    check_Ff(Fc, fc, F, f, max(1e-12, eF), 1e-9, max(1e-12, ef), lab)
    # the default, one evaluation (with the warning of a brittle regime where it lies in one, c0 > 0 at c < -15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F1 = ltratio_cdf(x, cas, c, c0, g0, method="contour")
        f1 = ltratio_pdf(x, cas, c, c0, g0, method="contour")
        assert ltratio_cdf(x, cas, c, c0, g0) == F1
    check_Ff(F1, f1, F, f, 5e-9, 1e-8, 5e-9, lab + " (one evaluation)")


def test_contour_scope():
    """method "contour" outside a fixed start in cases DU, DT is an error"""
    for args in (("DK", -5, 0, 1), ("DU", -5, 0, 0), ("SU", -5, 0, 0)):
        with pytest.raises(ValueError, match="applies to a fixed start"):
            ltratio_cdf(-2.0, args[0], args[1], args[2], args[3], method="contour")


def _nwarn(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return sum("one evaluation" in str(x.message) for x in w)


def test_brittle_warnings():
    """the warnings of the brittle regimes fire where they should, once per call, and not elsewhere"""
    assert _nwarn(lambda: ltratio_cdf(-3.0, "DU", -5, c0=2, method="double")) == 1
    assert _nwarn(lambda: ltratio_cdf([-3.0, 1.0], "DK", 2, c0=-5, method="double")) == 1
    assert _nwarn(lambda: ltratio_cdf(-5.0, "DU", -20, c0=2, g0=0.5)) == 1
    assert _nwarn(lambda: ltratio_cdf(-2.0, "SK", -0.1, c0=1, method="double")) == 1
    assert _nwarn(lambda: ltratio_icdf([0.05, 0.5], "DU", -20, c0=2, g0=0.5)) == 1
    assert _nwarn(lambda: ltratio_cdf(-3.0, "DU", -5, c0=2)) == 0
    assert _nwarn(lambda: ltratio_cdf(-2.0, "DU", -5)) == 0
    assert _nwarn(lambda: ltratio_cdf(-2.0, "DU", -5, g0=1.5)) == 0
    assert _nwarn(lambda: ltratio_cdf(-3.0, "DU", -5, c0=2, method="double", verified=True)) == 0
    with pytest.warns(RuntimeWarning, match="contour integral by one evaluation fails"):
        ltratio_cdf(-8.4, "DU", 2, c0=5, g0=1.5, method="contour")
    # case DU with a fixed start, c0 > 3 at c > 1.5: the contour integral fails there, "auto" takes the conditional integral
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Fa = ltratio_cdf(-8.4075, "DU", 2, c0=5, g0=1.5)
    assert abs(Fa - ltratio_cdf(-8.4075, "DU", 2, c0=5, g0=1.5, method="double", verified=True)) < 1e-9


def test_verified_reproduces():
    """verified=True reproduces the checked computation (the engine before the default became one evaluation)"""
    vr = [("DU", -5, 2, 0, -3, 0.1976948237625359, 0.4648778393052269),
          ("DK", -10, -5, 0, -2, 0.16726497409337868, 0.3125724838563969),
          ("SU", -5, -5, 0, -1, 0.44272919976045844, 0.4264723317698583),
          ("DU", -5, 0, 0, -4.5, 0.0005234905959262681, 0.0022074999304135296)]
    for cas, c, c0, g0, x, F, f in vr:
        assert abs(ltratio_cdf(x, cas, c, c0, g0, verified=True) - F) < 1e-12
        assert abs(ltratio_pdf(x, cas, c, c0, g0, verified=True) - f) < 1e-12
        assert abs(ltratio_cdf(x, cas, c, c0, g0) - F) < 1e-9
