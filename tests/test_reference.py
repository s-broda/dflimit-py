"""The stored Julia reference values (tests/testthat/test-reference.R of the R package dflimit)."""
import math
import warnings

import numpy as np
import pytest

from dflimit import (adfmax_cdf, adfmax_pdf, ltratio_cdf, ltratio_lincdf, ltratio_mom, ltratio_pdf, ltratio_region,
                     power_envelope)
from helpers import case_code, check_Ff, read_app_ref, read_ref

SINGLE = read_ref("single_ref.txt")       # m c st x F f
GEN = read_ref("gen_ref.txt")             # m c c0 st x F f
FIXED = read_ref("fixed_ref.txt")         # m c c0 g0 st x F f
DBL = read_ref("dbl_ref.txt")             # m c c0 g0 st x F f
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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Fc = ltratio_cdf(x, cas, c, c0, g0, method="double")
        fc = ltratio_pdf(x, cas, c, c0, g0, method="double")
    check_Ff(Fc, fc, F, f, 1e-10, 1e-8, 1e-10, f"{cas} c = {c:g} c0 = {c0:g} g0 = {g0:g} x = {x:g}")


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
