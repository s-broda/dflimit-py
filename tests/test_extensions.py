"""The extensions of the limit (volatility and persistence regimes, a normal component, the first seasonal t-ratio)
against the values of the Julia engine DFLimit.jl (ref/ext_ref.txt, written by dev/ext_ref.jl of DFLimit.jl; the second
seasonal t-ratio and the F statistic are not part of the release 0.5.0), the regimes with GLS detrending, with the
statistic that detrends first and with the stationary start of the first regime (the tags rval, rquant, rcase, rwarn),
the argument errors and the limiting cases (tests/testthat/test-extensions.R of the R package)."""
import math
import os
import re

import numpy as np
import pytest

import dflimit
from dflimit import (cadf_c0, detrend_first, dist_cdf, dist_pdf, dist_quantile, hegy_t3_cdf, hegy_t3_pdf, hegy_t3_quantile,
                     ltratio_cdf, ltratio_icdf, ltratio_pdf, perron, predictive_c0, trend, volpath)

REF = os.path.join(os.path.dirname(__file__), "ref", "ext_ref.txt")


def _lines():
    with open(REF) as f:
        return [ln.split() for ln in f if ln.strip() and not ln.startswith("#")]


LINES = _lines()
DESIGNS, VOLS = {}, {}
for w in LINES:
    if w[0] == "design":
        DESIGNS[int(w[1])] = trend(int(w[3])) if w[2] == "trend" else perron(w[3], [float(v) for v in w[5:5 + int(w[4])]])
    elif w[0] == "vol":
        nb = int(w[2])
        VOLS[int(w[1])] = volpath([float(v) for v in w[3:3 + nb]], [float(v) for v in w[3 + nb:4 + 2 * nb]])
# (the layers of the engine that the package does not reach, the transform, its roots and t_*, are not checked: vdval,
# vroots, vtstar, rdval, rroots, rtstar)
SKIPPED = ("vdval", "vroots", "vtstar", "rdval", "rroots", "rtstar")
CASES = [w for w in LINES if w[0] in ("vval", "vquant", "cval", "cquant", "rval", "rquant", "rcase", "rwarn")
         or (w[0] == "sval" and w[1].startswith("t3")) or (w[0] == "squant" and w[1] == "t3")]
ERRS = [" ".join(w[w.index("|") + 1:]) for w in LINES if w[0] == "err"]


@pytest.fixture(scope="module", autouse=True)
def _caches():
    yield
    dflimit.empty_caches()


def _vec(w, k):
    n = int(w[k])
    return np.array([float(v) for v in w[k + 1:k + 1 + n]]), k + 1 + n


def _var(w, k):
    """The variant of the regimes at w[k] (did vid nc c_1..c_nc start gls first: start 0 zero or 2 stationary, gls cbar or
    NaN for least squares, first 1 for detrend_first of the design): the design, c, the keywords, the next index."""
    did, vid, nc = int(w[k]), int(w[k + 1]), int(w[k + 2])
    c = [float(v) for v in w[k + 3:k + 3 + nc]]
    st, g, first = int(w[k + 3 + nc]), float(w[k + 4 + nc]), int(w[k + 5 + nc])
    D = detrend_first(DESIGNS[did]) if first == 1 else DESIGNS[did]
    kw = {"vol": VOLS[vid], "start": "stationary" if st == 2 else "zero"}
    if not math.isnan(g):
        kw["gls"] = g
    return D, c, kw, k + 6 + nc


def _id(w):
    return " ".join(w[:11] if w[0] in ("rval", "rquant", "rcase", "rwarn") else w[:6])


@pytest.mark.parametrize("w", CASES, ids=[_id(w) for w in CASES])
def test_reference(w):
    tag = w[0]
    if tag == "rval":
        D, c, kw, k = _var(w, 2)
        c0 = float(w[k]); x, k = _vec(w, k + 1); ref, k = _vec(w, k); tol = float(w[k])
        got = (dist_cdf if w[1] == "cdf" else dist_pdf)(D, x, c, c0=c0, **kw)
    elif tag == "rquant":
        D, c, kw, k = _var(w, 1)
        c0, p, ref, tol = (float(v) for v in w[k:k + 4])
        got = dist_quantile(D, p, c, c0=c0, **kw)
    elif tag == "rcase":
        # (the case strings SU, ST, DG with a volatility path; for icdf x holds the probabilities)
        f = {"cdf": ltratio_cdf, "pdf": ltratio_pdf, "icdf": ltratio_icdf}[w[1]]
        cas, V, nc = w[2], VOLS[int(w[3])], int(w[4])
        c = [float(v) for v in w[5:5 + nc]]
        k = 5 + nc
        c0 = float(w[k]); x, k = _vec(w, k + 1); ref, k = _vec(w, k); tol = float(w[k])
        got = f(x, cas, c, c0, vol=V)
    elif tag == "rwarn":
        # (dist_cdf at one point, which warns with a message that holds the fragment after the bar)
        D, c, kw, k = _var(w, 1)
        c0, x, ref, tol = (float(v) for v in w[k:k + 4])
        frag = " ".join(w[w.index("|") + 1:])
        with pytest.warns(RuntimeWarning, match=re.escape(frag)):
            got = dist_cdf(D, x, c, c0=c0, **kw)
    elif tag in ("vval", "vquant"):
        k0 = 2 if tag == "vval" else 1
        D, V, nc = DESIGNS[int(w[k0])], VOLS[int(w[k0 + 1])], int(w[k0 + 2])
        c = [float(v) for v in w[k0 + 3:k0 + 3 + nc]]
        k = k0 + 3 + nc
        if tag == "vval":
            c0 = float(w[k]); x, k = _vec(w, k + 1); ref, k = _vec(w, k); tol = float(w[k])
            got = (dist_cdf if w[1] == "cdf" else dist_pdf)(D, x, c, c0=c0, vol=V)
        else:
            c0, p, ref, tol = (float(v) for v in w[k:k + 4])
            got = dist_quantile(D, p, c, c0=c0, vol=V)
    elif tag == "cval":
        D = DESIGNS[int(w[2])]; st = ("zero", "fixed", "stationary")[int(w[3])]
        g0, c, c0, rho = (float(v) for v in w[4:8]); m = "auto" if rho == 0 else w[8]
        x, k = _vec(w, 9); ref, k = _vec(w, k); tol = float(w[k])
        got = (dist_cdf if w[1] == "cdf" else dist_pdf)(D, x, c, c0=c0, start=st, gamma0=g0, rho=rho, method=m)
    elif tag == "cquant":
        D = DESIGNS[int(w[1])]; c, c0, rho, p, ref, tol = (float(v) for v in w[4:10])
        got = dist_quantile(D, p, c, c0=c0, rho=rho)
    elif tag == "sval":
        D = DESIGNS[int(w[2])]; c, c0 = float(w[3]), float(w[4]); vid = int(w[5])
        x, k = _vec(w, 6); ref, k = _vec(w, k); tol = float(w[k])
        V = VOLS[vid] if vid else None
        got = (hegy_t3_cdf if w[1] == "t3cdf" else hegy_t3_pdf)(D, x, [c, c] if vid else c, c0=c0, vol=V)
    else:
        D = DESIGNS[int(w[2])]; c, p, ref, tol = (float(v) for v in w[3:7])
        got = hegy_t3_quantile(D, p, c)
    dev = float(np.max(np.abs(np.asarray(got) - np.asarray(ref))))
    assert dev <= tol, f"{' '.join(w[:8])}: deviation {dev:.2e} (tolerance {tol:g})"


def test_coverage():
    # vval, vquant, cval, cquant, sval, squant; rval, rquant, rcase, rwarn
    assert len(CASES) == 20 + 6 + 20 + 3 + 12 + 2 + 38 + 12 + 12 + 1
    # every tag of the file is checked or skipped on purpose (a new tag of dev/ext_ref.jl is noticed)
    assert {w[0] for w in LINES} <= {w[0] for w in CASES} | set(SKIPPED) | {"design", "vol", "err"}
    # the err lines: each fragment is checked by test_errors or test_errors_types
    assert len(ERRS) == 18
    assert set(ERRS) <= {frag for _, frag in ERRORS} | {"needs a volatility path"}


ERRORS = [
    (lambda: volpath([1.2]), "must lie in (0, 1)"),
    (lambda: volpath([0.5, 0.3]), "strictly increasing"),
    (lambda: volpath([0.5], [1.0]), "one standard deviation omega per regime"),
    (lambda: volpath([0.5], [1.0, -1.0]), "finite and positive"),
    (lambda: dist_cdf(trend(0), -2, 2, vol=volpath([0.3], [1, 3])), "must be at most 0"),
    # (the regime variants: a fixed start is refused under more than one regime, the stationary start is that of the
    # first regime, and GLS detrending, detrending first and a start exclude each other as without regimes)
    (lambda: dist_cdf(trend(0), -2, -5, vol=volpath([0.3], [1, 3]), start="fixed", gamma0=1),
     "fixed start (gamma0 != 0) is not computed"),
    (lambda: ltratio_cdf(-2, "DU", -5, g0=1, vol=volpath([0.3], [1, 3])), "fixed start (gamma0 != 0) is not computed"),
    (lambda: dist_cdf(trend(0), -2, [0, -5], vol=volpath([0.3], [1, 3]), start="stationary"),
     "needs c < 0 on the first regime"),
    (lambda: ltratio_cdf(-2, "SU", [0, -5], vol=volpath([0.3], [1, 3])), "needs c < 0 on the first regime"),
    (lambda: dist_cdf(detrend_first(trend(1)), -2, -5, vol=volpath([0.3], [1, 3]), gls=-13.5), "detrends first already"),
    (lambda: dist_cdf(trend(1), -2, -5, vol=volpath([0.3], [1, 3]), gls=-13.5, start="stationary"), "zero start only"),
    (lambda: dist_cdf(detrend_first(trend(1)), -2, -5, vol=volpath([0.3], [1, 3]), start="stationary"), "zero start only"),
    (lambda: dist_cdf(trend(0), -2, [0, -1, 0], vol=volpath([0.3], [1, 3])), "one value per regime"),
    (lambda: dist_cdf(trend(0), -2, -5, vol=volpath([0.3], [1, 3]), rho=0.5), "rho and vol together"),
    (lambda: dist_cdf(trend(0), -2, -5, rho=1.5), "rho must lie in [-1, 1]"),
    (lambda: dist_cdf(trend(1), -2, -5, rho=0.5, gls=-13.5), "GLS detrending"),
    (lambda: dist_cdf(trend(0), -2, -5, rho=0.5, gamma0=1, method="lines"), "zero or a stationary start"),
    (lambda: dist_cdf(trend(0), -2, -5, vol=volpath([0.3], [1, 3]), method="line"), "method must be 'auto' or 'paths'"),
    # (the methods of rho without it: the refusal and the message of DFLimit.jl, November 26, 2026)
    *[(f, "method must be 'auto', 'line' (the line integral with the explicit kernel) or 'paths' (the inversion on paths "
          "around the roots)")
      for m in ("lines", "mixture")
      for f in (lambda m=m: dist_cdf(trend(0), -2, -5, method=m), lambda m=m: dist_pdf(trend(0), -2, -5, method=m),
                lambda m=m: dist_quantile(trend(0), 0.05, -5, method=m))],
    (lambda: cadf_c0(-5, 0), "not be 0"),
    (lambda: predictive_c0(-5, 1.5), "not be 0"),
    (lambda: hegy_t3_cdf(trend(0), 1, 1), "must be at most 0"),
    (lambda: ltratio_cdf(-2, "DU", 0, vol=volpath([0.3], [1, 3]), engine="classical"), "engine classical"),
]


@pytest.mark.parametrize("call, frag", ERRORS)
def test_errors(call, frag):
    with pytest.raises(ValueError) as e:
        call()
    assert frag in str(e.value)


def test_errors_types():
    with pytest.raises(TypeError):
        dist_cdf(trend(0), -2, -5, vol=[0.3])
    # (c a vector without a volatility path: the message of DFLimit.jl and the MATLAB programs, a ValueError)
    for call in (lambda: dist_cdf(trend(0), -2, [0, -1]), lambda: ltratio_cdf(-2, "DU", [0, -1]),
                 lambda: dist_quantile(trend(0), 0.05, [0, -1])):
        with pytest.raises(ValueError, match=r"^(dist_cdf|ltratio_cdf|dist_quantile): c is a vector \(one value per regime\): "
                                             r"that needs a volatility path, vol = volpath\(breaks, omega\)$"):
            call()


def test_limiting_cases():
    xs = np.array([-4, -2.86, -1, 0.5])
    T0 = trend(0)
    # one regime is the engine itself; a dummy regime boundary at a constant variance is the engine's paths to rounding
    assert np.array_equal(dist_cdf(T0, xs, -10, vol=volpath([])), dist_cdf(T0, xs, -10))
    assert dist_quantile(trend(1), 0.05, 0, vol=volpath([])) == dist_quantile(trend(1), 0.05, 0)
    for c in (0, -10):
        assert np.max(np.abs(dist_cdf(T0, xs, c, vol=volpath([0.4])) - dist_cdf(T0, xs, c, method="paths"))) < 1e-12
        assert np.max(np.abs(dist_pdf(T0, xs, c, vol=volpath([0.4])) - dist_pdf(T0, xs, c, method="paths"))) < 1e-12
    # the regime variants at a dummy regime boundary, a constant variance and the same c on both regimes: the engine
    # itself (its paths) with the same detrending, design and start (GLS detrending, the statistic that detrends first,
    # the stationary start; within about 1e-14)
    V1 = volpath([0.4], [1, 1])
    for D, c, kw in ((trend(1), 0, {"gls": -13.5}), (perron("C", 0.4), -10, {"gls": -13.5}),
                     (detrend_first(trend(1)), 0, {}), (detrend_first(trend(2)), -10, {}),
                     (T0, -2, {"start": "stationary"}), (perron("A", 0.3), -10, {"start": "stationary"})):
        for f in (dist_cdf, dist_pdf):
            got, ref = f(D, xs, [c, c], vol=V1, **kw), f(D, xs, c, method="paths", **kw)
            assert np.max(np.abs(got - ref)) < 1e-10, (f.__name__, D.name, c, kw)
    # one regime with a fixed start is the engine itself (a fixed start is refused under more than one regime only)
    assert dist_cdf(T0, -2, -5, vol=volpath([]), gamma0=1) == dist_cdf(T0, -2, -5, gamma0=1)
    # the endpoints and the reflection of rho
    x4 = np.array([-3, -1, 0.5, 2])
    phi = np.array([0.5 * math.erfc(-v / math.sqrt(2)) for v in x4])
    assert np.max(np.abs(dist_cdf(T0, x4, -5, rho=0) - phi)) < 1e-15
    assert np.max(np.abs(dist_cdf(T0, x4, -5, rho=-1) - (1 - dist_cdf(T0, -x4, -5)))) < 1e-15
    assert np.max(np.abs(dist_cdf(T0, x4, -5, rho=-0.7, method="lines") - (1 - dist_cdf(T0, -x4, -5, rho=0.7, method="lines")))) < 1e-15
    assert abs(cadf_c0(-5, math.sqrt(0.5)) - 5) < 1e-13
    assert abs(predictive_c0(-5, -0.95, 1) - (-5 + 1 / 0.95)) < 1e-14
    # the classical functions take vol and rho (the general engine for the case's design)
    V2 = volpath([0.3], [1, 3])
    assert np.max(np.abs(ltratio_cdf([-3, -2], "DU", -5, vol=V2) - dist_cdf(T0, [-3, -2], -5, vol=V2))) < 1e-15
    q = ltratio_icdf(0.05, "DU", 0, vol=V2)
    assert abs(ltratio_cdf(q, "DU", 0, vol=V2) - 0.05) < 1e-9
    # (the case strings with their start and detrending: SU the stationary start of the first regime, DG GLS detrending at
    # cbar = -13.5)
    assert np.max(np.abs(ltratio_cdf([-3, -2], "SU", -5, vol=V2) - dist_cdf(T0, [-3, -2], -5, vol=V2, start="stationary"))) < 1e-15
    assert np.max(np.abs(ltratio_cdf([-3, -2], "DG", -5, vol=V2) - dist_cdf(trend(1), [-3, -2], -5, vol=V2, gls=-13.5))) < 1e-15
    assert abs(ltratio_cdf(1.645, "DU", 0, predictive_c0(0, -0.95), rho=-0.95) - 0.5788122) < 1e-7
    # non-finite points, the shapes
    assert list(dist_cdf(T0, [-math.inf, math.inf], -5, vol=V2)) == [0.0, 1.0]
    assert math.isnan(hegy_t3_cdf(T0, math.nan))
    assert isinstance(hegy_t3_quantile(T0, 0.05), float)


V3 = volpath([0.2, 0.7], [2, 1, 0.5])
UPPER = [(perron("B", 0.5), V3, [-5, 0, -10], {}), (perron("C", 0.4), volpath([0.3], [3, 1]), [-5, -5], {"start": "stationary"}),
         (perron("B", 0.3), volpath([0.2], [5, 1]), [-5, -20], {}), (perron("B", 0.5), V3, [-20, 0, -10], {"c0": 1})]


@pytest.mark.parametrize("D, V, c, kw", UPPER, ids=["B0.5", "C0.4-stationary", "B0.3", "B0.5-c0=1"])
def test_upper_tail_regimes(D, V, c, kw):
    # the outer contour under regimes: a root near the imaginary axis far out (the testset of test/extensions/regimes.jl
    # of DFLimit.jl: cases in which the thin step on the engine's 45-degree hyperbola lost 1.4e-6 to 3.8e-4 at every x,
    # and in the far upper tail the range check of dist_cdf refused the value: an early regime of high variance with a
    # break in the deterministic terms or a persistence that changes; the hyperbola under regimes starts at 70 degrees
    # since the C engine of dflimit-matlab ffeb48e)
    xs = np.array([-4, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 5.0])
    F = dist_cdf(D, xs, c, vol=V, **kw)
    assert np.max(np.abs(F - dist_cdf(D, xs, c, vol=V, verified=True, **kw))) < 1e-10
    assert np.all(np.diff(F) > -1e-11) and 1 - F[-1] < 1e-11
    assert np.min(dist_pdf(D, xs, c, vol=V, **kw)) > -1e-11
