"""The extensions of the limit (volatility and persistence regimes, a normal component, the first seasonal t-ratio)
against the values of the Julia engine DFLimit.jl (ref/ext_ref.txt, written by dev/ext_ref.jl of DFLimit.jl; the second
seasonal t-ratio and the F statistic are not part of the release 0.5.0), the argument errors and the
limiting cases (tests/testthat/test-extensions.R of the R package)."""
import math
import os

import numpy as np
import pytest

import dflimit
from dflimit import (cadf_c0, dist_cdf, dist_pdf, dist_quantile, hegy_t3_cdf, hegy_t3_pdf, hegy_t3_quantile, ltratio_cdf,
                     ltratio_icdf, perron, predictive_c0, trend, volpath)

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
CASES = [w for w in LINES if w[0] in ("vval", "vquant", "cval", "cquant")
         or (w[0] == "sval" and w[1].startswith("t3")) or (w[0] == "squant" and w[1] == "t3")]


@pytest.fixture(scope="module", autouse=True)
def _caches():
    yield
    dflimit.empty_caches()


def _vec(w, k):
    n = int(w[k])
    return np.array([float(v) for v in w[k + 1:k + 1 + n]]), k + 1 + n


@pytest.mark.parametrize("w", CASES, ids=[" ".join(w[:6]) for w in CASES])
def test_reference(w):
    tag = w[0]
    if tag in ("vval", "vquant"):
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
    assert len(CASES) == 20 + 6 + 20 + 3 + 12 + 2


@pytest.mark.parametrize("call, frag", [
    (lambda: volpath([1.2]), "must lie in (0, 1)"),
    (lambda: volpath([0.5, 0.3]), "strictly increasing"),
    (lambda: volpath([0.5], [1.0]), "one standard deviation omega per regime"),
    (lambda: volpath([0.5], [1.0, -1.0]), "finite and positive"),
    (lambda: dist_cdf(trend(0), -2, 2, vol=volpath([0.3], [1, 3])), "must be at most 0"),
    (lambda: dist_cdf(trend(0), -2, -5, vol=volpath([0.3], [1, 3]), start="stationary"), "zero start only"),
    (lambda: dist_cdf(trend(1), -2, -5, vol=volpath([0.3], [1, 3]), gls=-13.5), "GLS detrending"),
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
])
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
    assert abs(ltratio_cdf(1.645, "DU", 0, predictive_c0(0, -0.95), rho=-0.95) - 0.5788122) < 1e-7
    # non-finite points, the shapes
    assert list(dist_cdf(T0, [-math.inf, math.inf], -5, vol=V2)) == [0.0, 1.0]
    assert math.isnan(hegy_t3_cdf(T0, math.nan))
    assert isinstance(hegy_t3_quantile(T0, 0.05), float)
