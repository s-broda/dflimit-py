"""The general engine and the router: the stored values of the Julia general engine (ref/det_ref.txt, a subset of
c/test/det_ref.txt of dflimit-matlab, the reference of the C engine's own test dfl_test_det.c), and the interface of the
designs, the keyword engine, the warnings and the errors."""
import math
import pickle
import threading
import warnings

import numpy as np
import pytest

import dflimit
from dflimit import (Design, adfmax_cdf, coef_cdf, coef_icdf, design_from_matrix, detrend_first, dist_cdf, dist_moments,
                     dist_pdf, dist_quantile, fdesign, lin_cdf, lm_cdf, ltratio_cdf, ltratio_icdf, ltratio_lincdf,
                     ltratio_mom, ltratio_pdf, ltratio_region, ltrend, msb_cdf, perron, power_envelope, pt_cdf, trend,
                     v_cdf)
from helpers import det_check, det_designs, det_label, det_lines, det_route, det_value

DESIGNS = det_designs()
VAL = det_lines("val")
ROUTE = det_lines("route")


@pytest.fixture(scope="module", autouse=True)
def _caches():
    yield
    dflimit.empty_caches()


def _run(fun, r):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = fun(r, DESIGNS)
    dev, ok = det_check(got, r)
    assert ok, f"{det_label(r)}: {np.ravel(got)!r}, reference {r['want']!r} (largest deviation {dev:.2e}, tolerance {r['tol']:g})"
    if r["brittle"]:
        # (where the Julia engine warns that one evaluation is brittle, the C engine warns too)
        assert any(issubclass(x.category, RuntimeWarning) for x in w), f"{det_label(r)}: no warning"


@pytest.mark.parametrize("r", VAL, ids=det_label)
def test_general_engine(r):
    """the functions of the general engine reproduce the values of the Julia general engine"""
    _run(det_value, r)


@pytest.mark.parametrize("r", ROUTE, ids=det_label)
def test_router(r):
    """the router (engine auto, classical, general) reproduces the values of the Julia router"""
    _run(det_route, r)


def test_reference_coverage():
    """the stored values cover every constructor, the three starts, GLS detrending, a local null value, every function
    and companion, and the three settings of the router"""
    kinds = set()
    with open(__file__.replace("test_general.py", "ref/det_ref.txt")) as fh:
        for line in fh:
            v = line.split()
            if v and v[0] == "design":
                kinds.add(v[2])
    assert kinds >= {"trend", "perron", "fdesign", "ltrend", "matrix", "first", "raw"}
    fns = {r["fn"] for r in VAL}
    assert fns >= {"cdf", "pdf", "quant", "coefcdf", "coefpdf", "coefq", "coefgp", "mom", "lpow", "lincdf", "linpdf",
                   "linq", "adfcdf", "adfpdf", "adfq", "env", "ptcdf", "ptpdf", "ptq", "ptpow", "ptenv", "lmcdf", "lmq",
                   "lmpow", "mzacdf", "mzaq", "msbcdf", "msbq"}
    assert {r["st"] for r in VAL} == {0, 1, 2}
    assert any(not math.isnan(r["cb"]) for r in VAL) and any(r["c0"] != 0 for r in VAL) and any(r["ver"] for r in VAL)
    assert {r["e"] for r in ROUTE} == {0, 1, 2} and {r["spec"] for r in ROUTE} == {"case", "design"}


# ---------------------------------------------------------------------------------------------------------------------
# the designs
# ---------------------------------------------------------------------------------------------------------------------
def test_designs():
    D = perron("C", [0.7, 0.3])
    assert repr(D) == "Design('Perron model C, lambda = 0.3, 0.7': K = 6 regressors, 3 pieces of degree 1)"
    assert D.K == 6 and D.npieces == 3 and D.degree == 1 and not D.loc and not D.first
    np.testing.assert_array_equal(D.br, [0, 0.3, 0.7, 1])
    assert D.coef.shape == (3, 2, 6)
    np.testing.assert_array_equal(D.value(0.5), [1, 0.5, 1, 0.2, 0, 0])
    assert D.value([[0.1, 0.9]]).shape == (1, 2, 6)
    X = D.regmat(10)
    assert X.shape == (10, 6)
    np.testing.assert_allclose(X[:, 1], np.arange(1, 11) / 10)
    with pytest.raises(ValueError):
        D.br[0] = 1                                         # (read-only)
    with pytest.raises(AttributeError, match="immutable"):
        D.K = 3
    assert trend(1) == Design([0, 1], [[[1.0, 0.0], [0.0, 1.0]]]) and trend(1) != trend(2)
    assert len({trend(1), trend(1), trend(2)}) == 2
    E = pickle.loads(pickle.dumps(D))
    assert E == D and E.name == D.name
    assert repr(trend(-1)) == "Design('no deterministic terms (DK)': K = 0 regressors, 1 piece of degree 0)"
    assert trend(-1).value(0.5).shape == (0,)
    L = ltrend(3)
    assert L.loc and L.K == 4
    np.testing.assert_allclose(L.value(1.0), [1, math.sqrt(3), math.sqrt(5), math.sqrt(7)], rtol=1e-12)
    F = detrend_first(perron("B", 0.3))
    assert F.first and F.name.endswith("(detrended first)")
    assert F != perron("B", 0.3)


def test_design_errors():
    cases = [
        (lambda: trend(-2), ValueError, "p >= -1"),
        (lambda: trend(1.5), TypeError, "integer"),
        (lambda: ltrend(-1), ValueError, "p >= 0"),
        (lambda: perron("D", 0.3), ValueError, "model must be"),
        (lambda: perron("A", 1.2), ValueError, "0 < lambda_1"),
        (lambda: perron("A", [0.3, 0.3]), ValueError, "0 < lambda_1"),
        (lambda: perron("A", []), ValueError, "at least one"),
        (lambda: fdesign(lambda r: [1.0, r], K=3), ValueError, "returned 2 values"),
        (lambda: fdesign(lambda r: [1.0, r], [1.5]), ValueError, "breaks must lie"),
        (lambda: fdesign(lambda r: [1.0, math.nan]), ValueError, "not finite"),
        (lambda: fdesign(lambda r: [abs(r - 0.5)]), ValueError, "do not fall below"),
        (lambda: fdesign(3), TypeError, "callable"),
        (lambda: design_from_matrix(np.ones(5)), ValueError, "matrix"),
        (lambda: design_from_matrix(np.column_stack([np.ones(100), np.arange(100.0) == 50])), ValueError, "pulse"),
        (lambda: detrend_first("DT"), TypeError, "Design"),
        (lambda: Design([0, 0.5], [[[1.0]]]), ValueError, "break points"),
        (lambda: Design([0, 1], np.ones((2, 1, 1))), ValueError, "shape"),
        (lambda: Design([0, 1], [[[math.inf]]]), ValueError, "finite"),
        (lambda: Design([0, 1], [[[1.0]]], loc=1), TypeError, "True or False"),
        (lambda: dist_moments(detrend_first(trend(1))), ValueError, "not implemented"),
        (lambda: dist_cdf(trend(1), -2.0, 1.0, start="stationary"), ValueError, "needs c < 0"),
        (lambda: dist_cdf("DT", -2.0), TypeError, "must be a Design"),
        (lambda: dist_cdf(trend(1), -2.0, start="random"), ValueError, "start must be"),
        (lambda: dist_cdf(trend(1), -2.0, start="zero", gamma0=1.0), ValueError, "excludes a fixed start"),
        (lambda: dist_quantile(trend(1), 1.5), ValueError, r"\(0, 1\)"),
        (lambda: lm_cdf(trend(0), -2.0), ValueError, "trend function"),
    ]
    for fun, exc, msg in cases:
        with pytest.raises(exc, match=msg):
            fun()


def test_fdesign_callback():
    """the function of a design is called on the calling thread; it may call the package; its exceptions pass"""
    seen = []

    def f(r):
        seen.append(dflimit.ltratio_threads())              # (an engine call inside the callback: no deadlock)
        return [1.0, r]

    D = fdesign(f)
    assert len(seen) > 10 and D.K == 2
    assert dist_cdf(D, -3.0) == pytest.approx(dist_cdf(trend(1), -3.0), abs=1e-10)

    def g(r):
        raise KeyError("from the design")

    with pytest.raises(KeyError, match="from the design"):
        fdesign(g, K=1)


def test_design_from_matrix():
    t = np.arange(1, 501)
    X = np.column_stack([np.ones(500), t, t > 150, np.maximum(t - 150, 0)])
    D = design_from_matrix(X)                                   # (the break detected)
    np.testing.assert_allclose(D.br[[0, -1]], [0, 1])
    assert D.br[1] == pytest.approx(0.3)
    E = design_from_matrix(X, [150])
    assert dist_cdf(D, -4.0) == pytest.approx(dist_cdf(perron("C", 0.3), -4.0), abs=1e-9)
    assert dist_cdf(E, -4.0) == pytest.approx(dist_cdf(perron("C", 0.3), -4.0), abs=1e-9)
    with pytest.warns(RuntimeWarning, match="linearly dependent"):
        design_from_matrix(np.column_stack([X, 2 * X[:, 1]]))


# ---------------------------------------------------------------------------------------------------------------------
# the router and the keyword engine
# ---------------------------------------------------------------------------------------------------------------------
def same(a, b):
    """the same values (to rounding: with threads the order of summation in the engine can differ by one ulp)"""
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-300)


def test_case_strings_keep_the_classical_engine():
    """with a case string and no engine the values are the classical engine's (also through the C router's classical
    branch, and for the design of the case with engine classical)"""
    from dflimit import _engine
    x = np.array([-3.5, -2.0, -0.5, 1.0])
    for cas, D, c, c0, g0 in (("DU", trend(0), -5.0, 0.0, 0.0), ("DK", trend(-1), 2.0, 2.0, 0.0),
                              ("DT", trend(1), -5.0, -3.0, 0.0), ("SU", trend(0), -10.0, 0.0, 0.0),
                              ("DU", trend(0), -5.0, 0.0, 1.0)):
        F = ltratio_cdf(x, cas, c, c0, g0)
        same(F, ltratio_cdf(x, cas, c, c0, g0, engine="classical"))
        start = "stationary" if cas[0] == "S" else None
        same(F, ltratio_cdf(x, D, c, c0, g0, engine="classical", start=start))
        model, stat = "KUT".index(cas[1]) + 1, int(cas[0] == "S")
        Fr = np.frombuffer(_engine.route(1, (None, model, 2 if stat else int(g0 != 0), g0, math.nan), x.tobytes(), c, c0, 1,
                                         0, 0, 0.0)[0])
        same(Fr, F)
    same(ltratio_icdf([0.05, 0.5], "DT"), ltratio_icdf([0.05, 0.5], trend(1), engine="classical"))
    same(ltratio_mom(3, "DU", -5), ltratio_mom(3, trend(0), -5, engine="classical"))
    same(coef_cdf(-14.0, "DU"), coef_cdf(-14.0, trend(0), engine="classical"))
    same(adfmax_cdf(-2.5, "DU"), adfmax_cdf(-2.5, trend(0), engine="classical"))
    same(ltratio_lincdf(-0.26, -3.5, "DK"), ltratio_lincdf(-0.26, -3.5, trend(-1), engine="classical"))
    same(power_envelope(-7.0), power_envelope(-7.0, trend(-1), engine="classical"))
    bnd = {"type": "line", "a": -3.5, "b": -0.2639}
    same(ltratio_region(bnd), ltratio_region(bnd, trend(-1), engine="classical"))


def test_the_engines_agree():
    """the three settings give the same values to the engines' accuracy, and a design of a classical case gives the
    values of its case string"""
    x = [-3.0, -1.5]
    for eng in ("auto", "classical", "general"):
        np.testing.assert_allclose(ltratio_cdf(x, "DT", -5, engine=eng), ltratio_cdf(x, "DT", -5), atol=1e-9)
        np.testing.assert_allclose(ltratio_pdf(x, trend(1), -5, engine=eng), ltratio_pdf(x, "DT", -5), atol=1e-9)
    same(ltratio_cdf(x, trend(1), -5), ltratio_cdf(x, "DT", -5, engine="auto"))
    np.testing.assert_allclose(ltratio_cdf(x, trend(0), -10, start="stationary", engine="general"),
                               ltratio_cdf(x, "SU", -10), atol=1e-9)
    np.testing.assert_allclose(ltratio_cdf(x, trend(1), -5, gls=-13.5, engine="general"), ltratio_cdf(x, "DG", -5),
                               atol=1e-9)
    np.testing.assert_allclose(ltratio_cdf(x, trend(0), -5, g0=1.0, engine="general"), ltratio_cdf(x, "DU", -5, g0=1.0),
                               atol=1e-9)
    assert coef_icdf(0.05, trend(0), engine="general") == pytest.approx(coef_icdf(0.05, "DU"), abs=1e-7)
    assert ltratio_region({"type": "parabola", "x": -1.94}, trend(-1), engine="general") == pytest.approx(
        ltratio_cdf(-1.94, "DK"), abs=1e-9)
    assert v_cdf(trend(-1), 0.5) == pytest.approx(ltratio_lincdf(0.5, math.nan, "DK"), abs=1e-9)


def test_a_design_beyond_the_cases():
    """a design beyond the classical cases has the general engine (engine None or "auto"), the same as dist_cdf"""
    D = perron("A", 0.5)
    x = np.array([-4.5, -3.5])
    F = dist_cdf(D, x, -5)
    same(ltratio_cdf(x, D, -5), F)
    same(ltratio_cdf(x, D, -5, engine="general"), F)
    np.testing.assert_allclose(ltratio_cdf(ltratio_icdf([0.05, 0.1], D, -5), D, -5), [0.05, 0.1], atol=1e-9)
    r = ltratio_icdf(0.05, D, info=True)
    assert set(r) == {"q", "F"} and r.F == pytest.approx(0.05, abs=1e-9)
    m = ltratio_mom(2, D, info=True)
    np.testing.assert_allclose(m.m, dist_moments(D, p=2), rtol=1e-12)
    assert np.all(np.isnan(m.err))
    assert ltratio_cdf(-2.0, trend(1), -200.0) == pytest.approx(dist_cdf(trend(1), -2.0, -200.0), abs=1e-9)


def test_router_errors():
    D = perron("C", 0.3)
    cases = [
        (lambda: ltratio_cdf(0, "DU", engine="fast"), ValueError, "engine must be"),
        (lambda: ltratio_cdf(0, "DU", engine=1), TypeError, "engine must be"),
        (lambda: ltratio_cdf(0, D, engine="classical"), ValueError, "the classical engine has the cases"),
        (lambda: ltratio_cdf(0, D, method="double"), ValueError, "names a representation"),
        (lambda: ltratio_cdf(0, "DU", method="double", engine="general"), ValueError, "names a representation"),
        (lambda: ltratio_cdf(0, "DU", start="stationary"), ValueError, "apply to a design"),
        (lambda: ltratio_cdf(0, "DU", gls=-7), ValueError, "apply to a design"),
        (lambda: ltratio_cdf(0, D, g0=1, start="zero"), ValueError, "excludes a fixed start"),
        (lambda: ltratio_cdf(0, D, g0=1, start="stationary"), ValueError, "exclude each other"),
        (lambda: ltratio_cdf(0, "DU", -600, engine="auto"), ValueError, "at least -550"),
        (lambda: ltratio_icdf(0.05, D, verified=True), ValueError, "classical engine's"),
        (lambda: ltratio_icdf(1.5, D), ValueError, r"\(0, 1\)"),
        (lambda: ltratio_icdf(1.5, "DT", 3, engine="auto"), ValueError, r"\(0, 1\)"),
        (lambda: ltratio_mom(2, "DG", engine="classical"), ValueError, "not available"),
        (lambda: coef_icdf(0.05, D, method="single"), ValueError, "names a representation"),
        (lambda: adfmax_cdf(-2, "DK", engine="auto"), ValueError, "the case must be one of DU, DT, SU, ST"),
        (lambda: adfmax_cdf(-2, trend(-1), engine="classical"), ValueError, "ADFmax of the classical engine"),
        (lambda: ltratio_lincdf(0, -2, D, engine="classical"), ValueError, "the linear form of the classical engine"),
        (lambda: ltratio_lincdf(0, -2, "DK", g0=1), ValueError, "apply to a design"),
        (lambda: power_envelope(-5, D, engine="classical"), ValueError, "the envelope of the classical engine"),
        (lambda: ltratio_region({"type": "wap", "k": 3.6658, "omega": 10}, D), ValueError, "not available by the general"),
        (lambda: ltratio_region({"type": "line", "a": -3.5, "b": 0, "v0": 0.1}, "DK", engine="general"), ValueError,
         "classical engine's"),
        (lambda: ltratio_region({"type": "line", "a": -3.5, "b": 0}, D, engine="classical"), ValueError, "case DK only"),
    ]
    for fun, exc, msg in cases:
        with pytest.raises(exc, match=msg):
            fun()


def test_shapes_and_non_finite_points():
    D = perron("A", 0.5)
    x = np.array([[-math.inf, math.nan], [-3.0, math.inf]])
    for f, top in ((dist_cdf, 1), (dist_pdf, 0)):
        v = f(D, x)
        assert v.shape == (2, 2) and v[0, 0] == 0 and math.isnan(v[0, 1]) and v[1, 1] == top
    for v, top in ((lin_cdf(D, x, -2.0), 1), (v_cdf(D, x), 1), (pt_cdf(D, x, -7.0), 1), (msb_cdf(trend(1), x), 1),
                   (ltratio_cdf(x, D), 1), (ltratio_pdf(x, D), 0), (lm_cdf(trend(1), x), 1)):
        assert v.shape == (2, 2) and v[0, 0] == 0 and math.isnan(v[0, 1]) and v[1, 1] == top
    for v in (dist_cdf(D, -3), ltratio_cdf(-3, D), dist_quantile(D, 0.05), coef_cdf(-10, D), adfmax_cdf(-3, D),
              ltratio_lincdf(0.0, -2.0, D), power_envelope(1.0, D), ltratio_region({"type": "line", "a": -2, "b": 0}, D)):
        assert type(v) is float
    assert dist_cdf(D, []).shape == (0,)
    assert ltratio_icdf(np.array([[0.05, 0.1]]), D).shape == (1, 2)
    E = power_envelope([math.nan, 1.0], D)                  # (the point-optimal envelope of the design: nan, level)
    assert math.isnan(E[0]) and E[1] == 0.05


def test_warnings_of_the_engine():
    """the general engine's warnings are RuntimeWarnings reported at the caller, once per call, and do not outlive it"""
    import os
    x = [-3.5, -2.5, -1.8, -1.0, 0.3]
    with pytest.warns(RuntimeWarning, match="vertical line") as rec:
        dist_cdf(trend(1), x, -5, gls=-2)
    assert len(rec) == 1 and os.path.basename(rec[0].filename) == os.path.basename(__file__)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ltratio_cdf(x, "DU", -5, engine="classical")         # (the classical branch of the router after a warning)
        ltratio_cdf(x, "DU", -5, engine="auto")
        design_from_matrix(trend(1).regmat(100))
        dist_cdf(trend(1), x, 0)
    with pytest.warns(RuntimeWarning, match="vertical line"):
        ltratio_cdf(x, trend(1), -5, gls=-2)                 # (through the router)
    with pytest.raises(ValueError):
        ltratio_icdf([0.05, 1.5], "DT", 3, engine="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dist_cdf(trend(1), x, 0)


def test_brittle_regimes_of_the_classical_engine():
    """where the router takes the classical engine in a regime in which its one evaluation is brittle, the warning of
    the classical engine is given (and not where the general engine computes)"""
    with pytest.warns(RuntimeWarning, match="c0 > 0 far from the unit root"):
        Fc = ltratio_cdf(-2.0, "DU", -20, c0=1, g0=0.5, engine="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        F = ltratio_cdf(-2.0, "DU", -20, c0=1, g0=0.5, engine="general")
    assert F == pytest.approx(Fc, abs=1e-9)


def test_verified():
    D = perron("C", 0.3)
    np.testing.assert_allclose(dist_cdf(D, [-4.0, -3.0], -5, verified=True), dist_cdf(D, [-4.0, -3.0], -5), atol=1e-9)
    np.testing.assert_allclose(coef_cdf([-20.0, -5.0], D, -5, verified=True), coef_cdf([-20.0, -5.0], D, -5), atol=1e-9)
    np.testing.assert_allclose(ltratio_cdf([-4.0], D, -5, verified=True), dist_cdf(D, [-4.0], -5), atol=1e-9)


def test_threads_and_concurrent_calls():
    D = perron("B", 0.4)
    x = [-4.0, -3.0, -2.0]
    ref = dist_cdf(D, x, -5)
    old = dflimit.ltratio_threads()
    dflimit.ltratio_threads(1)
    one = dist_cdf(D, x, -5)
    dflimit.ltratio_threads(old)
    np.testing.assert_allclose(one, ref, rtol=1e-13)
    out = [None] * 3

    def work(k):
        out[k] = dist_cdf(D, x, -5)

    th = [threading.Thread(target=work, args=(k,)) for k in range(len(out))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for o in out:
        np.testing.assert_allclose(o, ref, rtol=1e-13)
    dflimit.empty_caches()
    np.testing.assert_allclose(dist_cdf(D, x, -5), ref, rtol=1e-12)
