"""The general engine and the router: the stored values of the Julia general engine (ref/det_ref.txt, a subset of
c/test/det_ref.txt of dflimit-matlab, the reference of the C engine's own test dfl_test_det.c), and the interface of the
designs, the keyword engine, the warnings and the errors."""
import math
import pickle
import re
import threading
import warnings

import numpy as np
import pytest

import dflimit
from dflimit import (Design, adfmax_cdf, adfmax_icdf, adfmax_pdf, adfmax_quantile, coef_cdf, coef_icdf, design_from_matrix, detrend_first, dist_cdf, dist_moments,
                     dist_pdf, dist_quantile, fdesign, lin_cdf, lin_pdf, lm_cdf, lm_quantile, ltratio_cdf, ltratio_icdf, ltratio_lincdf,
                     ltratio_mom, ltratio_pdf, ltratio_region, ltrend, msb_cdf, perron, power_envelope, pt_cdf,
                     pt_envelope, pt_quantile, trend, v_cdf, v_pdf, coef_cdf_gp, coef_quantile, lin_quantile)
from helpers import det_check, det_designs, det_label, det_lines, det_route, det_value

DESIGNS = det_designs()
VAL = det_lines("val")
ROUTE = det_lines("route")
ERR = det_lines("err")
RERR = det_lines("rerr")


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


@pytest.mark.parametrize("r", ERR, ids=det_label)
def test_general_engine_refuses(r):
    """the general engine refuses what it does not compute (p and levels outside (0, 1), the LM statistic above c = 10,
    the point-optimal statistic where its grid does not resolve the transform and beyond the range of q, the design
    whose roots run off), with an error that says so"""
    with pytest.raises(ValueError, match=re.escape(r["text"])):
        det_value(r, DESIGNS)


@pytest.mark.parametrize("r", RERR, ids=det_label)
def test_router_refuses(r):
    """the router refuses a fixed start with the classical engine's ADFmax and regions, gls with a case code, p and
    levels outside (0, 1)"""
    with pytest.raises(ValueError, match=re.escape(r["text"])):
        det_route(r, DESIGNS)


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
    assert {r["fn"] for r in ERR} >= {"lmcdf", "lmq", "lmpow", "cdf", "ptcdf", "linq", "adfq", "ptq", "quant", "coefq", "msbq",
                                      "mzaq", "ptpow", "env"}
    assert any(r["fn"] == "adf" and r["st"] == 1 for r in ROUTE) and any(r["fn"] == "lmcdf" and r["c"] == 5 for r in VAL)


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


def test_case_strings_classical_and_default():
    """(i) engine="classical" for a case string is the classical engine (also through the C router's classical branch,
    and for the design of the case with engine classical), the values of the versions before 0.5.0; (ii) the default,
    engine="auto" also for a case string, agrees with it to the accuracy of the reference values (1e-9) on the same calls"""
    from dflimit import _engine
    x = np.array([-3.5, -2.0, -0.5, 1.0])

    def close(a, b):
        assert np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) < 1e-9

    for cas, D, c, c0, g0 in (("DU", trend(0), -5.0, 0.0, 0.0), ("DK", trend(-1), 2.0, 2.0, 0.0),
                              ("DT", trend(1), -5.0, -3.0, 0.0), ("SU", trend(0), -10.0, 0.0, 0.0),
                              ("DU", trend(0), -5.0, 0.0, 1.0)):
        F = ltratio_cdf(x, cas, c, c0, g0, engine="classical")
        start = "stationary" if cas[0] == "S" else None
        same(F, ltratio_cdf(x, D, c, c0, g0, engine="classical", start=start))
        model, stat = "KUT".index(cas[1]) + 1, int(cas[0] == "S")
        Fr = np.frombuffer(_engine.route(1, (None, model, 2 if stat else int(g0 != 0), g0, math.nan), x.tobytes(), c, c0, 1,
                                         0, 0, 0.0)[0])
        same(Fr, F)
        close(ltratio_cdf(x, cas, c, c0, g0), F)
        same(ltratio_cdf(x, cas, c, c0, g0), ltratio_cdf(x, cas, c, c0, g0, engine="auto"))
    pairs = [(lambda e: ltratio_icdf([0.05, 0.5], "DT", engine=e), lambda: ltratio_icdf([0.05, 0.5], trend(1), engine="classical")),
             (lambda e: ltratio_mom(3, "DU", -5, engine=e), lambda: ltratio_mom(3, trend(0), -5, engine="classical")),
             (lambda e: coef_cdf(-14.0, "DU", engine=e), lambda: coef_cdf(-14.0, trend(0), engine="classical")),
             (lambda e: adfmax_cdf(-2.5, "DU", engine=e), lambda: adfmax_cdf(-2.5, trend(0), engine="classical")),
             (lambda e: ltratio_lincdf(-0.26, -3.5, "DK", engine=e), lambda: ltratio_lincdf(-0.26, -3.5, trend(-1), engine="classical")),
             (lambda e: power_envelope(-7.0, engine=e), lambda: power_envelope(-7.0, trend(-1), engine="classical"))]
    for f, g in pairs:
        same(f("classical"), g())
        v = np.asarray(f("auto"), float); w = np.asarray(g(), float)
        close(v / np.where(np.abs(w) > 1, w, 1), w / np.where(np.abs(w) > 1, w, 1))       # (the moments relative)
    bnd = {"type": "line", "a": -3.5, "b": -0.2639}
    same(ltratio_region(bnd, engine="classical"), ltratio_region(bnd, trend(-1), engine="classical"))
    close(ltratio_region(bnd), ltratio_region(bnd, engine="classical"))


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
        (lambda: ltratio_cdf(0, "DU", start="stationary"), ValueError, "start applies to a design"),
        (lambda: ltratio_cdf(0, "DU", gls=-7), ValueError, "gls applies to a design"),
        (lambda: ltratio_cdf(0, D, g0=1, start="zero"), ValueError, "excludes a fixed start"),
        (lambda: ltratio_cdf(0, D, g0=1, start="stationary"), ValueError, "exclude each other"),
        (lambda: ltratio_cdf(0, "DU", -600, engine="auto"), ValueError, "at least -550"),
        (lambda: ltratio_icdf(0.05, D, verified=True), ValueError, "classical engine's"),
        (lambda: ltratio_icdf(1.5, D), ValueError, r"\(0, 1\)"),
        (lambda: ltratio_icdf(1.5, "DT", 3, engine="auto"), ValueError, r"\(0, 1\)"),
        (lambda: ltratio_mom(2, "DG", engine="classical"), ValueError, "not available"),
        (lambda: coef_icdf(0.05, D, method="single"), ValueError, "names a representation"),
        (lambda: adfmax_cdf(-2, "DK", engine="auto"), ValueError, "the case must be one of DU, DT, SU, ST"),
        (lambda: adfmax_cdf(-2, trend(-1), engine="classical"), ValueError, "the classical engine's ADFmax"),
        (lambda: adfmax_cdf(-2, "DU", g0=1.0, engine="classical"), ValueError, "with a zero or a stationary start only"),
        (lambda: adfmax_icdf(0.05, "DU", g0=1.0, engine="classical"), ValueError, "with a zero or a stationary start"),
        (lambda: ltratio_lincdf(0, -2, D, engine="classical"), ValueError, "the linear form of the classical engine"),
        (lambda: ltratio_lincdf(0, -2, "DK", g0=1, engine="classical"), ValueError, "zero start only"),
        (lambda: ltratio_lincdf(0, -2, "DG", g0=1, engine="auto"), ValueError, "case DG has a zero start"),
        (lambda: ltratio_lincdf(0, -2, "DK", start="fixed"), ValueError, "apply to a design"),
        (lambda: ltratio_region({"type": "line", "a": -3.5, "b": 0}, "DK", g0=1.0, engine="classical"), ValueError, "case DK only"),
        (lambda: power_envelope(-5, "DK", alpha=1.5, engine="auto"), ValueError, r"must lie in \(0, 1\)"),
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


# ---------------------------------------------------------------------------------------------------------------------
# the repairs of the C engine of November 2026
# ---------------------------------------------------------------------------------------------------------------------
def test_adfmax_with_a_fixed_start():
    """ADFmax with a fixed start goes to the general engine through the router (a case code, a design), and agrees with
    the general engine forced; the classical engine refuses it"""
    x = [-3.0, -2.4, -1.5]
    G = adfmax_cdf(x, trend(0), -5, g0=1.5, engine="general")
    same(adfmax_cdf(x, "DU", -5, g0=1.5, engine="auto"), G)
    same(adfmax_cdf(x, trend(0), -5, g0=1.5), G)
    same(adfmax_cdf(x, trend(0), -5, g0=1.5, start="fixed"), G)
    zero = adfmax_cdf(x, "DU", -5)
    assert np.max(np.abs(G - zero)) > 1e-3                  # (not the zero start, which the router had computed)
    np.testing.assert_allclose(adfmax_pdf(x, "DT", -5, g0=1.0, engine="auto"),
                               adfmax_pdf(x, trend(1), -5, g0=1.0, engine="general"), rtol=1e-14)
    q = adfmax_icdf(0.05, "DU", -5, g0=1.5, engine="auto")
    assert q == pytest.approx(adfmax_quantile(trend(0), 0.05, -5, start="fixed", gamma0=1.5), abs=1e-12)
    assert adfmax_cdf(q, trend(0), -5, g0=1.5) == pytest.approx(0.05, abs=1e-9)


def test_a_fixed_start_of_case_dk():
    """the regions and the linear form of case DK with a fixed start take the general engine (the classical engine has
    the zero start only)"""
    bnd = {"type": "line", "a": -3.5, "b": -0.2639}
    P = ltratio_region(bnd, "DK", -5, g0=1.5, engine="auto")
    assert P == pytest.approx(ltratio_region(bnd, trend(-1), -5, g0=1.5), abs=1e-14)
    assert P == pytest.approx(lin_cdf(trend(-1), -0.2639, -3.5, -5, gamma0=1.5), abs=1e-14)
    assert abs(P - ltratio_region(bnd, "DK", -5)) > 0.1
    assert ltratio_lincdf(-0.2639, -3.5, "DK", -5, g0=1.5, engine="general") == pytest.approx(P, abs=1e-14)


def test_the_law_of_v_off_its_support():
    """P(V <= b) and its density are 0 at b <= 0"""
    for D in (trend(-1), trend(1), perron("A", 0.5)):
        np.testing.assert_array_equal(v_cdf(D, [-1.0, -1e-3, 0.0]), 0)
        np.testing.assert_array_equal(v_pdf(D, [-1.0, 0.0]), 0)
        np.testing.assert_array_equal(lin_cdf(D, [-0.5, 0.0], math.nan, -5), 0)
        np.testing.assert_array_equal(lin_pdf(D, [-0.5, 0.0], math.nan, -5), 0)
    np.testing.assert_array_equal(ltratio_lincdf([-0.5, 0.0], math.nan, perron("A", 0.5), -5), 0)


def test_refusals():
    """the refusals of the general engine are ValueErrors with messages that say what to change"""
    # (the design (1 + r, 1{r <= 0.4}), refused at every c: the err line of the reference file)
    with pytest.raises(ValueError, match="at most 10"):
        lm_cdf(trend(1), -0.5, 11)
    with pytest.raises(ValueError, match="at most 10"):
        lm_quantile(trend(1), 0.05, c=10.5)
    with pytest.raises(ValueError, match=r"does not resolve.*needs N >= (\d+)") as e:
        pt_cdf(trend(1), [4.0, 10.0], -13.5, 3.8)
    N = int(re.search(r"needs N >= (\d+)", str(e.value)).group(1))
    F = pt_cdf(trend(1), [4.0, 10.0, 30.0], -13.5, 3.8, N=N)       # (the grid it names serves)
    assert np.all((F >= 0) & (F <= 1)) and np.all(np.diff(F) > 0)
    with pytest.raises(ValueError, match="beyond the range"):
        pt_cdf(trend(1), 2000.0, -13.5)
    assert pt_cdf(trend(1), 999.0, -13.5) == pytest.approx(1, abs=1e-9)
    for f in (lambda: pt_quantile(trend(1), 1.0, -13.5), lambda: dist_quantile(trend(1), math.nan),
              lambda: pt_envelope(trend(1), [-5.0], level=0.0), lambda: adfmax_quantile(trend(0), -0.1)):
        with pytest.raises(ValueError, match=r"must lie in \(0, 1\)"):
            f()


def test_the_lm_statistic_above_the_unit_root():
    """the LM statistic above the unit root is a distribution function (on the path of V alone; it exceeded 1 on the
    vertical line)"""
    x = np.linspace(-1.5, -0.02, 12)
    for c in (2.5, 5.0, 10.0):
        F = lm_cdf(trend(1), x, c)
        assert np.all((F >= 0) & (F <= 1 + 1e-12)) and np.all(np.diff(F) >= -1e-12)



def test_nonfinite_arguments():
    """the general engine and the router at non-finite arguments: a limit, or an error that names the argument (none of
    them may hang: coef_cdf_gp at y = +-inf, the searches from x0 = +-inf and the point-optimal statistic at c = +-inf did
    not return in the C engine)"""
    D = trend(0)
    np.testing.assert_array_equal(coef_cdf_gp(D, [math.inf, -math.inf, math.nan], -5), [1, 0, math.nan])
    for f, text in [(lambda: dist_quantile(D, 0.05, -5, x0=math.inf), "x0 must be a finite"),
                    (lambda: coef_quantile(D, 0.05, -5, y0=-math.inf), "y0 must be a finite"),
                    (lambda: lin_quantile(D, 0.05, -3.5, -5, b0=math.inf), "b0 must be a finite"),
                    (lambda: adfmax_quantile(D, 0.05, -5, x0=-math.inf), "x0 must be a finite"),
                    (lambda: lin_cdf(D, -0.26, math.inf, -5), "a must be a finite"),
                    (lambda: pt_cdf(trend(1), 4.0, -13.5, math.inf), "c must be a finite"),
                    (lambda: dist_cdf(trend(1), -2.0, -5, gls=-math.inf), "cbar < 0"),
                    (lambda: ltratio_region({"type": "line", "a": math.inf, "b": -0.26}, "DK", -5), "slope a of the line"),
                    (lambda: ltratio_lincdf(-0.26, math.inf, "DK", -5), "a must be")]:
        with pytest.raises(ValueError, match=re.escape(text)):
            f()
    np.testing.assert_array_equal(power_envelope([-math.inf, math.nan, math.inf], "DK"), [1, math.nan, 0.05])
    np.testing.assert_array_equal(pt_envelope(trend(1), [-math.inf, math.nan, math.inf]), [1, math.nan, 0.05])
    P = [ltratio_region({"type": "line", "a": -3.5, "b": b}, "DK", -5) for b in (math.inf, -math.inf)]
    np.testing.assert_array_equal(P, [1, 0])


def test_line_integral_method():
    """The line integral with the explicit kernel for a design of rank two (method), through dist_* and ltratio_*."""
    import dflimit as d
    D = d.trend(2)
    x = np.array([-4.0, -3.2, -2.5, -1.8])
    Fl = d.dist_cdf(D, x, -5, method="line")
    Fp = d.dist_cdf(D, x, -5, method="paths")
    assert np.max(np.abs(Fl - Fp)) < 1e-10
    assert np.array_equal(d.dist_cdf(D, x, -5), Fl)                       # auto: the line integral at c0 = 0, c >= -20
    assert np.array_equal(d.dist_cdf(D, x, -50), d.dist_cdf(D, x, -50, method="paths"))
    assert np.array_equal(d.dist_pdf(D, x, -5, c0=-3), d.dist_pdf(D, x, -5, c0=-3, method="paths"))
    assert abs(d.dist_quantile(D, 0.05, -5, method="line") - d.dist_quantile(D, 0.05, -5, method="paths")) < 1e-9
    with pytest.raises(ValueError, match="GLS detrending is not covered"):
        d.dist_cdf(D, x, -5, gls=-13.5, method="line")
    with pytest.raises(ValueError, match="a jump"):
        d.dist_cdf(d.perron("A", 0.5), x, 0, method="line")
    with pytest.raises(ValueError, match="method must be"):
        d.dist_cdf(D, x, -5, method="single")
    assert np.array_equal(d.ltratio_cdf(x, D, -5, method="paths"), Fp)
    assert np.array_equal(d.ltratio_cdf(x, D, -5, method="line"), Fl)
    assert np.array_equal(d.ltratio_cdf(x, d.trend(0), -5, method="paths"), d.dist_cdf(d.trend(0), x, -5, method="paths"))
    assert np.array_equal(d.ltratio_cdf(x, d.trend(0), -5, method="line"), d.ltratio_cdf(x, "DU", -5, method="single"))
    with pytest.raises(ValueError):
        d.ltratio_cdf(x, "DU", -5, method="paths")


def test_above_the_unit_root():
    """The coefficient statistic at c = 20 (the segments of the paths around the roots shorter than 1e-12 got no panel: case
    DK gave 0.3128 for the 30% point), the moments (case DT at c = 0, c0 = -2 failed on the default route; far above the
    unit root the (1,1) entry of the matrix of the transform), the guard of coef_cdf_gp and the range of c (16 with
    deterministic terms, 20 without)."""
    assert abs(coef_cdf(19.999999940099205, "DK", 20, engine="general") - 0.3) < 2e-8
    mc = np.asarray(ltratio_mom(4, "DT", 0, c0=-2, engine="classical"))
    assert np.max(np.abs(np.asarray(ltratio_mom(4, "DT", 0, c0=-2)) - mc) / np.abs(mc)) < 1e-9
    mc = np.asarray(ltratio_mom(4, "DK", 20, engine="classical"))
    assert np.max(np.abs(np.asarray(dist_moments(trend(-1), 20)) - mc) / np.abs(mc)) < 1e-11
    with pytest.raises(ValueError, match="needs N >="):
        coef_cdf_gp(trend(-1), 11.999892584561692, 12)
    with pytest.raises(ValueError, match="up to c = 16"):
        coef_cdf(17, trend(0), 17, engine="general")
    with pytest.raises(ValueError, match=r"the t-ratio \(dist_cdf\) at c = 18"):
        dist_cdf(perron("A", 0.5), 0, 18, c0=2)
    with pytest.raises(ValueError, match="up to c = 20"):
        dist_moments(trend(-1), 21)
    assert np.allclose(ltratio_mom(2, "DU", 18), ltratio_mom(2, "DU", 18, engine="classical"), rtol=0, atol=0)
