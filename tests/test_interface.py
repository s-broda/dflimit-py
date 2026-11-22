"""The interface: shapes, non-finite points, errors, threads (tests/testthat/test-interface.R of the R package
dflimit), and what is particular to Python: scalar and array types, info results, warnings, concurrent calls."""
import importlib.metadata
import math
import os
import threading
import warnings

import numpy as np
import pytest

import dflimit
from dflimit import (adfmax_cdf, adfmax_icdf, adfmax_pdf, coef_cdf, ltratio_cdf, ltratio_icdf, ltratio_lincdf,
                     ltratio_mom, ltratio_pdf, ltratio_region, ltratio_threads, power_envelope)


def test_shape_and_non_finite_points():
    x = np.array([[-math.inf, math.nan], [-2, math.inf]])
    F = ltratio_cdf(x, "DK")
    assert isinstance(F, np.ndarray) and F.shape == x.shape and F.dtype == np.float64
    assert F[0, 0] == 0 and math.isnan(F[0, 1]) and F[1, 1] == 1
    np.testing.assert_array_equal(ltratio_pdf([-math.inf, math.inf], "DK"), [0, 0])
    assert ltratio_cdf([-1, 0], "dk").shape == (2,)                     # (the case code in lower case)
    e = ltratio_cdf([], "DU")
    assert isinstance(e, np.ndarray) and e.shape == (0,) and e.dtype == np.float64
    np.testing.assert_array_equal(adfmax_cdf([-math.inf, math.nan, math.inf], "DU"), [0, math.nan, 1])
    assert ltratio_icdf(np.array([[0.05, 0.1]]), "DK").shape == (1, 2)
    P = ltratio_lincdf(x, -3.5, "DK")
    assert P.shape == x.shape
    assert P[0, 0] == 0 and math.isnan(P[0, 1]) and P[1, 1] == 1
    assert ltratio_lincdf([0.1, 0.5], math.nan, "dk").shape == (2,)
    e = ltratio_lincdf([], 1, "DG")
    assert isinstance(e, np.ndarray) and e.shape == (0,)
    E = power_envelope(np.array([[0, math.nan], [1, 2]]))
    assert E.shape == (2, 2)
    assert E[0, 0] == 0.05 and E[1, 0] == 0.05 and math.isnan(E[0, 1])
    assert power_envelope(-7, "DU") == power_envelope(-7, "dk")


def test_scalars_give_floats_and_arrays_keep_their_shape():
    for x in (-1, -1.0, np.float64(-1), np.float32(-1), np.int64(-1), np.array(-1.0)):
        assert type(ltratio_cdf(x, "DU")) is float
    assert type(ltratio_pdf(-1, "DU")) is float
    assert type(ltratio_icdf(0.05, "DU")) is float
    assert type(adfmax_cdf(-2, "DU")) is float and type(adfmax_pdf(-2, "DU")) is float
    assert type(adfmax_icdf(0.05, "DU")) is float
    assert type(coef_cdf(-5, "DU")) is float
    assert type(ltratio_region({"type": "line", "a": -3.5, "b": -0.2639})) is float
    assert type(ltratio_lincdf(0.1, np.float64(-3.5), "DK")) is float and type(power_envelope(-7)) is float
    x = np.linspace(-3, 1, 6).reshape(2, 3)
    F = ltratio_cdf(x, "DU")
    assert F.shape == (2, 3)
    # (to rounding: with threads the order of summation in the engine can differ by one ulp)
    np.testing.assert_allclose(F.ravel(), ltratio_cdf(x.ravel(), "DU"), rtol=1e-14)
    np.testing.assert_allclose(ltratio_cdf(x.T, "DU"), F.T, rtol=1e-14)  # (a non-contiguous input)
    assert ltratio_cdf([[-1.0]], "DU").shape == (1, 1)
    m = ltratio_mom(1, "DU")
    assert isinstance(m, np.ndarray) and m.shape == (1,)


def test_info_results():
    # (the information of the classical engine, engine="classical"; the router's, the default, has q and F)
    r = ltratio_icdf(0.05, "DU", info=True, engine="classical")
    assert set(r) == {"q", "F", "iter", "conv"}
    assert r.q == r["q"] and type(r.q) is float and type(r.iter) is int and r.conv is True
    r = ltratio_icdf([0.05, 0.1], "DU", info=True, engine="classical")
    assert r.q.shape == r.F.shape == r.iter.shape == r.conv.shape == (2,) and r.conv.dtype == bool
    r = ltratio_mom(2, "DU", info=True)
    assert set(r) == {"m", "err"} and r.m.shape == r.err.shape == (2,)
    r = adfmax_cdf([-3, -2], "DU", info=True)
    assert set(r) == {"value", "err", "status"} and r.status.shape == (2,) and not r.status.any()
    r = ltratio_region({"type": "line", "a": -3.5, "b": -0.2639}, info=True)
    assert set(r) == {"P", "err", "status"} and r.status == 0
    with pytest.raises(AttributeError):
        r.nothing


def test_engine_errors_become_value_errors():
    cases = [
        (lambda: ltratio_cdf(0, "DU", -600), "at least -550"),
        (lambda: ltratio_cdf(0, "SU", 1), r"requires c <= -1e-4"),
        (lambda: ltratio_cdf(0, "DU", -5, g0=3), "fixed start"),
        (lambda: ltratio_cdf(0, "SU", -5, g0=1, engine="classical"), "applies to cases DK, DU, DT"),
        (lambda: ltratio_cdf(0, "SU", -5, g0=1), "exclude each other"),
        (lambda: ltratio_mom(2, "DG"), "not available"),
        (lambda: ltratio_cdf(0, "DG", method="double"), "single integral only"),
        (lambda: ltratio_icdf(1, "DU"), r"\(0, 1\)"),
        (lambda: ltratio_cdf(0, "DU", c=math.nan), "single number"),
        (lambda: ltratio_cdf(0, "XX"), "the case must be"),
        (lambda: ltratio_cdf(0, "DU", method="fast"), "method must be"),
        (lambda: adfmax_cdf(0, "DK"), "the case must be one of DU, DT, SU, ST"),
        (lambda: ltratio_region({"type": "fun", "g": abs}), "not available"),
        (lambda: ltratio_region({"type": "line", "a": 1}), "needs the key 'b'"),
        (lambda: ltratio_region({"type": "wap", "k": -1, "omega": 1}), "k > 0"),
        (lambda: ltratio_region({"type": "line", "a": 1, "b": 0, "v0": -1}), "nonnegative"),
        (lambda: ltratio_region({"type": "line", "a": 1, "b": 0, "brks": [0.5, -1]}), "positive and finite"),
        (lambda: ltratio_region({"type": "line", "a": 1, "b": 0}, "DU"), "must be DK"),
        (lambda: ltratio_mom(0, "DU"), "positive integer"),
        (lambda: ltratio_mom(2.5, "DU"), "positive integer"),
        (lambda: adfmax_icdf(1.5, "DU"), r"\(0, 1\)"),
        (lambda: ltratio_lincdf(0, math.nan, "DG"), "case DK only"),
        (lambda: ltratio_lincdf(0, 1, "DU"), "the case must be one of DK, DG"),
        (lambda: ltratio_lincdf(0, 1, "DK", c=math.inf), "finite"),
        (lambda: power_envelope(-5, "DT"), "the case must be one of DK, DU, DG"),
        (lambda: power_envelope(-5, alpha=2, engine="classical"), "no sign change"),
        (lambda: power_envelope(-5, alpha=2), r"the level must lie in \(0, 1\)"),
    ]
    for fun, msg in cases:
        with pytest.raises(ValueError, match=msg):
            fun()


def test_type_errors():
    cases = [
        (lambda: ltratio_cdf("a", "DU"), "x must be numeric"),
        (lambda: ltratio_cdf([1, None], "DU"), "x must be numeric"),
        (lambda: ltratio_cdf([True], "DU"), "x must be numeric"),
        (lambda: ltratio_cdf([1 + 1j], "DU"), "x must be numeric"),
        (lambda: ltratio_icdf("a", "DU"), "p must be numeric"),
        (lambda: ltratio_cdf(0, 1), "the case must be"),
        (lambda: ltratio_cdf(0, "DU", method=1), "method must be"),
        (lambda: ltratio_cdf(0, "DU", c=None), "c must be a single number"),
        (lambda: ltratio_cdf(0, "DU", c=[1, 2]), "c must be a single number"),
        (lambda: ltratio_cdf(0, "DU", c=True), "c must be a single number"),
        (lambda: ltratio_cdf(0, "DU", c0="1"), "c0 must be a single number"),
        (lambda: ltratio_mom("2", "DU"), "positive integer"),
        (lambda: ltratio_region([1, 2]), "dict with the key 'type'"),
        (lambda: ltratio_region({"type": "line", "a": "1", "b": 0}), r"bnd\['a'\] must be a single number"),
        (lambda: ltratio_threads("2"), "positive integer"),
        (lambda: ltratio_lincdf(0, None, "DK"), "a must be a single number"),
        (lambda: ltratio_lincdf(0, [1, 2], "DK"), "a must be a single number"),
        (lambda: ltratio_lincdf("b", 1, "DK"), "b must be numeric"),
        (lambda: power_envelope("c"), "c must be numeric"),
        (lambda: power_envelope(-5, alpha=None), "alpha must be a single number"),
    ]
    for fun, msg in cases:
        with pytest.raises(TypeError, match=msg):
            fun()


def test_threads_can_be_set_and_do_not_change_the_values():
    old = ltratio_threads()
    F2 = ltratio_cdf([-2, 0], "DT", -3)
    assert ltratio_threads(1) == 1
    assert ltratio_threads() == 1
    F1 = ltratio_cdf([-2, 0], "DT", -3)
    ltratio_threads(old)
    assert ltratio_threads() == old
    np.testing.assert_allclose(F1, F2, rtol=1e-14)
    ltratio_threads(1)
    P1 = ltratio_lincdf([-0.2639, 0.5], -3.5, "DK", -7)
    E1 = power_envelope([-7, -13.5], "DG")
    ltratio_threads(old)
    np.testing.assert_allclose(ltratio_lincdf([-0.2639, 0.5], -3.5, "DK", -7), P1, rtol=1e-14)
    np.testing.assert_allclose(power_envelope([-7, -13.5], "DG"), E1, rtol=1e-14)
    with pytest.raises(ValueError, match="positive integer"):
        ltratio_threads(0)


def test_concurrent_calls_from_python_threads():
    # the engine calls are serialized by the extension (the GIL is released while one runs)
    x = [-2.5, -1.5]
    ref = ltratio_cdf(x, "DU", -3)
    out = [None] * 4

    def work(k):
        out[k] = ltratio_cdf(x, "DU", -3)

    th = [threading.Thread(target=work, args=(k,)) for k in range(len(out))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for o in out:
        np.testing.assert_allclose(o, ref, rtol=1e-14)


class _FakeEngine:
    """The engine's entry points with flags set, for the warnings (which the real engine gives rarely)."""

    def dist(self, kind, x, *args):
        return bytearray(x), len(x) // 8, "", ""

    def icdf(self, p, *args):
        n = len(p) // 8
        return (bytearray(p), bytearray(p), bytearray(np.full(n, 60, np.intc).tobytes()), bytearray(np.zeros(n, np.intc).tobytes()),
                "", "")

    def adfmax(self, kind, x, *args):
        n = len(x) // 8
        return bytearray(x), bytearray(x), bytearray(np.ones(n, np.intc).tobytes())

    def region(self, *args):
        return 0.5, 1e-3, 1


def test_warnings(monkeypatch):
    from dflimit import _adfmax, _ltratio, _region
    fake = _FakeEngine()
    for mod in (_ltratio, _adfmax, _region):
        monkeypatch.setattr(mod, "_engine", fake)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ltratio_cdf([-1, 0], "DU", method="auto", engine="classical")    # (no warning without method double)
    # (the warnings of the classical entry points, engine="classical"; the fake engine has no router)
    cases = [
        (lambda: ltratio_cdf([-1, 0], "DU", method="double", verified=True), "did not meet its error estimate at 2 of 2 points"),
        (lambda: ltratio_icdf([0.1, 0.2, 0.3], "DU", engine="classical"), "Newton's method did not converge at 3 of 3 points"),
        (lambda: adfmax_cdf(-2, "DU", engine="classical"), "at 1 of 1 points; the result may be inaccurate there"),
        (lambda: ltratio_region({"type": "line", "a": -3.5, "b": 0}), "the error estimate 1.0e-03 is above"),
    ]
    for fun, msg in cases:
        with pytest.warns(RuntimeWarning, match=msg) as rec:
            fun()
        assert os.path.basename(rec[0].filename) == os.path.basename(__file__)   # (reported at the caller)


def test_version_and_upstream():
    assert dflimit.__version__ == importlib.metadata.version("dflimit")
    up = os.path.join(os.path.dirname(dflimit.__file__), "UPSTREAM")
    with open(up) as fh:
        text = fh.read()
    assert "upstream_commit: " in text


def test_stationary_start_near_the_unit_root():
    # for c > -1e-4 the engine refuses: the variance -1/(2c) of the start is too large for the quadrature
    for f in (lambda: ltratio_cdf(-1.94087, "SK", -9e-5), lambda: ltratio_pdf(-2, "SU", -9e-5),
              lambda: ltratio_mom(2, "ST", -9e-5), lambda: ltratio_icdf(0.05, "SK", -9e-5),
              lambda: coef_cdf(-8.039, "SK", -9e-5), lambda: adfmax_cdf(-2.4, "SU", -9e-5)):
        with pytest.raises(ValueError, match=r"requires c <= -1e-4"):
            f()
    # at c = -1e-4 the values of the Julia engine (test/engine/quickcheck.jl of DFLimit.jl, section 18b)
    assert abs(ltratio_cdf(-1.94087, "SK", -1e-4) - 0.028490044498374834) < 1e-11
    assert abs(coef_cdf(-8.039, "SK", -1e-4) - 0.00032997244558452543) < 1e-12
    assert abs(adfmax_cdf(-2.43542, "SU", -1e-4) - 0.050000718603976146) < 1e-11
    # case SK: the coefficient statistic at c = -0.001, which the lower end of the conditional integral had put at 6e-6
    assert abs(coef_cdf(-8.039, "SK", -0.001) - 0.0010438037922007037) < 1e-12
    assert abs(ltratio_cdf(-1.94087, "SK", -0.1) - 0.059510986573375835) < 1e-11
