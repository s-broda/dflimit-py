"""The coefficient statistic n(phi^ - phi0): coef_cdf, coef_icdf (R/coef.R of the R package dflimit).  With a design or
the keyword engine, the router (_route.py) or the general engine."""
import math
import warnings

import numpy as np

from . import _general
from ._ltratio import CASES_DOC, ERRORS_DOC, ROUTE_DOC, _classical, ltratio_cdf
from ._route import (AUTO, CLASSICAL, DESIGN_DOC, ENGINE_DOC, GENERAL, NO_METHOD, brittle, classical_ok, det_opts,
                     engine_of, route_tuple, route_x, spec_of, warn_lines)
from ._route import dfl_engine as dfl_engine_code
from ._utils import dfl_flag, dfl_imethod, dfl_num, dfl_posint, dfl_real, dfl_shape, dfl_tol, fill_doc, warn

# (the operations of _engine.route_x)
RX_MOM, RX_COEF, RX_COEF_ICDF, RX_COEF_PDF, RX_COEF_MOM, RX_LIN, RX_ENV = range(1, 8)

COEF_PARAMS_DOC = """\
cas : str or Design
    The case, one of ``"DK"``, ``"DU"``, ``"DT"``, ``"SK"``, ``"SU"``, ``"ST"``, ``"DG"`` (see Notes), or a design of
    the general engine.
c : float, default 0
    The local-to-unity parameter c, phi = 1 + c/n.
c0 : float, default 0
    The local null value c0 of the statistic n(phi^ - phi0), phi0 = 1 + c0/n.
g0 : float, default 0
    The fixed start, y_0 = g0 sigma sqrt(n) (cases DK, DU, DT).
method : {"auto", "inversion", "engine", "single", "line", "double", "conditional", "contour"}, default "auto"
    ``"inversion"`` forces the single inversion of Bao and Kan, ``"engine"`` the engines' computation of before; a
    representation of the t-ratio, as in `ltratio_cdf`, takes the engines too (see Notes).
verified : bool, default False
    One evaluation (False) or the checked computation (True): the values of the single inversion whose error estimate
    exceeds ``tol`` are recomputed by the engines' checked computation, as in `ltratio_cdf`."""

COEF_ROUTE_DOC = """\
With a design or ``engine="auto"`` the router evaluates the t-ratio at 0 with the null value c0 + y by the engine of
its rule; the general engine (a design beyond the classical cases, or ``engine="general"``) computes the coefficient
statistic directly, by the inversion on paths around the roots, and its quantiles by Newton's method from y = -10."""

COEF_NOTES_DOC = """\
The limit is (U - c0 V)/V, so that P[(U - c0 V)/V <= y] = P[U - (c0 + y)V <= 0]. In the cases DK, DU, DT (zero or
fixed start ``g0``) and SK, SU, ST the default (``method="auto"`` with ``engine="auto"``, up to c = 20) is the single
inversion of the joint characteristic function along the ray of the half-plane, the method of Yong Bao and Raymond Kan,
"Exact Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order Autoregressions: O(1)
Algorithms": adaptive quadrature over decades, the tail completed by integration by parts with the derivatives from
jets, the closed-form transform with the branch fixed at each point, one evaluation with an error estimate per value
(as ``coef_cdf`` of DFLimit.jl and of the MATLAB programs). ``tol`` is its absolute tolerance, ``err=True`` returns the
estimates as well, and a warning names the points at which they exceed it. ``method="inversion"`` forces it (also above
c = 20); ``method="engine"`` is the computation of before, the distribution function of the t-ratio at 0 with the local
null value c0 + y (``ltratio_cdf(0, cas, c, c0=c0 + y, g0=g0)``, with its choice of engine and representation), which
reproduces the values of earlier versions; a representation of the t-ratio, ``engine="classical"`` or ``"general"``
and case DG take that computation too. `coef_icdf` finds each quantile by the Illinois variant of regula falsi to the
width ``tol``: by default on the single inversion's distribution function (tolerance 1e-12) from the bracket
E[U - cV]/E[V] -/+ twice the scale of the ratio (``err=True``: the estimate of the distribution function at the
quantile divided by the density there), otherwise on the engines' distribution function from a bracket grown from
[-40, 10]."""

ERR_DOC = """\
tol : float, default 1e-12
    The absolute tolerance of each value of the single inversion.
err : bool, default False
    If true, return the tuple (values, error estimates); the estimates are ``nan`` where the engines computed."""


@fill_doc(params=COEF_PARAMS_DOC, errors=ERRORS_DOC, notes=COEF_NOTES_DOC, cases=CASES_DOC, engine=ENGINE_DOC,
          design=DESIGN_DOC, route=ROUTE_DOC, croute=COEF_ROUTE_DOC, errdoc=ERR_DOC)
def coef_cdf(y, cas, c=0, c0=0, g0=0, method="auto", verified=False, *, engine="auto", start=None, gls=None, tol=1e-12,
             err=False):
    """Distribution function of the limit of the Dickey-Fuller coefficient statistic.

    The limit of P[n(phi^ - phi0) <= y] for the least squares estimate phi^ of phi in the regression of
    `ltratio_cdf`, under phi = 1 + c/n, with phi0 = 1 + c0/n (``c0``, default 0: the coefficient test of Dickey and
    Fuller, n(phi^ - 1)).

    Parameters
    ----------
    y : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    {params}
    {engine}
    {design}
    {errdoc}

    Returns
    -------
    float or numpy.ndarray
        The distribution function at ``y``: a float for a scalar ``y``, otherwise an array of the shape of ``y`` (with
        ``err=True`` the tuple of the values and their error estimates).

    Raises
    ------
    {errors}

    See Also
    --------
    coef_icdf, ltratio_cdf, coef_pdf

    Notes
    -----
    {notes}

    {cases}

    {route}

    {croute}

    Examples
    --------
    >>> from dflimit import coef_cdf
    >>> coef_cdf(-14.0936, "DU", -10)    # power of the 5% coefficient test of rho_mu at c = -10
    0.4592525632...
    >>> from dflimit import perron
    >>> coef_cdf(-30.0, perron("A", 0.5))     # Perron's model A with a break at the middle
    0.018386...
    """
    S = spec_of(cas, g0, start, gls)
    e = engine_of(S, engine)
    im = dfl_imethod(method, "coef_cdf", True)
    ve = dfl_flag(verified, "verified")
    c = dfl_num(c, "c")
    c0 = dfl_num(c0, "c0")
    a = dfl_real(y, "y")
    if 1 <= im <= 4 or (e == CLASSICAL and im != 5):
        # (a representation named, or the classical engine: the computation of before)
        me = im if im <= 4 else 0
        if _classical(S, e, c, c0, me, ve):
            with _once():
                return _with_err(_coef_cdf(a, S.case, c, c0, S.g0, method if me else "auto", verified), None, a, err)
        if me != 0:
            raise ValueError(NO_METHOD)
    if im != 5 and (e == GENERAL or not (S.named or classical_ok(S, c, c0, 0))):
        return _with_err(_general._values(_general.DET_COEF_CDF, S.design, a, c, det_opts(S, c0, ve), name="y"), None, a, err)
    out, eo = route_x(RX_COEF, route_tuple(S), a.ravel(), c, c0, e, im if im >= 5 else 0, dfl_tol(tol), ve)
    if np.all(np.isnan(eo)) and e == AUTO and not ve:
        warn_lines(_coef_brittle(S, c, c0, a))
    return _with_err(dfl_shape(out, a), eo, a, err)


def _with_err(v, e, a, err):
    """The values, with err the tuple (values, error estimates; nan where the engines computed)."""
    if not err:
        return v
    if e is None:
        e = np.full(a.size, np.nan)
    return v, dfl_shape(np.asarray(e, dtype=np.float64), a)


def _coef_brittle(S, c, c0, ys):
    """The brittle regimes of the t-ratio at 0 with the null values c0 + y where the router takes the classical engine
    (each message once)."""
    return "\n".join(sorted({brittle(2, S, c, c0 + y, [0.0]) for y in np.unique(ys[np.isfinite(ys)])} - {""}))


def _coef_cdf(a, cas, c, c0, g0, method, verified):
    F = np.empty(a.size)
    for k, yk in enumerate(a.ravel()):
        if math.isnan(yk):
            F[k] = math.nan
        elif yk == math.inf:
            F[k] = 1.0
        elif yk == -math.inf:
            F[k] = 0.0
        else:
            F[k] = ltratio_cdf(0.0, cas, c, c0=c0 + yk, g0=g0, method=method, verified=verified, engine="classical")
    return dfl_shape(F, a)


@fill_doc(params=COEF_PARAMS_DOC, errors=ERRORS_DOC, notes=COEF_NOTES_DOC, cases=CASES_DOC, engine=ENGINE_DOC,
          design=DESIGN_DOC, route=ROUTE_DOC, croute=COEF_ROUTE_DOC, errdoc=ERR_DOC)
def coef_icdf(p, cas, c=0, c0=0, g0=0, method="auto", tol=1e-9, verified=False, *, engine="auto", start=None, gls=None,
              err=False):
    """Quantiles of the limit of the Dickey-Fuller coefficient statistic.

    The p-quantiles of the limit of n(phi^ - phi0) of `coef_cdf`.

    Parameters
    ----------
    p : array_like
        The probabilities, in (0, 1).
    {params}
    tol : float, default 1e-9
        The width of the final bracket (with the general engine: the tolerance of Newton's method).
    {engine}
    {design}
    err : bool, default False
        If true, return the tuple (quantiles, error estimates): by the single inversion the estimate of the distribution
        function at the quantile divided by the density there, ``nan`` where the engines computed.

    Returns
    -------
    float or numpy.ndarray
        The quantiles: a float for a scalar ``p``, otherwise an array of the shape of ``p`` (with ``err=True`` the
        tuple of the quantiles and their error estimates).

    Raises
    ------
    {errors}

    See Also
    --------
    coef_cdf, ltratio_icdf, coef_quantile

    Notes
    -----
    {notes}

    {cases}

    {route}

    {croute}

    Examples
    --------
    >>> from dflimit import coef_icdf
    >>> coef_icdf(0.05, "DK")            # -8.039, the 5% critical value of rho
    -8.03912890...
    >>> from dflimit import trend
    >>> coef_icdf(0.05, trend(2))        # with a quadratic trend
    -28.11298...
    """
    a = dfl_real(p, "p")
    if not np.all((a > 0) & (a < 1)):
        raise ValueError("p must lie in (0, 1)")
    tol = dfl_num(tol, "tol")
    S = spec_of(cas, g0, start, gls)
    e = engine_of(S, engine)
    im = dfl_imethod(method, "coef_icdf", True)
    ve = dfl_flag(verified, "verified")
    c = dfl_num(c, "c")
    c0 = dfl_num(c0, "c0")
    if 1 <= im <= 4 or (e == CLASSICAL and im != 5):
        me = im if im <= 4 else 0
        if _classical(S, e, c, c0, me, ve):
            y = np.empty(a.size)
            with _once():
                for k, pk in enumerate(a.ravel()):
                    y[k] = _coef_quantile(pk, S.case, c, c0, S.g0, method if me else "auto", tol, verified)
            return _with_err(dfl_shape(y, a), None, a, err)
        if me != 0:
            raise ValueError(NO_METHOD)
    gen = im != 5 and (e == GENERAL or not (S.named or classical_ok(S, c, c0, 0)))
    if ve and gen:
        raise ValueError("verified=True: the checked computation of the quantiles is the classical engine's; the general "
                         "engine has none")
    q, eo = route_x(RX_COEF_ICDF, route_tuple(S), a.ravel(), c, c0, e, im if im >= 5 else 0, math.nan, ve, tol)
    if np.all(np.isnan(eo)) and e == AUTO and not ve:
        warn_lines(_coef_brittle(S, c, c0, q))
    return _with_err(dfl_shape(q, a), eo, a, err)


@fill_doc(errors=ERRORS_DOC, cases=CASES_DOC, engine=ENGINE_DOC, design=DESIGN_DOC)
def coef_mom(p, cas, c=0, c0=0, g0=0, *, tol=1e-10, err=False, engine="auto", method="auto", start=None, gls=None):
    """Moments of the limit of the Dickey-Fuller coefficient statistic.

    m[j] = E[{(U - c0 V)/V}^j], j = 1, ..., p, the moments of the limit of n(phi^ - phi0) of `coef_cdf`, in the cases DK,
    DU, DT (zero or fixed start ``g0``) and SK, SU, ST (a design that is one of them), up to c = 20.

    Parameters
    ----------
    p : int
        The number of moments, a positive integer (at most 31).
    cas : str or Design
        The case, one of ``"DK"``, ``"DU"``, ``"DT"``, ``"SK"``, ``"SU"``, ``"ST"``, or a design that is one of them.
    c : float, default 0
        The local-to-unity parameter c, phi = 1 + c/n.
    c0 : float, default 0
        The local null value c0 of the statistic n(phi^ - phi0), phi0 = 1 + c0/n.
    g0 : float, default 0
        The fixed start, y_0 = g0 sigma sqrt(n) (cases DK, DU, DT).
    tol : float, default 1e-10
        The relative tolerance of the quadrature.
    err : bool, default False
        If true, return the tuple (moments, error estimates).
    {engine}
    method : {"auto", "inversion", "jets"}, default "auto"
        ``"jets"`` (or ``"inversion"``) also above c = 20.
    {design}

    Returns
    -------
    numpy.ndarray
        The moments, an array of length ``p`` (with ``err=True`` the tuple of the moments and their error estimates).

    Raises
    ------
    {errors}

    See Also
    --------
    coef_cdf, ltratio_mom

    Notes
    -----
    Sawa's representation m_j = int_0^inf [d^j M(s1, -s2)/ds1^j at s1 = 0] s2^(j-1)/Gamma(j) ds2 with the derivative
    from one jet of order j per node, the method of Yong Bao and Raymond Kan, "Exact Finite-Sample Distribution and
    Moments of the Least Squares Estimator in First-Order Autoregressions: O(1) Algorithms", adaptive quadrature in
    log s2 to the relative tolerance ``tol`` (as ``coef_mom`` of DFLimit.jl and of the MATLAB programs). The engines
    have no moments of the coefficient statistic: case DG, the other designs and ``engine="general"`` are refused.

    {cases}

    Examples
    --------
    >>> from dflimit import coef_mom
    >>> float(coef_mom(1, "DK")[0])       # the mean of n(phi^ - 1) at the unit root
    -1.7814...
    """
    p = dfl_posint(p, "p")
    S = spec_of(cas, g0, start, gls)
    e = dfl_engine_code(engine)
    im = dfl_imethod(method, "coef_mom")
    if im == 6:
        raise ValueError('coef_mom: method must be "auto" (the jets of Bao and Kan; the engines have no moments of the '
                         "coefficient statistic)")
    m, eo = route_x(RX_COEF_MOM, route_tuple(S), np.array([float(p)]), dfl_num(c, "c"), dfl_num(c0, "c0"), e, im,
                    dfl_tol(tol), 0)
    m = m.copy()
    return (m, eo.copy()) if err else m


def _coef_quantile(pk, cas, c, c0, g0, method, tol, verified=False):
    """One quantile by the Illinois variant of regula falsi (coef_icdf of the R package and of the MATLAB programs)."""
    def f(x):
        return coef_cdf(x, cas, c, c0, g0, method, verified, engine="classical") - pk

    lo, hi = -40.0, 10.0
    flo, fhi = f(lo), f(hi)
    while flo > 0:
        hi, fhi = lo, flo
        lo = 2 * lo - 10
        flo = f(lo)
    while fhi < 0:
        lo, flo = hi, fhi
        hi = 2 * hi + 10
        fhi = f(hi)
    side = 0
    x = (lo + hi) / 2
    for _ in range(200):
        x = (lo * fhi - hi * flo) / (fhi - flo)
        fx = f(x)
        if fx == 0:
            break
        if fx < 0:
            lo, flo = x, fx
            if side == -1:
                fhi /= 2
            side = -1
        else:
            hi, fhi = x, fx
            if side == 1:
                flo /= 2
            side = 1
        if hi - lo < tol:
            x = (lo * fhi - hi * flo) / (fhi - flo)
            break
    return x


class _once:
    """Gives each distinct warning once: the calls of ltratio_cdf inside coef_cdf and coef_icdf would repeat the warning of
    a brittle regime at every point."""
    def __enter__(self):
        self._cm = warnings.catch_warnings(record=True)
        self._log = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        seen = set()
        for w in self._log:
            m = str(w.message)
            if m not in seen:
                seen.add(m)
                warn(m)
        return False
