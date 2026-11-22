"""The coefficient statistic n(phi^ - phi0): coef_cdf, coef_icdf (R/coef.R of the R package dflimit).  With a design or
the keyword engine, the router (_route.py) or the general engine."""
import math
import warnings

import numpy as np

from . import _engine, _general
from ._ltratio import CASES_DOC, ERRORS_DOC, R_COEF, R_COEF_ICDF, ROUTE_DOC, _classical, ltratio_cdf
from ._route import (AUTO, DESIGN_DOC, ENGINE_DOC, GENERAL, NO_METHOD, brittle, classical_ok, det_opts, engine_of,
                     route_tuple, spec_of, warn_lines)
from ._utils import dfl_flag, dfl_method, dfl_num, dfl_real, dfl_shape, f64, fill_doc, warn

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
method : {"auto", "single", "line", "double", "conditional", "contour"}, default "auto"
    The representation, as in `ltratio_cdf`.
verified : bool, default False
    One evaluation (False) or the checked computation (True), as in `ltratio_cdf`."""

COEF_ROUTE_DOC = """\
With a design or ``engine="auto"`` the router evaluates the t-ratio at 0 with the null value c0 + y by the engine of
its rule; the general engine (a design beyond the classical cases, or ``engine="general"``) computes the coefficient
statistic directly, by the inversion on paths around the roots, and its quantiles by Newton's method from y = -10."""

COEF_NOTES_DOC = """\
The limit is (U - c0 V)/V, so that P[(U - c0 V)/V <= y] = P[U - (c0 + y)V <= 0]: the distribution function of the
t-ratio at 0 with the local null value c0 + y. The C engine has no entry point of its own for the coefficient
statistic: as ``coef_cdf`` and ``coef_icdf`` of the MATLAB programs, `coef_cdf` evaluates
``ltratio_cdf(0, cas, c, c0=c0 + y, g0=g0, method=method)`` at each ``y``, in every case and with every start and
option of `ltratio_cdf`, and `coef_icdf` finds each quantile by the Illinois variant of regula falsi on `coef_cdf`,
on a bracket grown from [-40, 10] until its width is below ``tol`` (about ten evaluations of `coef_cdf`)."""


@fill_doc(params=COEF_PARAMS_DOC, errors=ERRORS_DOC, notes=COEF_NOTES_DOC, cases=CASES_DOC, engine=ENGINE_DOC,
          design=DESIGN_DOC, route=ROUTE_DOC, croute=COEF_ROUTE_DOC)
def coef_cdf(y, cas, c=0, c0=0, g0=0, method="auto", verified=False, *, engine=None, start=None, gls=None):
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

    Returns
    -------
    float or numpy.ndarray
        The distribution function at ``y``: a float for a scalar ``y``, otherwise an array of the shape of ``y``.

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
    me = dfl_method(method)
    ve = dfl_flag(verified, "verified")
    c = dfl_num(c, "c")
    c0 = dfl_num(c0, "c0")
    a = dfl_real(y, "y")
    if _classical(S, e, c, c0, me, ve):
        with _once():
            return _coef_cdf(a, S.case, c, c0, S.g0, method, verified)
    if e == GENERAL or not (S.named or classical_ok(S, c, c0, 0)):
        if me != 0:
            raise ValueError(NO_METHOD)
        return _general._values(_general.DET_COEF_CDF, S.design, a, c, det_opts(S, c0, ve), name="y")
    out, wd, wc = _engine.route(R_COEF, route_tuple(S), a.tobytes(), c, c0, AUTO, 0, 0, 0.0)
    warn_lines(_coef_brittle(S, c, c0, a), wc, wd)
    return dfl_shape(f64(out), a)


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
            F[k] = ltratio_cdf(0.0, cas, c, c0=c0 + yk, g0=g0, method=method, verified=verified)
    return dfl_shape(F, a)


@fill_doc(params=COEF_PARAMS_DOC, errors=ERRORS_DOC, notes=COEF_NOTES_DOC, cases=CASES_DOC, engine=ENGINE_DOC,
          design=DESIGN_DOC, route=ROUTE_DOC, croute=COEF_ROUTE_DOC)
def coef_icdf(p, cas, c=0, c0=0, g0=0, method="auto", tol=1e-9, verified=False, *, engine=None, start=None, gls=None):
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

    Returns
    -------
    float or numpy.ndarray
        The quantiles: a float for a scalar ``p``, otherwise an array of the shape of ``p``.

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
    me = dfl_method(method)
    ve = dfl_flag(verified, "verified")
    c = dfl_num(c, "c")
    c0 = dfl_num(c0, "c0")
    if _classical(S, e, c, c0, me, ve):
        y = np.empty(a.size)
        with _once():
            for k, pk in enumerate(a.ravel()):
                y[k] = _coef_quantile(pk, S.case, c, c0, S.g0, method, tol, verified)
        return dfl_shape(y, a)
    if ve:
        raise ValueError("verified=True: the checked computation of the quantiles is the classical engine's; the general "
                         "engine has none")
    if me != 0:
        raise ValueError(NO_METHOD)
    q, wd, wc = _engine.route(R_COEF_ICDF, route_tuple(S), a.tobytes(), c, c0, e, 0, 0, tol)
    qv = f64(q)
    warn_lines(_coef_brittle(S, c, c0, qv) if e == AUTO else "", wc, wd)
    return dfl_shape(qv, a)


def _coef_quantile(pk, cas, c, c0, g0, method, tol, verified=False):
    """One quantile by the Illinois variant of regula falsi (coef_icdf of the R package and of the MATLAB programs)."""
    def f(x):
        return coef_cdf(x, cas, c, c0, g0, method, verified) - pk

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
