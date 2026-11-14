"""The coefficient statistic n(phi^ - phi0): coef_cdf, coef_icdf (R/coef.R of the R package dflimit)."""
import math

import numpy as np

from ._ltratio import CASES_DOC, ERRORS_DOC, ltratio_cdf
from ._utils import dfl_case, dfl_method, dfl_num, dfl_real, dfl_shape, fill_doc

COEF_PARAMS_DOC = """\
cas : str
    The case, one of ``"DK"``, ``"DU"``, ``"DT"``, ``"SK"``, ``"SU"``, ``"ST"``, ``"DG"`` (see Notes).
c : float, default 0
    The local-to-unity parameter c, phi = 1 + c/n.
c0 : float, default 0
    The local null value c0 of the statistic n(phi^ - phi0), phi0 = 1 + c0/n.
g0 : float, default 0
    The fixed start, y_0 = g0 sigma sqrt(n) (cases DK, DU, DT).
method : {"auto", "single", "line", "double", "conditional"}, default "auto"
    The representation, as in `ltratio_cdf`."""

COEF_NOTES_DOC = """\
The limit is (U - c0 V)/V, so that P[(U - c0 V)/V <= y] = P[U - (c0 + y)V <= 0]: the distribution function of the
t-ratio at 0 with the local null value c0 + y. The C engine has no entry point of its own for the coefficient
statistic: as ``coef_cdf`` and ``coef_icdf`` of the MATLAB programs, `coef_cdf` evaluates
``ltratio_cdf(0, cas, c, c0=c0 + y, g0=g0, method=method)`` at each ``y``, in every case and with every start and
option of `ltratio_cdf`, and `coef_icdf` finds each quantile by the Illinois variant of regula falsi on `coef_cdf`,
on a bracket grown from [-40, 10] until its width is below ``tol`` (about ten evaluations of `coef_cdf`)."""


@fill_doc(params=COEF_PARAMS_DOC, errors=ERRORS_DOC, notes=COEF_NOTES_DOC, cases=CASES_DOC)
def coef_cdf(y, cas, c=0, c0=0, g0=0, method="auto"):
    """Distribution function of the limit of the Dickey-Fuller coefficient statistic.

    The limit of P[n(phi^ - phi0) <= y] for the least squares estimate phi^ of phi in the regression of
    `ltratio_cdf`, under phi = 1 + c/n, with phi0 = 1 + c0/n (``c0``, default 0: the coefficient test of Dickey and
    Fuller, n(phi^ - 1)).

    Parameters
    ----------
    y : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    {params}

    Returns
    -------
    float or numpy.ndarray
        The distribution function at ``y``: a float for a scalar ``y``, otherwise an array of the shape of ``y``.

    Raises
    ------
    {errors}

    See Also
    --------
    coef_icdf, ltratio_cdf

    Notes
    -----
    {notes}

    {cases}

    Examples
    --------
    >>> from dflimit import coef_cdf
    >>> coef_cdf(-14.0936, "DU", -10)    # power of the 5% coefficient test of rho_mu at c = -10
    0.4592525632...
    """
    dfl_case(cas)
    c = dfl_num(c, "c")
    c0 = dfl_num(c0, "c0")
    g0 = dfl_num(g0, "g0")
    dfl_method(method)
    a = dfl_real(y, "y")
    F = np.empty(a.size)
    for k, yk in enumerate(a.ravel()):
        if math.isnan(yk):
            F[k] = math.nan
        elif yk == math.inf:
            F[k] = 1.0
        elif yk == -math.inf:
            F[k] = 0.0
        else:
            F[k] = ltratio_cdf(0.0, cas, c, c0=c0 + yk, g0=g0, method=method)
    return dfl_shape(F, a)


@fill_doc(params=COEF_PARAMS_DOC, errors=ERRORS_DOC, notes=COEF_NOTES_DOC, cases=CASES_DOC)
def coef_icdf(p, cas, c=0, c0=0, g0=0, method="auto", tol=1e-9):
    """Quantiles of the limit of the Dickey-Fuller coefficient statistic.

    The p-quantiles of the limit of n(phi^ - phi0) of `coef_cdf`.

    Parameters
    ----------
    p : array_like
        The probabilities, in (0, 1).
    {params}
    tol : float, default 1e-9
        The width of the final bracket.

    Returns
    -------
    float or numpy.ndarray
        The quantiles: a float for a scalar ``p``, otherwise an array of the shape of ``p``.

    Raises
    ------
    {errors}

    See Also
    --------
    coef_cdf, ltratio_icdf

    Notes
    -----
    {notes}

    {cases}

    Examples
    --------
    >>> from dflimit import coef_icdf
    >>> coef_icdf(0.05, "DK")            # -8.039, the 5% critical value of rho
    -8.03912890...
    """
    a = dfl_real(p, "p")
    if not np.all((a > 0) & (a < 1)):
        raise ValueError("p must lie in (0, 1)")
    tol = dfl_num(tol, "tol")
    dfl_case(cas)
    c = dfl_num(c, "c")
    c0 = dfl_num(c0, "c0")
    g0 = dfl_num(g0, "g0")
    dfl_method(method)
    y = np.empty(a.size)
    for k, pk in enumerate(a.ravel()):
        y[k] = _coef_quantile(pk, cas, c, c0, g0, method, tol)
    return dfl_shape(y, a)


def _coef_quantile(pk, cas, c, c0, g0, method, tol):
    """One quantile by the Illinois variant of regula falsi (coef_icdf of the R package and of the MATLAB programs)."""
    def f(x):
        return coef_cdf(x, cas, c, c0, g0, method) - pk

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
