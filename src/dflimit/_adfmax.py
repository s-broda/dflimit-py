"""The ADFmax statistic of Leybourne (1995): adfmax_cdf, adfmax_pdf, adfmax_icdf (R/adfmax.R of the R package
dflimit)."""
import math

import numpy as np

from . import _engine
from ._ltratio import ERRORS_DOC
from ._utils import Result, cint, dfl_case, dfl_num, dfl_real, dfl_shape, f64, fill_doc, warn

ADFMAX_CASES = ("DU", "DT", "SU", "ST")

ADFMAX_NOTES_DOC = """\
ADFmax is the larger of the Dickey-Fuller t-ratios of the forward regression (y_t on y_{t-1}) and of the reverse
regression (y_t on y_{t+1}), under phi = 1 + c/n, no drift. In the limit both regressions have the denominator V and
the numerators U and -U - 1, so that ADFmax converges to (|R| - 1/2)/sqrt(V), R = U + 1/2, whose distribution is
computed by the conditional integral (the ADFmax test of the paper).

The cases are ``"DU"`` and ``"DT"`` (a constant, or a constant and a trend, zero start) and ``"SU"`` and ``"ST"``
(the same with a stationary start, ``c <= -1e-4``). The values are accurate to about 1e-11."""

ADFMAX_PARAMS_DOC = """\
cas : str
    The case, one of ``"DU"``, ``"DT"``, ``"SU"``, ``"ST"``.
c : float, default 0
    The local-to-unity parameter c, phi = 1 + c/n.
info : bool, default False
    If true, return a `Result` (a dict whose keys are also attributes) with the values ``value``, the error
    estimates ``err`` and the statuses ``status`` (nonzero: the error estimate is not met) of the conditional
    integral, each of the shape of ``x``."""

ADFMAX_REFERENCES_DOC = """\
.. [1] Leybourne, S. J. (1995), "Testing for Unit Roots Using Forward and Reverse Dickey-Fuller Regressions", Oxford
   Bulletin of Economics and Statistics 57, 559-571."""


@fill_doc(params=ADFMAX_PARAMS_DOC, errors=ERRORS_DOC, notes=ADFMAX_NOTES_DOC, references=ADFMAX_REFERENCES_DOC)
def adfmax_cdf(x, cas, c=0, info=False):
    """Distribution function of the limit of the ADFmax statistic of Leybourne (1995).

    Parameters
    ----------
    x : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    {params}

    Returns
    -------
    float or numpy.ndarray or Result
        The distribution function at ``x``: a float for a scalar ``x``, otherwise an array of the shape of ``x``
        (or the `Result` described under ``info``).

    Warns
    -----
    RuntimeWarning
        If the conditional integral reports, at some points, that its error estimate is not met.

    Raises
    ------
    {errors}

    See Also
    --------
    adfmax_pdf, adfmax_icdf, ltratio_cdf

    Notes
    -----
    {notes}

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import adfmax_cdf
    >>> adfmax_cdf(-2.43542, "DU")       # 0.05 (the quantiles of the ADFmax statistic in the paper)
    0.0499995786...
    """
    return _adfmax(2, x, cas, c, info)


@fill_doc(params=ADFMAX_PARAMS_DOC, errors=ERRORS_DOC, notes=ADFMAX_NOTES_DOC, references=ADFMAX_REFERENCES_DOC)
def adfmax_pdf(x, cas, c=0, info=False):
    """Density of the limit of the ADFmax statistic of Leybourne (1995).

    Parameters
    ----------
    x : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 0 and ``nan``.
    {params}

    Returns
    -------
    float or numpy.ndarray or Result
        The density at ``x``: a float for a scalar ``x``, otherwise an array of the shape of ``x`` (or the
        `Result` described under ``info``).

    Warns
    -----
    RuntimeWarning
        If the conditional integral reports, at some points, that its error estimate is not met.

    Raises
    ------
    {errors}

    See Also
    --------
    adfmax_cdf, adfmax_icdf

    Notes
    -----
    {notes}

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import adfmax_pdf
    >>> adfmax_pdf([-3, -2], "DT", -5)
    array([0.315718..., 0.599475...])
    """
    return _adfmax(1, x, cas, c, info)


def _adfmax(kind, x, cas, c, info):
    _, model, stat = dfl_case(cas, ADFMAX_CASES)
    a = dfl_real(x, "x")
    out, err, status = _engine.adfmax(kind, a.tobytes(), model, dfl_num(c, "c"), stat)
    st = cint(status)
    nbad = int(np.count_nonzero(st))
    if nbad:
        warn(f"the conditional integral did not meet its error estimate at {nbad} of {a.size} points; "
             "the result may be inaccurate there")
    r = Result(value=dfl_shape(f64(out), a), err=dfl_shape(f64(err), a), status=dfl_shape(st, a, int))
    return r if info else r["value"]


@fill_doc(errors=ERRORS_DOC, notes=ADFMAX_NOTES_DOC, references=ADFMAX_REFERENCES_DOC)
def adfmax_icdf(p, cas, c=0, tol=1e-10, x0=-2):
    """Quantiles of the limit of the ADFmax statistic of Leybourne (1995).

    `adfmax_icdf` has no entry point of its own in the C engine: as ``adfmax_icdf`` of the MATLAB and Julia programs,
    it finds each quantile by Newton's method on `adfmax_cdf` with `adfmax_pdf` as the derivative, safeguarded by
    bisection on a bracket grown from [x0 - 1, x0 + 1].

    Parameters
    ----------
    p : array_like
        The probabilities, in (0, 1).
    cas : str
        The case, one of ``"DU"``, ``"DT"``, ``"SU"``, ``"ST"``.
    c : float, default 0
        The local-to-unity parameter c, phi = 1 + c/n.
    tol : float, default 1e-10
        The tolerance on the quantile.
    x0 : float, default -2
        The starting point of the search.

    Returns
    -------
    float or numpy.ndarray
        The quantiles: a float for a scalar ``p``, otherwise an array of the shape of ``p``.

    Warns
    -----
    RuntimeWarning
        If Newton's method did not converge at a point.

    Raises
    ------
    {errors}

    See Also
    --------
    adfmax_cdf, adfmax_pdf

    Notes
    -----
    {notes}

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import adfmax_icdf
    >>> adfmax_icdf(0.05, "DT")          # -3.11891 (the quantiles of the ADFmax statistic in the paper)
    -3.11891085...
    """
    a = dfl_real(p, "p")
    if not np.all((a > 0) & (a < 1)):
        raise ValueError("p must lie in (0, 1)")
    dfl_case(cas, ADFMAX_CASES)
    c = dfl_num(c, "c")
    tol = dfl_num(tol, "tol")
    x0 = dfl_num(x0, "x0")
    q = np.empty(a.size)
    for k, pk in enumerate(a.ravel()):
        q[k] = _adfmax_quantile(pk, cas, c, tol, x0)
    return dfl_shape(q, a)


def _adfmax_quantile(pk, cas, c, tol, x0):
    """One quantile by safeguarded Newton (adfmax_icdf of the R package and of the MATLAB and Julia programs)."""
    def cdf(y):
        return adfmax_cdf(y, cas, c)

    lo, hi = x0 - 1, x0 + 1
    while cdf(lo) > pk:
        lo -= 1
    while cdf(hi) < pk:
        hi += 1
    x = (lo + hi) / 2
    for _ in range(100):
        F = cdf(x)
        f = adfmax_pdf(x, cas, c)
        if F > pk:
            hi = x
        else:
            lo = x
        xn = x - (F - pk) / f if f != 0 else math.nan     # (R: Inf or NaN, which the safeguard below catches)
        if not (math.isfinite(xn) and lo < xn < hi):
            xn = (lo + hi) / 2
        if abs(xn - x) < tol:
            return xn
        x = xn
    warn(f"no convergence at p = {pk:g}")
    return x
