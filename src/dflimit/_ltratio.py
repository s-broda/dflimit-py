"""The limiting distribution of the Dickey-Fuller t-ratio: ltratio_cdf, ltratio_pdf, ltratio_icdf, ltratio_mom
(R/ltratio.R of the R package dflimit)."""
import numpy as np

from . import _engine
from ._utils import Result, cint, dfl_case, dfl_method, dfl_num, dfl_posint, dfl_real, dfl_shape, f64, fill_doc, warn

CASES_DOC = """\
**Cases.** ``cas`` is a two-letter code, the initial observation (``D``: deterministic, i.e. fixed; ``S``: drawn
from the stationary distribution, which requires ``c < 0``) followed by the deterministic terms of the regression
(``K``: none, the intercept is known to be zero; ``U``: an unknown intercept; ``T``: an intercept and a trend):

- ``"DK"``, ``"DU"``, ``"DT"``: the statistics tau, tau_mu, tau_tau, with the start y_0 = g0 sigma sqrt(n) (``g0``,
  default 0);
- ``"SK"``, ``"SU"``, ``"ST"``: the same regressions with a stationary start;
- ``"DG"``: the GLS-detrended trend case of Elliott, Rothenberg and Stock (1996), detrended at cbar = -13.5 (the
  DF-GLS tests): zero start, the line integral only, no moments."""

METHODS_DOC = """\
**Methods and ranges.** Two exact representations are implemented: the line integral of Theorem 2 of the paper
(``method="single"``, or ``"line"``) and the conditional integral of Theorem S1 of its Supplementary Material
(``"double"``, or ``"conditional"``). ``"auto"`` (the default) takes the faster of the two at each point: the line
integral at ``c0 = 0`` and with a fixed start, and with ``c0 != 0`` and a zero start the conditional integral, except
for ``c0 > 0`` at ``x >= 0``. With ``"auto"`` the values of the line integral below 1e-3 are recomputed by the
conditional integral (whose accuracy is relative where that of the line integral is absolute), a point at which the
conditional integral reports that its error estimate is not met gets the line integral's value, and a computed
density below zero is returned as zero. Far from the unit root (a stationary start at ``c < -50``; ``c0 != 0`` with a
zero start at ``c < -100``) the line integral serves alone.

``c`` must be at least -550 (the range in which the programs are checked); the rules are set for ``c`` from about
-480 to 10. With a fixed start ``g0 != 0`` the functions are checked, and an error is raised outside, in case DK for
``c <= 2``, ``|g0| <= 2`` (``|g0| <= 1`` for ``c > 0``), in case DU for ``-100 <= c <= 2``, ``|g0| <= 1.5`` and
``(c g0)^2/2 <= 4050``, in case DT for ``-100 <= c <= 2``, ``|g0| <= 3`` and ``(c g0)^2/2 <= 7200``.

In case DK with ``c0 > 0`` the density is infinite at ``x = -sqrt(2 c0)``, and ``inf`` is returned there.

The computation runs on the engine's worker threads; see `ltratio_threads`."""

PARAMS_DOC = """\
cas : str
    The case, one of ``"DK"``, ``"DU"``, ``"DT"``, ``"SK"``, ``"SU"``, ``"ST"``, ``"DG"`` (see Notes).
c : float, default 0
    The local-to-unity parameter c, phi = 1 + c/n.
c0 : float, default 0
    The local null value c0, phi0 = 1 + c0/n.
g0 : float, default 0
    The fixed start, y_0 = g0 sigma sqrt(n) (cases DK, DU, DT).
method : {"auto", "single", "line", "double", "conditional"}, default "auto"
    The representation; see Notes."""

ERRORS_DOC = """\
TypeError
    If an argument has the wrong type.
ValueError
    If an argument is out of range (the engine's checks, with its message)."""

REFERENCES_DOC = """\
.. [1] Bao, Y., Broda, S. A. and Kan, R., "The Exact Limiting Distribution of the Dickey-Fuller t-Ratio and Some
   Related Statistics", working paper."""


@fill_doc(params=PARAMS_DOC, errors=ERRORS_DOC, cases=CASES_DOC, methods=METHODS_DOC, references=REFERENCES_DOC)
def ltratio_cdf(x, cas, c=0, c0=0, g0=0, method="auto"):
    """Distribution function of the limit of the Dickey-Fuller t-ratio.

    The limit, as the sample size grows, of the distribution function of the least squares t-ratio of the
    hypothesis phi = phi0 in the Gaussian first-order autoregression y_t = alpha + delta t + phi y_{t-1} + sigma e_t
    under the local alternatives phi = 1 + c/n, phi0 = 1 + c0/n, with no drift (alpha = delta = 0). With ``c0 = 0``
    (the default) these are the Dickey-Fuller tests of a unit root; ``c = 0`` is the null hypothesis.

    The limit is tau = (U - c0 V)/sqrt(V), with U = int J~ dJ and V = int J~^2 dr, J the Ornstein-Uhlenbeck process
    dJ = cJ dr + dW and J~ its residual on the deterministic terms of the regression.

    Parameters
    ----------
    x : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 (density 0) and ``nan``.
    {params}

    Returns
    -------
    float or numpy.ndarray
        The distribution function at ``x``: a float for a scalar ``x``, otherwise an array of the shape of ``x``.

    Warns
    -----
    RuntimeWarning
        With ``method="double"``, if the conditional integral reports, at some points, that its error estimate is
        not met.

    Raises
    ------
    {errors}

    See Also
    --------
    ltratio_pdf, ltratio_icdf, ltratio_mom, coef_cdf, adfmax_cdf, ltratio_region, ltratio_threads

    Notes
    -----
    {cases}

    {methods}

    References
    ----------
    {references}
    .. [2] Elliott, G., Rothenberg, T. J. and Stock, J. H. (1996), "Efficient Tests for an Autoregressive Unit
       Root", Econometrica 64, 813-836.

    Examples
    --------
    >>> from dflimit import ltratio_cdf
    >>> ltratio_cdf(-2.86157, "DU", -5)      # limiting power of the 5% test of tau_mu at c = -5
    0.1165135620...
    >>> ltratio_cdf(-2.86157, "SU", -5)      # the same with a stationary start
    0.1249641227...
    >>> ltratio_cdf(0, "DK", 2, c0=2)        # t-ratio centred at the true value
    0.6504355619...
    """
    return _dist(2, x, cas, c, c0, g0, method)


@fill_doc(params=PARAMS_DOC, errors=ERRORS_DOC, cases=CASES_DOC, methods=METHODS_DOC, references=REFERENCES_DOC)
def ltratio_pdf(x, cas, c=0, c0=0, g0=0, method="auto"):
    """Density of the limit of the Dickey-Fuller t-ratio.

    The density of the limit tau = (U - c0 V)/sqrt(V) of `ltratio_cdf`, under phi = 1 + c/n, phi0 = 1 + c0/n.

    Parameters
    ----------
    x : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 0 and ``nan``.
    {params}

    Returns
    -------
    float or numpy.ndarray
        The density at ``x``: a float for a scalar ``x``, otherwise an array of the shape of ``x``.

    Warns
    -----
    RuntimeWarning
        With ``method="double"``, if the conditional integral reports, at some points, that its error estimate is
        not met.

    Raises
    ------
    {errors}

    See Also
    --------
    ltratio_cdf, ltratio_icdf

    Notes
    -----
    {cases}

    {methods}

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import ltratio_pdf
    >>> ltratio_pdf([-3, -2, -1], "DT", -5)  # density
    array([0.405856..., 0.508800..., 0.040584...])
    """
    return _dist(1, x, cas, c, c0, g0, method)


def _dist(kind, x, cas, c, c0, g0, method):
    _, model, stat = dfl_case(cas)
    me = dfl_method(method)
    a = dfl_real(x, "x")
    out, ns = _engine.dist(kind, a.tobytes(), model, dfl_num(c, "c"), dfl_num(c0, "c0"), dfl_num(g0, "g0"), stat, me)
    if me == 2 and ns > 0:
        warn(f"the conditional integral did not meet its error estimate at {ns} of {a.size} points")
    return dfl_shape(f64(out), a)


@fill_doc(params=PARAMS_DOC, errors=ERRORS_DOC, cases=CASES_DOC, references=REFERENCES_DOC)
def ltratio_icdf(p, cas, c=0, c0=0, g0=0, method="auto", tol=1e-10, info=False):
    """Quantiles of the limit of the Dickey-Fuller t-ratio.

    The p-quantiles of the limit of `ltratio_cdf`, by Newton's method with `ltratio_pdf` as the derivative,
    safeguarded by bisection, started at the normal approximation from the first two moments (`ltratio_mom`).

    Parameters
    ----------
    p : array_like
        The probabilities, in (0, 1).
    {params}
    tol : float, default 1e-10
        The tolerance on the quantile.
    info : bool, default False
        If true, return a `Result` (a dict whose keys are also attributes) with the quantiles ``q``, the
        distribution function ``F`` at ``q``, the numbers of iterations ``iter`` and the convergence flags ``conv``,
        each of the shape of ``p``.

    Returns
    -------
    float or numpy.ndarray or Result
        The quantiles: a float for a scalar ``p``, otherwise an array of the shape of ``p`` (or the `Result`
        described under ``info``).

    Warns
    -----
    RuntimeWarning
        If Newton's method did not converge at some points.

    Raises
    ------
    {errors}

    See Also
    --------
    ltratio_cdf

    Notes
    -----
    {cases}

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import ltratio_icdf
    >>> ltratio_icdf([0.01, 0.05, 0.10], "DU")   # asymptotic critical values of tau_mu
    array([-3.430353..., -2.861568..., -2.566791...])
    """
    _, model, stat = dfl_case(cas)
    a = dfl_real(p, "p")
    q, F, it, cv = _engine.icdf(a.tobytes(), model, dfl_num(c, "c"), dfl_num(c0, "c0"), dfl_num(g0, "g0"), stat,
                                dfl_method(method), dfl_num(tol, "tol"))
    conv = cint(cv) != 0
    if not conv.all():
        warn(f"Newton's method did not converge at {int(np.count_nonzero(~conv))} of {conv.size} points")
    r = Result(q=dfl_shape(f64(q), a), F=dfl_shape(f64(F), a), iter=dfl_shape(cint(it), a, int),
               conv=dfl_shape(conv, a, bool))
    return r if info else r["q"]


@fill_doc(errors=ERRORS_DOC, cases=CASES_DOC, references=REFERENCES_DOC)
def ltratio_mom(p, cas, c=0, c0=0, g0=0, info=False):
    """Moments of the limit of the Dickey-Fuller t-ratio.

    The moments E[tau^j], j = 1, ..., p, of the limit tau = (U - c0 V)/sqrt(V) of `ltratio_cdf`, as one-dimensional
    integrals of the joint moment generating function of (U, V) (Proposition S1 of the Supplementary Material of the
    paper). Not available for case DG.

    Parameters
    ----------
    p : int
        The number of moments, a positive integer.
    cas : str
        The case, one of ``"DK"``, ``"DU"``, ``"DT"``, ``"SK"``, ``"SU"``, ``"ST"`` (see Notes; not ``"DG"``).
    c : float, default 0
        The local-to-unity parameter c, phi = 1 + c/n.
    c0 : float, default 0
        The local null value c0, phi0 = 1 + c0/n.
    g0 : float, default 0
        The fixed start, y_0 = g0 sigma sqrt(n) (cases DK, DU, DT).
    info : bool, default False
        If true, return a `Result` with the moments ``m`` and the error estimates ``err`` of the integrals.

    Returns
    -------
    numpy.ndarray or Result
        The moments, an array of length ``p`` (or the `Result` described under ``info``).

    Raises
    ------
    {errors}

    See Also
    --------
    ltratio_cdf

    Notes
    -----
    {cases}

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import ltratio_mom
    >>> ltratio_mom(4, "DT")        # the first four moments of tau_tau
    array([ -2.181355...,   5.320675..., -14.035800...,  39.554430...])
    >>> float(ltratio_mom(2, "DU")[1])     # 3.0559953450 (Nabeya, 1999)
    3.0559953450...
    """
    pm = dfl_posint(p, "p")
    _, model, stat = dfl_case(cas)
    m, e = _engine.moments(pm, model, dfl_num(c, "c"), dfl_num(c0, "c0"), dfl_num(g0, "g0"), stat)
    r = Result(m=f64(m), err=f64(e))
    return r if info else r["m"]
