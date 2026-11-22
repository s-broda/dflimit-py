"""The limiting distribution of the Dickey-Fuller t-ratio: ltratio_cdf, ltratio_pdf, ltratio_icdf, ltratio_mom
(R/ltratio.R of the R package dflimit).  With a design in place of the case, or with the keyword engine, they go to
the router over the classical and the general engine (_route.py); with a case string and no engine, to the classical
engine as before."""
import math

import numpy as np

from . import _engine
from ._route import (AUTO, CLASSICAL, DESIGN_DOC, ENGINE_DOC, F_ICDF, brittle, classical_ok, engine_of, route_tuple,
                     spec_of, warn_lines)
from ._utils import Result, cint, dfl_case, dfl_flag, dfl_method, dfl_num, dfl_posint, dfl_real, dfl_shape, f64, fill_doc, warn

# the operations of _engine.route (py_route in _engine.c)
R_CDF, R_PDF, R_ICDF, R_MOM, R_COEF, R_COEF_ICDF, R_ADF_CDF, R_ADF_PDF = range(1, 9)

CASES_DOC = """\
**Cases.** ``cas`` is a two-letter code, the initial observation (``D``: deterministic, i.e. fixed; ``S``: drawn
from the stationary distribution, which requires ``c <= -1e-4``: closer to zero the variance -1/(2c) of the start
is too large for the quadrature, and the functions raise an error) followed by the deterministic terms of the regression
(``K``: none, the intercept is known to be zero; ``U``: an unknown intercept; ``T``: an intercept and a trend):

- ``"DK"``, ``"DU"``, ``"DT"``: the statistics tau, tau_mu, tau_tau, with the start y_0 = g0 sigma sqrt(n) (``g0``,
  default 0);
- ``"SK"``, ``"SU"``, ``"ST"``: the same regressions with a stationary start;
- ``"DG"``: the GLS-detrended trend case of Elliott, Rothenberg and Stock (1996), detrended at cbar = -13.5 (the
  DF-GLS tests): zero start, the line integral only, no moments."""

METHODS_DOC = """\
**Methods and ranges.** Three exact representations of the paper are implemented: the line integral representation
(``method="single"``, or ``"line"``), the conditional integral representation (``"double"``, or ``"conditional"``) and,
for a fixed start ``g0 != 0`` in cases DU and DT only, the contour integral representation (``"contour"``): a double
integral from the inversion formula on paths around the singularities of the moment generating function,
with the integral over v in closed form and the paths in s and t deformed, an explicit integrand where the line
integral needs a noncentral kernel that is itself an integral.

By default (``verified=False``) every value is one evaluation by one method: no second evaluation, no refinement driven
by an error estimate, no fallback to another representation and no recomputation. ``"auto"`` (the default) takes at
each point the fastest representation that one evaluation gives to well below 1e-5: the contour integral with a fixed
start in cases DU, DT (the conditional integral there for ``c > 1`` with ``c0 < 0`` and, in case DU, for ``c > 1.5``
with ``c0 > 3``, where the contour integral fails), the line integral with a fixed start in case DK, at ``c0 = 0``, with a
zero start and ``c0 > 0`` (but the conditional integral at ``x < 0`` for ``c >= 0`` in cases DU, DT), in case DK with
``c0 < 0`` for ``c > -5``, with a stationary start in cases SU, ST and in case
SK for ``c > -0.5``, and far from the unit root (``c < -100`` with a zero start, ``c < -50`` with a stationary start), and
the conditional integral otherwise. Where the parameters lie in a regime in which one evaluation is known to be brittle a
``RuntimeWarning`` names it (once per call). ``verified=True`` gives the checked computation, at 1 to 4.5 times the
cost of the default (about 1.8 with the contour integral; more in the tails below 1e-3, which it recomputes): the
conditional integral with the agreement of two paths and the refinement of their rules, the contour integral
as two evaluations on different paths refined until they agree, a warning where the error estimate of ``"double"`` or
``"contour"`` is not met, and with ``"auto"`` the rule before (the conditional integral with ``c0 != 0`` and a zero start
except for ``c0 > 0`` at ``x >= 0``), the flagged values given another representation's value and the values below
1e-3 recomputed by the conditional integral. A computed density below zero is returned as zero.

``c`` must be at least -550 (the range in which the programs are checked); the rules are set for ``c`` from about
-480 to 10. With a fixed start ``g0 != 0`` the functions are checked, and an error is raised outside, in case DK for
``c <= 2``, ``|g0| <= 2`` (``|g0| <= 1`` for ``c > 0``), in case DU for ``-100 <= c <= 2``, ``|g0| <= 1.5`` and
``(c g0)^2/2 <= 4050``, in case DT for ``-100 <= c <= 2``, ``|g0| <= 3`` and ``(c g0)^2/2 <= 7200``.

In case DK with ``c0 > 0`` the density is infinite at ``x = -sqrt(2 c0)``, and ``inf`` is returned there.

In two regions the line integral would need more quadrature nodes than the engine allows: with ``c0 < 0`` near zero and
``x > 0``, where a root of the parabola X(v) = c0 v^2 + x v + 1/2 lies far out (near ``x/|c0|``), more than two million
nodes on one path of its inner integrals (from ``c0 = -1e-4`` at ``x = 0.05``, at ``c0 = -1e-3`` near ``x = 1.5``, at
``c0 = -0.01`` in places, at ``c0 = -0.1`` only far above the unit root, as at ``c = 5``, ``x = 14``), and above the
unit root at large ``x``, a Gauss-Legendre rule of more than 4096 nodes (at ``c = 7`` from ``x`` of about 660 to 780).
The line integral refuses such a point before evaluating it: ``"auto"`` takes the conditional integral there, with its
checks, and gives a ``RuntimeWarning`` where such a value near ``c0 = 0`` is flagged; ``method="single"``, and case DG,
which has the line integral only, raise a ``ValueError`` that names the arguments.

The computation runs on the engine's worker threads; see `ltratio_threads`."""

PARAMS_DOC = """\
cas : str or Design
    The case, one of ``"DK"``, ``"DU"``, ``"DT"``, ``"SK"``, ``"SU"``, ``"ST"``, ``"DG"`` (see Notes), or a design of
    the general engine (`trend`, `perron`, `fdesign`, `ltrend`, `design_from_matrix`, `detrend_first`).
c : float, default 0
    The local-to-unity parameter c, phi = 1 + c/n.
c0 : float, default 0
    The local null value c0, phi0 = 1 + c0/n.
g0 : float, default 0
    The fixed start, y_0 = g0 sigma sqrt(n) (cases DK, DU, DT).
method : {"auto", "single", "line", "double", "conditional", "contour"}, default "auto"
    The representation of the classical engine (``"contour"``: a fixed start in cases DU, DT only); see Notes. A
    method other than ``"auto"`` keeps the classical engine with ``engine="auto"``.
verified : bool, default False
    False: one evaluation by one method; True: the checked computation (see Notes; with ``engine="auto"`` the
    classical engine's)."""

ROUTE_DOC = """\
**The general engine.** With a design in place of the case, or with ``engine="auto"`` or ``"general"``, the call
goes to the router over the classical engine and the general engine (any piecewise smooth deterministic terms, the
inversion on paths around the roots of the determinant of the moment generating function): ``engine="auto"`` takes,
at each point, the faster engine where the two agree and the more accurate where they do not, and the general engine
where the classical one does not apply (the README, "The general engine"). With a case string and no ``engine`` the
classical engine computes, as before. The general engine's warnings (where one evaluation may be inaccurate) are
``RuntimeWarning``s."""

ERRORS_DOC = """\
TypeError
    If an argument has the wrong type.
ValueError
    If an argument is out of range (the engine's checks, with its message)."""

REFERENCES_DOC = """\
.. [1] Bao, Y., Broda, S. A. and Kan, R., "The Exact Limiting Distribution of the Dickey-Fuller t-Ratio and Some
   Related Statistics", working paper."""


@fill_doc(params=PARAMS_DOC, errors=ERRORS_DOC, cases=CASES_DOC, methods=METHODS_DOC, references=REFERENCES_DOC,
          engine=ENGINE_DOC, design=DESIGN_DOC, route=ROUTE_DOC)
def ltratio_cdf(x, cas, c=0, c0=0, g0=0, method="auto", verified=False, *, engine=None, start=None, gls=None):
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
    {engine}
    {design}

    Returns
    -------
    float or numpy.ndarray
        The distribution function at ``x``: a float for a scalar ``x``, otherwise an array of the shape of ``x``.

    Warns
    -----
    RuntimeWarning
        In a regime in which one evaluation is known to be brittle; where points that the line integral refuses get
        flagged values from the conditional integral (see Notes); with ``verified=True`` and ``method="double"`` or
        ``"contour"``, if the integral reports at some points that its error estimate is not met; where the general
        engine reports that one evaluation may be inaccurate.

    Raises
    ------
    {errors}

    See Also
    --------
    ltratio_pdf, ltratio_icdf, ltratio_mom, coef_cdf, adfmax_cdf, ltratio_region, ltratio_threads, dist_cdf, perron

    Notes
    -----
    {cases}

    {methods}

    {route}

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
    >>> from dflimit import perron
    >>> ltratio_cdf(-4.5, perron("C", 0.3))  # Perron's model C with a break at 0.3 (the general engine)
    0.014908314...
    >>> ltratio_cdf(-2.86157, "DU", -5, engine="auto")    # the router's rule for a case string
    0.1165135620...
    """
    return _dist(2, x, cas, c, c0, g0, method, verified, engine, start, gls)


@fill_doc(params=PARAMS_DOC, errors=ERRORS_DOC, cases=CASES_DOC, methods=METHODS_DOC, references=REFERENCES_DOC,
          engine=ENGINE_DOC, design=DESIGN_DOC, route=ROUTE_DOC)
def ltratio_pdf(x, cas, c=0, c0=0, g0=0, method="auto", verified=False, *, engine=None, start=None, gls=None):
    """Density of the limit of the Dickey-Fuller t-ratio.

    The density of the limit tau = (U - c0 V)/sqrt(V) of `ltratio_cdf`, under phi = 1 + c/n, phi0 = 1 + c0/n.

    Parameters
    ----------
    x : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 0 and ``nan``.
    {params}
    {engine}
    {design}

    Returns
    -------
    float or numpy.ndarray
        The density at ``x``: a float for a scalar ``x``, otherwise an array of the shape of ``x``.

    Warns
    -----
    RuntimeWarning
        In a regime in which one evaluation is known to be brittle; where points that the line integral refuses get
        flagged values from the conditional integral (see Notes); with ``verified=True`` and ``method="double"`` or
        ``"contour"``, if the integral reports at some points that its error estimate is not met.

    Raises
    ------
    {errors}

    See Also
    --------
    ltratio_cdf, ltratio_icdf, dist_pdf

    Notes
    -----
    {cases}

    {methods}

    {route}

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import ltratio_pdf
    >>> ltratio_pdf([-3, -2, -1], "DT", -5)  # density
    array([0.405856..., 0.508800..., 0.040584...])
    >>> from dflimit import trend
    >>> ltratio_pdf(-3.5, trend(2), -5)      # a quadratic trend
    0.326334...
    """
    return _dist(1, x, cas, c, c0, g0, method, verified, engine, start, gls)


def _classical(S, e, c, c0, me, ve):
    """Whether the classical engine computes: engine "classical", or "auto" with a representation named or the checked
    computation, where the classical engine accepts the arguments (for a case string its checks give the errors)."""
    return e == CLASSICAL or (e == AUTO and (me != 0 or ve) and (S.named or classical_ok(S, c, c0, me)))


def _dist(kind, x, cas, c, c0, g0, method, verified=False, engine=None, start=None, gls=None):
    S = spec_of(cas, g0, start, gls)
    e = engine_of(S, engine)
    me = dfl_method(method)
    ve = dfl_flag(verified, "verified")
    if _classical(S, e, c, c0, me, ve):
        return _dist_classical(kind, x, S.case, c, c0, S.g0, me, ve)
    a = dfl_real(x, "x")
    c, c0 = dfl_num(c, "c"), dfl_num(c0, "c0")
    out, wd, wc = _engine.route(R_CDF if kind == 2 else R_PDF, route_tuple(S), a.tobytes(), c, c0, e, me, ve, 0.0)
    v = f64(out)
    warn_lines(brittle(kind, S, c, c0, a, v) if e == AUTO else "", wc, wd)
    return dfl_shape(v, a)


def _dist_classical(kind, x, cas, c, c0, g0, me, ve):
    """The classical engine (dfl_dist), with its warnings."""
    _, model, stat = dfl_case(cas)
    a = dfl_real(x, "x")
    out, ns, bm, wm = _engine.dist(kind, a.tobytes(), model, dfl_num(c, "c"), dfl_num(c0, "c0"), dfl_num(g0, "g0"),
                                   stat, me, ve)
    if ve and me >= 2 and ns > 0:
        name = "conditional" if me == 2 else "contour"
        warn(f"the {name} integral did not meet its error estimate at {ns} of {a.size} points")
    if bm:
        warn(bm)
    if wm:
        warn(wm)
    return dfl_shape(f64(out), a)


@fill_doc(params=PARAMS_DOC, errors=ERRORS_DOC, cases=CASES_DOC, references=REFERENCES_DOC, engine=ENGINE_DOC,
          design=DESIGN_DOC, route=ROUTE_DOC)
def ltratio_icdf(p, cas, c=0, c0=0, g0=0, method="auto", tol=1e-10, info=False, verified=False, *, engine=None, start=None,
                 gls=None):
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
        each of the shape of ``p`` (by the router: ``q`` and ``F`` only).
    {engine}
    {design}

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
    ltratio_cdf, dist_quantile

    Notes
    -----
    {cases}

    {route}

    By the router the quantiles of the general engine start at the normal approximation from its moments too;
    ``verified=True`` is the classical engine's.

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import ltratio_icdf
    >>> ltratio_icdf([0.01, 0.05, 0.10], "DU")   # asymptotic critical values of tau_mu
    array([-3.430353..., -2.861568..., -2.566791...])
    >>> from dflimit import perron
    >>> ltratio_icdf(0.05, perron("C", [0.3, 0.7]))     # Perron's model C with two breaks
    -4.750569...
    """
    S = spec_of(cas, g0, start, gls)
    e = engine_of(S, engine)
    me = dfl_method(method)
    ve = dfl_flag(verified, "verified")
    if _classical(S, e, c, c0, me, ve):
        return _icdf_classical(p, S.case, c, c0, S.g0, me, tol, info, ve)
    if ve:
        raise ValueError("verified=True: the checked computation of the quantiles is the classical engine's; the general "
                         "engine has none")
    a = dfl_real(p, "p")
    c, c0, tol = dfl_num(c, "c"), dfl_num(c0, "c0"), dfl_num(tol, "tol")
    q, wd, wc = _engine.route(R_ICDF, route_tuple(S), a.tobytes(), c, c0, e, me, 0, tol)
    qv = f64(q)
    # (the brittle regime of the classical search, where the rule takes it)
    warn_lines(brittle(2, S, c, c0, [0.0], f=F_ICDF) if e == AUTO else "", wc, wd)
    q = dfl_shape(qv, a)
    if not info:
        return q
    return Result(q=q, F=_dist(2, q, cas, c, c0, g0, method, False, engine, start, gls))


def _icdf_classical(p, cas, c, c0, g0, me, tol, info, ve):
    """The quantiles of the classical engine (dfl_icdf), with its warnings."""
    _, model, stat = dfl_case(cas)
    a = dfl_real(p, "p")
    q, F, it, cv, bm, wm = _engine.icdf(a.tobytes(), model, dfl_num(c, "c"), dfl_num(c0, "c0"), dfl_num(g0, "g0"), stat,
                                        me, dfl_num(tol, "tol"), ve)
    if bm:
        warn(bm)
    if wm:
        warn(wm)
    conv = cint(cv) != 0
    if not conv.all():
        warn(f"Newton's method did not converge at {int(np.count_nonzero(~conv))} of {conv.size} points")
    r = Result(q=dfl_shape(f64(q), a), F=dfl_shape(f64(F), a), iter=dfl_shape(cint(it), a, int),
               conv=dfl_shape(conv, a, bool))
    return r if info else r["q"]


@fill_doc(errors=ERRORS_DOC, cases=CASES_DOC, references=REFERENCES_DOC, engine=ENGINE_DOC, design=DESIGN_DOC,
          route=ROUTE_DOC)
def ltratio_mom(p, cas, c=0, c0=0, g0=0, info=False, *, engine=None, start=None, gls=None):
    """Moments of the limit of the Dickey-Fuller t-ratio.

    The moments E[tau^j], j = 1, ..., p, of the limit tau = (U - c0 V)/sqrt(V) of `ltratio_cdf`, as one-dimensional
    integrals of the joint moment generating function of (U, V) (the moments of the limit in the Supplementary
    Material of the paper). Not available for case DG.

    Parameters
    ----------
    p : int
        The number of moments, a positive integer.
    cas : str or Design
        The case, one of ``"DK"``, ``"DU"``, ``"DT"``, ``"SK"``, ``"SU"``, ``"ST"`` (see Notes; not ``"DG"``), or a
        design of the general engine.
    c : float, default 0
        The local-to-unity parameter c, phi = 1 + c/n.
    c0 : float, default 0
        The local null value c0, phi0 = 1 + c0/n.
    g0 : float, default 0
        The fixed start, y_0 = g0 sigma sqrt(n) (cases DK, DU, DT).
    info : bool, default False
        If true, return a `Result` with the moments ``m`` and the error estimates ``err`` of the integrals (by the
        router ``nan``; `dist_moments` has those of the general engine).
    {engine}
    {design}

    Returns
    -------
    numpy.ndarray or Result
        The moments, an array of length ``p`` (or the `Result` described under ``info``).

    Raises
    ------
    {errors}

    See Also
    --------
    ltratio_cdf, dist_moments

    Notes
    -----
    {cases}

    {route}

    The general engine has the moments of case DG and of other designs too (not of the statistic that detrends first).

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
    >>> from dflimit import trend
    >>> ltratio_mom(2, trend(2))           # a quadratic trend
    array([-2.631415..., 7.445401...])
    """
    pm = dfl_posint(p, "p")
    S = spec_of(cas, g0, start, gls)
    e = engine_of(S, engine)
    if e == CLASSICAL:
        _, model, stat = dfl_case(S.case)
        m, er = _engine.moments(pm, model, dfl_num(c, "c"), dfl_num(c0, "c0"), S.g0, stat)
        r = Result(m=f64(m), err=f64(er))
        return r if info else r["m"]
    m, wd, wc = _engine.route(R_MOM, route_tuple(S), np.array([float(pm)]).tobytes(), dfl_num(c, "c"), dfl_num(c0, "c0"), e,
                              0, 0, 0.0)
    warn_lines(wc, wd)
    r = Result(m=f64(m), err=np.full(pm, math.nan))
    return r if info else r["m"]
