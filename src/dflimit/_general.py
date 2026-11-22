"""The general engine's functions of a design (the names of DFLimit.jl): the t-ratio (dist_cdf, dist_pdf,
dist_quantile, dist_moments), the coefficient statistic (coef_pdf, coef_quantile, coef_cdf_gp), local power, the
linear form U - aV and the law of V (lin_cdf, lin_pdf, lin_quantile, v_cdf, v_pdf), ADFmax (adfmax_quantile), regions
(region_prob), and the companions: the point-optimal statistic (pt_*), the LM statistic (lm_*), MZ_alpha (mza_*) and
MSB (msb_*); empty_caches.  The distribution function of the coefficient statistic and of ADFmax, ADFmax's density and
the power envelope of a design are the classical functions coef_cdf, adfmax_cdf, adfmax_pdf and power_envelope, which
take a design (as in the Julia package, where these names are the classical engine's).

The engine is csrc/dfl_det_*.c: the joint moment generating function of numerator and denominator as a small
determinant, and the inversion formula on paths around its singularities in the first argument.
"""
import math
from collections.abc import Mapping

import numpy as np

from . import _engine
from ._design import Design
from ._route import gls_value, start_code, warn_lines
from ._utils import Result, dfl_flag, dfl_num, dfl_posint, dfl_real, dfl_shape, f64, fill_doc

# the operations of _engine.det (det_run in _engine.c)
(DET_CDF, DET_PDF, DET_COEF_CDF, DET_COEF_PDF, DET_COEF_GP, DET_LIN_CDF, DET_LIN_PDF, DET_ADF_CDF, DET_ADF_PDF, DET_LPOW,
 DET_ENV, DET_PT_CDF, DET_PT_PDF, DET_PT_ENV, DET_LM_CDF, DET_MZA_CDF, DET_MSB_CDF) = range(1, 18)
DET_Q, DET_COEF_Q, DET_LIN_Q, DET_ADF_Q, DET_PT_Q, DET_LM_Q, DET_MZA_Q, DET_MSB_Q = range(30, 38)
DET_PT_POW, DET_LM_POW = 50, 51

OPTS_DOC = """\
c : float, default 0
    The local-to-unity parameter, phi = 1 + c/n.
c0 : float, default 0
    The local null value, phi0 = 1 + c0/n.
start : {None, "zero", "fixed", "stationary"}, optional
    The start: J(0) = 0 (``"zero"``), J(0) = ``gamma0`` (``"fixed"``; y_0 = gamma0 sigma sqrt(n)) or J(0) ~ N(0,
    -1/(2c)) (``"stationary"``, c < 0); ``None``: ``"fixed"`` when ``gamma0 != 0``, else ``"zero"``.
gamma0 : float, default 0
    The fixed start.
gls : float, optional
    GLS detrending at cbar = ``gls`` < 0 (the data quasi-differenced at 1 + gls/n; zero start); default least
    squares."""

VERIFIED_DOC = """\
verified : bool, default False
    False: one evaluation; True: the checked computation (two evaluations on different paths with different rules, a
    third where they differ by more than 1e-9, and a warning where it does not settle)."""

NOTES_DOC = """\
The general engine computes the limit for any piecewise smooth design by the inversion formula on paths around the
singularities of the joint moment generating function of numerator and denominator, a small determinant. Points of
one call share their paths: pass a whole vector in one call. The prepared designs and the paths are cached (see
`empty_caches`). Where the roots of a design run far out on the vertical line (``c > 0``, exotic designs such as
(1 + r, 1{r < 0.4}), which has no proper intercept) the paths around them would need more than a million nodes at a node
of the outer contour, and the engine refuses with a ``ValueError``."""


def _design(D):
    if not isinstance(D, Design):
        raise TypeError("D must be a Design (trend, perron, fdesign, ltrend, design_from_matrix, detrend_first)")
    return D._tuple()


def _opts(c0=0.0, start=None, gamma0=0.0, gls=None, verified=False):
    g0 = dfl_num(gamma0, "gamma0")
    s = start_code(start, g0, "gamma0")
    return (dfl_num(c0, "c0"), s, g0 if s == 1 else 0.0, gls_value(gls), dfl_flag(verified, "verified"))


_NOOPTS = (0.0, 0, 0.0, math.nan, 0)


def _call(op, D, a, c, opts, *par):
    """The engine's values at the flat points a (one call)."""
    out, w = _engine.det(op, _design(D), np.ascontiguousarray(a, dtype=np.float64).tobytes(), dfl_num(c, "c"), opts,
                         *[float(p) for p in par])
    warn_lines(w)
    return f64(out)


def _values(op, D, x, c, opts, *par, name="x", cdf=None):
    """The values at the points x in the shape of x; cdf True/False: the non-finite points set here (0 at -inf, 1 or 0
    at inf, nan at nan) and the finite ones computed (for the functions of the engine that do not set them)."""
    a = dfl_real(x, name)
    flat = a.ravel()
    if cdf is None:
        out = _call(op, D, flat, c, opts, *par).copy()
    else:
        out = np.where(np.isnan(flat), math.nan, np.where(flat > 0, 1.0 if cdf else 0.0, 0.0))
        fin = np.isfinite(flat)
        if fin.any():
            out[fin] = _call(op, D, flat[fin], c, opts, *par)
    return dfl_shape(out, a)


def _quantiles(op, D, p, c, opts, *par):
    """The quantiles at the probabilities p (in (0, 1): the engine's check) in the shape of p."""
    a = dfl_real(p, "p")
    return dfl_shape(_call(op, D, a.ravel(), c, opts, *par).copy(), a)


@fill_doc(opts=OPTS_DOC, verified=VERIFIED_DOC, notes=NOTES_DOC)
def dist_cdf(D, x, c=0, *, c0=0, start=None, gamma0=0, gls=None, verified=False):
    """Distribution function of the limit of the Dickey-Fuller t-ratio for a design, by the general engine.

    The limit (U - c0 V)/sqrt(V) of the t-ratio of phi = phi0 = 1 + c0/n in the regression of y_t on d(t/n) and
    y_{t-1}, d the design, under phi = 1 + c/n.

    Parameters
    ----------
    D : Design
        The design.
    x : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    {opts}
    {verified}

    Returns
    -------
    float or numpy.ndarray
        The distribution function at ``x``: a float for a scalar ``x``, otherwise an array of the shape of ``x``.

    Warns
    -----
    RuntimeWarning
        Where the engine reports that one evaluation may be inaccurate.

    See Also
    --------
    dist_pdf, dist_quantile, ltratio_cdf

    Notes
    -----
    {notes}

    Examples
    --------
    >>> from dflimit import dist_cdf, perron
    >>> dist_cdf(perron("C", [0.3, 0.7]), [-5, -4, -3], -5)
    array([0.030755..., 0.292180..., 0.790536...])
    """
    return _values(DET_CDF, D, x, c, _opts(c0, start, gamma0, gls, verified))


@fill_doc(opts=OPTS_DOC, verified=VERIFIED_DOC, notes=NOTES_DOC)
def dist_pdf(D, x, c=0, *, c0=0, start=None, gamma0=0, gls=None, verified=False):
    """Density of the limit of the Dickey-Fuller t-ratio for a design, by the general engine.

    For a design whose numerator has rank one (no deterministic terms, GLS detrending) with ``c0 > 0`` the density is
    infinite at the tangency x = -sqrt(4 c0 a) (a = 1/2, or (1 + gamma0^2)/2 with a fixed start), and ``inf`` is
    returned there.

    Parameters
    ----------
    D : Design
        The design.
    x : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 0 and ``nan``.
    {opts}
    {verified}

    Returns
    -------
    float or numpy.ndarray
        The density at ``x``.

    Warns
    -----
    RuntimeWarning
        Where the engine reports that one evaluation may be inaccurate.

    See Also
    --------
    dist_cdf, ltratio_pdf

    Notes
    -----
    {notes}

    Examples
    --------
    >>> from dflimit import dist_pdf, trend
    >>> dist_pdf(trend(2), -3.8)               # the quadratic trend at the unit root
    0.142477...
    """
    return _values(DET_PDF, D, x, c, _opts(c0, start, gamma0, gls, verified))


@fill_doc(opts=OPTS_DOC, notes=NOTES_DOC)
def dist_quantile(D, p, c=0, *, x0=-2, tol=1e-9, c0=0, start=None, gamma0=0, gls=None):
    """Quantiles of the limit of the Dickey-Fuller t-ratio for a design, by the general engine.

    Newton's method from ``x0`` on `dist_cdf` with `dist_pdf` from the same evaluation (the step limited to 1), the
    safeguarded search where it does not converge. `ltratio_icdf` with a design starts each quantile at the normal
    approximation from the moments instead.

    Parameters
    ----------
    D : Design
        The design.
    p : array_like
        The probabilities, in (0, 1).
    {opts}
    x0 : float, default -2
        The start of the search.
    tol : float, default 1e-9
        The tolerance on the quantile.

    Returns
    -------
    float or numpy.ndarray
        The quantiles: a float for a scalar ``p``, otherwise an array of the shape of ``p``.

    See Also
    --------
    dist_cdf, ltratio_icdf

    Notes
    -----
    {notes}

    Examples
    --------
    >>> from dflimit import dist_quantile, perron
    >>> dist_quantile(perron("C", 0.3), [0.01, 0.05, 0.10])     # critical values of Perron's model C
    array([-4.625894..., -4.072544..., -3.782146...])
    """
    return _quantiles(DET_Q, D, p, c, _opts(c0, start, gamma0, gls), dfl_num(x0, "x0"), dfl_num(tol, "tol"))


@fill_doc(opts=OPTS_DOC)
def dist_moments(D, c=0, *, p=4, c0=0, start=None, gamma0=0, gls=None, info=False):
    """Moments of the limit of the Dickey-Fuller t-ratio for a design, by the general engine.

    E tau^k, k = 1, ..., p, from Cauchy integrals of the moment generating function on circles whose logarithm is
    continued around them, and a trapezoidal rule in log s. Not for the statistic that detrends first.

    Parameters
    ----------
    D : Design
        The design.
    {opts}
    p : int, default 4
        The number of moments.
    info : bool, default False
        If true, return a `Result` with the moments ``m`` and ``err``, the differences to the rule of twice the step.

    Returns
    -------
    numpy.ndarray or Result
        The moments, an array of length ``p`` (or the `Result` described under ``info``).

    See Also
    --------
    ltratio_mom

    Examples
    --------
    >>> from dflimit import dist_moments, trend
    >>> dist_moments(trend(1), p=2)          # E tau_tau, E tau_tau^2
    array([-2.181355..., 5.320675...])
    """
    pm = dfl_posint(p, "p")
    m, e, w = _engine.det_moments(_design(D), dfl_num(c, "c"), pm, _opts(c0, start, gamma0, gls))
    warn_lines(w)
    r = Result(m=f64(m).copy(), err=f64(e).copy())
    return r if info else r["m"]


@fill_doc(opts=OPTS_DOC, verified=VERIFIED_DOC)
def coef_pdf(D, y, c=0, *, c0=0, start=None, gamma0=0, gls=None, verified=False):
    """Density of the limit of the coefficient statistic n(phi^ - phi0) for a design, by the general engine.

    The limit is (U - c0 V)/V; its distribution function is `coef_cdf` with the design.

    Parameters
    ----------
    D : Design
        The design.
    y : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 0 and ``nan``.
    {opts}
    {verified}

    Returns
    -------
    float or numpy.ndarray
        The density at ``y``.

    See Also
    --------
    coef_cdf, coef_quantile

    Examples
    --------
    >>> from dflimit import coef_pdf, trend
    >>> coef_pdf(trend(0), [-12.0, -6.0])
    array([0.020277..., 0.079232...])
    """
    return _values(DET_COEF_PDF, D, y, c, _opts(c0, start, gamma0, gls, verified), name="y")


@fill_doc(opts=OPTS_DOC)
def coef_quantile(D, p, c=0, *, y0=-10, tol=1e-9, c0=0, start=None, gamma0=0, gls=None):
    """Quantiles of the limit of the coefficient statistic n(phi^ - phi0) for a design, by the general engine.

    Newton's method from ``y0`` (the step limited to 5), the safeguarded search where it does not converge.

    Parameters
    ----------
    D : Design
        The design.
    p : array_like
        The probabilities, in (0, 1).
    {opts}
    y0 : float, default -10
        The start of the search.
    tol : float, default 1e-9
        The tolerance on the quantile.

    Returns
    -------
    float or numpy.ndarray
        The quantiles.

    See Also
    --------
    coef_cdf, coef_icdf

    Examples
    --------
    >>> from dflimit import coef_quantile, trend
    >>> coef_quantile(trend(1), 0.05)       # the 5% critical value of the coefficient test with a trend
    -21.71122...
    """
    return _quantiles(DET_COEF_Q, D, p, c, _opts(c0, start, gamma0, gls), dfl_num(y0, "y0"), dfl_num(tol, "tol"))


@fill_doc(opts=OPTS_DOC)
def coef_cdf_gp(D, y, c=0, *, c0=0, start=None, gamma0=0, gls=None, N=60000, umax=4000):
    """Distribution function of the limit of the coefficient statistic by the inversion formula of Gil-Pelaez.

    P[(U - c0 V)/V <= y] = P[U - (c0 + y) V <= 0] from the characteristic function on the trapezoidal rule in u =
    sinh(v) with ``N`` steps up to ``umax`` (a check of `coef_cdf`, which inverts on paths around the roots).

    Parameters
    ----------
    D : Design
        The design.
    y : array_like
        The points (finite).
    {opts}
    N : int, default 60000
        The number of steps.
    umax : float, default 4000
        The end of the grid.

    Returns
    -------
    float or numpy.ndarray
        The distribution function at ``y``.

    See Also
    --------
    coef_cdf

    Examples
    --------
    >>> from dflimit import coef_cdf_gp, trend
    >>> coef_cdf_gp(trend(0), -14.0936)      # 0.05
    0.04999988...
    """
    return _values(DET_COEF_GP, D, y, c, _opts(c0, start, gamma0, gls), dfl_posint(N, "N"), dfl_num(umax, "umax"),
                   name="y", cdf=True)


def local_power(D, cs, *, level=0.05, statistic="tratio", c0=0, start=None, gamma0=0, gls=None):
    """Asymptotic local power of the left-tailed test of phi = 1 + c0/n for a design.

    P_c(stat <= q) at each c of ``cs``, q the ``level``-quantile under c = c0 (with the same start; a stationary
    start needs c0 < 0), stat the t-ratio (``statistic="tratio"``) or the coefficient statistic n(phi^ - phi0)
    (``"coef"``).

    Parameters
    ----------
    D : Design
        The design.
    cs : array_like
        The local alternatives c.
    level : float, default 0.05
        The level of the test.
    statistic : {"tratio", "coef"}, default "tratio"
        The statistic.
    c0, start, gamma0, gls
        As in `dist_cdf`.

    Returns
    -------
    float or numpy.ndarray
        The power at ``cs``.

    See Also
    --------
    power_envelope, dist_cdf

    Examples
    --------
    >>> from dflimit import local_power, trend
    >>> local_power(trend(1), [-5.0, -13.5])        # the power of the 5% test of tau_tau
    array([0.082946..., 0.302181...])
    """
    if statistic not in ("tratio", "coef"):
        raise ValueError('statistic must be "tratio" or "coef"')
    return _values(DET_LPOW, D, cs, 0.0, _opts(c0, start, gamma0, gls), dfl_num(level, "level"),
                   0 if statistic == "tratio" else 1, name="cs")


def _a(a):
    """The coefficient a of the linear form: a number, or nan for the law of V."""
    v = a[()] if isinstance(a, np.ndarray) and a.ndim == 0 else a
    if isinstance(v, (float, np.floating)) and math.isnan(v):
        return math.nan
    try:
        return dfl_num(a, "a")
    except (TypeError, ValueError):
        raise TypeError("a must be a single number (nan: the law of V)") from None


@fill_doc(opts=OPTS_DOC, verified=VERIFIED_DOC)
def lin_cdf(D, b, a, c=0, *, start=None, gamma0=0, gls=None, verified=False):
    """Limiting distribution of the linear form U - aV for a design, by the general engine.

    P(U - aV <= b), U and V the numerator and the denominator of the limit of the t-ratio; ``a = nan`` gives P(V <= b)
    (`v_cdf`). The point-optimal test against cbar rejects for small U - aV.

    Parameters
    ----------
    D : Design
        The design.
    b : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    a : float
        The coefficient of V (``nan``: the law of V).
    {opts}
    {verified}

    Returns
    -------
    float or numpy.ndarray
        The probabilities at ``b``.

    See Also
    --------
    lin_pdf, lin_quantile, ltratio_lincdf

    Examples
    --------
    >>> from dflimit import lin_cdf, trend
    >>> lin_cdf(trend(-1), -0.2639, -3.5)    # the size of the point-optimal test against cbar = -7
    0.04999...
    """
    av = _a(a)
    return _values(DET_LIN_CDF, D, b, c, _opts(0.0, start, gamma0, gls, verified), av, name="b")


@fill_doc(opts=OPTS_DOC, verified=VERIFIED_DOC)
def lin_pdf(D, b, a, c=0, *, start=None, gamma0=0, gls=None, verified=False):
    """Density of the limit of the linear form U - aV for a design (``a = nan``: of V), by the general engine.

    Parameters
    ----------
    D : Design
        The design.
    b : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 0 and ``nan``.
    a : float
        The coefficient of V (``nan``: the law of V).
    {opts}
    {verified}

    Returns
    -------
    float or numpy.ndarray
        The density at ``b``.

    See Also
    --------
    lin_cdf

    Examples
    --------
    >>> from dflimit import lin_pdf, trend
    >>> lin_pdf(trend(-1), [-0.2639, 0.0], -3.5)
    array([0.521943..., 0.541021...])
    """
    av = _a(a)
    return _values(DET_LIN_PDF, D, b, c, _opts(0.0, start, gamma0, gls, verified), av, name="b")


@fill_doc(opts=OPTS_DOC)
def lin_quantile(D, p, a, c=0, *, b0=0, tol=1e-10, start=None, gamma0=0, gls=None):
    """Quantiles of the limit of the linear form U - aV for a design, by the general engine.

    Newton's method on `lin_cdf` with `lin_pdf`, safeguarded by bisection on a bracket grown from [b0 - 1, b0 + 1].

    Parameters
    ----------
    D : Design
        The design.
    p : array_like
        The probabilities, in (0, 1).
    a : float
        The coefficient of V.
    {opts}
    b0 : float, default 0
        The start of the search.
    tol : float, default 1e-10
        The tolerance.

    Returns
    -------
    float or numpy.ndarray
        The quantiles.

    See Also
    --------
    lin_cdf

    Examples
    --------
    >>> from dflimit import lin_quantile, trend
    >>> lin_quantile(trend(-1), 0.05, -3.5)   # the critical value of the point-optimal test against cbar = -7
    -0.2638...
    """
    av = dfl_num(a, "a")
    return _quantiles(DET_LIN_Q, D, p, c, _opts(0.0, start, gamma0, gls), av, dfl_num(b0, "b0"), dfl_num(tol, "tol"))


@fill_doc(opts=OPTS_DOC)
def v_cdf(D, y, c=0, *, start=None, gamma0=0, gls=None):
    """Limiting distribution of the denominator V = int J~^2 of the t-ratio for a design.

    P(V <= y) by the single inversion of the transform of V on the outer contour; 0 at y <= 0 (V > 0).

    Parameters
    ----------
    D : Design
        The design.
    y : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    {opts}

    Returns
    -------
    float or numpy.ndarray
        The probabilities at ``y``.

    See Also
    --------
    v_pdf, lin_cdf, msb_cdf

    Examples
    --------
    >>> from dflimit import v_cdf, trend
    >>> v_cdf(trend(-1), [0.1, 0.5, 2])       # P(V <= y) at the unit root, no deterministic terms
    array([0.161002..., 0.677827..., 0.969685...])
    """
    return _values(DET_LIN_CDF, D, y, c, _opts(0.0, start, gamma0, gls), math.nan, name="y")


@fill_doc(opts=OPTS_DOC)
def v_pdf(D, y, c=0, *, start=None, gamma0=0, gls=None):
    """Limiting density of the denominator V of the t-ratio for a design.

    Parameters
    ----------
    D : Design
        The design.
    y : array_like
        The points (0 at y <= 0). ``-inf``, ``inf`` and ``nan`` give 0, 0 and ``nan``.
    {opts}

    Returns
    -------
    float or numpy.ndarray
        The density at ``y``.

    See Also
    --------
    v_cdf

    Examples
    --------
    >>> from dflimit import v_pdf, trend
    >>> v_pdf(trend(0), 0.06)
    6.300094...
    """
    return _values(DET_LIN_PDF, D, y, c, _opts(0.0, start, gamma0, gls), math.nan, name="y")


def adfmax_quantile(D, p, c=0, *, x0=-2, tol=1e-10, start=None, gamma0=0):
    """Quantiles of the limit of the ADFmax statistic of Leybourne (1995) for a design, by the general engine.

    Newton's method on `adfmax_cdf` with `adfmax_pdf` of the design, safeguarded by bisection on a bracket grown from
    [x0 - 1, x0 + 1]. The data are detrended by least squares.

    Parameters
    ----------
    D : Design
        The design.
    p : array_like
        The probabilities, in (0, 1).
    c : float, default 0
        The local-to-unity parameter.
    x0 : float, default -2
        The start of the search.
    tol : float, default 1e-10
        The tolerance.
    start, gamma0
        As in `dist_cdf`.

    Returns
    -------
    float or numpy.ndarray
        The quantiles.

    See Also
    --------
    adfmax_cdf, adfmax_icdf

    Examples
    --------
    >>> from dflimit import adfmax_quantile, trend
    >>> adfmax_quantile(trend(0), 0.05)       # -2.43542 (the quantiles of the ADFmax statistic in the paper)
    -2.4354...
    """
    return _quantiles(DET_ADF_Q, D, p, c, _opts(0.0, start, gamma0, None), dfl_num(x0, "x0"), dfl_num(tol, "tol"))


def region_prob(D, bnd, c=0, *, start=None, gamma0=0, gls=None):
    """Limiting probability of a region {U <= g(V)} of the plane for a design, by the general engine.

    The boundaries whose kernels the inversion on paths around the roots has in closed form: ``{"type": "line", "a":
    a, "b": b}`` (g(v) = b + a v, `lin_cdf`) and ``{"type": "parabola", "x": x, "c0": c0}`` (g(v) = x sqrt(v) + c0
    v, the t-ratio, `dist_cdf`). The weighted-average-power boundary is the classical engine's (`ltratio_region`).

    Parameters
    ----------
    D : Design
        The design.
    bnd : dict
        The boundary.
    c : float, default 0
        The local-to-unity parameter.
    start, gamma0, gls
        As in `dist_cdf`.

    Returns
    -------
    float
        The probability.

    See Also
    --------
    ltratio_region

    Examples
    --------
    >>> from dflimit import region_prob, trend
    >>> region_prob(trend(-1), {"type": "line", "a": -3.5, "b": -0.2639})
    0.04999...
    """
    if not isinstance(bnd, Mapping) or not isinstance(bnd.get("type"), str):
        raise TypeError("bnd must be a dict with the key 'type'")
    t = bnd["type"].lower()

    def el(name, default=None):
        v = bnd.get(name)
        if v is None:
            if default is None:
                raise ValueError(f"a boundary of type {t} needs the key '{name}'")
            v = default
        return dfl_num(v, f"bnd['{name}']")

    if bnd.get("v0") is not None or bnd.get("brks") is not None:
        raise ValueError("the keys 'v0' and 'brks' of a boundary are the classical engine's (ltratio_region)")
    if t == "line":
        return lin_cdf(D, el("b"), el("a"), c, start=start, gamma0=gamma0, gls=gls)
    if t == "parabola":
        return dist_cdf(D, el("x"), c, c0=el("c0", 0), start=start, gamma0=gamma0, gls=gls)
    if t in ("wap", "fun"):
        raise ValueError(f"region_prob: the boundary type {t} is not available by the general engine (its kernels are in "
                         "closed form for boundaries quadratic in sqrt(V) only); the type wap is the classical "
                         "engine's (ltratio_region)")
    raise ValueError('the boundary type must be "line" or "parabola"')


# ---------------------------------------------------------------------------------------------------------------------
# the companions: the point-optimal statistic, the LM statistic, MZ_alpha, MSB
# ---------------------------------------------------------------------------------------------------------------------
GP_DOC = """\
N : int, default 60000
    The number of steps of the trapezoidal rule of the inversion formula of Gil-Pelaez (in u = sinh(v)).
umax : float, default 4000
    The end of its grid."""


@fill_doc(gp=GP_DOC)
def pt_cdf(D, q, cbar, c=0, *, N=60000, umax=4000):
    """Limiting distribution of the point-optimal statistic of Elliott, Rothenberg and Stock for a design.

    P(P_T <= q), P_T = {S(abar) - abar S(1)}/s^2 the point-optimal statistic against cbar (abar = 1 + cbar/n, S(a)
    the sum of squared residuals of the data quasi-differenced at a on the trend functions transformed in the same
    way), under phi = 1 + c/n, zero start, by one inversion of the characteristic function of its limit, a quadratic
    form. The test rejects for small values.

    The inversion is the trapezoidal rule of Gil-Pelaez on a grid of ``N`` steps in u = sinh(v) up to ``umax``. Where
    that grid does not resolve the transform (its singularity near the real line, far above the unit root: with a
    linear trend at cbar = -13.5 from c = 3.7) the functions of the point-optimal statistic raise a ``ValueError`` that
    names the ``N`` that would serve, and so they do at |q| beyond 0.15 N/asinh(umax) (1000 with the default grid).

    Parameters
    ----------
    D : Design
        The design.
    q : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    cbar : float
        The alternative of the test, cbar < 0.
    c : float, default 0
        The local-to-unity parameter.
    {gp}

    Returns
    -------
    float or numpy.ndarray
        The probabilities at ``q``.

    See Also
    --------
    pt_pdf, pt_quantile, pt_power, pt_envelope

    Examples
    --------
    >>> from dflimit import pt_cdf, trend
    >>> pt_cdf(trend(1), [4.0, 7.0], -13.5)      # the trend case at cbar = -13.5
    array([0.009748..., 0.100187...])
    """
    return _values(DET_PT_CDF, D, q, c, _NOOPTS, dfl_num(cbar, "cbar"), dfl_posint(N, "N"), dfl_num(umax, "umax"),
                   name="q")


@fill_doc(gp=GP_DOC)
def pt_pdf(D, q, cbar, c=0, *, N=60000, umax=4000):
    """Limiting density of the point-optimal statistic for a design (arguments as `pt_cdf`).

    Parameters
    ----------
    D : Design
        The design.
    q : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 0 and ``nan``.
    cbar : float
        The alternative of the test, cbar < 0.
    c : float, default 0
        The local-to-unity parameter.
    {gp}

    Returns
    -------
    float or numpy.ndarray
        The density at ``q``.

    Examples
    --------
    >>> from dflimit import pt_pdf, trend
    >>> pt_pdf(trend(1), 7.0, -13.5)
    0.0...
    """
    return _values(DET_PT_PDF, D, q, c, _NOOPTS, dfl_num(cbar, "cbar"), dfl_posint(N, "N"), dfl_num(umax, "umax"),
                   name="q")


@fill_doc(gp=GP_DOC)
def pt_quantile(D, p, cbar, *, N=60000, umax=4000):
    """Quantiles at c = 0 of the limit of the point-optimal statistic against cbar: the critical values.

    Parameters
    ----------
    D : Design
        The design.
    p : array_like
        The probabilities (the levels), in (0, 1).
    cbar : float
        The alternative of the test, cbar < 0.
    {gp}

    Returns
    -------
    float or numpy.ndarray
        The quantiles.

    Examples
    --------
    >>> from dflimit import pt_quantile, trend
    >>> pt_quantile(trend(1), 0.05, -13.5)    # (5.62 in the simulated table of Elliott, Rothenberg and Stock, 1996)
    5.704481...
    """
    return _quantiles(DET_PT_Q, D, p, 0.0, _NOOPTS, dfl_num(cbar, "cbar"), dfl_posint(N, "N"), dfl_num(umax, "umax"))


@fill_doc(gp=GP_DOC)
def pt_power(D, cbar, c, *, level=0.05, N=60000, umax=4000):
    """The critical value of the level-``level`` point-optimal test against cbar and its limiting power at c.

    Parameters
    ----------
    D : Design
        The design.
    cbar : float
        The alternative of the test, cbar < 0.
    c : float
        The local alternative.
    level : float, default 0.05
        The level.
    {gp}

    Returns
    -------
    tuple of float
        ``(b, power)``: the critical value and the power.

    Examples
    --------
    >>> from dflimit import pt_power, perron
    >>> b, power = pt_power(perron("A", 0.4), -10.0, -10.0)
    >>> round(power, 4)
    0.3101
    """
    out = _call(DET_PT_POW, D, np.empty(1), dfl_num(c, "c"), _NOOPTS, dfl_num(cbar, "cbar"), dfl_num(level, "level"),
                dfl_posint(N, "N"), dfl_num(umax, "umax"))
    return float(out[0]), float(out[1])


@fill_doc(gp=GP_DOC)
def pt_envelope(D, cs, *, level=0.05, N=60000, umax=4000):
    """The Gaussian power envelope for a design: the power at c of the point-optimal test against cbar = c.

    ``level`` for c >= 0. For no deterministic terms, an intercept and a linear trend it is `power_envelope` (to
    1e-8); `power_envelope` with any other design is this function.

    Parameters
    ----------
    D : Design
        The design.
    cs : array_like
        The local alternatives c. ``nan`` gives ``nan``.
    level : float, default 0.05
        The level.
    {gp}

    Returns
    -------
    float or numpy.ndarray
        The envelope at ``cs``.

    See Also
    --------
    power_envelope, pt_power

    Examples
    --------
    >>> from dflimit import pt_envelope, perron
    >>> pt_envelope(perron("C", 0.3), [-5.0, -10.0, 1.0])
    array([0.081798..., 0.178810..., 0.05    ...])
    """
    return _values(DET_PT_ENV, D, cs, 0.0, _NOOPTS, dfl_num(level, "level"), dfl_posint(N, "N"), dfl_num(umax, "umax"),
                   name="cs")


def lm_cdf(D, x, c=0):
    """Limiting distribution of the LM statistic of Schmidt and Phillips for a design (with breaks: Amsler and Lee, Lee
    and Strazicich).

    The trend coefficients are estimated from the first differences; with a linear trend among the trend functions the
    statistic converges to -1/(2 sqrt V) for every c, so that F(x) = P(V <= 1/(4 x^2)) for x < 0 and 1 for x >= 0
    (one inversion of the transform of V, on the path of V alone, a hyperbola above the unit root). ``c`` must be at
    most 10 (above, the inversion of its V is not accurate; a ``ValueError``), as for `lm_quantile` and `lm_power`.

    Parameters
    ----------
    D : Design
        The design (a linear trend among its trend functions).
    x : array_like
        The points. ``-inf`` and ``nan`` give 0 and ``nan``.
    c : float, default 0
        The local-to-unity parameter.

    Returns
    -------
    float or numpy.ndarray
        The probabilities at ``x``.

    See Also
    --------
    lm_quantile, lm_power

    Examples
    --------
    >>> from dflimit import lm_cdf, trend
    >>> lm_cdf(trend(1), [-3.0, -2.2])
    array([0.053182..., 0.312176...])
    """
    return _values(DET_LM_CDF, D, x, c, _NOOPTS)


def lm_quantile(D, p, *, c=0):
    """Quantiles of the limit of the LM statistic for a design (the critical values; it rejects for small values).

    Parameters
    ----------
    D : Design
        The design (a linear trend among its trend functions).
    p : array_like
        The probabilities, in (0, 1).
    c : float, default 0
        The local-to-unity parameter, at most 10.

    Returns
    -------
    float or numpy.ndarray
        The quantiles.

    Examples
    --------
    >>> from dflimit import lm_quantile, trend
    >>> lm_quantile(trend(1), 0.05)          # -3.02 (Schmidt and Phillips, 1992)
    -3.0...
    """
    return _quantiles(DET_LM_Q, D, p, c, _NOOPTS)


def lm_power(D, cs, *, level=0.05):
    """The critical value of the level-``level`` LM test and its limiting power at each c of ``cs``.

    Parameters
    ----------
    D : Design
        The design (a linear trend among its trend functions).
    cs : array_like
        The local alternatives, at most 10.
    level : float, default 0.05
        The level, in (0, 1).

    Returns
    -------
    tuple
        ``(q, power)``: the critical value (a float) and the power (a float for a scalar ``cs``, otherwise an array of
        its shape).

    Examples
    --------
    >>> from dflimit import lm_power, perron
    >>> q, power = lm_power(perron("B", 0.5), [-5.0, -10.0])
    >>> power
    array([0.075895..., 0.153290...])
    """
    a = dfl_real(cs, "cs")
    out = _call(DET_LM_POW, D, a.ravel(), 0.0, _NOOPTS, dfl_num(level, "level"))
    return float(out[0]), dfl_shape(out[1:].copy(), a)


def mza_cdf(D, y, c=0, *, gls=-13.5, c0=0, verified=False):
    """Limiting distribution of MZ_alpha of Ng and Perron with the data GLS-detrended at cbar = ``gls``.

    Its limit is that of the coefficient statistic of the GLS-detrended regression, (U - c0 V)/V.

    Parameters
    ----------
    D : Design
        The design.
    y : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    c : float, default 0
        The local-to-unity parameter.
    gls : float, default -13.5
        The GLS detrending, cbar < 0.
    c0 : float, default 0
        The local null value.
    verified : bool, default False
        One evaluation (False) or the checked computation (True).

    Returns
    -------
    float or numpy.ndarray
        The probabilities at ``y``.

    See Also
    --------
    mza_quantile, msb_cdf, coef_cdf

    Examples
    --------
    >>> from dflimit import mza_cdf, trend
    >>> mza_cdf(trend(1), -16.591)             # -16.591: the 5% critical value of MZ_alpha with a trend
    0.050001...
    """
    return _values(DET_MZA_CDF, D, y, c, _opts(c0, None, 0.0, None, verified), dfl_num(gls, "gls"), name="y")


def mza_quantile(D, p, c=0, *, gls=-13.5, y0=-20, tol=1e-9, c0=0):
    """Quantiles of the limit of MZ_alpha with the data GLS-detrended at cbar = ``gls``.

    Parameters
    ----------
    D : Design
        The design.
    p : array_like
        The probabilities, in (0, 1).
    c : float, default 0
        The local-to-unity parameter.
    gls : float, default -13.5
        The GLS detrending.
    y0 : float, default -20
        The start of the search.
    tol : float, default 1e-9
        The tolerance.
    c0 : float, default 0
        The local null value.

    Returns
    -------
    float or numpy.ndarray
        The quantiles.

    Examples
    --------
    >>> from dflimit import mza_quantile, trend
    >>> mza_quantile(trend(1), 0.05)          # (-17.3 in the simulated table of Ng and Perron, 2001)
    -16.591095...
    """
    return _quantiles(DET_MZA_Q, D, p, c, _opts(c0), dfl_num(gls, "gls"), dfl_num(y0, "y0"), dfl_num(tol, "tol"))


def msb_cdf(D, x, c=0, *, gls=-13.5):
    """Limiting distribution of MSB of Ng and Perron (Stock's modified Sargan-Bhargava statistic) with the data
    GLS-detrended at cbar = ``gls``: MSB => sqrt(V_g), so that F(x) = P(V_g <= x^2).

    Parameters
    ----------
    D : Design
        The design.
    x : array_like
        The points (0 for x <= 0). ``inf`` and ``nan`` give 1 and ``nan``.
    c : float, default 0
        The local-to-unity parameter.
    gls : float, default -13.5
        The GLS detrending.

    Returns
    -------
    float or numpy.ndarray
        The probabilities at ``x``.

    See Also
    --------
    msb_quantile, mza_cdf

    Examples
    --------
    >>> from dflimit import msb_cdf, trend
    >>> msb_cdf(trend(1), 0.171)               # 0.171: the 5% critical value of MSB with a trend (msb_quantile)
    0.0499...
    """
    return _values(DET_MSB_CDF, D, x, c, _NOOPTS, dfl_num(gls, "gls"))


def msb_quantile(D, p, c=0, *, gls=-13.5):
    """Quantiles of the limit of MSB with the data GLS-detrended at cbar = ``gls`` (by bisection; it rejects for small
    values).

    Parameters
    ----------
    D : Design
        The design.
    p : array_like
        The probabilities, in (0, 1).
    c : float, default 0
        The local-to-unity parameter.
    gls : float, default -13.5
        The GLS detrending.

    Returns
    -------
    float or numpy.ndarray
        The quantiles.

    Examples
    --------
    >>> from dflimit import msb_quantile, trend
    >>> msb_quantile(trend(1), 0.05)           # (0.168 in the simulated table of Ng and Perron, 2001)
    0.171040...
    """
    return _quantiles(DET_MSB_Q, D, p, c, _NOOPTS, dfl_num(gls, "gls"))


def empty_caches():
    """Drop the general engine's cached prepared designs, outer paths and refined lines.

    The engine keeps up to 64 of each, reference-counted; the values do not change, the next call builds them again.

    Examples
    --------
    >>> from dflimit import empty_caches
    >>> empty_caches()
    """
    _engine.det_empty()
