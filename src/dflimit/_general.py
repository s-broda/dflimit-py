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
from ._ext import RHO_DOC, VOL_DOC, ext_active, ext_args
from ._route import engine_of, gls_value, route_tuple, route_x, spec_of, start_code, warn_lines
from ._utils import Result, dfl_flag, dfl_imethod, dfl_num, dfl_posint, dfl_real, dfl_shape, engine_call, f64, fill_doc

# the operations of _engine.det (det_run in _engine.c)
(DET_CDF, DET_PDF, DET_COEF_CDF, DET_COEF_PDF, DET_COEF_GP, DET_LIN_CDF, DET_LIN_PDF, DET_ADF_CDF, DET_ADF_PDF, DET_LPOW,
 DET_ENV, DET_PT_CDF, DET_PT_PDF, DET_PT_ENV, DET_LM_CDF, DET_MZA_CDF, DET_MSB_CDF) = range(1, 18)
DET_Q, DET_COEF_Q, DET_LIN_Q, DET_ADF_Q, DET_PT_Q, DET_LM_Q, DET_MZA_Q, DET_MSB_Q = range(30, 38)
DET_PT_POW, DET_LM_POW = 50, 51
DET_T3_CDF, DET_T3_PDF, DET_T3_Q = 60, 61, 62

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


def _opts(c0=0.0, start=None, gamma0=0.0, gls=None, verified=False, method=None, rule=None, ext=None):
    g0 = dfl_num(gamma0, "gamma0")
    s = start_code(start, g0, "gamma0")
    o = (dfl_num(c0, "c0"), s, g0 if s == 1 else 0.0, gls_value(gls), dfl_flag(verified, "verified"))
    if method is None and rule is None and ext is None:
        return o
    o = o + (0 if method is None else _gmethod(method),)
    if rule is None and ext is None:
        return o
    o = o + (0 if rule is None else _grule(rule),)
    # (the extensions of the limit, ext_args: rho and the volatility path)
    return o if ext is None or not ext_active(ext[1], ext[2]) else o + (ext[1], ext[2])


_GRULES = {"auto": 0, "present": 1, "thin": 2, "inner": 3, "outer": 4, "fine": 5, "finestep": 6}
_GRULE_MSG = ('rule must be "auto", "present", "thin", "inner", "outer", "fine" or "finestep" (the rules of one evaluation '
              'of the inversion on paths)')


def _grule(rule):
    """The rule of one evaluation of the general engine's inversion on paths (DFL_RULE_* of the C engine)."""
    if not isinstance(rule, str):
        raise TypeError(_GRULE_MSG)
    try:
        return _GRULES[rule.lower()]
    except KeyError:
        raise ValueError(_GRULE_MSG) from None


_GMETHODS = {"auto": 0, "line": 1, "paths": 2, "lines": 3, "mixture": 4}
_GMETHOD_MSG = ('method must be "auto", "line" (the line integral with the explicit kernel) or "paths" (the inversion on '
                'paths around the roots), with rho "lines" or "mixture"')


def _gmethod(method):
    """The method of the general engine's t-ratio: 0 automatic, 1 the line integral with the explicit kernel, 2 the
    inversion on paths around the roots."""
    if not isinstance(method, str):
        raise TypeError(_GMETHOD_MSG)
    try:
        return _GMETHODS[method.lower()]
    except KeyError:
        raise ValueError(_GMETHOD_MSG) from None


METHOD_DOC = """method : {"auto", "line", "paths"}, default "auto"
        The line integral with the explicit kernel (the classical engine's line integral with P(t) and the two roots
        supplied by the general engine's transform) or the inversion on paths around the roots. The line integral
        covers the designs whose numerator has rank one or two: no deterministic terms, continuous designs that do not
        vanish at the origin and whose span is closed under differentiation (polynomial trends of any order, an
        intercept with sines and cosines), and `detrend_first` of a continuous design that does not vanish at the
        origin, with a zero or a stationary start and least squares detrending. ``"auto"`` takes it at ``c0 = 0`` for
        ``c >= -20``, where it is 1.0 to 1.6 times faster than the paths per value for ``-20 <= c <= 0``, except just
        above the unit root with deterministic terms (``0 < c <= 0.5``, where the paths are 1.5 to 2.3 times faster),
        and with ``c0 > 0`` at the points ``x < 0`` where the outer contour of the paths is the vertical line (there the
        paths do not resolve the two roots of the parabola of such a point; the quantiles keep the paths); C engine,
        November 2026; the paths otherwise. Above
        ``c = 5.5`` the paths warn for a design beyond the classical cases and a fixed start with deterministic terms,
        where one evaluation may be off by more than 1e-5 in the upper tail. ``"line"`` raises ``ValueError`` with the
        reason where it does not cover the design, the start or the options (GLS detrending, a fixed start, a jump).
        With ``rho`` the methods are "auto", "lines" and "mixture", with ``vol`` "auto" and "paths"."""

RULE_DOC = """rule : {"auto", "present", "thin", "inner", "outer", "fine", "finestep"}, default "auto"
        The rule of one evaluation of the inversion on paths. ``"auto"`` takes by regions the outer hyperbola with 1.5
        times the step and the paths around the roots with 16 nodes per panel and 32 on the tails of their rays (24 and
        40 before November 2026; 32 and 48 for 48 and 64 with a fixed start), about half the time, where that holds, and
        near the unit root with ``c0 < 0`` the rule of before or a finer one (the rules and regions of DFLimit.jl,
        within 7.6e-10 of the rule of before in the distribution function and 4e-9 of the peak in the density on a
        dense grid); ``"present"`` is the rule before November 2026 (also taken by ``verified = True``), ``"thin"``,
        ``"inner"`` (the thin nodes on the present step), ``"outer"`` (the present nodes on the longer step),
        ``"fine"`` (the thin nodes on half the present step) and ``"finestep"`` (a fixed start without deterministic
        terms: the present nodes on min(1/2, 2.5/(gamma0 |c0|)) times the present step where the outer hyperbola is
        flatter than 70 degrees) name a rule. The line integral (see `method`) is not affected. Under regimes
        (``vol``) the outer hyperbola starts at 70 degrees, where the engine's designs start at 45, and ``"thin"`` and
        ``"outer"`` take twice the step of that hyperbola."""


_NOOPTS = (0.0, 0, 0.0, math.nan, 0)


def _dname(m, D):
    """The name of the design in the engine's messages that refer to it (the engine has no names: "the design does not
    contain it" of the LM statistic's refusal becomes "the design <name> does not contain it", as in DFLimit.jl)."""
    return m.replace("; the design does not contain it", f"; the design {D.name} does not contain it")


def _call(op, D, a, c, opts, *par):
    """The engine's values at the flat points a (one call)."""
    Dt = _design(D)
    try:
        out, w = _engine.det(op, Dt, np.ascontiguousarray(a, dtype=np.float64).tobytes(), dfl_num(c, "c"), opts,
                             *[float(p) for p in par])
    except ValueError as e:
        raise ValueError(_dname(str(e), D)) from None
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


@fill_doc(opts=OPTS_DOC, verified=VERIFIED_DOC, notes=NOTES_DOC, method=METHOD_DOC, rule=RULE_DOC, vol=VOL_DOC, rho=RHO_DOC)
def dist_cdf(D, x, c=0, *, c0=0, start=None, gamma0=0, gls=None, verified=False, method="auto", rule="auto", vol=None, rho=1):
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
    {method}
    {rule}
    {vol}
    {rho}

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
    e = ext_args(vol, rho, c, "dist_cdf")
    return _values(DET_CDF, D, x, e[0], _opts(c0, start, gamma0, gls, verified, method, rule, e))


@fill_doc(opts=OPTS_DOC, verified=VERIFIED_DOC, notes=NOTES_DOC, method=METHOD_DOC, rule=RULE_DOC, vol=VOL_DOC, rho=RHO_DOC)
def dist_pdf(D, x, c=0, *, c0=0, start=None, gamma0=0, gls=None, verified=False, method="auto", rule="auto", vol=None, rho=1):
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
    {method}
    {rule}
    {vol}
    {rho}

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
    e = ext_args(vol, rho, c, "dist_pdf")
    return _values(DET_PDF, D, x, e[0], _opts(c0, start, gamma0, gls, verified, method, rule, e))


@fill_doc(opts=OPTS_DOC, notes=NOTES_DOC, method=METHOD_DOC, rule=RULE_DOC, vol=VOL_DOC, rho=RHO_DOC)
def dist_quantile(D, p, c=0, *, x0=-2, tol=1e-9, c0=0, start=None, gamma0=0, gls=None, method="auto", rule="auto", vol=None,
                  rho=1):
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
    {method}
    {vol}
    {rho}

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
    e = ext_args(vol, rho, c, "dist_quantile")
    return _quantiles(DET_Q, D, p, e[0], _opts(c0, start, gamma0, gls, method=method, rule=rule, ext=e), dfl_num(x0, "x0"),
                      dfl_num(tol, "tol"))


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
def coef_pdf(D, y, c=0, *, c0=0, start=None, gamma0=0, gls=None, verified=False, method="auto", engine="auto", tol=1e-12,
             err=False):
    """Density of the limit of the coefficient statistic n(phi^ - phi0) for a design or a classical case.

    The limit is (U - c0 V)/V; its distribution function is `coef_cdf`. In the cases DK, DU, DT (zero or fixed start
    ``gamma0``) and SK, SU, ST, and for a design that is one of them, the default (``method="auto"`` with
    ``engine="auto"``, up to c = 20) is the single inversion of Yong Bao and Raymond Kan, "Exact Finite-Sample
    Distribution and Moments of the Least Squares Estimator in First-Order Autoregressions: O(1) Algorithms": the
    density of the ratio by the formula of Geary, one integral of the derivative of the joint characteristic function
    along the ray, from jets, the tail completed by integration by parts (as ``coef_pdf`` of DFLimit.jl and of the
    MATLAB programs). ``tol`` is its tolerance relative to the scale max(1, E[V]/sd(U)) of the density, and a warning
    names the points at which the estimate exceeds it (``verified=True`` recomputes those by the general engine's checked
    computation). ``method="inversion"`` forces it, ``method="engine"`` (and ``engine="general"``, case DG, the other
    designs) is the general engine's density, the kernel of its contour integral. Near y = -c0, where the integrand
    decays slowly, one value of the single inversion can take a second, with a warning.

    Parameters
    ----------
    D : Design or str
        The design, or a case code (``"DK"``, ``"DU"``, ``"DT"``, ``"SK"``, ``"SU"``, ``"ST"``, ``"DG"``; the fixed start
        of cases DK, DU, DT is ``gamma0``).
    y : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 0 and ``nan``.
    {opts}
    {verified}
    method : {"auto", "inversion", "engine"}, default "auto"
        ``"inversion"`` forces the single inversion of Bao and Kan, ``"engine"`` the general engine.
    engine : {"auto", "classical", "general"}, default "auto"
        ``"classical"``: the six classical cases by the single inversion; ``"general"``: the general engine.
    tol : float, default 1e-12
        The tolerance of the single inversion.
    err : bool, default False
        If true, return the tuple (values, error estimates; ``nan`` where the general engine computed).

    Returns
    -------
    float or numpy.ndarray
        The density at ``y`` (with ``err=True`` the tuple of the values and their error estimates).

    See Also
    --------
    coef_cdf, coef_quantile

    Examples
    --------
    >>> from dflimit import coef_pdf, trend
    >>> coef_pdf(trend(0), [-12.0, -6.0])
    array([0.020277..., 0.079232...])
    >>> float(coef_pdf("DK", -8.039))        # the density at the 5% critical value of rho
    0.01463777932...
    """
    S = spec_of(D, gamma0, start, gls)
    e = engine_of(S, engine)
    im = dfl_imethod(method, "coef_pdf")
    ve = dfl_flag(verified, "verified")
    a = dfl_real(y, "y")
    flat = a.ravel()
    out = np.where(np.isnan(flat), math.nan, 0.0)
    eo = np.where(np.isfinite(flat), math.nan, 0.0)
    fin = np.isfinite(flat)
    if fin.any():
        v, ev = route_x(4, route_tuple(S), flat[fin], dfl_num(c, "c"), dfl_num(c0, "c0"), e, im,
                        math.nan if tol is None else dfl_num(tol, "tol"), ve)
        out[fin] = v
        eo[fin] = ev
    return (dfl_shape(out, a), dfl_shape(eo, a)) if err else dfl_shape(out, a)


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
def coef_cdf_gp(D, y, c=0, *, c0=0, start=None, gamma0=0, gls=None, N=None, umax=None, method="auto", tol=1e-12,
                err=False):
    """Distribution function of the limit of the coefficient statistic by the inversion formula of Gil-Pelaez.

    P[(U - c0 V)/V <= y] = P[U - (c0 + y) V <= 0] from the characteristic function (a check of `coef_cdf`, which
    inverts on paths around the roots). By default (a zero or stationary start, up to c = 5) the single inversion of
    Yong Bao and Raymond Kan, "Exact Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order
    Autoregressions: O(1) Algorithms", for the general engine's transform, with the branch of the square root of its
    determinant fixed at each point from the pivots of the precision matrix and the tail completed by integration by
    parts; a warning names the points at which the error estimate exceeds ``tol``. ``method="grid"`` (and ``N`` or
    ``umax`` given, a fixed start, c above 5) is the trapezoidal rule of before in u = sinh(v) with ``N`` steps (60000)
    up to ``umax`` (4000), refused where the grid does not resolve the singularity of the transform nearest the real
    line.

    Parameters
    ----------
    D : Design
        The design.
    y : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    {opts}
    N : int, optional
        The number of steps of the fixed grid (60000 where it computes); given, it chooses the grid.
    umax : float, optional
        The end of the grid (4000); given, it chooses the grid.
    method : {"auto", "inversion", "grid"}, default "auto"
        ``"inversion"`` forces the single inversion of Bao and Kan, ``"grid"`` the fixed grid.
    tol : float, default 1e-12
        The absolute tolerance of the single inversion.
    err : bool, default False
        If true, return the tuple (values, error estimates; ``nan`` on the fixed grid).

    Returns
    -------
    float or numpy.ndarray
        The distribution function at ``y`` (with ``err=True`` the tuple of the values and their error estimates).

    See Also
    --------
    coef_cdf

    Examples
    --------
    >>> from dflimit import coef_cdf_gp, trend
    >>> coef_cdf_gp(trend(0), -14.0936)      # 0.05
    0.04999988...
    """
    a = dfl_real(y, "y")
    flat = a.ravel()
    out = np.where(np.isnan(flat), math.nan, np.where(flat > 0, 1.0, 0.0))
    eo = np.where(np.isfinite(flat), math.nan, 0.0)
    fin = np.isfinite(flat)
    if fin.any():
        v, ev = _ptx(DET_COEF_GP, D, flat[fin], c, _opts(c0, start, gamma0, gls), method, N, umax, tol, name="y")
        out[fin] = v
        eo[fin] = ev
    return (dfl_shape(out, a), dfl_shape(eo, a)) if err else dfl_shape(out, a)


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
    >>> adfmax_quantile(trend(0), 0.05)       # -2.43542 (the ADFmax quantiles of the companion paper)
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
N : int, optional
    The number of steps of the fixed grid of the inversion formula of Gil-Pelaez (in u = sinh(v); 60000 where the grid
    computes); given, it chooses the grid.
umax : float, optional
    The end of that grid (4000); given, it chooses the grid.
method : {"auto", "inversion", "grid"}, default "auto"
    ``"auto"``: the single inversion of Bao and Kan up to c = 4, the fixed grid beyond (see `pt_cdf`).
tol : float, default 1e-12
    The absolute tolerance of the single inversion."""

_PTMETHODS = {"auto": 0, "inversion": 5, "grid": 7}


def _ptmethod(method):
    """The method of the point-optimal functions and coef_cdf_gp: "auto" 0, "inversion" 5, "grid" 7."""
    msg = 'method must be "auto", "inversion" (the single inversion of Bao and Kan) or "grid" (the fixed grid of Gil-Pelaez)'
    if not isinstance(method, str):
        raise TypeError(msg)
    m = _PTMETHODS.get(method.lower())
    if m is None:
        raise ValueError(msg)
    return m


def _ptx(op, D, x, c, opts, method, N, umax, tol, *par, name="x"):
    """The values and the error estimates of the point-optimal functions and coef_cdf_gp (_engine.det_x) at the points
    x, in the shape of x (the flat values for DET_PT_POW)."""
    a = dfl_real(x, name)
    Nv = 0 if N is None else dfl_posint(N, "N")
    uv = 0.0 if umax is None else dfl_num(umax, "umax")
    tv = math.nan if tol is None else dfl_num(tol, "tol")
    out, eo, w = engine_call(_engine.det_x, op, _design(D), np.ascontiguousarray(a.ravel(), dtype=np.float64).tobytes(),
                             dfl_num(c, "c"), opts, _ptmethod(method), Nv, uv, tv, *[float(p) for p in par])
    warn_lines(w)
    out, eo = f64(out).copy(), f64(eo).copy()
    if op == DET_PT_POW:
        return out, eo
    return dfl_shape(out, a), dfl_shape(eo, a)


def _err(r, err):
    return r if err else r[0]


@fill_doc(gp=GP_DOC)
def pt_cdf(D, q, cbar, c=0, *, N=None, umax=None, method="auto", tol=1e-12, err=False):
    """Limiting distribution of the point-optimal statistic of Elliott, Rothenberg and Stock for a design.

    P(P_T <= q), P_T = {S(abar) - abar S(1)}/s^2 the point-optimal statistic against cbar (abar = 1 + cbar/n, S(a)
    the sum of squared residuals of the data quasi-differenced at a on the trend functions transformed in the same
    way), under phi = 1 + c/n, zero start, by one inversion of the characteristic function of its limit, a quadratic
    form. The test rejects for small values.

    By default (``method="auto"``, up to c = 4) the inversion is the single inversion of Yong Bao and Raymond Kan,
    "Exact Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order Autoregressions: O(1)
    Algorithms": adaptive quadrature over decades along the real axis, with the branch of the square root of the
    determinant fixed at each point from the pivots of the precision matrix, and the tail completed by integration by
    parts (as ``pt_cdf`` of DFLimit.jl and of the MATLAB programs); a warning names the points at which the error
    estimate exceeds ``tol``, and the quantiles are found by regula falsi on it with the values of the transform along
    the ray kept across the search. Above c = 4, with ``method="grid"`` and wherever ``N`` or ``umax`` is given, the
    inversion is that of before, the trapezoidal rule of Gil-Pelaez on a grid of ``N`` steps in u = sinh(v) up to
    ``umax``. Where that grid does not resolve the transform (its singularity near the real line, far above the unit
    root: with a linear trend at cbar = -13.5 from c = 3.7) the functions of the point-optimal statistic raise a
    ``ValueError`` that names the ``N`` that would serve, and so they do at |q| beyond 0.15 N/asinh(umax) (1000 with
    the default grid); `pt_power` and `pt_envelope` take the grid for the critical value too where the alternative c is
    above 4.

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
    err : bool, default False
        If true, return the tuple (values, error estimates; ``nan`` on the fixed grid).

    Returns
    -------
    float or numpy.ndarray
        The probabilities at ``q`` (with ``err=True`` the tuple of the values and their error estimates).

    See Also
    --------
    pt_pdf, pt_quantile, pt_power, pt_envelope

    Examples
    --------
    >>> from dflimit import pt_cdf, trend
    >>> pt_cdf(trend(1), [4.0, 7.0], -13.5)      # the trend case at cbar = -13.5
    array([0.009748..., 0.100187...])
    """
    return _err(_ptx(DET_PT_CDF, D, q, c, _NOOPTS, method, N, umax, tol, dfl_num(cbar, "cbar"), name="q"), err)


@fill_doc(gp=GP_DOC)
def pt_pdf(D, q, cbar, c=0, *, N=None, umax=None, method="auto", tol=1e-12, err=False):
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
    err : bool, default False
        If true, return the tuple (values, error estimates; ``nan`` on the fixed grid).

    Returns
    -------
    float or numpy.ndarray
        The density at ``q`` (with ``err=True`` the tuple of the values and their error estimates).

    Examples
    --------
    >>> from dflimit import pt_pdf, trend
    >>> pt_pdf(trend(1), 7.0, -13.5)
    0.0...
    """
    return _err(_ptx(DET_PT_PDF, D, q, c, _NOOPTS, method, N, umax, tol, dfl_num(cbar, "cbar"), name="q"), err)


@fill_doc(gp=GP_DOC)
def pt_quantile(D, p, cbar, *, N=None, umax=None, method="auto", tol=1e-12):
    """Quantiles at c = 0 of the limit of the point-optimal statistic against cbar: the critical values.

    By default regula falsi on the single inversion of Bao and Kan (see `pt_cdf`); with the fixed grid Newton's method.

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
    return _ptx(DET_PT_Q, D, p, 0.0, _NOOPTS, method, N, umax, tol, dfl_num(cbar, "cbar"), name="p")[0]


@fill_doc(gp=GP_DOC)
def pt_power(D, cbar, c, *, level=0.05, N=None, umax=None, method="auto", tol=1e-12):
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
    out, _ = _ptx(DET_PT_POW, D, np.empty(1), dfl_num(c, "c"), _NOOPTS, method, N, umax, tol, dfl_num(cbar, "cbar"),
                  dfl_num(level, "level"))
    return float(out[0]), float(out[1])


@fill_doc(gp=GP_DOC)
def pt_envelope(D, cs, *, level=0.05, N=None, umax=None, method="auto", tol=1e-12):
    """The Gaussian power envelope for a design: the power at c of the point-optimal test against cbar = c.

    ``level`` for c >= 0. For no deterministic terms, an intercept and a linear trend it is `power_envelope` (to
    1e-8); `power_envelope` with any other design is this function. By default by the single inversion of Bao and Kan
    up to c = 4 (see `pt_cdf`).

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
    return _ptx(DET_PT_ENV, D, cs, 0.0, _NOOPTS, method, N, umax, tol, dfl_num(level, "level"), name="cs")[0]


def lm_cdf(D, x, c=0):
    """Limiting distribution of the LM statistic of Schmidt and Phillips for a design (with breaks: Amsler and Lee, Lee
    and Strazicich).

    The trend coefficients are estimated from the first differences; with a linear trend among the trend functions the
    statistic converges to -1/(2 sqrt V) for every c, so that F(x) = P(V <= 1/(4 x^2)) for x < 0 and 1 for x >= 0
    (one inversion of the transform of V, on the path of V alone, a hyperbola above the unit root). ``c`` must be at
    most 10 (above, the inversion of its V is not accurate; a ``ValueError``), as for `lm_quantile` and `lm_power`.
    The limit holds only for trend functions whose span contains the constant (the LM regression estimates the level
    from the first observation): for any other design, e^r alone or r alone, say, the three functions raise a
    ``ValueError`` (decided on the values of the piecewise polynomial design at Chebyshev points of each piece: exactly
    for polynomial trends, breaks and dummies, to 1e-6 for the interpolants of `fdesign` and `design_from_matrix`).

    Parameters
    ----------
    D : Design
        The design (a linear trend among its trend functions, the constant in their span).
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
        The design (a linear trend among its trend functions, the constant in their span).
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
        The design (a linear trend among its trend functions, the constant in their span).
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


T3_DOC = """\
D : Design
    The design of the deterministic terms (real trend functions entering with complex coefficients): ``trend(-1)`` for a
    regression without seasonal terms (a constant or a linear trend leave the harmonic statistics alone), ``trend(0)``
    for seasonal dummies (with or without a trend), ``trend(1)`` for seasonal dummies and seasonal trends, and any design
    of the general engine (breaks in the seasonal means, ...).
c : float or sequence, default 0
    The local-to-unity parameter, c <= 0, in the units of the roots' modulus 1 + c/n (n the number of years); with
    ``vol`` a sequence of one value per regime.
c0 : float, default 0
    The local null value.
vol : VolPath, optional
    A volatility path (`volpath`)."""


@fill_doc(t3=T3_DOC)
def hegy_t3_cdf(D, x, c=0, *, c0=0, vol=None, verified=False):
    """Distribution function of the limit of the first seasonal t-ratio of Hylleberg, Engle, Granger and Yoo.

    The t-ratio of Dickey-Fuller shape of the regression at a harmonic frequency (strictly between 0 and pi: pi/2 for
    quarterly data, where t3 is the t-ratio of pi_3, the coefficient of y_{3,t-2}; each pair of harmonic frequencies of
    monthly data): t3 = (R - c0 V)/sqrt(V), R = Re int Z~* dZ, V = int |Z~|^2 of the complex Ornstein-Uhlenbeck process
    dZ = c Z dr + dW, Z(0) = 0, and Z~ its residual on the deterministic terms. The moment generating function of (R, V)
    is e^{-s}/D(s, t) with the general engine's determinant, whose singularities in s are poles: the distribution
    function is one integral over the engine's outer contour of the residues at the roots (where two roots of a group
    come within 1e-4 of each other, the engine's inversion on paths of the doubled transform). Zero start, least squares
    detrending.

    Parameters
    ----------
    {t3}
    x : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    verified : bool, default False
        True: the residues against the inversion on paths of the doubled transform, the latter returned with a warning
        where they differ by more than 1e-9.

    Returns
    -------
    float or numpy.ndarray
        The distribution function at ``x``.

    See Also
    --------
    hegy_t3_pdf, hegy_t3_quantile, volpath

    Examples
    --------
    >>> from dflimit import hegy_t3_cdf, trend
    >>> hegy_t3_cdf(trend(0), -3.3442)       # the 5% critical value with seasonal dummies
    0.0500...
    """
    e = ext_args(vol, 1, c, "hegy_t3_cdf")
    return _values(DET_T3_CDF, D, x, e[0], _opts(c0, verified=verified, method="auto", rule="auto", ext=e), cdf=True)


@fill_doc(t3=T3_DOC)
def hegy_t3_pdf(D, x, c=0, *, c0=0, vol=None, verified=False):
    """Density of the limit of the first seasonal t-ratio of Hylleberg, Engle, Granger and Yoo (`hegy_t3_cdf`).

    Parameters
    ----------
    {t3}
    x : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 0 and ``nan``.
    verified : bool, default False
        As in `hegy_t3_cdf`.

    Returns
    -------
    float or numpy.ndarray
        The density at ``x``.

    Examples
    --------
    >>> from dflimit import hegy_t3_pdf, trend
    >>> hegy_t3_pdf(trend(0), -1.0)
    0.2274...
    """
    e = ext_args(vol, 1, c, "hegy_t3_pdf")
    return _values(DET_T3_PDF, D, x, e[0], _opts(c0, verified=verified, method="auto", rule="auto", ext=e), cdf=False)


@fill_doc(t3=T3_DOC)
def hegy_t3_quantile(D, p, c=0, *, x0=-2, tol=1e-10, c0=0, vol=None):
    """Quantiles of the limit of the first seasonal t-ratio of Hylleberg, Engle, Granger and Yoo (`hegy_t3_cdf`).

    Newton's method with the density from ``x0``, the step limited to 1, then the safeguarded search.

    Parameters
    ----------
    {t3}
    p : array_like
        The probabilities, in (0, 1).
    x0 : float, default -2
        The start of the search.
    tol : float, default 1e-10
        The tolerance on the quantile.

    Returns
    -------
    float or numpy.ndarray
        The quantiles.

    Examples
    --------
    >>> from dflimit import hegy_t3_quantile, trend
    >>> hegy_t3_quantile(trend(0), 0.05)
    -3.3442...
    """
    e = ext_args(vol, 1, c, "hegy_t3_quantile")
    return _quantiles(DET_T3_Q, D, p, e[0], _opts(c0, method="auto", rule="auto", ext=e), dfl_num(x0, "x0"), dfl_num(tol, "tol"))
