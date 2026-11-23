"""The linear form U - aV and the power envelope: ltratio_lincdf, power_envelope (R/lincdf.R of the R package
dflimit)."""
import numbers

import numpy as np

from . import _engine, _general
from ._design import Design, is_trend
from ._route import CLASSICAL, ENGINE_DOC, design_tuple, dfl_engine, route_x, spec_of
from ._ltratio import ERRORS_DOC
from ._utils import dfl_case, dfl_flag, dfl_imethod, dfl_num, dfl_real, dfl_shape, dfl_tol, f64, fill_doc

# (the operations of _engine.route_x: the linear form, the envelope)
_RX_LIN, _RX_ENV = 6, 7

INV_DOC = """\
method : {"auto", "inversion", "engine"}, default "auto"
    ``"inversion"`` forces the single inversion of Bao and Kan, ``"engine"`` the engines' computation of before.
tol : float, default 1e-12
    The absolute tolerance of the single inversion.
verified : bool, default False
    If true, the values whose error estimate exceeds ``tol`` are recomputed by the engines.
err : bool, default False
    If true, return the tuple (values, error estimates); the estimates are ``nan`` where the engines computed."""


def _with_err(v, e, x, err):
    return (v, dfl_shape(np.asarray(e, dtype=np.float64).copy(), x)) if err else v

LIN_ROUTE_DOC = """\
With a design, or with ``engine="auto"`` or ``"general"``, the router computes: for cases DK and DG ``"auto"`` takes
the general engine for P(V <= b) (case DK) and in case DG for ``a != 0`` above the unit root (c > 2, or c > 0 with
cbar >= -5), the classical engine otherwise; a fixed start and every other design (any start and GLS detrending) have
the general engine, `lin_cdf`. By the general engine P(V <= b) is 0 at b <= 0."""
ERS_REFERENCE_DOC = """\
.. [1] Elliott, G., Rothenberg, T. J. and Stock, J. H. (1996), "Efficient Tests for an Autoregressive Unit Root",
   Econometrica 64, 813-836."""


@fill_doc(errors=ERRORS_DOC, references=ERS_REFERENCE_DOC, engine=ENGINE_DOC, route=LIN_ROUTE_DOC, inv=INV_DOC)
def ltratio_lincdf(b, a, cas, c=0, cbar=-13.5, *, engine="auto", start=None, g0=0, gls=None, method="auto", tol=1e-12,
                   verified=False, err=False):
    """Limiting distribution of the linear form U - aV.

    The limit of P(U - aV <= b) under phi = 1 + c/n, zero start, no drift, with U and V as in `ltratio_cdf`: case
    ``"DK"`` (no deterministic terms, and the constant case with GLS demeaning, whose limit is the same) or ``"DG"``
    (the GLS-detrended trend case of Elliott, Rothenberg and Stock (1996), detrended at ``cbar``). The
    point-optimal test of Elliott, Rothenberg and Stock against cbar < 0 rejects for U - aV < b: in case DK with
    a = cbar/2 (its log likelihood ratio is cbar U - cbar^2 V/2), in case DG, with the data detrended at cbar, with
    a = cbar^2/(2 (cbar - 1)) (it rejects for small cbar^2 V + (1 - cbar) J(1)^2, J(1)^2 = 2U + 1); see
    `power_envelope`. ``a = nan`` gives P(V <= b) (case DK).

    Parameters
    ----------
    b : array_like
        The points. ``-inf``, ``inf`` and ``nan`` give 0, 1 and ``nan``.
    a : float
        The coefficient of V, a single number; ``nan`` for the distribution of V (case DK).
    cas : str or Design
        The case, ``"DK"`` or ``"DG"``, or a design of the general engine.
    c : float, default 0
        The local-to-unity parameter, finite.
    cbar : float, default -13.5
        The detrending parameter cbar of case DG (-13.5: the DF-GLS tests); not used in case DK (with a design: the
        keyword ``gls``).
    {engine}
    start, g0, gls
        With a design: the start (``None``, ``"zero"``, ``"fixed"`` with ``g0``, ``"stationary"``) and the GLS
        detrending (cbar < 0; default least squares), as in `ltratio_cdf`; with case DK a fixed start ``g0`` (its
        design, ``engine="auto"`` or ``"general"``).
    {inv}

    Returns
    -------
    float or numpy.ndarray
        The probabilities: a float for a scalar ``b``, otherwise an array of the shape of ``b`` (with ``err=True`` the
        tuple of the values and their error estimates).

    Raises
    ------
    {errors}

    See Also
    --------
    power_envelope, ltratio_region, ltratio_cdf

    Notes
    -----
    By default (``method="auto"`` with ``engine="auto"``, a finite a != 0) the probability is the single inversion of
    the joint characteristic function along the ray of the half-plane, the method of Yong Bao and Raymond Kan, "Exact
    Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order Autoregressions: O(1)
    Algorithms", with the tail beyond S = 400/|w| (w = 1/2 + b) completed by integration by parts: in case DK for
    -100 <= c <= 4, and for a design that is one of the six classical cases (``trend(-1)``, ``trend(0)``, ``trend(1)``
    with any start) for -100 <= c <= 2 (as ``ltratio_lincdf`` of DFLimit.jl and of the MATLAB programs); a warning
    names the points at which the error estimate exceeds ``tol``. ``method="inversion"`` forces it (any c);
    ``method="engine"``, case DG, ``a = nan``, ``engine="classical"`` or ``"general"`` and the points outside those
    ranges take the computation of before, which reproduces its values and is described next.

    The engines compute the probability by the inversion formula of Gil-Pelaez,
    P(X < b) = 1/2 - (1/pi) int_0^inf Im[phi(u) exp(-iub)]/u du, from the joint moment generating function of
    (U, V), by the trapezoidal rule in u = sinh(v) with 60000 steps up to u = 4000, as ``ltratio_lincdf`` of the
    MATLAB and Julia programs.

    Where the transform has not fallen below 5e-9 by the end of that grid, or ``c > 0`` (|a| below about 0.3 to 3,
    depending on ``c``, which includes ``a = 0``, the law of U alone), that inversion is not accurate, and the
    probability comes from a conditional integral instead: in case DK that of the region U <= aV + b
    (`ltratio_region`), in case DG the conditional integral given U + 1/2, whose law is a gamma law of shape 1/2
    under every tilt in V; P(V <= b) takes a longer grid for ``c <= -50`` and a finer one for ``c > 2``. The values
    are accurate to about 1e-10 and lie in [0, 1]. ``c`` must be at least -550, in case DG at most about 3.5 (at
    ``cbar = -13.5``; less for ``cbar`` nearer 0), and for ``a = nan`` at most about 7.

    {route}

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import ltratio_lincdf
    >>> ltratio_lincdf(-0.2639, -3.5, "DK", 0)      # 0.05: the size of the point-optimal test at cbar = -7
    0.0499967819...
    >>> ltratio_lincdf(-0.2639, -3.5, "DK", -7)     # its power at c = -7
    0.4993815721...
    >>> ltratio_lincdf([0.1, 0.5, 2], float("nan"), "DK")    # P(V <= b) at the unit root
    array([0.161002..., 0.677827..., 0.969685...])
    >>> from dflimit import perron
    >>> ltratio_lincdf(0.0, -2.0, perron("A", 0.5), -5)      # a design with a break (the general engine)
    0.994468...
    """
    msg = "a must be a single number (NaN: the distribution of V)"
    if isinstance(a, np.ndarray):
        if a.ndim != 0 or a.dtype.kind not in "iuf":
            raise TypeError(msg)
        a = a[()]
    if isinstance(a, (bool, np.bool_)) or not isinstance(a, numbers.Real):
        raise TypeError(msg)
    e = dfl_engine(engine)
    im = dfl_imethod(method, "ltratio_lincdf")
    ve = dfl_flag(verified, "verified")
    if isinstance(cas, Design) or (dfl_case(cas, ("DK", "DG"))[0] == "DK" and dfl_num(g0, "g0") != 0):
        # (a design, or case DK with a fixed start (its design): the router, with the single inversion for the designs of
        # the six classical cases, the classical linear form for cases DK and DG with a zero start, the general engine
        # otherwise; the general engine's points at -inf, inf, nan are 0, 1, nan)
        if not isinstance(cas, Design) and (start is not None or gls is not None):
            raise ValueError("start and gls apply to a design (case DG has cbar)")
        S = spec_of(cas, g0, start, gls) if isinstance(cas, Design) else spec_of("DK", g0)
        if e == CLASSICAL and S.case is None:
            raise ValueError("engine='classical': the linear form of the classical engine has the cases DK and DG only "
                             "(trend(-1), and trend(1) with gls = -13.5, zero start)")
        if e == CLASSICAL and not isinstance(cas, Design):
            raise ValueError("the classical engine's linear form has a zero start only (a fixed start g0 in case DK: "
                             "engine='auto' or 'general', or a design)")
        x = dfl_real(b, "b")
        xv = x.ravel()
        out = np.full(xv.size, np.nan)
        out[~np.isnan(xv) & (xv > 0)] = 1.0
        out[~np.isnan(xv) & (xv < 0)] = 0.0
        eo = np.where(np.isfinite(xv), np.nan, 0.0)
        ok = np.isfinite(xv)
        if ok.any():
            v, ev = route_x(_RX_LIN, design_tuple(S), xv[ok], dfl_num(c, "c"), 0.0, e, im, dfl_tol(tol), ve,
                            _general._a(a), float("nan"))
            out[ok] = v
            eo[ok] = ev
        return _with_err(dfl_shape(out, x), eo, x, err)
    elif start is not None or gls is not None:
        raise ValueError("start and gls apply to a design (case DG has cbar)")
    case, model, _ = dfl_case(cas, ("DK", "DG"))
    if case == "DG" and dfl_num(g0, "g0") != 0:
        raise ValueError("case DG has a zero start; a fixed start g0 applies to case DK and to designs")
    x = dfl_real(b, "b")
    if e == CLASSICAL and im != 5:
        P = _engine.lincdf(x.tobytes(), float(a), model, dfl_num(c, "c"), dfl_num(cbar, "cbar"))
        return _with_err(dfl_shape(f64(P), x), np.full(x.size, np.nan), x, err)
    P, eo = route_x(_RX_LIN, (None, model, 0, 0.0, float("nan")), x.ravel(), dfl_num(c, "c"), 0.0, e, im, dfl_tol(tol), ve,
                    float(a), dfl_num(cbar, "cbar"))
    return _with_err(dfl_shape(P, x), eo, x, err)


def _general_lin(b, a, S, c):
    """The linear form (a = nan: the law of V) of the computation S by the general engine (lin_cdf)."""
    st = ("zero", "fixed", "stationary")[S.start]
    return _general._values(_general.DET_LIN_CDF, S.design, b, c,
                            _general._opts(0.0, st, S.g0, None if np.isnan(S.gls) else S.gls), _general._a(a), name="b")


@fill_doc(errors=ERRORS_DOC, references=ERS_REFERENCE_DOC, engine=ENGINE_DOC, inv=INV_DOC)
def power_envelope(c, cas="DK", alpha=0.05, *, engine="auto", method="auto", tol=1e-12, verified=False, err=False):
    """Asymptotic power envelope of Elliott, Rothenberg and Stock (1996).

    The power at ``c`` of the point-optimal level-``alpha`` test against ``c`` itself, the envelope of the powers
    of all tests of the unit root, zero start, no drift: case ``"DK"`` (no deterministic terms; also the constant
    case, ``"DU"``, where the point-optimal test uses the GLS-demeaned data and the envelope is the same curve) and
    case ``"DG"`` (the trend case, the point-optimal test against ``c`` with the data GLS-detrended at ``c``).

    Parameters
    ----------
    c : array_like
        The local-to-unity parameters. ``nan`` gives ``nan``.
    cas : str or Design, default "DK"
        The case, ``"DK"``, ``"DU"`` (the same as ``"DK"``) or ``"DG"``, or a design of the general engine:
        ``trend(-1)`` and ``trend(0)`` are case DK, ``trend(1)`` case DG, and every other design has the envelope of
        `pt_envelope` (the point-optimal test of the design against cbar = c).
    alpha : float, default 0.05
        The level of the tests.
    {engine}
    {inv}

    Returns
    -------
    float or numpy.ndarray
        The envelope: a float for a scalar ``c``, otherwise an array of the shape of ``c`` (with ``err=True`` the
        tuple of the values and their error estimates, the sum of the two inversions' estimates).

    Raises
    ------
    {errors}

    See Also
    --------
    ltratio_lincdf, ltratio_cdf

    Notes
    -----
    The test against cbar rejects for small U - aV (`ltratio_lincdf`), a = cbar/2 in case DK and
    a = cbar^2/(2 (cbar - 1)) in case DG (where the limit of the invariant statistic of Elliott, Rothenberg and Stock
    is cbar^2 V + (1 - cbar) J(1)^2, J(1)^2 = 2U + 1): its critical value is the ``alpha``-quantile at ``c = 0``, found by bisection on [-50, 50] to the width 1e-9, and the envelope is its power
    at cbar = c; ``alpha`` for ``c >= 0``. Near ``c = 0``, where a is small and that inversion is not accurate, the
    critical value and the power come from the conditional integral of `ltratio_lincdf`, the critical value by
    bisection in b + 1/2 to 1e-12 of its scale. As ``power_envelope`` of the MATLAB and Julia programs.

    With ``engine="general"`` (or ``"auto"``, whose rule takes the classical engine here) the general engine finds the
    critical value by Newton's method on its own linear form, to 1e-10.

    In case DK (and DU) the default (``method="auto"`` with ``engine="auto"``, c <= -1e-4) computes the critical value
    and the power by the single inversion of Bao and Kan (see `ltratio_lincdf`), the critical value by regula falsi to
    1e-12; a warning names the points at which the sum of the two error estimates exceeds ``tol``.
    ``method="inversion"`` forces it; ``method="engine"``, case DG, ``engine="classical"`` or ``"general"`` and
    -1e-4 < c < 0 take the computation described above, which reproduces the values of earlier versions.

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import power_envelope
    >>> power_envelope(-7, "DK")            # 0.4994: at c = -7 the envelope has power one half
    0.4994019828...
    >>> power_envelope([-5, -13.5], "DG")
    array([0.114845..., 0.514413...])
    >>> from dflimit import perron
    >>> power_envelope(-10.0, perron("C", 0.3))     # a design with a break: the point-optimal envelope of the design
    0.178810...
    """
    e = dfl_engine(engine)
    im = dfl_imethod(method, "power_envelope")
    ve = dfl_flag(verified, "verified")
    if isinstance(cas, Design):
        case = "DK" if is_trend(cas, -1) or is_trend(cas, 0) else "DG" if is_trend(cas, 1) else None
        if case is None:
            if e == CLASSICAL:
                raise ValueError("engine='classical': the envelope of the classical engine has the cases DK (DU) and DG "
                                 "only (trend(-1), trend(0), trend(1))")
            if im != 0:
                raise ValueError("power_envelope: method applies to the cases DK (DU) and DG; the envelope of a design is "
                                 "pt_envelope")
            v = _general.pt_envelope(cas, c, level=alpha)
            x = dfl_real(c, "c")
            return _with_err(v, np.full(x.size, np.nan), x, err)
        cas = case
    _, model, _ = dfl_case(cas, ("DK", "DU", "DG"))
    x = dfl_real(c, "c")
    if e == CLASSICAL and im != 5:
        E = _engine.envelope(x.tobytes(), 4 if model == 4 else 1, dfl_num(alpha, "alpha"))
        return _with_err(dfl_shape(f64(E), x), np.full(x.size, np.nan), x, err)
    E, eo = route_x(_RX_ENV, (None, 1, 0, 0.0, float("nan")), x.ravel(), 0.0, 0.0, e, im, dfl_tol(tol), ve,
                    4 if model == 4 else 1, dfl_num(alpha, "alpha"))
    return _with_err(dfl_shape(E, x), eo, x, err)

