"""The linear form U - aV and the power envelope: ltratio_lincdf, power_envelope (R/lincdf.R of the R package
dflimit)."""
import numbers

import numpy as np

from . import _engine, _general
from ._design import Design, is_trend
from ._route import AUTO, CLASSICAL, ENGINE_DOC, GENERAL, dfl_engine, spec_of, warn_lines
from ._ltratio import ERRORS_DOC
from ._utils import dfl_case, dfl_num, dfl_real, dfl_shape, f64, fill_doc

LIN_ROUTE_DOC = """\
With a design, or with ``engine="auto"`` or ``"general"``, the router computes: for cases DK and DG ``"auto"`` takes
the general engine for P(V <= b) (case DK) and in case DG for ``a != 0`` above the unit root (c > 2, or c > 0 with
cbar >= -5), the classical engine otherwise; every other design (any start and GLS detrending) has the general
engine, `lin_cdf`."""
ERS_REFERENCE_DOC = """\
.. [1] Elliott, G., Rothenberg, T. J. and Stock, J. H. (1996), "Efficient Tests for an Autoregressive Unit Root",
   Econometrica 64, 813-836."""


@fill_doc(errors=ERRORS_DOC, references=ERS_REFERENCE_DOC, engine=ENGINE_DOC, route=LIN_ROUTE_DOC)
def ltratio_lincdf(b, a, cas, c=0, cbar=-13.5, *, engine=None, start=None, g0=0, gls=None):
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
        detrending (cbar < 0; default least squares), as in `ltratio_cdf`.

    Returns
    -------
    float or numpy.ndarray
        The probabilities: a float for a scalar ``b``, otherwise an array of the shape of ``b``.

    Raises
    ------
    {errors}

    See Also
    --------
    power_envelope, ltratio_region, ltratio_cdf

    Notes
    -----
    The probability is computed by the inversion formula of Gil-Pelaez,
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
    if isinstance(cas, Design):
        S = spec_of(cas, g0, start, gls)
        case = S.case if S.case in ("DK", "DG") and S.start == 0 else None
        e = AUTO if e is None else e
        if case is None or e == GENERAL:
            if e == CLASSICAL:
                raise ValueError("engine='classical': the linear form of the classical engine has the cases DK and DG only "
                                 "(trend(-1), and trend(1) with gls = -13.5, zero start)")
            st = {0: "zero", 1: "fixed", 2: "stationary"}[S.start]
            return _general._values(_general.DET_LIN_CDF, S.design, b, c, _general._opts(0.0, st, S.g0, gls),
                                    _general._a(a), name="b", cdf=True)
        cas, cbar = case, -13.5
    elif start is not None or gls is not None or g0 != 0:
        raise ValueError("start, g0 and gls apply to a design (case DG has cbar)")
    _, model, _ = dfl_case(cas, ("DK", "DG"))
    x = dfl_real(b, "b")
    if e is None or e == CLASSICAL:
        P = _engine.lincdf(x.tobytes(), float(a), model, dfl_num(c, "c"), dfl_num(cbar, "cbar"))
        return dfl_shape(f64(P), x)
    P, wd, wc = _engine.route_lin(x.tobytes(), float(a), model, dfl_num(c, "c"), dfl_num(cbar, "cbar"), e)
    warn_lines(wc, wd)
    return dfl_shape(f64(P), x)


@fill_doc(errors=ERRORS_DOC, references=ERS_REFERENCE_DOC, engine=ENGINE_DOC)
def power_envelope(c, cas="DK", alpha=0.05, *, engine=None):
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

    Returns
    -------
    float or numpy.ndarray
        The envelope: a float for a scalar ``c``, otherwise an array of the shape of ``c``.

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

    References
    ----------
    {references}

    Examples
    --------
    >>> from dflimit import power_envelope
    >>> power_envelope(-7, "DK")            # 0.4994: at c = -7 the envelope has power one half
    0.4994019826...
    >>> power_envelope([-5, -13.5], "DG")
    array([0.114845..., 0.514413...])
    >>> from dflimit import perron
    >>> power_envelope(-10.0, perron("C", 0.3))     # a design with a break: the point-optimal envelope of the design
    0.178810...
    """
    e = dfl_engine(engine)
    if isinstance(cas, Design):
        case = "DK" if is_trend(cas, -1) or is_trend(cas, 0) else "DG" if is_trend(cas, 1) else None
        e = AUTO if e is None else e
        if case is None:
            if e == CLASSICAL:
                raise ValueError("engine='classical': the envelope of the classical engine has the cases DK (DU) and DG "
                                 "only (trend(-1), trend(0), trend(1))")
            return _general.pt_envelope(cas, c, level=alpha)
        cas = case
    _, model, _ = dfl_case(cas, ("DK", "DU", "DG"))
    x = dfl_real(c, "c")
    if e is None or e == CLASSICAL:
        E = _engine.envelope(x.tobytes(), 4 if model == 4 else 1, dfl_num(alpha, "alpha"))
        return dfl_shape(f64(E), x)
    E, wd, wc = _engine.route_env(x.tobytes(), 4 if model == 4 else 1, dfl_num(alpha, "alpha"), e)
    warn_lines(wc, wd)
    return dfl_shape(f64(E), x)

