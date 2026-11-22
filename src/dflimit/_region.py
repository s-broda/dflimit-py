"""The probability of a region of the plane: ltratio_region (R/region.R of the R package dflimit)."""
import math
from collections.abc import Mapping

import numpy as np

from . import _engine, _general
from ._design import Design
from ._route import CLASSICAL, ENGINE_DOC, GENERAL, dfl_engine, spec_of
from ._utils import Result, dfl_num, dfl_real, fill_doc, warn

_TYPES = ("line", "parabola", "wap")
_TYPE_MSG = ('the boundary type must be "line", "parabola" or "wap" (the type "fun" of the MATLAB programs is not '
             'available in the C engine)')


@fill_doc(engine=ENGINE_DOC)
def ltratio_region(bnd, cas="DK", c=0, info=False, *, engine=None, start=None, g0=0, gls=None):
    """Limiting probability of a region of the plane.

    The limit of the probability P{U <= g(V)} under phi = 1 + c/n, no drift, zero start, case DK (no deterministic
    terms; also the constant case with GLS demeaning), with U and V as in `ltratio_cdf`. Every test of the unit root
    in the limit experiment of case DK rejects on such a region (the paper's section on the
    weighted-average-power test), and the first part of the conditional integral representation of the paper, the
    probability of a region of the plane, gives its probability by one outer integral over V. Accurate to about 1e-11.

    Parameters
    ----------
    bnd : dict
        The boundary: the key ``"type"`` and the parameters of the boundary,

        - ``"line"``: g(v) = b + a v, keys ``"a"`` and ``"b"`` (the point-optimal test, U - aV <= b);
        - ``"parabola"``: g(v) = x sqrt(v) + c0 v, keys ``"x"`` and ``"c0"`` (default 0): the t-ratio, the same
          probability as ``ltratio_cdf(x, "DK", c, c0=c0)``;
        - ``"wap"``: the weighted-average-power test with the half-normal weight of scale omega on c < 0,
          (1 + omega^2 V)^(-1/2) erfcx(U/sqrt(2 (V + 1/omega^2))) >= k, keys ``"k"`` and ``"omega"``;

        and optionally ``"v0"`` (g = +inf on (0, v0), whose probability P(V < v0) is added) and ``"brks"`` (kinks
        of g, breakpoints of the outer rule). The boundary type ``"fun"`` of the MATLAB programs (any function g) is
        not available: the C engine has the three families above only.
    cas : str or Design, default "DK"
        The case: ``"DK"`` only, or a design of the general engine (``trend(-1)`` with a zero start is case DK).
    c : float, default 0
        The local-to-unity parameter.
    info : bool, default False
        If true, return a `Result` (a dict whose keys are also attributes) with the probability ``P``, the error
        estimate ``err`` and the status ``status`` (nonzero: the error estimate is not met); with the general engine
        ``err`` is nan and ``status`` 0.
    {engine}
    start, g0, gls
        With a design: the start and the GLS detrending, as in `ltratio_cdf`; with case DK a fixed start ``g0`` (the
        general engine: ``engine="auto"`` or ``"general"``).

    Returns
    -------
    float or Result
        The probability (or the `Result` described under ``info``).

    Warns
    -----
    RuntimeWarning
        If the error estimate is not met.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If an argument is out of range (the engine's checks, with its message).

    See Also
    --------
    ltratio_cdf, region_prob

    Notes
    -----
    The rule of ``engine="auto"`` takes the classical engine for case DK with a zero start (2.7 to 3.1 times faster,
    and the only engine for the weighted-average-power boundary); a fixed start, other designs, and
    ``engine="general"`` have the general engine (`region_prob`: the line and the parabola, without ``"v0"`` and
    ``"brks"``).

    Examples
    --------
    >>> from dflimit import ltratio_region
    >>> ltratio_region({"type": "wap", "k": 3.6658, "omega": 10}, "DK", 0)      # 0.05 (the weighted-average-power test of the paper)
    0.0499996228...
    >>> # the point-optimal test against cbar = -7: size 0.05
    >>> ltratio_region({"type": "line", "a": -3.5, "b": -0.2639}, "DK", 0)
    0.0499967819...
    >>> from dflimit import trend
    >>> ltratio_region({"type": "parabola", "x": -2.86157}, trend(0))     # the t-ratio of case DU at 5%
    0.049999857...
    """
    e = dfl_engine(engine)
    if isinstance(cas, Design):
        S = spec_of(cas, g0, start, gls)
        if not (S.case == "DK" and S.start == 0) or e == GENERAL:
            if e == CLASSICAL:
                raise ValueError("engine='classical': the regions of the classical engine have case DK only (trend(-1), zero "
                                 "start)")
            return _general_region(bnd, S, c, info)
    else:
        if start is not None or gls is not None:
            raise ValueError("start and gls apply to a design")
        if not isinstance(cas, str):
            raise TypeError("the case must be DK (zero start)")
        if cas.upper() != "DK":
            raise ValueError("the case must be DK (zero start)")
        g0 = dfl_num(g0, "g0")
        if g0 != 0:
            if e is None or e == CLASSICAL:
                raise ValueError("the regions of the classical engine have case DK only with a zero start (a fixed start g0: "
                                 "engine='auto' or 'general', or a design)")
            return _general_region(bnd, spec_of("DK", g0), c, info)
        if e == GENERAL:
            return _general_region(bnd, spec_of("DK"), c, info)
    if not isinstance(bnd, Mapping) or bnd.get("type") is None:
        raise TypeError("bnd must be a dict with the key 'type'")
    btype = bnd["type"]

    def el(name, default=None):
        v = bnd.get(name)
        if v is None:
            if default is None:
                raise ValueError(f"a boundary of type {btype} needs the key '{name}'")
            v = default
        return dfl_num(v, f"bnd['{name}']")

    t = btype.lower() if isinstance(btype, str) else None
    if t not in _TYPES:
        raise (ValueError if isinstance(btype, str) else TypeError)(_TYPE_MSG)
    if t == "line":
        par = (el("a"), el("b"))
    elif t == "parabola":
        par = (el("x"), el("c0", 0))
    else:
        par = (el("k"), el("omega"))
    v0 = 0.0 if bnd.get("v0") is None else el("v0")
    brks = np.empty(0) if bnd.get("brks") is None else dfl_real(bnd["brks"], "bnd['brks']").ravel()
    if not np.all(np.isfinite(brks) & (brks > 0)):
        raise ValueError("bnd['brks'] must be positive and finite")
    if not math.isfinite(v0) or v0 < 0:
        raise ValueError("bnd['v0'] must be finite and nonnegative")
    P, err, status = _engine.region(_TYPES.index(t) + 1, par[0], par[1], v0, brks.tobytes(), dfl_num(c, "c"))
    if status != 0:
        warn(f"the error estimate {err:.1e} is above the tolerance; the result may be inaccurate")
    return Result(P=P, err=err, status=status) if info else P


def _general_region(bnd, S, c, info):
    """The region by the general engine (region_prob)."""
    st = {0: "zero", 1: "fixed", 2: "stationary"}[S.start]
    P = _general.region_prob(S.design, bnd, c, start=st, gamma0=S.g0, gls=None if math.isnan(S.gls) else S.gls)
    return Result(P=P, err=math.nan, status=0) if info else P
