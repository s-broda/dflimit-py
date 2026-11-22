"""What the user-level functions share for the two engines: the keyword engine, the computation of a case string or a
design (its start, fixed start and GLS detrending), the options of the general engine, the checks of the classical
engine, and the warnings of the engine (the router: dfl_engine_* in csrc/dfl_det_route.c).

A computation names a case string of the classical engine ("DK", ..., "DG") or a Design.  engine=None (the default)
keeps the classical engine for a case string, so that its values are those of before, and takes the router's rule for a
design; "auto" is the rule (dfl_autoengine: the classical engine where it accepts the arguments and is the faster or the
more accurate, the general engine elsewhere), "classical" and "general" force one engine.
"""
import math
from collections import namedtuple

import numpy as np

from . import _engine
from ._design import Design, classical_case, trend
from ._utils import CASES, dfl_case, dfl_num, warn

AUTO, CLASSICAL, GENERAL = 0, 1, 2
_ENGINES = {"auto": AUTO, "classical": CLASSICAL, "general": GENERAL}
_ENGINE_MSG = 'engine must be "auto", "classical" or "general"'
_STARTS = {"zero": 0, "fixed": 1, "stationary": 2}
_START_MSG = 'start must be "zero", "fixed" or "stationary"'
NOT_CLASSICAL = ("engine='classical': the classical engine has the cases DK, DU, DT, SK, SU, ST and DG only (trend(-1), "
                 "trend(0), trend(1), the last GLS-detrended at gls = -13.5, with a zero start)")
NO_METHOD = ("method names a representation of the classical engine; it does not apply with the general engine "
             "(method='auto')")

# (the functions of the rule, f of dfl_autoengine)
F_DIST, F_ICDF, F_MOM, F_LINCDF, F_ADFMAX, F_ENVELOPE, F_REGION = 1, 2, 3, 4, 5, 6, 7

ENGINE_DOC = """\
engine : {None, "auto", "classical", "general"}, default None
    The engine: ``None`` takes the classical engine for a case string (the values of before) and ``"auto"`` for a
    design; ``"auto"`` takes the router's rule, the faster or the more accurate engine where the classical engine
    accepts the arguments and the general engine elsewhere; ``"classical"`` and ``"general"`` force one (see the
    README, "The general engine")."""

DESIGN_DOC = """\
start : {None, "zero", "fixed", "stationary"}, optional
    With a design: the start (``None``: ``"fixed"`` when ``g0 != 0``, else ``"zero"``; ``"stationary"`` needs
    ``c < 0``); with a case string the start is that of the case and ``start`` must not be given.
gls : float, optional
    With a design: GLS detrending at cbar = ``gls`` < 0 (the data quasi-differenced at 1 + gls/n; zero start); the
    default is least squares. Not with a case string (case DG is the GLS-detrended trend at cbar = -13.5)."""


def dfl_engine(engine):
    """The engine code, or None."""
    if engine is None:
        return None
    if not isinstance(engine, str):
        raise TypeError(_ENGINE_MSG)
    try:
        return _ENGINES[engine.lower()]
    except KeyError:
        raise ValueError(_ENGINE_MSG) from None


def start_code(start, g0, name="g0"):
    """The start of a design: 0 zero, 1 fixed (g0 != 0), 2 stationary (start None: fixed when g0 != 0, else zero)."""
    if start is None:
        return 1 if g0 != 0 else 0
    if not isinstance(start, str):
        raise TypeError(_START_MSG)
    try:
        s = _STARTS[start.lower()]
    except KeyError:
        raise ValueError(_START_MSG) from None
    if s == 0 and g0 != 0:
        raise ValueError(f'start="zero" excludes a fixed start {name} != 0')
    if s == 2 and g0 != 0:
        raise ValueError(f'{name} (a fixed start) and start="stationary" exclude each other')
    return 0 if s == 1 and g0 == 0 else s


def gls_value(gls):
    """GLS detrending: NaN (least squares) or cbar."""
    return math.nan if gls is None else dfl_num(gls, "gls")


# a computation: the design (for a case string its design: trend(p), case DG trend(1) with gls -13.5), the classical
# case code or None, model and stat of the case (0 without one), the start (0, 1, 2), the fixed start, gls (NaN: least
# squares), and whether a case string named it
Spec = namedtuple("Spec", "design case model stat start g0 gls named")
_TRENDS = {}


def _trend(p):
    if p not in _TRENDS:
        _TRENDS[p] = trend(p)
    return _TRENDS[p]


def _model(case):
    if case is None:
        return 0, 0
    _, model, stat = dfl_case(case)
    return model, stat


def spec_of(cas, g0=0.0, start=None, gls=None, allowed=CASES):
    """The computation of a case string or a design (the start and gls only with a design)."""
    g0 = dfl_num(g0, "g0")
    if isinstance(cas, Design):
        s = start_code(start, g0)
        gl = gls_value(gls)
        case = classical_case(cas, s, gl)
        model, stat = _model(case)
        return Spec(cas, case, model, stat, s, g0 if s == 1 else 0.0, gl, False)
    if start is not None or gls is not None:
        raise ValueError("start and gls apply to a design; a case string has its start in its code (D: zero or fixed g0, "
                         "S: stationary) and case DG its GLS detrending")
    case, model, stat = dfl_case(cas, allowed)
    s = 2 if stat else (1 if g0 != 0 else 0)
    D = _trend(1) if model == 4 else _trend(model - 2)
    return Spec(D, case, model, stat, s, g0, -13.5 if model == 4 else math.nan, True)


def route_tuple(S):
    """The computation for the router (_engine.route): a case string as its model, a design as itself."""
    if S.named:
        return (None, S.model, S.start, S.g0, math.nan)
    return (S.design._tuple(), 0, S.start, S.g0, S.gls)


def det_opts(S, c0=0.0, verified=False):
    """The options of the general engine for the computation S (c0, start, gamma0, gls, verified)."""
    return (dfl_num(c0, "c0"), S.start, S.g0, S.gls, int(bool(verified)))


def method_ok(model, g0, stat, me):
    """The classical engine's check of the method (check_method of dfl_api.c)."""
    if model == 4 and me == 2:
        return False
    if me == 3 and not (model in (2, 3) and g0 != 0 and not stat):
        return False
    return True


def classical_ok(S, c, c0, me=0):
    """Whether the classical engine accepts the computation (classical_valid of the router)."""
    if S.case is None:
        return False
    if _engine.check_args(S.model, dfl_num(c, "c"), dfl_num(c0, "c0"), S.g0, S.stat):
        return False
    return method_ok(S.model, S.g0, S.stat, me)


def engine_of(S, engine):
    """The engine code of a call: engine=None is the classical engine for a case string, the rule for a design."""
    e = dfl_engine(engine)
    if e is None:
        e = CLASSICAL if S.named else AUTO
    if e == CLASSICAL and S.case is None:
        raise ValueError(NOT_CLASSICAL)
    return e


def warn_lines(*msgs):
    """The warnings of the engine (each line once) as RuntimeWarnings."""
    seen = set()
    for m in msgs:
        for line in (m or "").splitlines():
            if line and line not in seen:
                seen.add(line)
                warn(line)


def warn_brittle(kind, S, c, c0, xs, f=F_DIST):
    """The warning of a regime in which one evaluation of the classical engine is brittle, where the router takes the
    classical engine at some of the points xs (the rule f; the classical engine's own calls give it once per call)."""
    if S.case is None:
        return
    bm = _engine.brittle(kind, S.model, dfl_num(c, "c"), dfl_num(c0, "c0"), S.g0, S.stat, 0)
    if not bm:
        return
    x = np.asarray(xs, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    gen = np.frombuffer(_engine.autoengine(f, S.model, S.stat, S.g0, S.gls, c, c0, x.tobytes()), dtype=np.intc)
    if not np.all(gen):
        warn(bm)
