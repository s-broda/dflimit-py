"""Argument conversion shared by the user-level functions (R/utils.R of the R package dflimit).

The checks of the ranges are the engine's (dfl_check_args in csrc/dfl_api.c); here are the types, the case code, the
method name, and the shapes of the results.
"""
import math
import numbers
import os
import sys
import warnings

import numpy as np

CASES = ("DK", "DU", "DT", "SK", "SU", "ST", "DG")
_METHODS = {"auto": 0, "single": 1, "line": 1, "double": 2, "conditional": 2, "contour": 3}
_METHOD_MSG = 'method must be "auto", "single" ("line"), "double" ("conditional") or "contour"'


def dfl_case(cas, allowed=CASES):
    """The case code: (code, model, stat).

    model 1, 2, 3: no deterministic terms, intercept, intercept and trend; 4: the GLS-detrended trend case DG.
    stat 1: a stationary start.
    """
    msg = "the case must be one of " + ", ".join(allowed)
    if not isinstance(cas, str):
        raise TypeError(msg)
    cas = cas.upper()
    if cas not in allowed:
        raise ValueError(msg)
    if cas == "DG":
        return cas, 4, 0
    return cas, "KUT".index(cas[1]) + 1, int(cas[0] == "S")


def dfl_flag(v, name):
    """A boolean flag, as 0 or 1 for the engine."""
    if not isinstance(v, (bool, np.bool_)):
        raise TypeError(f"{name} must be True or False")
    return int(bool(v))


def dfl_num(v, name):
    """A real scalar, as a float (finiteness and range are checked by the engine)."""
    msg = f"{name} must be a single number"
    if isinstance(v, np.ndarray):
        if v.ndim != 0 or v.dtype.kind not in "iuf":
            raise TypeError(msg)
        v = v[()]
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise TypeError(msg)
    v = float(v)
    if math.isnan(v):
        raise ValueError(msg)
    return v


def dfl_posint(v, name):
    """A positive integer (an integral float such as 4.0 is accepted, as in R)."""
    msg = f"{name} must be a positive integer"
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v[()]
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise TypeError(msg)
    if not (math.isfinite(v) and v == math.floor(v) and v >= 1):
        raise ValueError(msg)
    return int(v)


def dfl_method(method):
    """The method: 0 automatic, 1 the line (single) integral, 2 the conditional (double) integral, 3 the contour
    integral."""
    if not isinstance(method, str):
        raise TypeError(_METHOD_MSG)
    try:
        return _METHODS[method.lower()]
    except KeyError:
        raise ValueError(_METHOD_MSG) from None


def dfl_real(x, name):
    """A numeric array (any shape) as a C-contiguous float64 array."""
    try:
        a = np.asarray(x)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be numeric") from None
    if a.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric")
    return np.asarray(a, dtype=np.float64, order="C")     # (not ascontiguousarray, which makes a 0-d array 1-d)


def f64(buf):
    """The float64 values of a bytearray of the engine (no copy)."""
    return np.frombuffer(buf, dtype=np.float64) if len(buf) else np.empty(0)


def cint(buf):
    """The C int values of a bytearray of the engine (no copy)."""
    return np.frombuffer(buf, dtype=np.intc) if len(buf) else np.empty(0, dtype=np.intc)


def dfl_shape(out, a, kind=float):
    """The flat result out in the shape of the input a: a Python scalar (float, or kind) for a 0-d input."""
    if a.ndim == 0:
        return kind(out[0])
    return out.reshape(a.shape)


class Result(dict):
    """The results of ``info=True``: a dict whose keys are also attributes (``r["q"]`` or ``r.q``)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self):
        return list(self.keys()) + dir(dict)


_PKG = os.path.dirname(os.path.abspath(__file__))


def warn(msg):
    """A RuntimeWarning (a warning of the R package), reported at the first caller outside the package."""
    level, f = 2, sys._getframe(1)              # (stacklevel 2: the caller of warn)
    while f is not None and os.path.dirname(os.path.abspath(f.f_code.co_filename)) == _PKG:
        f = f.f_back
        level += 1
    warnings.warn(msg, RuntimeWarning, stacklevel=level)


def fill_doc(**parts):
    """A decorator: the lines {name} of the docstring become the text parts[name], at the indentation of the line."""
    def deco(fun):
        lines = []
        for line in (fun.__doc__ or "").splitlines():
            s = line.strip()
            if len(s) > 2 and s[0] == "{" and s[-1] == "}" and s[1:-1] in parts:
                ind = line[: len(line) - len(line.lstrip())]
                lines += [ind + t if t else t for t in parts[s[1:-1]].splitlines()]
            else:
                lines.append(line)
        fun.__doc__ = "\n".join(lines)
        return fun
    return deco
