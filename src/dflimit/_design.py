"""The designs of the general engine: the class Design and its constructors trend, perron, fdesign, ltrend,
design_from_matrix and detrend_first (the names of the Julia package DFLimit.jl).

A design is the piecewise smooth deterministic part d(r), r in [0, 1], of the regression of y_t on d(t/n) and y_{t-1}
(the limit of the regressors of a sample of size n): K regressors, polynomial on each of the pieces br[j] < r <= br[j+1]
between the break points 0 = br[0] < ... < br[np] = 1.  The engine builds the designs (csrc/dfl_det_design.c); the
class only holds them, immutable, and hands them to the engine as the tuple of _engine.c.
"""
import math
import numbers

import numpy as np

from . import _engine
from ._utils import dfl_num, dfl_posint, dfl_real, warn


def _readonly(a):
    a = np.array(a, dtype=np.float64)
    a.setflags(write=False)
    return a


def _fmt(v):
    return f"{v:g}"


class Design:
    """A deterministic design of the general engine.

    The deterministic terms d(r) = (d_1(r), ..., d_K(r)), r in [0, 1], of the regression of y_t on d(t/n) and y_{t-1}:
    K regressors, each a polynomial on every piece ``br[j] < r <= br[j+1]`` between the break points
    ``0 = br[0] < br[1] < ... < br[np] = 1`` (r = 0 belongs to the first piece; a regressor is left-continuous at a
    break, so that the level-shift dummy 1{r > lambda} is 0 at r = lambda, as 1{t > T_B} in the regression).

    Designs are made by the constructors `trend`, `perron`, `fdesign`, `ltrend`, `design_from_matrix` and
    `detrend_first`, or directly from the break points and the coefficients (the class ``Determ`` of the Julia
    package). A design is immutable and hashable; two designs with the same break points, coefficients and statistic
    are equal.

    Parameters
    ----------
    br : array_like
        The break points, ``np + 1`` increasing numbers from 0 to 1.
    coef : array_like
        The coefficients, of shape ``(np, deg + 1, K)``: ``coef[j, k, i]`` is the coefficient of r^k (of
        ``(r - br[j])^k`` with ``loc=True``) of regressor i on piece j.
    loc : bool, default False
        Whether the coefficients of piece j are those of the local variable ``r - br[j]``.
    first : bool, default False
        The statistic that detrends first (see `detrend_first`) instead of the one-step regression.
    name : str, optional
        A description, shown by ``repr``.

    Attributes
    ----------
    br : numpy.ndarray
        The break points (read-only).
    coef : numpy.ndarray
        The coefficients, shape ``(np, deg + 1, K)`` (read-only).
    K : int
        The number of regressors.
    npieces : int
        The number of pieces.
    degree : int
        The degree of the polynomials.
    loc, first : bool
        As the parameters.
    name : str
        The description.

    Examples
    --------
    >>> import dflimit
    >>> D = dflimit.perron("C", 0.3)
    >>> D
    Design('Perron model C, lambda = 0.3': K = 4 regressors, 2 pieces of degree 1)
    >>> D.value(0.5)                          # (1, r, 1{r > 0.3}, (r - 0.3)+) at r = 0.5
    array([1. , 0.5, 1. , 0.2])
    >>> E = dflimit.Design([0, 1], [[[1.0, 0.0], [0.0, 1.0]]], name="linear trend")
    >>> E == dflimit.trend(1)
    True
    """

    __slots__ = ("_br", "_cb", "_K", "_deg", "_loc", "_first", "_name")

    def __init__(self, br, coef, *, loc=False, first=False, name=None):
        b = dfl_real(br, "br").ravel()
        C = dfl_real(coef, "coef")
        if b.size < 2 or not np.all(np.isfinite(b)) or b[0] != 0 or b[-1] != 1 or not np.all(np.diff(b) > 0):
            raise ValueError("br must be increasing break points from 0 to 1")
        npc = b.size - 1
        if C.ndim == 2 and npc == 1:
            C = C[None, :, :]
        if C.ndim != 3 or C.shape[0] != npc or C.shape[1] < 1:
            raise ValueError("coef must have the shape (np, deg + 1, K), np the number of pieces")
        if not np.all(np.isfinite(C)):
            raise ValueError("coef must be finite")
        if not isinstance(loc, (bool, np.bool_)) or not isinstance(first, (bool, np.bool_)):
            raise TypeError("loc and first must be True or False")
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string")
        # (the engine's layout: piece j, regressor i, power k at j (deg + 1) K + i (deg + 1) + k)
        cb = np.ascontiguousarray(np.transpose(C, (0, 2, 1))).tobytes()
        self._set(b.tobytes(), cb, C.shape[2], C.shape[1] - 1, bool(loc), bool(first), name or "design")

    def _set(self, brb, cb, K, deg, loc, first, name):
        for k, v in zip(Design.__slots__, (brb, cb, int(K), int(deg), bool(loc), bool(first), name)):
            object.__setattr__(self, k, v)

    @classmethod
    def _of(cls, t, name):
        """The design of the engine's tuple (br, C, K, deg, loc, first)."""
        D = object.__new__(cls)
        D._set(t[0], t[1], t[2], t[3], t[4], t[5], name)
        return D

    def _tuple(self):
        """The tuple of the gateway (_engine.c)."""
        return (self._br, self._cb, self._K, self._deg, int(self._loc), int(self._first))

    def __setattr__(self, name, value):
        raise AttributeError("a Design is immutable")

    def __delattr__(self, name):
        raise AttributeError("a Design is immutable")

    def __reduce__(self):
        return (_rebuild, (self._tuple(), self._name))

    @property
    def br(self):
        return _readonly(np.frombuffer(self._br, dtype=np.float64))

    @property
    def coef(self):
        C = np.frombuffer(self._cb, dtype=np.float64).reshape(self.npieces, self._K, self._deg + 1)
        return _readonly(np.transpose(C, (0, 2, 1)))

    @property
    def K(self):
        return self._K

    @property
    def npieces(self):
        return len(self._br) // 8 - 1

    @property
    def degree(self):
        return self._deg

    @property
    def loc(self):
        return self._loc

    @property
    def first(self):
        return self._first

    @property
    def name(self):
        return self._name

    def __eq__(self, other):
        if not isinstance(other, Design):
            return NotImplemented
        return self._tuple() == other._tuple()

    def __hash__(self):
        return hash(self._tuple())

    def __repr__(self):
        npc = self.npieces
        s = (f"Design({self._name!r}: K = {self._K} regressor{'' if self._K == 1 else 's'}, {npc} piece"
             f"{'' if npc == 1 else 's'} of degree {self._deg}")
        if self._loc:
            s += ", local coefficients"
        if self._first:
            s += ", the statistic that detrends first"
        return s + ")"

    def value(self, r):
        """The regressors d(r): an array of shape ``r.shape + (K,)`` (``(K,)`` for a scalar ``r``).

        Examples
        --------
        >>> import dflimit
        >>> dflimit.trend(2).value([0.0, 0.5])
        array([[1.  , 0.  , 0.  ],
               [1.  , 0.5 , 0.25]])
        """
        a = dfl_real(r, "r")
        out = np.frombuffer(_engine.design_value(self._tuple(), a.ravel().tobytes()), dtype=np.float64)
        return out.reshape(a.shape + (self._K,)).copy()

    def regmat(self, n):
        """The n x K matrix of the regressors d(t/n), t = 1, ..., n, of a sample of size n.

        Examples
        --------
        >>> import dflimit
        >>> dflimit.perron("A", 0.5).regmat(4)
        array([[1.  , 0.25, 0.  ],
               [1.  , 0.5 , 0.  ],
               [1.  , 0.75, 1.  ],
               [1.  , 1.  , 1.  ]])
        """
        n = dfl_posint(n, "n")
        out = np.frombuffer(_engine.design_regmat(self._tuple(), n), dtype=np.float64)
        return out.reshape(self._K, n).T.copy()


def _int(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    return int(v)


def trend(p):
    """The polynomial trend d(r) = (1, r, ..., r^p).

    ``p = -1``: no deterministic terms (case DK), ``p = 0``: an intercept (DU), ``p = 1``: an intercept and a linear
    trend (DT); higher orders give the regressions with polynomial trends.

    Parameters
    ----------
    p : int
        The order, at least -1.

    Returns
    -------
    Design

    See Also
    --------
    ltrend, perron, ltratio_cdf

    Examples
    --------
    >>> from dflimit import trend, ltratio_cdf
    >>> trend(2)
    Design('quadratic trend': K = 3 regressors, 1 piece of degree 2)
    >>> ltratio_cdf(-3.41, trend(1))         # the classical case DT, by the router
    0.0500...
    """
    p = _int(p, "p")
    names = {-1: "no deterministic terms (DK)", 0: "intercept (DU)", 1: "linear trend (DT)", 2: "quadratic trend",
             3: "cubic trend"}
    return Design._of(_engine.design_trend(p), names.get(p, f"polynomial trend of order {p}"))


def ltrend(p):
    """The polynomial trend of order p in the orthonormal (shifted Legendre) basis sqrt(2k + 1) P_k(2r - 1).

    The same span as ``trend(p)``, hence the same statistic, with a well-conditioned Gram matrix at high order (the
    monomial basis has the Hilbert matrix, whose condition number reaches 5e11 at p = 8). Built by `fdesign`.

    Parameters
    ----------
    p : int
        The order, at least 0.

    Returns
    -------
    Design

    Examples
    --------
    >>> from dflimit import ltrend
    >>> ltrend(3)
    Design('polynomial trend of order 3 (Legendre basis)': K = 4 regressors, 1 piece of degree 3, local coefficients)
    """
    p = _int(p, "p")
    return Design._of(_engine.design_trend(p, 1), f"polynomial trend of order {p} (Legendre basis)")


def perron(model, lam):
    """Perron's models of a trend with breaks at known fractions.

    Model ``"A"``: (1, r, 1{r > lambda_i}) (breaks in the level), ``"B"``: (1, r, (r - lambda_i)+) (breaks in the
    slope, the trend joined at the breaks), ``"C"``: both for every break.

    Parameters
    ----------
    model : {"A", "B", "C"}
        The model.
    lam : float or array_like
        The break fractions, in (0, 1) and distinct.

    Returns
    -------
    Design

    See Also
    --------
    detrend_first : the statistic that detrends first, the valid one for model B

    Examples
    --------
    >>> from dflimit import perron, ltratio_icdf
    >>> perron("C", [0.3, 0.7])
    Design('Perron model C, lambda = 0.3, 0.7': K = 6 regressors, 3 pieces of degree 1)
    >>> ltratio_icdf(0.05, perron("A", 0.5))       # the 5% critical value
    -3.714662...
    """
    if not isinstance(model, str) or model.upper() not in ("A", "B", "C"):
        raise ValueError('model must be "A", "B" or "C"')
    lv = dfl_real(lam, "lam").ravel()
    if lv.size == 0:
        raise ValueError("lam must hold at least one break fraction")
    name = f"Perron model {model.upper()}, lambda = " + ", ".join(_fmt(v) for v in np.sort(lv))
    return Design._of(_engine.design_perron(model.upper(), lv.tobytes()), name)


def fdesign(f, breaks=(), *, K=None, tol=1e-14, maxdeg=96, sub=None, name="function design"):
    """A design given by a function of r.

    ``f(r)`` returns the K regressors at r, smooth on each piece between the breaks (jumps and kinks only there;
    left-continuous at a break). On each piece every regressor is replaced by its Chebyshev interpolant of the
    smallest degree 2^m - 1 >= 15 whose coefficients beyond the last that exceeds ``tol`` times the largest fall below
    it (an error beyond ``maxdeg``), re-expanded about the left ends of ``sub`` computational sub-pieces (default
    ceil(deg/4)) in local coefficients. ``f`` is called from the engine on the calling thread, a few hundred times.

    Parameters
    ----------
    f : callable
        ``f(r)`` -> the K regressors at r (a sequence of numbers).
    breaks : array_like, default ()
        The breaks, in (0, 1) and distinct.
    K : int, optional
        The number of regressors (default: the length of ``f(0.5)``).
    tol : float, default 1e-14
        The relative size of the last Chebyshev coefficient kept.
    maxdeg : int, default 96
        The largest degree of the interpolants.
    sub : int, optional
        The number of sub-pieces of each piece (default ceil(deg/4)).
    name : str, default "function design"
        A description.

    Returns
    -------
    Design

    Examples
    --------
    >>> import math
    >>> from dflimit import fdesign, ltratio_cdf
    >>> D = fdesign(lambda r: [1.0, math.sin(2 * math.pi * r)], name="intercept and a sine")
    >>> D.K, D.loc
    (2, True)
    >>> ltratio_cdf(-3.0, D, -5)
    0.237275...
    """
    if not callable(f):
        raise TypeError("f must be callable")
    if K is None:
        K = np.asarray(f(0.5), dtype=np.float64).size
    K = _int(K, "K")
    if K < 1:
        raise ValueError("K must be a positive integer")
    bk = dfl_real(breaks, "breaks").ravel()
    tol = dfl_num(tol, "tol")
    maxdeg = _int(maxdeg, "maxdeg")
    sub = 0 if sub is None else _int(sub, "sub")
    if not isinstance(name, str):
        raise TypeError("name must be a string")

    def g(r):
        v = np.asarray(f(r), dtype=np.float64).ravel()
        if v.size != K:
            raise ValueError(f"f({r:g}) returned {v.size} values, the design has K = {K}")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"f({r:g}) is not finite")
        return v.tobytes()

    return Design._of(_engine.design_fdesign(g, K, bk.tobytes(), tol, maxdeg, sub), name)


def design_from_matrix(X, breaks=None, *, tol=1e-8, maxdeg=40, minlen=10, name="design from a matrix"):
    """The design of the limit of the deterministic regressors of a sample.

    ``X`` is the n x K matrix of the deterministic regressors of a regression (column k sampled at t = 1, ..., n, any
    scaling). The known breaks, if given, are the indices T_B after which the regressors change (a regime of t <= T_B
    and one of t > T_B, as 1{t > T_B}), at lambda = T_B/n; without them the breaks are detected from the sixth
    differences of the columns. On each piece every column is fitted by Chebyshev polynomials of increasing degree up
    to ``maxdeg`` until the largest residual is below ``tol``, and the design is the function design of these fits
    (`fdesign`). Columns that are not piecewise smooth at the sample's resolution (a piece shorter than ``minlen``
    observations, as a pulse dummy gives, or seasonal dummies) raise an error: they do not affect the limit and should
    be left out. Columns linearly dependent on the others are dropped, with a ``RuntimeWarning``.

    Parameters
    ----------
    X : array_like
        The n x K regressor matrix.
    breaks : array_like of int, optional
        The break indices T_B (1 <= T_B < n); detected if not given.
    tol : float, default 1e-8
        The tolerance of the fits.
    maxdeg : int, default 40
        The largest degree of the fits.
    minlen : int, default 10
        The shortest piece, in observations.
    name : str, default "design from a matrix"
        A description.

    Returns
    -------
    Design

    Examples
    --------
    >>> import numpy as np
    >>> from dflimit import design_from_matrix, perron
    >>> t = np.arange(1, 201)
    >>> X = np.column_stack([np.ones(200), t, t > 80])     # an intercept, a trend and a level shift after t = 80
    >>> D = design_from_matrix(X)
    >>> D.br
    array([0. , 0.4, 1. ])
    """
    a = dfl_real(X, "X")
    if a.ndim != 2:
        raise ValueError("X must be a matrix (n x K)")
    n, K = a.shape
    if not np.all(np.isfinite(a)):
        raise ValueError("X must be finite")
    if breaks is None:
        tb = None
    else:
        b = np.asarray(breaks)
        if b.dtype.kind not in "iu" and not (b.dtype.kind == "f" and np.all(b == np.floor(b))):
            raise TypeError("breaks must be integer indices")
        tb = np.asarray(b, dtype=np.intc).ravel().tobytes()
    tol = dfl_num(tol, "tol")
    maxdeg = _int(maxdeg, "maxdeg")
    minlen = _int(minlen, "minlen")
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    t, w = _engine.design_matrix(np.asfortranarray(a).tobytes(order="F"), n, K, tb, tol, maxdeg, minlen)
    if w:
        warn(w)
    return Design._of(t, name)


def detrend_first(D):
    """The design D for the statistic that detrends first.

    y_t is regressed on d(t/n), t = 1, ..., n, and the statistic is the t-ratio of the lagged residual in the
    regression of the differenced residual on it (Perron and Vogelsang), with a dummy for the observation after each
    jump of d. The one-step regression is invariant to the trend coefficients only when the span of d is closed under
    differentiation (polynomial trends, Perron's models A and C); for a break in the slope alone (model B) the test
    that detrends first is the valid one. Zero start only; no moments; GLS detrending detrends first already.

    Parameters
    ----------
    D : Design

    Returns
    -------
    Design

    Examples
    --------
    >>> from dflimit import detrend_first, perron
    >>> detrend_first(perron("B", 0.3))
    Design('Perron model B, lambda = 0.3 (detrended first)': K = 3 regressors, 2 pieces of degree 1, the statistic that detrends first)
    """
    if not isinstance(D, Design):
        raise TypeError("D must be a Design")
    return Design._of(_engine.design_first(D._tuple()), D.name + " (detrended first)")


def is_trend(D, p):
    """Whether D is trend(p) (the monomial basis, the one-step regression), as the engine's router decides."""
    if D.first or D.loc or D.K != p + 1 or D.npieces != 1 or D.degree != max(p, 0):
        return False
    return D._cb == _engine.design_trend(p)[1]


def classical_case(D, start, gls):
    """The case code of the classical engine for the design D with the start (0 zero, 1 fixed, 2 stationary) and the
    GLS detrending gls (NaN: least squares), or None (the router's rule)."""
    for p, l in ((-1, "K"), (0, "U"), (1, "T")):
        if not is_trend(D, p):
            continue
        if math.isnan(gls):
            return ("S" if start == 2 else "D") + l
        if l == "T" and start == 0 and gls == -13.5:
            return "DG"
        return None
    return None


def _rebuild(t, name):
    """A design from its tuple (for pickle)."""
    return Design._of(t, name)
