"""The extensions of the limit (the names of DFLimit.jl): volatility and persistence regimes (`volpath`, the keyword
``vol`` of the t-ratio functions), a normal component (the keyword ``rho``: the covariate-augmented t-ratio and the
predictive regression; `cadf_c0`, `predictive_c0`) and the first seasonal t-ratio of Hylleberg, Engle, Granger and Yoo
(`hegy_t3_cdf`, `hegy_t3_pdf`, `hegy_t3_quantile`), computed by the general engine (csrc/dfl_det_ext.c).
"""
import math
import numbers

import numpy as np

from ._utils import dfl_num

VOL_DOC = """\
vol : VolPath, optional
    Volatility and persistence regimes (`volpath`): the limit of the process dJ = c(r) J dr + omega(r) dW with the
    standard deviation omega and, with ``c`` a sequence of one value per regime, the local-to-unity parameter piecewise
    constant between known dates (zero start, least squares detrending, the one-step regression, every c in [-480, 0];
    ``method`` "auto" or "paths": the inversion on paths of the volatility transform). One regime with a scalar ``c`` is
    the engine itself."""

RHO_DOC = """\
rho : float, default 1
    A normal component: the limit rho tau_c0 + sqrt(1 - rho^2) Z with Z standard normal and independent of the
    t-ratio's limit tau_c0, the limit of the covariate-augmented t-ratio of Hansen (1995) at ``c0 = cadf_c0(c, rho, R)``
    and of the t-ratio of the slope in a predictive regression with a regressor local to unity at ``c0 =
    predictive_c0(c, rho, b)``. ``rho`` in [-1, 1]: 1 the t-ratio itself, 0 the standard normal law, a negative ``rho``
    by the reflection; ``method`` "auto", "lines" (the inversion on two lines with the normal component in the kernel; a
    zero or stationary start, c <= 0) or "mixture" (the normal average of the t-ratio's distribution function); "auto"
    takes the lines for 0.1 <= rho^2 <= 0.99 with a zero or stationary start at c <= 0 and the mixture otherwise. Least
    squares detrending and the one-step regression; not together with ``vol``."""


class VolPath:
    """Volatility regimes: the boundaries ``lam`` (0, the break dates, 1) and the standard deviations ``om`` of the
    innovations on the regimes, scaled so that the integral of omega^2 is one (`volpath`)."""

    def __init__(self, breaks, omega):
        self.breaks = breaks
        self.omega = omega
        self.lam = np.concatenate(([0.0], breaks, [1.0]))
        self.om = omega / math.sqrt(float(np.sum(omega ** 2 * np.diff(self.lam))))

    @property
    def nregimes(self):
        return len(self.om)

    def __repr__(self):
        return f"volpath({list(self.breaks)}, {[round(float(w), 6) for w in self.om]})"

    def _vec(self, c=None):
        """The doubles of the engine: nreg, the break dates, the standard deviations (as given) and the c of the
        regimes (left out: the c of the call)."""
        v = [float(self.nregimes)] + list(self.breaks) + list(self.omega)
        if c is not None:
            v += list(c)
        return np.asarray(v, dtype=np.float64).tobytes()


def volpath(breaks=(), omega=None):
    """Volatility (and persistence) regimes between known dates.

    Parameters
    ----------
    breaks : array_like
        The interior regime boundaries, strictly increasing in (0, 1) (fractions of the sample); empty: one regime, the
        engine itself.
    omega : array_like, optional
        The standard deviations of the innovations on the ``len(breaks) + 1`` regimes, finite and positive, of any scale
        (they are scaled so that the integral of omega^2 is one; only the ratios matter); default all equal.

    Returns
    -------
    VolPath
        The keyword ``vol`` of `dist_cdf`, `dist_pdf`, `dist_quantile`, `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`
        and `hegy_t3_cdf`, `hegy_t3_pdf`, `hegy_t3_quantile`.

    Examples
    --------
    >>> from dflimit import volpath, dist_cdf, trend
    >>> V = volpath([0.3], [1, 3])            # the standard deviation triples at 30% of the sample
    >>> dist_cdf(trend(0), -1.5, 0, vol=V)
    0.386893...
    """
    br = np.atleast_1d(np.asarray(breaks, dtype=np.float64))
    if not np.all(np.isfinite(br) & (br > 0) & (br < 1)):
        raise ValueError("volpath: the break dates must lie in (0, 1)")
    if br.size > 1 and np.any(np.diff(br) <= 0):
        raise ValueError("volpath: the break dates must be strictly increasing")
    om = np.ones(br.size + 1) if omega is None else np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if om.size != br.size + 1:
        raise ValueError(f"volpath: one standard deviation omega per regime: {br.size + 1} for {br.size} break dates, "
                         f"{om.size} given")
    if not np.all(np.isfinite(om) & (om > 0)):
        raise ValueError("volpath: the standard deviations omega must be finite and positive")
    return VolPath(br, om)


def _rho(rho, fn):
    if isinstance(rho, (bool, np.bool_)) or not isinstance(rho, numbers.Real):
        raise TypeError(f"{fn}: rho must be a number")
    rho = float(rho)
    if not (math.isfinite(rho) and -1 <= rho <= 1 and rho != 0):
        raise ValueError(f"{fn}: rho must lie in [-1, 1] and not be 0")
    return rho


def cadf_c0(c, rho, R=None):
    """The local null value of the covariate-augmented Dickey-Fuller t-ratio of Hansen (1995): c (1 - 1/(rho R)).

    ``rho`` the long-run correlation of the innovation of the series and the error of the augmented regression, ``R``
    the ratio of their long-run standard deviations (``abs(rho)``, the default, for the well-specified regression;
    Hansen's rho^2 and R^2 are the squares); zero at c = 0. With ``rho`` and ``c0 = cadf_c0(c, rho, R)`` the t-ratio
    functions give the limit of the statistic, rho tau_c0 + sqrt(1 - rho^2) Z.

    >>> from dflimit import cadf_c0
    >>> round(cadf_c0(-5, 0.5 ** 0.5), 12)     # -c (1 - rho^2)/rho^2 when R = rho
    5.0
    """
    rho = _rho(rho, "cadf_c0")
    R = abs(rho) if R is None else R
    if isinstance(R, (bool, np.bool_)) or not isinstance(R, numbers.Real) or not (math.isfinite(R) and R > 0):
        raise ValueError("cadf_c0: R must be finite and positive")
    if isinstance(c, (bool, np.bool_)) or not isinstance(c, numbers.Real) or not math.isfinite(c):
        raise ValueError("cadf_c0: c must be finite")
    return 0.0 if c == 0 else float(c) * (1 - 1 / (rho * float(R)))


def predictive_c0(c, rho, b=0):
    """The local null value of the t-ratio of the slope in the predictive regression y_t = mu + beta x_{t-1} + u_t,
    x_t = phi x_{t-1} + v_t, phi = 1 + c/n: c - b/rho, with ``rho`` the correlation of the innovations and the local
    alternative beta = b sigma_u/(sigma_v n). With ``rho`` and ``c0 = predictive_c0(c, rho, b)`` the t-ratio functions
    of the design with an intercept (``trend(0)``, case "DU") give the limit of the statistic, rho tau_c0 + sqrt(1 -
    rho^2) Z.

    >>> from dflimit import ltratio_cdf, predictive_c0
    >>> ltratio_cdf(1.645, "DU", 0, predictive_c0(0, -0.95), rho=-0.95)
    0.5788...
    """
    rho = _rho(rho, "predictive_c0")
    ok = all(not isinstance(v, (bool, np.bool_)) and isinstance(v, numbers.Real) and math.isfinite(v) for v in (c, b))
    if not ok:
        raise ValueError("predictive_c0: c and b must be finite")
    return float(c) - float(b) / rho


def ext_args(vol, rho, c):
    """The extensions of a call for the engine: (the scalar c of the call, rho, the volatility path's doubles or None);
    c a sequence needs vol (one value per regime)."""
    if isinstance(rho, (bool, np.bool_)) or not isinstance(rho, numbers.Real):
        raise TypeError("rho must be a number")
    rho = float(rho)
    seq = not isinstance(c, (numbers.Real, str, bytes)) and c is not None and np.ndim(c) > 0
    if vol is None:
        if seq and np.size(c) > 1:
            raise TypeError("c must be a single number (a vector of one value per regime needs a volatility path, "
                            "vol=volpath(breaks, omega))")
        return (dfl_num(np.ravel(c)[0] if seq else c, "c"), rho, None)
    if not isinstance(vol, VolPath):
        raise TypeError("vol must be a volatility path, volpath(breaks, omega)")
    if not seq or np.size(c) == 1:
        return (dfl_num(np.ravel(c)[0] if seq else c, "c"), rho, vol._vec())
    cs = np.asarray(c)
    if cs.dtype.kind not in "iuf":
        raise TypeError("c must be numeric")
    cs = np.ravel(cs).astype(np.float64)
    if cs.size != vol.nregimes:
        raise ValueError(f"c must be a number or a vector with one value per regime of the volatility path ({vol.nregimes}), "
                         f"not {cs.size}")
    return (float(cs[0]), rho, vol._vec(cs))


def ext_active(rho, vb):
    """Whether a call has an extension (rho != 1 or a volatility path)."""
    return vb is not None or rho != 1
