"""The linear form U - aV and the power envelope: ltratio_lincdf, power_envelope (R/lincdf.R of the R package
dflimit)."""
import numbers

import numpy as np

from . import _engine
from ._ltratio import ERRORS_DOC
from ._utils import dfl_case, dfl_num, dfl_real, dfl_shape, f64, fill_doc

ERS_REFERENCE_DOC = """\
.. [1] Elliott, G., Rothenberg, T. J. and Stock, J. H. (1996), "Efficient Tests for an Autoregressive Unit Root",
   Econometrica 64, 813-836."""


@fill_doc(errors=ERRORS_DOC, references=ERS_REFERENCE_DOC)
def ltratio_lincdf(b, a, cas, c=0, cbar=-13.5):
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
    cas : str
        The case, ``"DK"`` or ``"DG"``.
    c : float, default 0
        The local-to-unity parameter, finite.
    cbar : float, default -13.5
        The detrending parameter cbar of case DG (-13.5: the DF-GLS tests); not used in case DK.

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
    """
    _, model, _ = dfl_case(cas, ("DK", "DG"))
    msg = "a must be a single number (NaN: the distribution of V)"
    if isinstance(a, np.ndarray):
        if a.ndim != 0 or a.dtype.kind not in "iuf":
            raise TypeError(msg)
        a = a[()]
    if isinstance(a, (bool, np.bool_)) or not isinstance(a, numbers.Real):
        raise TypeError(msg)
    x = dfl_real(b, "b")
    P = _engine.lincdf(x.tobytes(), float(a), model, dfl_num(c, "c"), dfl_num(cbar, "cbar"))
    return dfl_shape(f64(P), x)


@fill_doc(errors=ERRORS_DOC, references=ERS_REFERENCE_DOC)
def power_envelope(c, cas="DK", alpha=0.05):
    """Asymptotic power envelope of Elliott, Rothenberg and Stock (1996).

    The power at ``c`` of the point-optimal level-``alpha`` test against ``c`` itself, the envelope of the powers
    of all tests of the unit root, zero start, no drift: case ``"DK"`` (no deterministic terms; also the constant
    case, ``"DU"``, where the point-optimal test uses the GLS-demeaned data and the envelope is the same curve) and
    case ``"DG"`` (the trend case, the point-optimal test against ``c`` with the data GLS-detrended at ``c``).

    Parameters
    ----------
    c : array_like
        The local-to-unity parameters. ``nan`` gives ``nan``.
    cas : str, default "DK"
        The case, ``"DK"``, ``"DU"`` (the same as ``"DK"``) or ``"DG"``.
    alpha : float, default 0.05
        The level of the tests.

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
    at cbar = c; ``alpha`` for ``c >= 0``. As ``power_envelope`` of the MATLAB and Julia programs.

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
    """
    _, model, _ = dfl_case(cas, ("DK", "DU", "DG"))
    x = dfl_real(c, "c")
    E = _engine.envelope(x.tobytes(), 4 if model == 4 else 1, dfl_num(alpha, "alpha"))
    return dfl_shape(f64(E), x)

