"""dflimit: the exact limiting distribution of the Dickey-Fuller t-ratio.

The distribution function, density, quantiles and moments of the limits of the Dickey-Fuller t-ratios under local
alternatives, by the exact representations of Bao, Broda and Kan, "The Exact Limiting Distribution of the
Dickey-Fuller t-Ratio with General Deterministic Terms" (the lead paper, working paper, SSRN 7745921: the inversion
formula on paths and the limit for any piecewise smooth deterministic terms, the general engine) and, for the classical
cases, its companion, "Low-Rank Representations of the Classical Dickey-Fuller Limits, with an Extension to Predictive
Regressions" (working paper, SSRN 7725619: the line and the conditional integral, the classical engine), computed by
their C engine (a port of the Julia reference implementation; the same functions, with the same names and arguments,
exist in R, MATLAB and Julia).

- `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`, `ltratio_mom`: the t-ratio, cases DK, DU, DT (zero or fixed start),
  SK, SU, ST (stationary start) and DG (GLS-detrended trend case);
- `coef_cdf`, `coef_icdf`, `coef_mom`: the coefficient statistic n(phi^ - phi0) (its density `coef_pdf` also takes a
  case code);
- `adfmax_cdf`, `adfmax_pdf`, `adfmax_icdf`: the ADFmax test of Leybourne (1995);
- `ltratio_region`: the probability of a region {U <= g(V)}, case DK;
- `ltratio_lincdf`: the linear form U - aV, cases DK and DG;
- `power_envelope`: the asymptotic power envelope of Elliott, Rothenberg and Stock (1996);
- `ltratio_threads`: the number of worker threads.

The general engine (any piecewise smooth deterministic terms: polynomial trends, breaks in level and slope, designs
given by a function or by the regressor matrix of a sample; the inversion on paths around the roots of the determinant
of the joint moment generating function):

- the designs `Design`, `trend`, `perron`, `fdesign`, `ltrend`, `design_from_matrix`, `detrend_first`, which the
  functions above take in place of the case, with the keyword ``engine`` ("auto", "classical", "general") of the
  router over the two engines;
- `dist_cdf`, `dist_pdf`, `dist_quantile`, `dist_moments`, `coef_pdf`, `coef_quantile`, `coef_cdf_gp`,
  `local_power`, `lin_cdf`, `lin_pdf`, `lin_quantile`, `v_cdf`, `v_pdf`, `adfmax_quantile`, `region_prob`: its
  functions of a design, with the start, the fixed start, a local null value, GLS detrending and ``verified``;
- the companions `pt_cdf`, `pt_pdf`, `pt_quantile`, `pt_power`, `pt_envelope` (the point-optimal statistic),
  `lm_cdf`, `lm_quantile`, `lm_power` (the LM statistic), `mza_cdf`, `mza_quantile` (MZ_alpha), `msb_cdf`,
  `msb_quantile` (MSB); `empty_caches`.

The extensions of the limit, computed by the general engine:

- `volpath` (`VolPath`) and the keyword ``vol`` of `dist_cdf`, `dist_pdf`, `dist_quantile`, `ltratio_cdf`,
  `ltratio_pdf`, `ltratio_icdf`: volatility and persistence regimes (``c`` a sequence of one value per regime);
- the keyword ``rho`` of the same functions: a normal component, the covariate-augmented t-ratio (`cadf_c0`) and the
  predictive regression (`predictive_c0`);
- `hegy_t3_cdf`, `hegy_t3_pdf`, `hegy_t3_quantile`: the first seasonal t-ratio of Hylleberg, Engle, Granger and Yoo.

Not in the package, because the C engine does not have it: the boundary type ``"fun"`` (any function g) of
``ltratio_region``.  (A new entry point of the engine: a function in _engine.c, see the comment at its top, and a
function in a module like _lincdf.py, exported here.)
"""
from ._adfmax import adfmax_cdf, adfmax_icdf, adfmax_pdf
from ._coef import coef_cdf, coef_icdf, coef_mom
from ._design import Design, design_from_matrix, detrend_first, fdesign, ltrend, perron, trend
from ._ext import VolPath, cadf_c0, predictive_c0, volpath
from ._general import (adfmax_quantile, coef_cdf_gp, coef_pdf, coef_quantile, dist_cdf, dist_moments, dist_pdf,
                       dist_quantile, empty_caches, hegy_t3_cdf, hegy_t3_pdf, hegy_t3_quantile, lin_cdf, lin_pdf,
                       lin_quantile, lm_cdf, lm_power, lm_quantile,
                       local_power, msb_cdf, msb_quantile, mza_cdf, mza_quantile, pt_cdf, pt_envelope, pt_pdf, pt_power,
                       pt_quantile, region_prob, v_cdf, v_pdf)
from ._lincdf import ltratio_lincdf, power_envelope
from ._ltratio import ltratio_cdf, ltratio_icdf, ltratio_mom, ltratio_pdf
from ._region import ltratio_region
from ._threads import default_threads as _default_threads
from ._threads import ltratio_threads
from ._utils import Result

__version__ = "0.5.0"

__all__ = [
    "ltratio_cdf", "ltratio_pdf", "ltratio_icdf", "ltratio_mom",
    "coef_cdf", "coef_icdf", "coef_mom",
    "adfmax_cdf", "adfmax_pdf", "adfmax_icdf",
    "ltratio_region",
    "ltratio_lincdf", "power_envelope",
    "ltratio_threads",
    "Result",
    "Design", "trend", "perron", "fdesign", "ltrend", "design_from_matrix", "detrend_first",
    "dist_cdf", "dist_pdf", "dist_quantile", "dist_moments", "coef_pdf", "coef_quantile", "coef_cdf_gp", "local_power",
    "lin_cdf", "lin_pdf", "lin_quantile", "v_cdf", "v_pdf", "adfmax_quantile", "region_prob",
    "pt_cdf", "pt_pdf", "pt_quantile", "pt_power", "pt_envelope", "lm_cdf", "lm_quantile", "lm_power",
    "mza_cdf", "mza_quantile", "msb_cdf", "msb_quantile", "empty_caches",
    "volpath", "VolPath", "cadf_c0", "predictive_c0", "hegy_t3_cdf", "hegy_t3_pdf", "hegy_t3_quantile",
]

_n = _default_threads()
if _n is not None:
    ltratio_threads(_n)
del _n
