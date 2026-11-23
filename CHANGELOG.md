# Changelog

## 0.5.0

- The C engine of dflimit-matlab 45159c9: the upper tail above the unit root. Above the unit root the t-ratio with
  `c0 != c` has its upper tail at x of the order (c - c0) e^c/(2c), and the default route returned values of the
  distribution function outside [0, 1] there without a warning (case DK at c = 10, c0 = 2: 0.8415 for 0.7436 at
  x = 1e4, -1.3674 for 0.9993 at x = 3e4; at c = 10, c0 = 9: 1.0479 for 0.9936 at x = 3e3; also with `verified=True`
  and `engine="classical"`): beyond the line integral's bound the conditional integral returned its one evaluation
  where its checked value was flagged, off by up to 14 in case DK. Case DK now takes the checked value (within 1.3e-8
  of an independent computation up to c = 14); for c > 4 the three cases take the conditional integral at x > 300 too
  (the line integral of cases DU, DT exhausted the memory at c = 16, c0 = 0 at the median, and its size overflowed an
  int in the C engine), and case DK with `c0 > c` at x < -20 (where the line integral was off by up to 1.5e-3); above
  c = 14 these points raise `ValueError` naming the case, c, c0, x and the region computed, with `verified=True` too
  (the conditional integral is off by up to 7.7e-5 in case DK at c = 16 and gives 1e44 in case DT). The general
  engine (`engine="general"`) cut its line as if c0 were 0 and was off by 1.1e-3 at c = 10, c0 = 9, x = 1000: now
  within 2e-13. Every distribution function of both engines ends with a check of its values: outside [0, 1] by more
  than 1e-8 a `ValueError` that names the call.
- The C engine of dflimit-matlab 06bb2cd: the general engine above the unit root. The coefficient statistic far above
  the unit root was silently wrong in the general engine (the segment of the path around the positive root, shorter than
  1e-12 from c = 16 on, got no panel; case DK at c = 20 gave 0.3128 for the 30% point with `engine="general"`; the
  default route takes the classical engine there): now within 2.4e-9 of the classical engine up to c = 20 without
  deterministic terms and 1.5e-8 up to c = 16 with them. The moments of the general engine, which the default route
  takes at `c0=0`, were off by 6e-4 relative at c = 15 and 0.95 at c = 20 (an absolute rounding error of 1e-16 in the
  matrix of the transform): now within 1.1e-14 of the classical engine. `ltratio_mom(4, "DT", 0, c0=-2)` raised
  `ValueError` on the default route (the circle of the Cauchy integrals continued the wrong logarithm): now within
  2.4e-14. The density of the coefficient statistic with a fixed start above the unit root (`coef_pdf`) was off by up to
  0.12 at c from 1 to 5 without a warning (the path around the positive roots came too close to the double pole of its
  kernel): now within about 1e-8 of the single inversion of DFKan (DFLimit.jl). `coef_cdf_gp` raises `ValueError` that
  names the grid size `N` that serves where its grid does not resolve the transform (it was off by 1.1e-2 at c = 12 and
  by 15 at c = 20 for the 30% point of case DK). Every function of the general engine raises `ValueError` above c = 16
  with deterministic terms (10 with them and a fixed start) and above c = 20 without them, where the roots of its
  transform are no longer resolved in double precision; `engine="auto"` takes the classical engine there.
- The C engine of dflimit-matlab 4910ea2: the line integral with the explicit kernel for the general designs of rank one
  or two (polynomial trends of any order, an intercept with sines and cosines, the statistic that detrends first for a
  continuous design with d(0) != 0; zero or stationary start, least squares detrending): `dist_cdf`, `dist_pdf` and
  `dist_quantile` take `method="auto"` (the default), `"line"` or `"paths"`; `"auto"` takes the line integral at
  `c0 = 0` for `c >= -20`, 2.2 to 3.5 times faster than the inversion on paths per value and 1.8 to 2.7 times per
  quantile, and the paths elsewhere; `"line"` raises `ValueError` with the reason where it does not cover the design.
  `ltratio_cdf`, `ltratio_pdf` and `ltratio_icdf` with a design take `method="line"` (for a design that is a classical
  case, the classical line integral) or `"paths"` (the general engine's paths). The values at `c0 = 0` for these designs
  move within 2e-11. The classical line integral takes its inner integrals on fixed node layouts at `c <= 0`, `c0 = 0`
  (as the Julia engine; 8 to 20 times faster in case DK, 2 to 3 times in the other cases, the values within 3.2e-12 of
  before), so that the router keeps the classical engine at `c0 = 0` below `c = -20` again, and the coefficient
  statistic (`coef_cdf`, `coef_icdf`) takes the classical engine at its points with `c0 + y > 0` and in case DU (its
  slowdown by three times at `c = -5` in case DU is gone). A quantile search whose last Newton step was below half an
  ulp no longer falls into bisection.

The general engine of the C engine (dflimit-matlab, its branch `general-engine`; the port of the general engine of
DFLimit.jl): the limits for any piecewise smooth deterministic terms, by the inversion formula on paths around the roots
of the determinant of the joint moment generating function. The engine is chosen automatically by default
(`engine="auto"`, also for a case string): where the router's rule takes the general engine for a case string the two
engines agree to about 1e-9, and the values move in the tenth digit or below against 0.4.x; `engine="classical"` gives
the classical values, bit for bit those of before.

- Designs: the immutable class `Design` and its constructors `trend`, `perron`, `fdesign`, `ltrend`,
  `design_from_matrix` and `detrend_first` (the names of the Julia package), with `Design.value` and `Design.regmat`.
- The general engine's functions of a design: `dist_cdf`, `dist_pdf`, `dist_quantile`, `dist_moments`, `coef_pdf`,
  `coef_quantile`, `coef_cdf_gp`, `local_power`, `lin_cdf`, `lin_pdf`, `lin_quantile`, `v_cdf`, `v_pdf`,
  `adfmax_quantile`, `region_prob`, and the companions `pt_cdf`, `pt_pdf`, `pt_quantile`, `pt_power`, `pt_envelope`
  (the point-optimal statistic and the power envelope of any design), `lm_cdf`, `lm_quantile`, `lm_power` (the LM
  statistic), `mza_cdf`, `mza_quantile` (MZ_alpha), `msb_cdf`, `msb_quantile` (MSB); `empty_caches`. The options are
  keywords: `c0`, `start` (`"zero"`, `"fixed"`, `"stationary"`), `gamma0`, `gls` and `verified`.
- The router: `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`, `ltratio_mom`, `coef_cdf`, `coef_icdf`, `adfmax_cdf`,
  `adfmax_pdf`, `adfmax_icdf`, `ltratio_lincdf`, `power_envelope` and `ltratio_region` take a design in place of the
  case (with the keywords `start` and `gls`) and the keyword `engine`: `"auto"` (the default, also given as `None`)
  the rule of the router for a case string as for a design (the faster engine where the two agree, the more accurate
  where they do not, the general engine where the classical one does not apply), `"classical"` and `"general"` force
  one engine. The general engine's warnings are `RuntimeWarning`s.
- ADFmax with a fixed start (`g0` of `adfmax_cdf`, `adfmax_pdf`, `adfmax_icdf`, with `engine="auto"` or
  `"general"` for a case string, or a design) and the regions and the linear form of case DK with a fixed start (`g0`
  of `ltratio_region` and `ltratio_lincdf`) take the general engine; the classical engine, which has the zero and the
  stationary start only there, refuses a fixed start.
- The general engine refuses, with a `ValueError`: probabilities and levels outside (0, 1); the LM statistic above
  c = 10; the point-optimal statistic where the grid of its inversion does not resolve the transform (far above the
  unit root; the message names the grid size `N` that serves) and at |q| beyond 0.15 N/asinh(umax); a design whose
  roots run off on the vertical line (paths of more than a million nodes, such as (1 + r, 1{r < 0.4})). The law of V
  is 0 at b <= 0; above the unit root the LM statistic is computed on the path of V alone.
- The C engine of dflimit-matlab 8f78233: a design whose roots run off along the outer contour (as (1 + r,
  1{r <= 0.4})) is refused before any inner integral, in 0.1 s and a few MB (it took 1.4 s and 1.2 GB before the
  refusal); the outer hyperbola refuses more than 1e5 nodes and the moments a grid of more than 1e6 points; at
  non-finite arguments every function of the general engine and of the router gives the limit or an error that names
  the argument (`coef_cdf_gp` at y = +-inf, the searches from a start value of +-inf and the point-optimal statistic at
  c = +-inf did not return in C); the rules of the general engine far from the unit root start at c = -60 (they started
  at c = -100, and on -100 <= c < -60 the general engine was off by up to 1.4e-9, now 5e-13). `ltratio_region` with
  case DK gives 1, 0 at b = inf, -inf (the classical code gave 0) and refuses a non-finite slope.
- The tests compare 299 values of the general engine and 118 of the router with the stored values of the Julia
  engine, and 44 calls that must fail with the expected errors (a subset of `c/test/det_ref.txt` of dflimit-matlab,
  copied by `tools/det-ref.py`).

Four errors of the engine are repaired (the C engine of dflimit-matlab 0.5.0, the repairs of DFLimit.jl 0.5.0). Values
outside the regions named below are unchanged.

- Large `x` above the unit root crashed Python. In the bulk of the distribution at `c = 7` and beyond (`x` in the
  hundreds) the line integral needs a Gauss-Legendre rule of more than 4096 nodes, which the engine does not hold:
  `ltratio_cdf(751, "DT", 7)` ended the interpreter with a segmentation fault. Such a point is now refused before it
  is evaluated, and `method="auto"` takes the conditional integral there (0.990824, in 0.2 seconds).
- A negative local null value near zero took unbounded memory. With `c0 < 0` close to zero and `x > 0`, in every case
  and with every start, the paths of the line integral grow like `(x/c0)^2` and every node was stored:
  `ltratio_cdf(0.05, "SK", -1e-3, c0=-1e-3)` took more than 2 GB within two seconds, with no bound. Paths are now
  summed as they are built and limited to two million nodes; beyond that `method="auto"` takes the conditional
  integral, with a `RuntimeWarning` where its value is flagged, and `method="single"` (and case DG, which has the line
  integral only) raises a `ValueError` that names the arguments. Below the limit such a call needs a few megabytes
  instead of up to 1 GB.
- Wrong values with a fixed start. With a fixed start (`g0 != 0`), `c0` of -0.01 or -0.001, `x > 0` and `c >= 0`, the
  line integral returned wrong values, from errors of 1e-7 and 2.4e-3 that look plausible to 1e74 and beyond:
  `ltratio_cdf(1.5, "DK", 0, c0=-0.01, g0=1)` gave 6.03e74 for 0.960992, `ltratio_cdf(1.5, "DK", 2, c0=-0.01, g0=1)`
  0.08440 for 0.08676. Case DK was affected by default, cases DU and DT with `method="single"`. A piece of the path
  that should descend climbed and cancelled against its neighbour; such a piece is now cut back. The same fault gave
  -3.5e86 for 0.48 in case DU with `c=-20`, `c0=1`, `g0=1`, `method="single"`, and 1.1e13 for 0.68267 for
  `ltratio_cdf(0, "DK", 0, c0=-1e-4)`.
- Wrong values of the linear form for small `|a|`. `ltratio_lincdf` inverted the characteristic function of U - aV by
  the formula of Gil-Pelaez on a grid that ends at u = 4000. At `a = 0` (the law of U, whose density is infinite at
  the edge b = -1/2 of its support) the transform decays only like u^(-1/2), and the values were off by up to 7e-2
  within 1e-3 of the edge, 1e-2 within 1e-2 of it and 1e-4 in the bulk, in cases DK and DG alike, as low as -1.5e-2
  below the support and above 1 by up to 5e-5: `ltratio_lincdf(-0.4, 0, "DK")` gave 0.345291 for 0.345279
  (`math.erf(math.sqrt(0.1))`), `ltratio_lincdf(-0.5, 0, "DG")` 0.0197 for 0. The error stays at that size for `|a|`
  up to about 3e-3, falls to 1e-3 in case DK (4e-5 in case DG) at `|a| = 0.3`, and is larger where U and V are
  concentrated (`c` of -50 and below) and where U - aV is wide (up to 1.2e-2 in case DK at `c = 5`); `P(V <= b)` (`a =
  NaN`) was off by 4e-2 at `c = -200`. Where the transform has not decayed by the end of the grid, or `c > 0`, the
  value now comes from a conditional integral (the region {U <= aV + b} in case DK, the conditional integral given U +
  1/2 in case DG), to about 1e-10, within [0, 1] and 0 below the support; `P(V <= b)` takes a longer or finer grid.
  `power_envelope` near `c = 0` takes the conditional integral too (in case DG it fell below `alpha` there).
  `ltratio_lincdf` now requires `c >= -550`, in case DG `c` up to about 3.5 at `cbar = -13.5` and for `a = NaN` `c` up
  to about 7 (a `ValueError` beyond). Other values are unchanged; `ltratio_region` with `v0 > 0` changes with P(V <
  v0) at `c <= -50` and `c > 2`.

## 0.4.1

- Documentation only; the code and the results are those of 0.4.0. The package is on PyPI (`pip install dflimit`); the
  README gives the paper's address on SSRN; `CITATION.cff` names the paper as the suggested citation, and
  `.zenodo.json` describes the package for its archive on Zenodo.

## 0.4.0

- The documentation and the comments refer to the representations by name (the line integral, the contour integral
  and the conditional integral representation) instead of by the paper's theorem and equation numbers. The results are
  unchanged.
- The C engine is now copied from dflimit-matlab, the repository of the MATLAB package, which holds the C engine since
  it left the paper's repository (`tools/sync-engine.sh /path/to/dflimit-matlab`); the sources are unchanged.

## 0.3.0

- Every value is now one evaluation by one method by default: no second evaluation, no refinement driven by an error
  estimate, no fallback to another representation and no recomputation of the tails. `"auto"` takes at each point the
  fastest representation that one evaluation gives to well below 1e-5 (the Notes of `ltratio_cdf`). The new argument
  `verified=True` of `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`, `coef_cdf` and `coef_icdf` gives the checked
  computation of before (error estimates, the agreement of two evaluations, the fallbacks and the recomputation of the
  tails), and reproduces the values of before. Where the parameters lie in a regime in which one evaluation is known to
  be brittle (with the conditional integral: a zero start with `c0 > 0` below the unit root, case DK with `c0 != 0` for
  `c > -5`, case SK for `c > -0.5`, `c < -100` with a zero start and `c < -50` with a stationary one, a fixed start below
  `c = -20` in case DK and `c = -30` in cases DU, DT; with the contour integral: `c0 > 0` for `c < -15`, `c0 < 0` for
  `c > 1`, and in case DU `c0 > 3` for `c > 1.5`) a `RuntimeWarning` names the regime, once per call. Away from these
  regimes the default values agree with the verified ones to well below 1e-5 (to 5e-9 at the reference points of the
  contour integral).
- A third representation, the contour integral (`method="contour"`), for a fixed start `g0 != 0` in cases DU and DT:
  a double integral from the inversion formula of the paper on paths around the singularities of the moment
  generating function, with the integral over v in closed form, the paths in s around the roots of the determinant and a hyperbola in t, an explicit integrand where the line
  integral needs the noncentral kernel, itself an integral. `"auto"` takes it there (one evaluation takes 5 to 12 ms a value on ten threads, against 8 to 89 ms for the line integral with its noncentral kernel and 0.24 to 0.28 s for the conditional integral), except for `c > 1` with `c0 < 0` and, in case DU,
  for `c > 1.5` with `c0 > 3`, where it fails and the conditional integral serves. Where the line integral was
  accurate the values change by at most 4e-13 (3.5e-10 at the corner `(c g0)^2/2 = 7200` of case DT, where the line
  integral is the less accurate); with `c0 > 0` at `c` below about -60 the line integral, which `"auto"` took before,
  is wrong (`ltratio_cdf` gave values up to 1e83 there), and the values are now right. `method="contour"` elsewhere
  raises a `ValueError`.
- The moments of case DK with a fixed start (`ltratio_mom`) were NaN below `c` of about -280 (a factor of the moment
  generating function overflowed); it is now computed in a form that stays finite.
- The C engine re-synced from DF-Limit (November 2026). The power envelope of the trend case,
  `power_envelope(c, "DG")`, uses the weight `a = c^2/(2 (c - 1))` of the point-optimal invariant test of Elliott,
  Rothenberg and Stock (it used `c/2`, the weight of case DK; the curve rises by at most 3e-5). With a stationary
  start the functions raise a `ValueError` for `c > -1e-4`, where the variance `-1/(2c)` of the start is too large
  for the quadrature. In case SK the conditional integral no longer cuts off the mass of `V` near zero as `c` tends
  to zero: the coefficient statistic of case SK was off by up to 1e-3 for `c` between -0.01 and 0
  (`coef_cdf(-8.039, "SK", -0.001)` gave 6e-6 for 0.0010438).
