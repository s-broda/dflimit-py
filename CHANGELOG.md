# Changelog

## 0.5.0

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
