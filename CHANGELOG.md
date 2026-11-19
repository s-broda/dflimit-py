# Changelog

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
  the double integral of Theorem 1 of the paper with the integral over v in closed form, the paths in s around the
  roots of the determinant and a hyperbola in t, an explicit integrand where the line integral needs the noncentral
  kernel, itself an integral. `"auto"` takes it there (one evaluation takes 5 to 12 ms a value on ten threads, against 8 to 89 ms for the line integral with its noncentral kernel and 0.24 to 0.28 s for the conditional integral), except for `c > 1` with `c0 < 0` and, in case DU,
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
