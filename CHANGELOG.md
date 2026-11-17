# Changelog

## Unreleased (after 0.2.0)

- The C engine re-synced from DF-Limit (November 2026). The power envelope of the trend case,
  `power_envelope(c, "DG")`, uses the weight `a = c^2/(2 (c - 1))` of the point-optimal invariant test of Elliott,
  Rothenberg and Stock (it used `c/2`, the weight of case DK; the curve rises by at most 3e-5). With a stationary
  start the functions raise a `ValueError` for `c > -1e-4`, where the variance `-1/(2c)` of the start is too large
  for the quadrature. In case SK the conditional integral no longer cuts off the mass of `V` near zero as `c` tends
  to zero: the coefficient statistic of case SK was off by up to 1e-3 for `c` between -0.01 and 0
  (`coef_cdf(-8.039, "SK", -0.001)` gave 6e-6 for 0.0010438).
