# dflimit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24404482.svg)](https://doi.org/10.5281/zenodo.24404482)

The exact limiting distribution of the Dickey-Fuller t-ratio in Python.

`dflimit` computes the distribution function, density, quantiles and moments of the limits of the Dickey-Fuller
t-ratios under local alternatives, phi = 1 + c/n, by the exact representations of two papers. The lead paper,

> Bao, Y., Broda, S. A. and Kan, R., "Exact Limiting Distributions of Unit Root t-Ratios", working paper, 2026,
> available at <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7745921>,

gives the inversion formula on paths around the singularities of the joint moment generating function and with it the
limit for any piecewise smooth deterministic terms (the general engine); its companion,

> Bao, Y., Broda, S. A. and Kan, R., "Low-Rank Representations of the Classical Dickey-Fuller Limits, with an Extension
> to Predictive Regressions", working paper, 2026, available at <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7725619>,

gives the representations of the classical cases, in which the numerator has rank one or two (the line integral and the
conditional integral), and the probabilities of regions of the plane of the numerator and the denominator.

The package covers the regression without deterministic terms, with an intercept and with an intercept and a trend; a
zero, a fixed or a stationary initial observation; the t-ratio centred at the unit root or at a local null value c0;
and the GLS-detrended trend case of Elliott, Rothenberg and Stock (1996). It also has the ADFmax test of
Leybourne (1995), the probabilities of regions {U <= g(V)} of the limit experiment (point-optimal and
weighted-average-power tests), the distribution of the linear form U - aV and the asymptotic power envelope of
Elliott, Rothenberg and Stock, and, by the general engine, the same for any piecewise smooth deterministic terms
(polynomial trends, breaks in level and slope) with their companions (see "The general engine" below). The numbers
come from the multithreaded C engine (a port of the Julia
reference implementation, checked against it to 1e-10 or better); the same functions, with the same names and
arguments, exist in R, MATLAB and Julia.

| function | computes |
|---|---|
| `ltratio_cdf(x, cas, c=0, c0=0, g0=0, method="auto")` | distribution function P[t <= x] of the limit |
| `ltratio_pdf(x, cas, c=0, c0=0, g0=0, method="auto")` | its density |
| `ltratio_icdf(p, cas, c=0, c0=0, g0=0, method="auto", tol=1e-10)` | quantiles (asymptotic critical values) |
| `ltratio_mom(p, cas, c=0, c0=0, g0=0)` | moments E[t^j], j = 1..p |
| `coef_cdf(y, cas, c, c0, g0, method)`, `coef_icdf(p, ...)`, `coef_pdf(cas, y, ...)`, `coef_mom(p, cas, ...)` | the coefficient statistic n(phi^ - phi0) |
| `adfmax_cdf`, `adfmax_pdf`, `adfmax_icdf` | the ADFmax statistic (cases DU, DT, SU, ST) |
| `ltratio_region(bnd, "DK", c)` | P{U <= g(V)} for a line, the parabola of the t-ratio, the WAP boundary |
| `ltratio_lincdf(b, a, cas, c=0, cbar=-13.5)` | P(U - aV <= b), cases DK and DG (a = nan: P(V <= b)) |
| `power_envelope(c, cas="DK", alpha=0.05)` | the asymptotic power envelope of Elliott, Rothenberg and Stock (cases DK, DU, DG) |
| `ltratio_threads(n)` | the number of worker threads of the engine |
| `trend`, `perron`, `fdesign`, `ltrend`, `design_from_matrix`, `detrend_first`; `dist_cdf`, ..., `pt_*`, `lm_*`, `mza_*`, `msb_*` | the general engine: any deterministic terms, a design in place of `cas`, the keyword `engine` (see "The general engine") |

The case `cas` is a two-letter code: `D` (fixed start, `g0`, default 0) or `S` (stationary start, c <= -1e-4),
followed by `K` (no deterministic terms, tau), `U` (intercept, tau_mu) or `T` (intercept and trend, tau_tau);
`"DG"` is the GLS-detrended trend case (DF-GLS, cbar = -13.5). `method` is `"auto"` (at each point the fastest
representation that one evaluation gives to well below 1e-5), `"single"` (the line integral representation of the
companion paper), `"double"` (its conditional integral representation) or `"contour"` (the contour integral
representation, the inversion formula on paths of the lead paper, a fixed start in cases DU, DT only). Every
value is one evaluation by one method, with a warning where the parameters lie in a regime in which one evaluation is
known to be brittle; `verified=True` gives the checked computation (error estimates and fallbacks). The
boundary `bnd` of `ltratio_region` is a dict: `{"type": "line", "a": a, "b": b}`, `{"type": "parabola", "x": x,
"c0": c0}` or `{"type": "wap", "k": k, "omega": omega}`, with the optional keys `"v0"` and `"brks"`.

A scalar argument gives a float and an array (or a list) a NumPy array of the same shape; `info=True` gives a
`Result`, a dict whose keys are also attributes (`r.q`, `r["q"]`). The help of each function (`help(ltratio_cdf)`)
has the details of the R help pages. Arguments of the wrong type raise `TypeError`, arguments out of range
`ValueError` (outside the ranges in which the programs are checked, with the engine's message), and the warnings of
the R package are `RuntimeWarning`s.

All numbers come from the C engine. The half-plane statistics of the six classical cases (the coefficient statistic,
the linear form, the power envelope) and the moments of the t-ratio come by default from the devices of Bao and Kan
(see "The devices of Bao and Kan" below); `adfmax_icdf` and, with `method="engine"` or a representation of the t-ratio,
`coef_cdf` and `coef_icdf` are short Python-level wrappers, as in the MATLAB programs and the R package (the
distribution function of the t-ratio at 0 with a shifted local null value, and root searches on the engine's
distribution functions). Not in the package, because the C engine does not have it: the boundary type `"fun"` (any
function g) of `ltratio_region`.

## The general engine

Since 0.5.0 the package also has the general engine of the C engine: the same limits for any piecewise smooth
deterministic terms of the regression of y_t on d(t/n) and y_{t-1} (polynomial trends of any order, breaks in level and
slope at several dates, a design given by a function or by the regressor matrix of a sample), with a zero, fixed or
stationary start, a local null value c0, GLS detrending and the statistic that detrends first, by the inversion
formula on paths around the roots of the determinant of the joint moment generating function of numerator and
denominator. A design is an immutable `Design`, made by

| constructor | design |
|---|---|
| `trend(p)` | (1, r, ..., r^p); p = -1, 0, 1 are the cases DK, DU, DT |
| `perron(model, lam)` | Perron's models `"A"` (breaks in level), `"B"` (in slope), `"C"` (both) at the fractions `lam` |
| `fdesign(f, breaks)` | the regressors `f(r)`, smooth between the breaks (Chebyshev interpolants) |
| `ltrend(p)` | the polynomial trend in the orthonormal Legendre basis (high orders) |
| `design_from_matrix(X, breaks=None)` | the limit of the n x K regressor matrix of a sample (the breaks detected if not given) |
| `detrend_first(D)` | the statistic that detrends first (Perron and Vogelsang), for D |
| `Design(br, coef)` | the break points and the coefficients of the polynomials on the pieces |

The user-level functions above take a design in place of the case string (`ltratio_cdf(x, perron("C", 0.3), c)`),
with the keywords `start` (`"zero"`, `"fixed"` with `g0`, `"stationary"`) and `gls` (GLS detrending at cbar), and all
of them take the keyword `engine`:

- `engine="auto"` (the default, for a case string as for a design) is the router's rule: where the classical engine accepts the arguments
  it takes, at each point, the faster engine where the two agree to 1e-10 and the more accurate one where they do not
  (for example the general engine for the t-ratio with a stationary start and c0 != 0, with a zero start and c0 > 0
  for -100 < c <= 1; the classical engine for the regions and for the ADFmax test with a zero start), and the general
  engine for every other design; a `method` other than `"auto"`, or `verified=True`, keeps the classical engine for a
  classical case; where it takes the general engine for a case string the two engines agree to about 1e-9, so that the
  values can move in the tenth digit against the versions before 0.5.0;
- `engine="classical"` and `engine="general"` force one engine (`"classical"` for the cases DK, DU, DT, SK, SU, ST,
  DG only, with the values of the versions before 0.5.0; `"general"` with the general engine's methods `"auto"`,
  `"line"` and `"paths"`).

For the designs whose numerator has rank one or two (polynomial trends of any order, an intercept with sines and
cosines, `detrend_first` of a continuous design that does not vanish at the origin; a zero or a stationary start and
least squares detrending) the general engine also has the line integral with the explicit kernel: the classical line
integral with P(t) and the two roots supplied by the general engine's transform. `dist_cdf`, `dist_pdf` and
`dist_quantile` choose by `method`: `"auto"` (the default) takes it at c0 = 0 for c >= -20, except just above the unit
root with deterministic terms, 0 < c <= 0.5, where the paths are 1.5 to 2.3 times faster, and with c0 > 0 at the points
x < 0 where the outer contour of the paths is the vertical line (there the paths do not resolve the two roots of the
parabola of such a point, and their one evaluation was off by up to 1e-3 in the lower tail); below the unit root it is 1.0 to 1.6 times faster than
the paths per value for -20 <= c <= 0 (`c/test/line_timing.c` of dflimit-matlab, November 27, 2026, and interleaved
timings, November 26), and the paths otherwise; `"line"`
forces it (`ValueError` with the reason where it does not cover the design) and `"paths"` the paths. `ltratio_cdf`,
`ltratio_pdf` and `ltratio_icdf` with a design pass `method="line"` or `"paths"` on (for a design that is a classical
case, `"line"` is the classical line integral). The classical engine's line integral takes its inner integrals on
fixed node layouts at c <= 0, c0 = 0 (15 to 20 times faster than the adaptive rules in case DK on one thread, 8 to 13
on four, 1.9 to 3.2 times in the other cases, 1.7 to 2.9 on four; November 27, 2026).

The general engine's own functions of a design have the names of the Julia package: `dist_cdf`, `dist_pdf`,
`dist_quantile`, `dist_moments` (the t-ratio), `coef_pdf`, `coef_quantile`, `coef_cdf_gp` (the coefficient statistic;
its distribution function is `coef_cdf` with the design), `local_power`, `lin_cdf`, `lin_pdf`, `lin_quantile`,
`v_cdf`, `v_pdf` (the linear form U - aV and the law of V), `adfmax_quantile`, `region_prob` (lines and parabolas),
and the companions `pt_cdf`, `pt_pdf`, `pt_quantile`, `pt_power`, `pt_envelope` (the point-optimal statistic of
Elliott, Rothenberg and Stock and the power envelope of any design), `lm_cdf`, `lm_quantile`, `lm_power` (the LM
statistic of Schmidt and Phillips; for trend functions whose span contains the constant and a linear trend, a
`ValueError` otherwise), `mza_cdf`, `mza_quantile` (MZ_alpha) and `msb_cdf`, `msb_quantile` (MSB), with
the options `c0`, `start`, `gamma0` (the fixed start), `gls`, `method` (the t-ratio) and, for the distribution
functions and densities, `verified` (the checked computation: two evaluations on different paths, a third where they differ by more than
1e-9) and `rule` (the rule of one evaluation of the t-ratio's paths: `"auto"`, the default, the thinner rules of
DFLimit.jl by regions of the start, the deterministic terms, `c` and `c0`, with the outer step 1.5 times as long and 16
and 32 nodes on the paths around the roots where they hold; `"present"` the rule before November 2026). Every value is
one evaluation by default. The engine's warnings (where one evaluation may be inaccurate) are
`RuntimeWarning`s.

```python
>>> from dflimit import *
>>> D = perron("C", [0.3, 0.7])                 # breaks in level and slope at 0.3 and 0.7
>>> ltratio_icdf([0.01, 0.05, 0.10], D)         # its asymptotic critical values
array([-5.28735943, -4.75056961, -4.46887591])
>>> ltratio_cdf(-4.75057, D, -10)               # the power of the 5% test at c = -10
0.09235786644709337
>>> ltratio_cdf(-2.86157, "DU", -5, engine="auto")    # a case string by the router's rule
0.11651356209338493
>>> pt_quantile(trend(1), 0.05, -13.5)          # the 5% critical value of the point-optimal test with a trend
5.704481449372031
>>> mza_quantile(trend(1), 0.05)                # MZ_alpha with a trend
-16.591095285246947

```

Far above the unit root, for c > 5.5, the general engine's inversion on paths by one evaluation may be off by more than
1e-5 in the upper tail for a design beyond the classical cases and for a fixed start with deterministic terms (Perron's
model C at c = 7 by 1.5e-5 in F, a linear trend with a fixed start at c = 7 by 8.7e-5), and it warns there
(`verified=True` checks it).

The general engine also gives ADFmax with a fixed start (through the router: `adfmax_cdf(x, "DU", c, g0=1.5,
engine="auto")`, or a design), and the regions and the linear form of case DK with a fixed start; the classical engine
has the zero and the stationary start only there, and refuses a fixed start. It refuses, with a `ValueError` that says
why: probabilities and levels outside (0, 1); every function of the general engine above c = 16 with deterministic terms
(10 with them and a fixed start) and above c = 20 without them (the range in which it is checked above the unit root;
there the roots of its transform are no longer resolved in double precision, and `engine="auto"` takes the classical
engine for the classical cases); the LM statistic above c = 10; the point-optimal statistic and `coef_cdf_gp` on their
fixed grid where the grid does not resolve the transform (far above the unit root; the message names the grid size `N`
that serves), the point-optimal statistic also at |q| beyond 0.15 N/asinh(umax) (1000 with the default grid); and a
design whose roots run off on the vertical line, where the paths around them would need more than a million nodes (such
as (1 + r, 1{r < 0.4}), which has no proper intercept); the classical cases far above the unit root at large |x| (x > 300,
in case DK with `c0 > c` at x < -20, and where the line integral exceeds its bounds) above c = 14, where the conditional
integral that computes them up to c = 14 (within 1.3e-8 of an independent computation in case DK, 5e-7 of the large-c law
in cases DU, DT) is off by up to 7.7e-5 in case DK and gives 1e44 in case DT; and any value of a
distribution function outside [0, 1] by more than 1e-8, which means a computation that failed. The law of V is 0 at
b <= 0.

Points of one call share the paths of the inversion, so pass a whole vector in one call. The engine caches up to 64
prepared designs, paths and refined lines; `empty_caches()` frees them.

## The extensions of the limit

Three extensions of the limit, ported from DFLimit.jl to the C engine, with the same names and arguments as in Julia, R
and MATLAB:

- **Volatility and persistence regimes**: `volpath(breaks, omega)` and the keyword `vol` of `dist_cdf`, `dist_pdf`,
  `dist_quantile`, `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`: the limit of the t-ratio when the process is
  dJ = c(r) J dr + omega(r) dW with the standard deviation omega and, with `c` a sequence of one value per regime, the
  local-to-unity parameter piecewise constant between known dates (every `c` in [-480, 0]; the inversion on paths of
  the volatility transform). The detrending and the start are the engine's: least squares or GLS detrending
  (`gls=cbar`), the one-step regression or the statistic that detrends first (`detrend_first(D)`), and a zero start
  or, with least squares in the one-step regression, the stationary start of the first regime (`start="stationary"`:
  J(0) ~ N(0, omega_1^2/(-2 c_1)), which needs c < 0 on the first regime); the case strings `"SK"`, `"SU"`, `"ST"` and
  `"DG"` take `vol` with their start and detrending. A fixed start (`gamma0 != 0`, or `g0` with a case string) is
  refused with more than one regime: the engine's paths of a fixed start are laid for a constant variance and
  persistence, and under regimes they fail or lose accuracy for starts of moderate size.
- **A normal component**: the keyword `rho` of the same functions, the limit rho tau_c0 + sqrt(1 - rho^2) Z of the
  covariate-augmented t-ratio of Hansen (1995) at `c0=cadf_c0(c, rho, R)` and of the t-ratio of the slope in a
  predictive regression with a regressor local to unity at `c0=predictive_c0(c, rho, b)` (`method` `"auto"`, `"lines"`,
  the inversion on two lines with the normal component in the kernel, or `"mixture"`, the normal average of the t-ratio's
  distribution function).
- **The first seasonal t-ratio** of Hylleberg, Engle, Granger and Yoo at a harmonic frequency (the t-ratio of pi_3 for
  quarterly data), for any design: `hegy_t3_cdf`, `hegy_t3_pdf`, `hegy_t3_quantile`, by the sum of the residues at the
  roots of the engine's determinant on its outer contour, with `c0` and `vol`.

```python
from dflimit import volpath, dist_cdf, dist_quantile, ltratio_cdf, ltratio_icdf, predictive_c0, hegy_t3_quantile, trend
V = volpath([0.3], [1, 3])                           # the standard deviation triples at 30% of the sample
dist_cdf(trend(0), -2.86, 0, vol=V)
dist_quantile(trend(1), 0.05, [-3, 0, -10], vol=volpath([0.2, 0.7], [2, 1, 0.5]))
dist_quantile(trend(1), 0.05, 0, vol=V, gls=-13.5)   # DF-GLS (cbar = -13.5) under the variance break
ltratio_icdf(0.05, "SU", -5, vol=V)                  # the stationary start of the first regime
ltratio_cdf(1.645, "DU", 0, predictive_c0(0, -0.95), rho=-0.95)   # the predictive regression
hegy_t3_quantile(trend(0), 0.05)                     # the 5% critical value with seasonal dummies
```

## The devices of Bao and Kan

The half-plane statistics of the six classical cases and the moments of the t-ratio are computed by default with the
numerical devices of Yong Bao and Raymond Kan, "Exact Finite-Sample Distribution and Moments of the Least Squares
Estimator in First-Order Autoregressions: O(1) Algorithms", as in the Julia package (its module `DFKan`), the MATLAB
programs and the R package (the C engine's `dfl_kan.c` and `dfl_det_kan.c`). The joint characteristic function of the
limit pair comes in closed forms that are free of cancellation on both sides of the unit root, with the branch of the
square root fixed at each point, and in Taylor-mode arithmetic (jets), so that its derivatives come out of the same
assembly. A statistic that rejects on a half-plane is one inversion along a ray, by adaptive quadrature over decades with
the tail completed by repeated integration by parts; the moments of the t-ratio are one-dimensional integrals of the
j-th derivative of the moment generating function, which one jet of order j gives at each node.

| function | the default | warm time in C against `method="engine"` |
|---|---|---|
| `coef_cdf`, `coef_icdf`, `coef_pdf`, `coef_mom` | the cases DK, DU, DT (zero or fixed start), SK, SU, ST, c <= 20 | 0.005 to 0.18 (quantiles 0.02; the density 0.03 against the general engine's) |
| `ltratio_lincdf` | case DK, a != 0, -100 <= c <= 4; the six cases through their designs, -100 <= c <= 2 | 0.07 to 0.24 |
| `power_envelope` | case DK (DU), c <= -1e-4 | 0.13 to 0.16 |
| `ltratio_mom` | the six cases, any c0 and fixed start, c <= 20, up to 31 moments | 0.013 to 0.23 |
| `pt_cdf`, `pt_pdf`, `pt_quantile`, `pt_power`, `pt_envelope` | any design, c <= 4 | 0.08 to 0.15 (quantiles 0.3 to 0.4, envelopes 0.2) |
| `coef_cdf_gp` | zero or stationary start, c <= 5 | 0.05 to 0.1 |

`method="inversion"` (`"jets"` for the moments) forces them, `method="engine"` is the computation of before, which
reproduces those values (`method="grid"` the fixed grid of the point-optimal functions and of `coef_cdf_gp`, also taken
where `N` or `umax` is given); `tol` is the tolerance (1e-12; 1e-10 relative for the moments), `verified=True`
recomputes by the engines the values whose error estimate exceeds it, `err=True` returns the tuple of the values and
their estimates (`info=True` for the moments), and a `RuntimeWarning` names the values whose estimate exceeds the
tolerance. The regions, the tolerances and the warnings are those of the Julia package, whose README lists the
comparisons behind them and the cases where the engines' one evaluation was wrong (the coefficient statistic of case DU
with a fixed start far from the unit root, the linear form and the moments with a fixed start above the unit root). The
values agree with DFLimit.jl to 6e-15 in the single inversions and the moments (`tests/test_kan.py`). The same devices
fix the branches inside the engines: P(t), M_0(t) and the labels of the roots of the classical line and contour
integrals and the argument of log P(t) in the general engine's line integral are determined at each point instead of
being continued along the grids, to 4e-13 the values of before, at the same cost.

## Example

```python
>>> from dflimit import *
>>> ltratio_icdf([0.01, 0.05, 0.10], "DU")     # asymptotic critical values of tau_mu
array([-3.430353  , -2.86156886, -2.56679107])
>>> ltratio_cdf(-2.86157, "DU", -5)             # limiting power of the 5% test at c = -5
0.11651356209338493
>>> ltratio_cdf(-2.86157, "SU", -5)             # the same with a stationary start
0.12496412271269253
>>> ltratio_cdf(0, "DK", 2, c0=2)               # t-ratio centred at the true value
0.6504355619776853
>>> ltratio_cdf(-2, "DU", -5, g0=1.5)           # fixed start
0.8649772400770597
>>> ltratio_pdf(-2, "DT", -5)                   # density
0.5088008223323347
>>> ltratio_mom(4, "DT")                        # first four moments of tau_tau
array([ -2.18135582,   5.32067592, -14.0358001 ,  39.5544305 ])
>>> coef_cdf(-14.0936, "DU", -10)               # power of the 5% coefficient test at c = -10
0.4592525632076304
>>> ltratio_region({"type": "wap", "k": 3.6658, "omega": 10})    # size of the WAP test of the companion paper
0.04999962283465172
>>> ltratio_lincdf(-0.2639, -3.5, "DK", -7)     # power of the point-optimal test against cbar = -7 at c = -7
0.49938157210469375
>>> power_envelope([-5, -10], "DG")             # the power envelope of the trend case
array([0.11484596, 0.31005163])

```

(The last digits can differ by rounding between machines and numbers of threads.)

The engine uses all processors by default; `ltratio_threads(n)` changes that, and so does the environment variable
`DFL_NTHREADS` (set before the first computation); `OMP_THREAD_LIMIT`, if set when the package is imported, caps the
default. The engine releases the GIL while it computes; calls from several Python threads run one after another,
each on all the worker threads.

## Installation

`dflimit` is an ordinary Python package with a C extension (the engine is C11 and the C++ of the Faddeeva package);
it needs Python 3.9 or later and NumPy.

```sh
pip install dflimit
```

installs it from PyPI, which carries wheels for Linux (x86_64 and aarch64, manylinux), macOS (Apple silicon and Intel) and
Windows (x86_64); there is one wheel per platform, which serves every CPython from 3.9 on (the tag `cp39-abi3`), and no
compiler is needed. The same files are on the Releases page of the GitHub repository `s-broda/dflimit-py`.

**From source** (on other platforms, or for the development version; needs a C and C++ compiler: the Xcode command line
tools on macOS (`xcode-select --install`), the compilers of the distribution on Linux, e.g. `build-essential`, the
Microsoft C++ Build Tools on Windows):

```sh
pip install git+https://github.com/s-broda/dflimit-py
```

The tests (`pip install pytest`, then `pytest tests` in a copy of the repository) compare the installed package with
the stored reference values of the Julia engine, as the tests of the R package do.

## Sibling implementations

The same functions, with the same names and arguments: the R package
[`dflimit`](https://github.com/s-broda/dflimit) (the same C engine, the same tests), the MATLAB package dflimit-matlab
(the MATLAB programs, and the C engine with a MEX gateway) and the Julia package
[`DFLimit.jl`](https://github.com/s-broda/DFLimit.jl), the Julia reference implementation.

## Licence and citation

MIT (see `LICENSE`); the Faddeeva package in `csrc/Faddeeva.cc` is by Steven G. Johnson, MIT licence (see
`COPYRIGHTS`). Please cite the lead paper when using the package, and the companion paper as well where you use the
representations of the classical cases (`CITATION.cff` has both).

The C engine in `csrc/` is copied from dflimit-matlab, the repository of the MATLAB package, which holds the C engine
(its directory `c/`), by `tools/sync-engine.sh` (the upstream commit is in `src/dflimit/UPSTREAM`, which is installed
with the package); do not edit those files here. The engine was developed in the companion paper's repository up to
commit b484d4f.
