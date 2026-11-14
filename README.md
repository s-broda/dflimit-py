# dflimit

The exact limiting distribution of the Dickey-Fuller t-ratio in Python.

`dflimit` computes the distribution function, density, quantiles and moments of the limits of the Dickey-Fuller
t-ratios under local alternatives, phi = 1 + c/n, by the exact representations of

> Bao, Y., Broda, S. A. and Kan, R., "The Exact Limiting Distribution of the Dickey-Fuller t-Ratio and Some Related
> Statistics", working paper.

for the regression without deterministic terms, with an intercept and with an intercept and a trend; with a zero,
a fixed or a stationary initial observation; for the t-ratio centred at the unit root or at a local null value c0;
and for the GLS-detrended trend case of Elliott, Rothenberg and Stock (1996). It also has the ADFmax test of
Leybourne (1995) and the probabilities of regions {U <= g(V)} of the limit experiment (point-optimal and
weighted-average-power tests). The numbers come from the multithreaded C engine of the paper (a port of its Julia
reference implementation, checked against it to 1e-10 or better); the same functions, with the same names and
arguments, exist in R, MATLAB and Julia.

| function | computes |
|---|---|
| `ltratio_cdf(x, cas, c=0, c0=0, g0=0, method="auto")` | distribution function P[t <= x] of the limit |
| `ltratio_pdf(x, cas, c=0, c0=0, g0=0, method="auto")` | its density |
| `ltratio_icdf(p, cas, c=0, c0=0, g0=0, method="auto", tol=1e-10)` | quantiles (asymptotic critical values) |
| `ltratio_mom(p, cas, c=0, c0=0, g0=0)` | moments E[t^j], j = 1..p |
| `coef_cdf(y, cas, c, c0, g0, method)`, `coef_icdf(p, ...)` | the coefficient statistic n(phi^ - phi0) |
| `adfmax_cdf`, `adfmax_pdf`, `adfmax_icdf` | the ADFmax statistic (cases DU, DT, SU, ST) |
| `ltratio_region(bnd, "DK", c)` | P{U <= g(V)} for a line, the parabola of the t-ratio, the WAP boundary |
| `ltratio_threads(n)` | the number of worker threads of the engine |

The case `cas` is a two-letter code: `D` (fixed start, `g0`, default 0) or `S` (stationary start, c < 0),
followed by `K` (no deterministic terms, tau), `U` (intercept, tau_mu) or `T` (intercept and trend, tau_tau);
`"DG"` is the GLS-detrended trend case (DF-GLS, cbar = -13.5). `method` is `"auto"` (the faster representation at
each point), `"single"` (the line integral, Theorem 2) or `"double"` (the conditional integral, Theorem S1). The
boundary `bnd` of `ltratio_region` is a dict: `{"type": "line", "a": a, "b": b}`, `{"type": "parabola", "x": x,
"c0": c0}` or `{"type": "wap", "k": k, "omega": omega}`, with the optional keys `"v0"` and `"brks"`.

A scalar argument gives a float and an array (or a list) a NumPy array of the same shape; `info=True` gives a
`Result`, a dict whose keys are also attributes (`r.q`, `r["q"]`). The help of each function (`help(ltratio_cdf)`)
has the details of the R help pages. Arguments of the wrong type raise `TypeError`, arguments out of range
`ValueError` (outside the ranges in which the programs are checked, with the engine's message), and the warnings of
the R package are `RuntimeWarning`s.

All numbers come from the C engine. `coef_cdf`, `coef_icdf` and `adfmax_icdf` have no entry point of their own
there: as in the MATLAB programs and the R package, they are short Python-level wrappers (the distribution function
of the t-ratio at 0 with a shifted local null value, and root searches on the engine's distribution functions). Not
in the package, because the C engine does not have them: `ltratio_lincdf` (the linear form U - aV) and
`power_envelope` of the MATLAB programs, and the boundary type `"fun"` (any function g) of `ltratio_region`.

## Example

```python
>>> from dflimit import *
>>> ltratio_icdf([0.01, 0.05, 0.10], "DU")     # asymptotic critical values of tau_mu
array([-3.430353  , -2.86156886, -2.56679107])
>>> ltratio_cdf(-2.86157, "DU", -5)             # limiting power of the 5% test at c = -5
0.11651356209305042
>>> ltratio_cdf(-2.86157, "SU", -5)             # the same with a stationary start
0.12496412271251967
>>> ltratio_cdf(0, "DK", 2, c0=2)               # t-ratio centred at the true value
0.6504355619776852
>>> ltratio_cdf(-2, "DU", -5, g0=1.5)           # fixed start
0.8649772400771647
>>> ltratio_pdf(-2, "DT", -5)                   # density
0.5088008223324625
>>> ltratio_mom(4, "DT")                        # first four moments of tau_tau
array([ -2.18135582,   5.32067592, -14.0358001 ,  39.5544305 ])
>>> coef_cdf(-14.0936, "DU", -10)               # power of the 5% coefficient test at c = -10
0.45925256320730823
>>> ltratio_region({"type": "wap", "k": 3.6658, "omega": 10})    # size of the WAP test (Table V of the paper)
0.04999962283465172

```

(The last digits can differ by rounding between machines and numbers of threads.)

The engine uses all processors by default; `ltratio_threads(n)` changes that, and so does the environment variable
`DFL_NTHREADS` (set before the first computation); `OMP_THREAD_LIMIT`, if set when the package is imported, caps the
default. The engine releases the GIL while it computes; calls from several Python threads run one after another,
each on all the worker threads.

## Installation

`dflimit` is an ordinary Python package with a C extension (the engine is C11 and the C++ of the Faddeeva package);
it needs Python 3.9 or later and NumPy.

**From source** (needs a C and C++ compiler: the Xcode command line tools on macOS (`xcode-select --install`), the
compilers of the distribution on Linux, e.g. `build-essential`, the Microsoft C++ Build Tools on Windows):

```sh
pip install git+https://github.com/s-broda/dflimit-py
```

or, from a downloaded source distribution, `pip install dflimit-0.1.0.tar.gz`.

**From a wheel** (no compiler needed): the releases of the GitHub repository (the Releases page of
`s-broda/dflimit-py`) carry wheels for Linux (x86_64 and aarch64, manylinux), macOS (Apple silicon and Intel) and
Windows (x86_64). There is one wheel per platform, which serves every CPython from 3.9 on (the tag `cp39-abi3`).
Download the one for your platform and

```sh
pip install dflimit-0.1.0-cp39-abi3-macosx_11_0_arm64.whl      # macOS, Apple silicon (the name of the file you downloaded)
```

The package is not on PyPI (yet).

The tests (`pip install pytest`, then `pytest tests` in a copy of the repository) compare the installed package with
the stored reference values of the Julia engine, as the tests of the R package do.

## Sibling implementations

The same functions, with the same names and arguments: the R package
[`dflimit`](https://github.com/s-broda/dflimit) (the same C engine, the same tests), and the MATLAB programs and
the Julia reference implementation in the paper's repository.

## Licence and citation

MIT (see `LICENSE`); the Faddeeva package in `csrc/Faddeeva.cc` is by Steven G. Johnson, MIT licence (see
`COPYRIGHTS`). Please cite the paper when using the package.

The C engine in `csrc/` is copied from the paper's repository by `tools/sync-engine.sh` (the upstream commit is in
`src/dflimit/UPSTREAM`, which is installed with the package); do not edit those files here.
