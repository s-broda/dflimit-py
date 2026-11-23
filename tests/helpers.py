"""Helpers of the tests (tests/testthat/helper-ref.R of the R package dflimit).

The reference files in ref/ are copied unchanged from tests/testthat/ref/ of the R package dflimit: subsets of the
Julia reference values in c/test/*_ref.txt of dflimit-matlab, the home of the C engine (see the first line of each file).
"""
import math
import os

import numpy as np

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")


def read_ref(name):
    """The rows of a reference file (numbers only), as a 2-d array."""
    return np.loadtxt(os.path.join(REF, name), comments="#", ndmin=2)


def read_app_ref():
    """The rows of app_ref.txt: (tag, numbers)."""
    rows = []
    with open(os.path.join(REF, "app_ref.txt")) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            rows.append((v[0], [float(t) for t in v[1:]]))
    return rows


def case_code(m, st):
    m = int(m)
    return "DG" if m == 4 else ("S" if st else "D") + "KUT"[m - 1]


def check_Ff(F, f, Fr, fr, tolF, relf, absf, label):
    """The criterion of the C engine's own tests: F to tolF absolute; f to relf relative or absf absolute (inf
    matching inf)."""
    assert abs(F - Fr) < tolF, f"{label}: F = {F:.15g}, reference {Fr:.15g}"
    okf = (abs(f - fr) / max(abs(fr), 1e-300) < relf or abs(f - fr) < absf
           or (math.isinf(f) and math.isinf(fr)))
    assert okf, f"{label}: f = {f:.15g}, reference {fr:.15g}"


def expect_equal(obj, expected, tolerance):
    """testthat's expect_equal(obj, expected, tolerance = ...): the mean absolute difference, relative to the mean
    absolute value of expected when that is above the tolerance, below the tolerance."""
    x = np.asarray(obj, dtype=float).ravel()
    y = np.asarray(expected, dtype=float).ravel()
    assert x.shape == y.shape, f"lengths differ: {x.size} and {y.size}"
    diff = np.mean(np.abs(x - y))
    avg = np.mean(np.abs(y))
    if math.isfinite(avg) and avg > tolerance:
        diff /= avg
    assert diff < tolerance, f"{obj!r} differs from {expected!r} by {diff:.3g} (tolerance {tolerance:g})"


# ---------------------------------------------------------------------------------------------------------------------
# The general engine: the lines of ref/det_ref.txt (a subset of c/test/det_ref.txt of dflimit-matlab, the values of the
# Julia general engine of DFLimit.jl, copied by tools/det-ref.py): 'design id spec...' defines a design, 'val' a value
# of a function of the general engine, 'route' one of the router (the format is in the header of the file).
# ---------------------------------------------------------------------------------------------------------------------
STARTS = ("zero", "fixed", "stationary")
ENGINES = ("auto", "classical", "general")


def _f(s):
    return float(s)          # (Python reads NaN, Inf, -Inf as Julia prints them)


def det_designs(path=None):
    """The designs of ref/det_ref.txt (or of the file path) by their ids (built by the package's constructors)."""
    import dflimit
    funs = {"sincos": (lambda r: [math.sin(2 * math.pi * r), math.cos(2 * math.pi * r)], ()),
            "perronC04": (lambda r: [1.0, r, 1.0 if r > 0.4 else 0.0, r - 0.4 if r > 0.4 else 0.0], (0.4,)),
            "expr": (lambda r: [math.exp(r)], ()),
            "trig": (lambda r: [1.0, math.sin(2 * math.pi * r), math.cos(2 * math.pi * r)], ()),
            "rlin": (lambda r: [r], ())}
    D = {}
    with open(path or os.path.join(REF, "det_ref.txt")) as fh:
        for line in fh:
            v = line.split()
            if not v or v[0] != "design":
                continue
            i, kind, a = int(v[1]), v[2], v[3:]
            if kind == "trend":
                D[i] = dflimit.trend(int(a[0]))
            elif kind == "ltrend":
                D[i] = dflimit.ltrend(int(a[0]))
            elif kind == "perron":
                D[i] = dflimit.perron(a[0], [_f(t) for t in a[2:2 + int(a[1])]])
            elif kind == "fdesign":
                f, br = funs[a[0]]
                D[i] = dflimit.fdesign(f, br, name=a[0])
            elif kind == "matrix":
                D[i] = dflimit.design_from_matrix(D[int(a[0])].regmat(int(a[1])))
            elif kind == "first":
                D[i] = dflimit.detrend_first(D[int(a[0])])
            elif kind == "raw":
                npc, deg, K, loc = (int(t) for t in a[:4])
                nums = [_f(t) for t in a[4:]]
                br, C = nums[:npc + 1], np.array(nums[npc + 1:]).reshape(npc, K, deg + 1).transpose(0, 2, 1)
                D[i] = dflimit.Design(br, C, loc=bool(loc))
    return D


def det_lines(tag, path=None):
    """The lines of ref/det_ref.txt (or of the file path) with the tag 'val', 'route' (values), 'err' or 'rerr' (calls
    that must fail with an error whose message contains the text after ' | '), parsed."""
    out = []
    with open(path or os.path.join(REF, "det_ref.txt")) as fh:
        for line in fh:
            text = None
            if " | " in line:
                line, text = line.split(" | ", 1)
                text = text.strip()
            v = line.split()
            if not v or v[0] != tag:
                continue
            r = {"fn": v[1], "brittle": v[-1] == "brittle", "text": text}
            if r["brittle"]:
                v = v[:-1]
            if tag in ("val", "err"):
                r.update(id=int(v[2]), cb=_f(v[3]), st=int(v[4]), g0=_f(v[5]), c=_f(v[6]), c0=_f(v[7]), ver=bool(int(v[8])),
                         par=_f(v[9]))
                k = 10
            else:
                r.update(e=int(v[2]), spec=v[3], a=int(v[4]), st=int(v[5]), g0=_f(v[6]), gl=_f(v[7]), c=_f(v[8]), c0=_f(v[9]),
                         par=_f(v[10]))
                k = 11
            n = int(v[k])
            r["x"] = np.array([_f(t) for t in v[k + 1:k + 1 + n]])
            k += 1 + n
            if text is None:
                nw = int(v[k])
                r["want"] = np.array([_f(t) for t in v[k + 1:k + 1 + nw]])
                r["tol"] = _f(v[k + 1 + nw])
            out.append(r)
    return out


def det_label(r):
    if r.get("text"):
        return det_label({**r, "text": None}) + f" (error: {r['text']})"
    if "id" in r:
        s = f"{r['fn']} design {r['id']} c={r['c']:g}"
        if not math.isnan(r["cb"]):
            s += f" gls={r['cb']:g}"
        if r["st"]:
            s += f" {STARTS[r['st']]}" + (f" g0={r['g0']:g}" if r["st"] == 1 else "")
        if r["c0"]:
            s += f" c0={r['c0']:g}"
        return s + (" verified" if r["ver"] else "")
    s = f"{r['fn']} {ENGINES[r['e']]} {r['spec']} {r['a']} c={r['c']:g}"
    if r["st"]:
        s += f" {STARTS[r['st']]}"
    if r["c0"]:
        s += f" c0={r['c0']:g}"
    return s


def det_value(r, D):
    """The package's values for a 'val' line (the general engine's functions of a design)."""
    import dflimit as d
    G = D[r["id"]]
    fn, x, c, c0, par = r["fn"], r["x"], r["c"], r["c0"], r["par"]
    kw = dict(start=STARTS[r["st"]], gamma0=r["g0"] if r["st"] == 1 else 0.0, gls=None if math.isnan(r["cb"]) else r["cb"])
    fn, _, me = fn.partition("@")                     # ('cdf@line', 'pdf@paths', 'quant@auto': the method of the t-ratio)
    me = me or "auto"
    if fn == "cdf":
        return d.dist_cdf(G, x, c, c0=c0, verified=r["ver"], method=me, **kw)
    if fn == "pdf":
        return d.dist_pdf(G, x, c, c0=c0, verified=r["ver"], method=me, **kw)
    if fn == "quant":
        return d.dist_quantile(G, x, c, x0=par, tol=1e-9, c0=c0, method=me, **kw)
    if fn == "coefcdf":
        return d.coef_cdf(x, G, c, c0=c0, g0=kw["gamma0"], start=kw["start"], gls=kw["gls"], engine="general",
                          verified=r["ver"])
    if fn == "coefpdf":
        return d.coef_pdf(G, x, c, c0=c0, verified=r["ver"], **kw)
    if fn == "coefq":
        return d.coef_quantile(G, x, c, y0=par, tol=1e-9, c0=c0, **kw)
    if fn == "coefgp":
        return d.coef_cdf_gp(G, x, c, c0=c0, **kw)
    if fn == "mom":
        return d.dist_moments(G, c, p=x.size, c0=c0, **kw)
    if fn == "lpow":
        return d.local_power(G, x, level=0.05, statistic="tratio" if par == 0 else "coef", **kw)
    if fn == "lincdf":
        return d.lin_cdf(G, x, par, c, verified=r["ver"], **kw)
    if fn == "linpdf":
        return d.lin_pdf(G, x, par, c, verified=r["ver"], **kw)
    if fn == "linq":
        return d.lin_quantile(G, x[0], par, c, b0=0.0, tol=1e-10, **kw)
    if fn in ("adfcdf", "adfpdf"):
        f = d.adfmax_cdf if fn == "adfcdf" else d.adfmax_pdf
        return f(x, G, c, start=kw["start"], engine="general")
    if fn == "adfq":
        return d.adfmax_quantile(G, x[0], c, x0=par, tol=1e-10, start=kw["start"])
    if fn == "env":                                  # (the general engine's envelope, as the C test calls it)
        from dflimit import _general
        return _general._values(_general.DET_ENV, G, x, 0.0, _general._NOOPTS, par, name="cs")
    if fn == "ptcdf":
        return d.pt_cdf(G, x, par, c)
    if fn == "ptcdfN":                               # (the field c0 carries the grid size N)
        return d.pt_cdf(G, x, par, c, N=int(c0))
    if fn == "ptcdfG":                               # (the fixed grid of before)
        return d.pt_cdf(G, x, par, c, method="grid")
    if fn == "ptpdf":
        return d.pt_pdf(G, x, par, c)
    if fn == "ptq":
        return d.pt_quantile(G, x[0], par)
    if fn == "ptpow":
        return np.array(d.pt_power(G, par, c, level=x[0]))
    if fn == "ptenv":
        return d.pt_envelope(G, x, level=par)
    if fn == "lmcdf":
        return d.lm_cdf(G, x, c)
    if fn == "lmq":
        return d.lm_quantile(G, x[0], c=c)
    if fn == "lmpow":
        q, power = d.lm_power(G, x, level=par)
        return np.concatenate([[q], power])
    if fn == "mzacdf":
        return d.mza_cdf(G, x, c, gls=par)
    if fn == "mzaq":
        return d.mza_quantile(G, x[0], c, gls=par)
    if fn == "msbcdf":
        return d.msb_cdf(G, x, c, gls=par)
    if fn == "msbq":
        return d.msb_quantile(G, x[0], c, gls=par)
    raise ValueError(f"unknown function {fn}")


def det_route(r, D):
    """The package's values for a 'route' line (the router of the user-level functions; engine auto, classical,
    general)."""
    import dflimit as d
    fn, x, c, c0, par, eng = r["fn"], r["x"], r["c"], r["c0"], r["par"], ENGINES[r["e"]]
    if r["spec"] == "case":
        cas = "DG" if r["a"] == 4 else ("S" if r["st"] == 2 else "D") + "KUT"[r["a"] - 1]
        kw = dict(g0=r["g0"]) if math.isnan(r["gl"]) or fn == "lin" else dict(g0=r["g0"], gls=r["gl"])
    else:
        cas = D[r["a"]]
        kw = dict(g0=r["g0"], start=STARTS[r["st"]], gls=None if math.isnan(r["gl"]) else r["gl"])
    if fn in ("cdf", "pdf"):
        f = d.ltratio_cdf if fn == "cdf" else d.ltratio_pdf
        return f(x, cas, c, c0=c0, engine=eng, **kw)
    if fn == "icdf":
        return d.ltratio_icdf(x, cas, c, c0=c0, engine=eng, **kw)
    if fn == "mom":
        return d.ltratio_mom(int(x[0]), cas, c, c0=c0, engine=eng, **kw)
    if fn == "coef":
        return d.coef_cdf(x, cas, c, c0=c0, engine=eng, **kw)
    if fn == "coefq":
        return d.coef_icdf(x, cas, c, c0=c0, engine=eng, **kw)
    if fn == "lin":
        return d.ltratio_lincdf(x, par, cas, c, cbar=-13.5 if math.isnan(r["gl"]) else r["gl"], engine=eng)
    if fn == "env":
        return d.power_envelope(x, cas, alpha=par, engine=eng)
    if fn in ("adf", "adfpdf"):
        f = d.adfmax_cdf if fn == "adf" else d.adfmax_pdf
        return f(x, cas, c, engine=eng, g0=r["g0"], **({} if r["spec"] == "case" else {"start": kw["start"]}))
    if fn == "region1":
        return d.ltratio_region({"type": "line", "a": par, "b": x[0]}, cas, c, engine=eng, **kw)
    if fn == "region2":
        return d.ltratio_region({"type": "parabola", "x": x[0]}, cas, c, engine=eng, **kw)
    raise ValueError(f"unknown function {fn}")


def det_check(got, r):
    """The largest deviation from the reference values of the line r and whether it is within the line's tolerance
    (relative to max(1, |value|) for the moments, as in the C engine's test)."""
    got = np.atleast_1d(np.asarray(got, dtype=float)).ravel()
    want = r["want"]
    assert got.shape == want.shape, f"{det_label(r)}: {got.size} values, reference {want.size}"
    with np.errstate(invalid="ignore"):
        d = np.abs(got - want)
    d[(np.isnan(got) & np.isnan(want)) | (np.isinf(want) & (got == want))] = 0
    tol = r["tol"] * np.maximum(1, np.abs(want)) if r["fn"] == "mom" else np.full(want.shape, r["tol"])
    return float(np.max(d)) if d.size else 0.0, bool(np.all(d <= tol))
