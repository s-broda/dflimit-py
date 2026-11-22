#!/usr/bin/env python3
"""tools/det-ref.py: copy a subset of the reference values of the general engine into the package's tests.

    python tools/det-ref.py /path/to/dflimit-matlab

The script only reads the dflimit-matlab checkout.  Its file c/test/det_ref.txt (the values of the Julia general engine
of DFLimit.jl, written by c/test/det_ref.jl, and the test of the C engine, c/dfl_test_det.c) is copied line by line,
unchanged, into tests/ref/det_ref.txt: every design line, the user-level values ('val') chosen by the rules below,
every line of the router ('route'), and the calls that must fail ('err' of the general engine's functions, 'rerr' of
the router) except those of ADFmax with GLS detrending or a local null value c0 != 0, which the package's ADFmax
functions have no arguments for.  The lines of the engine's layers (prep, model, dval, roots, tstar, path, lmtstar) and
of its rule (auto) are left out: the package does not reach those layers, and the C engine's own test checks them.

The rules keep the suite short (about 30 seconds on two threads) and cover every constructor, the three starts, GLS
detrending, a local null value, each function of the general engine and each companion: the distribution function and
the density in every cell except those with GLS detrending at designs other than trend(0), trend(1) (the heavy ones;
the GLS cells of trend(1) stay), the cells of designs 7 to 21 at c = -13.5 other than designs 10 and 16, and the bulk
above the unit root at c = 4 (the vertical line stays at c = 5); the quantiles at the level 0.05 of every cell; and of
the companions and the inversions of Gil-Pelaez the cheaper cells (see KEEP).
"""
import os
import subprocess
import sys

# (function, design id) of the val lines kept for the functions listed; the others of these functions are left out
KEEP = {
    "coefgp": {2},
    "env": {1},
    "ptcdf": {1, 2, 3},
    "ptpdf": {1, 2, 3},
    "mzacdf": {3},
    "msbcdf": {3, 5},
}


def keep_err(tag, v):
    """Whether an err or rerr line (split, without the text) is kept: not ADFmax with gls or c0 != 0."""
    if tag == "err":
        return not (v[1].startswith("adf") and (v[3] != "NaN" or float(v[7]) != 0))
    return not (v[1].startswith("adf") and v[7] != "NaN")


def keep_val(v):
    """Whether a val line (split) is kept."""
    fn, i, cb, c, x = v[1], int(v[2]), v[3], float(v[6]), v[11:]
    if fn in KEEP:
        return i in KEEP[fn]
    if fn in ("cdf", "pdf"):
        if cb != "NaN" and i not in (2, 3):
            return False
        if 7 <= i <= 21 and c == -13.5 and i not in (10, 16) and cb == "NaN" and v[4] == "0" and float(v[7]) == 0:
            return False
        if i == 2 and c == 4:
            return False
        return True
    if fn == "quant":
        return float(x[0]) == 0.05
    return True


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python tools/det-ref.py /path/to/dflimit-matlab")
    up = os.path.abspath(sys.argv[1])
    src = os.path.join(up, "c", "test", "det_ref.txt")
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dst = os.path.join(pkg, "tests", "ref", "det_ref.txt")
    try:
        commit = subprocess.run(["git", "--no-optional-locks", "-C", up, "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    out, counts = [], {"design": 0, "val": 0, "route": 0, "err": 0, "rerr": 0}
    total = {"val": 0, "route": 0, "err": 0, "rerr": 0}
    with open(src) as fh:
        for line in fh:
            v = line.split(" | ")[0].split()
            if not v:
                continue
            tag = v[0]
            if tag in total:
                total[tag] += 1
            if (tag == "design" or tag == "route" or (tag == "val" and keep_val(v))
                    or (tag in ("err", "rerr") and keep_err(tag, v))):
                out.append(line if line.endswith("\n") else line + "\n")
                counts[tag] += 1
    head = [
        f"# a subset of c/test/det_ref.txt of dflimit-matlab (commit {commit}): the values of the Julia general engine of\n",
        f"# DFLimit.jl, copied unchanged by tools/det-ref.py: {counts['design']} designs, {counts['val']} of {total['val']} "
        f"val lines, {counts['route']} of {total['route']} route lines, {counts['err']} of {total['err']} err lines and "
        f"{counts['rerr']} of {total['rerr']} rerr lines.\n",
        "# design id spec: trend p | perron M q lam... | ltrend p | fdesign name | matrix base n | first base | raw np deg K loc br... C...\n",
        "# val func id cbar start g0 c c0 verified par n x... nv values... tol [brittle]   (start 0 zero, 1 fixed, 2 stationary)\n",
        "# route func engine case|design a b g0 gls c c0 par n x... nv values... tol   (engine 0 auto, 1 classical, 2 general;\n",
        "#   a, b: the model and the start of a case, or the design id and its start)\n",
        "# err and rerr: the arguments of val and route without the values, then ' | ' and a text of the error's message\n",
    ]
    with open(dst, "w", newline="\n") as fh:
        fh.writelines(head + out)
    print(f"wrote {dst}: {counts['design']} designs, {counts['val']} val lines, {counts['route']} route lines, "
          f"{counts['err']} err lines, {counts['rerr']} rerr lines")


if __name__ == "__main__":
    main()
