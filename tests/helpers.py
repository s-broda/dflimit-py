"""Helpers of the tests (tests/testthat/helper-ref.R of the R package dflimit).

The reference files in ref/ are copied unchanged from tests/testthat/ref/ of the R package dflimit: subsets of the
Julia reference values in c/test/*_ref.txt of DF-Limit, the paper's repository (see the first line of each file).
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
