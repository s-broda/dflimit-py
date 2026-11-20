#!/bin/sh
# tools/sync-engine.sh: copy the C engine from a checkout of dflimit-matlab, the repository of the MATLAB package that
# holds the C engine, into csrc/ of this package.
#
#   sh tools/sync-engine.sh /path/to/dflimit-matlab
#
# The script only reads the dflimit-matlab checkout.  It
#   1. copies the engine sources of c/ (and c/faddeeva/, flattened into csrc/) into csrc/,
#   2. converts them to LF line endings (a Windows checkout with core.autocrlf=true has CRLF),
#   3. applies the patches in tools/patches/*.patch, if there are any, in name order (a patch is a change that the
#      Python package requires and that should go upstream into c/ of dflimit-matlab; once it is upstream, delete the patch
#      file; there are none at present),
#   4. records the upstream commit in src/dflimit/UPSTREAM (installed with the package).
# Left out: the test programs (dfl_test_*.c), dfl_probe.c, gen_tables.jl, the MEX gateway (c/mex/) and c/test/.
# The Python gateway, src/dflimit/_engine.c, is not part of the copy.  Do not edit the files of csrc/ by hand.
# (The same script as tools/sync-engine.sh of the R package dflimit, with the paths of this package.)
set -e

if [ $# -ne 1 ]; then echo "usage: sh tools/sync-engine.sh /path/to/dflimit-matlab" >&2; exit 2; fi
UP=$(cd "$1" && pwd)
PKG=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$UP/c/dflimit.h" ] || { echo "$UP does not look like a dflimit-matlab checkout (no c/dflimit.h)" >&2; exit 1; }
[ -f "$PKG/pyproject.toml" ] && [ -d "$PKG/src/dflimit" ] || { echo "$PKG is not the package root" >&2; exit 1; }

ENGINE="dflimit.h dfl_lock.h dfl_vmath.h dfl_api.c dfl_double.c dfl_gen.c dfl_gilpelaez.c dfl_mgf.c dfl_moments.c
        dfl_single.c dfl_special.c dfl_threads.c tables.c"
FADDEEVA="Faddeeva.cc Faddeeva.h faddeeva_shim.cpp"

mkdir -p "$PKG/csrc"
lf() { tr -d '\r' < "$1" > "$2"; }
for f in $ENGINE; do lf "$UP/c/$f" "$PKG/csrc/$f"; done
for f in $FADDEEVA; do lf "$UP/c/faddeeva/$f" "$PKG/csrc/$f"; done

# the patches (unified diffs relative to the package root, paths csrc/...)
PATCHES=""
for p in "$PKG"/tools/patches/*.patch; do
    [ -f "$p" ] || continue
    patch -p1 -s --forward --no-backup-if-mismatch -d "$PKG" < "$p"
    PATCHES="$PATCHES $(basename "$p")"
done

# the upstream commit (read-only git commands: --no-optional-locks keeps git from refreshing the index of the checkout)
G="git --no-optional-locks -C $UP"
COMMIT=$($G rev-parse HEAD 2>/dev/null || echo unknown)
CDATE=$($G log -1 --format=%cI 2>/dev/null || echo unknown)
if $G diff --quiet HEAD -- c 2>/dev/null; then DIRTY=no; else DIRTY=yes; fi
{
    echo "# The C engine in csrc/ was copied from dflimit-matlab (its directory c/) by tools/sync-engine.sh"
    echo "upstream_commit: $COMMIT"
    echo "upstream_commit_date: $CDATE"
    echo "upstream_c_modified: $DIRTY"
    echo "patches:$PATCHES"
} > "$PKG/src/dflimit/UPSTREAM"
echo "synced the engine from $UP at $COMMIT (c/ modified: $DIRTY); patches:${PATCHES:- none}"
