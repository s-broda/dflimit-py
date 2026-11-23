/* _engine.c: the Python gateway to the C engine (the files of csrc/ are copied from c/ of dflimit-matlab by
 * tools/sync-engine.sh and are not edited here; this file is the counterpart of src/dflimit_r.c of the R package).
 * The functions of the Python layer (dflimit/_*.py) check and convert the arguments (the case code to the model and
 * the start, the method name to its number); this file checks their types again and calls the engine.
 *
 * The limited API of CPython 3.9 (Py_LIMITED_API, set by setup.py): one abi3 wheel per platform, and no numpy
 * headers.  The buffer protocol (PyObject_GetBuffer) is in the limited API only from CPython 3.11, so the arrays come
 * in as bytes (ndarray.tobytes() of a float64 array) and go out as bytearrays, which the Python layer wraps with
 * numpy.frombuffer; every length is checked here.
 *
 * The engine runs its own worker threads (dfl_threads.c) and never calls into Python: the inputs are taken and the
 * scratch space allocated before the call, the GIL is released around every engine call, and an error of the engine
 * (a nonzero return code with a message in err[]) becomes a ValueError after the call has returned.  The engine calls
 * are serialized by a lock: each call starts its own workers on all the threads set by ltratio_threads(), and the
 * engine has not been checked for concurrent calls.
 *
 * To add an entry point of the engine: a function like py_dist or py_lincdf below (in_doubles for the inputs,
 * scratch() for the outputs, ENGINE_CALL around the call, check() for the return code, out_bytes() for the results),
 * a line in METHODS, and the user-level function in the Python layer.
 *
 * The general engine (dfl_det_*: any deterministic design) and the router over the two engines (dfl_engine_*) are
 * reached through the second half of this file.  A design crosses the gateway as the tuple (br, C, K, deg, loc, first)
 * of the Python class Design (br and C as bytes in the engine's layout of dfl_design); it is only read by the engine.
 * Their warnings (dfl_det_warning and, for the router, dfl_warning; thread-local, on the calling thread; cleared before
 * each call) are returned as strings with the results and become RuntimeWarnings in the Python layer.  The design given by a Python function
 * (fdesign) is built with the GIL held and outside the engine lock: its callback runs on the calling thread only, and
 * the construction of a design uses neither the worker threads nor the caches. */
#include <Python.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include "dflimit.h"
#include "dfl_lock.h"

#define ERRLEN 512

/* (the engine's own mutex, not a PyThread lock: python3.dll of CPython 3.9 on Windows does not export PyThread_*) */
static dfl_lock_t engine_lock = DFL_LOCK_INIT;

/* the engine call stmt without the GIL, under the engine lock */
#define ENGINE_CALL(stmt) do { \
        Py_BEGIN_ALLOW_THREADS \
        dfl_lock(&engine_lock); \
        stmt; \
        dfl_unlock(&engine_lock); \
        Py_END_ALLOW_THREADS \
    } while (0)

/* the n doubles of the bytes object b (ndarray.tobytes() of a float64 array).  *copy receives a copy that the caller
 * frees with PyMem_Free when the data of b are not aligned for double (otherwise NULL). */
static const double *in_doubles(PyObject *b, const char *name, int *n, double **copy)
{
    *copy = NULL;
    if (!PyBytes_Check(b)) { PyErr_Format(PyExc_TypeError, "%s must be bytes", name); return NULL; }
    Py_ssize_t len = PyBytes_Size(b);
    if (len < 0) return NULL;
    if (len % (Py_ssize_t) sizeof(double)) { PyErr_Format(PyExc_ValueError, "%s: the length is not a multiple of %d", name, (int) sizeof(double)); return NULL; }
    Py_ssize_t m = len / (Py_ssize_t) sizeof(double);
    if (m > INT_MAX / 2) { PyErr_Format(PyExc_ValueError, "%s is too long", name); return NULL; }
    const char *p = PyBytes_AsString(b);
    if (p == NULL) return NULL;
    *n = (int) m;
    if ((uintptr_t) p % sizeof(double)) {
        *copy = PyMem_Malloc((size_t) len + sizeof(double));
        if (*copy == NULL) { PyErr_NoMemory(); return NULL; }
        memcpy(*copy, p, (size_t) len);
        return *copy;
    }
    return (const double *) p;
}

/* scratch space for n + 1 elements of the given size (n + 1: never a request for zero bytes) */
static void *scratch(int n, size_t size)
{
    void *p = PyMem_Malloc(((size_t) n + 1) * size);
    if (p == NULL) PyErr_NoMemory();
    return p;
}

/* the n elements of the given size at p, as a bytearray (a new reference, NULL with an exception) */
static PyObject *out_bytes(const void *p, int n, size_t size)
{
    return PyByteArray_FromStringAndSize((const char *) p, (Py_ssize_t) n * (Py_ssize_t) size);
}

/* an error of the engine: a ValueError with its message (0 if rc is 0) */
static int check(int rc, const char *err)
{
    if (rc == 0) return 0;
    PyErr_SetString(PyExc_ValueError, err[0] ? err : "the engine failed");
    return -1;
}

/* dist(kind, x, model, c, c0, g0, stat, method, verified) -> (bytearray out, int nstatus, str brittle-regime message or
 * "", str the warning of the points that the line integral refused or ""): dfl_dist, dfl_brittle, dfl_warning */
static PyObject *py_dist(PyObject *self, PyObject *args)
{
    int kind, model, stat, method, verified = 0, n = 0, ns = 0, rc = 0;
    double c, c0, g0, *xc = NULL, *out = NULL;
    PyObject *xb, *res = NULL;
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "iOidddii|i:dist", &kind, &xb, &model, &c, &c0, &g0, &stat, &method, &verified)) return NULL;
    const double *x = in_doubles(xb, "x", &n, &xc);
    if (x == NULL || (out = scratch(n, sizeof(double))) == NULL) goto done;
    dfl_double_opts dv; dfl_double_defaults(&dv); dv.verified = verified != 0;
    ENGINE_CALL(rc = dfl_dist(kind, n, x, model, c, c0, g0, stat != 0, method, NULL, &dv, out, &ns, err, sizeof err));
    if (check(rc, err)) goto done;
    const char *bm = verified ? NULL : dfl_brittle(kind, model, c, c0, g0, stat != 0, method), *wm = dfl_warning();
    res = Py_BuildValue("(Niss)", out_bytes(out, n, sizeof(double)), ns, bm ? bm : "", wm ? wm : "");
done:
    PyMem_Free(out); PyMem_Free(xc);
    return res;
}

/* icdf(p, model, c, c0, g0, stat, method, tol, verified) -> (q, F, iter, conv) as bytearrays (iter, conv: C int), the
 * brittle-regime message and the warning of the points that the line integral refused: dfl_icdf, dfl_warning */
static PyObject *py_icdf(PyObject *self, PyObject *args)
{
    int model, stat, method, verified = 0, n = 0, rc = 0, *it = NULL, *cv = NULL;
    double c, c0, g0, tol, *pc = NULL, *q = NULL, *F = NULL;
    PyObject *pb, *res = NULL;
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "Oidddiid|i:icdf", &pb, &model, &c, &c0, &g0, &stat, &method, &tol, &verified)) return NULL;
    const double *p = in_doubles(pb, "p", &n, &pc);
    if (p == NULL || (q = scratch(n, sizeof(double))) == NULL || (F = scratch(n, sizeof(double))) == NULL ||
        (it = scratch(n, sizeof(int))) == NULL || (cv = scratch(n, sizeof(int))) == NULL) goto done;
    dfl_double_opts dv; dfl_double_defaults(&dv); dv.verified = verified != 0;
    ENGINE_CALL(rc = dfl_icdf(n, p, model, c, c0, g0, stat != 0, method, tol, NULL, &dv, q, F, it, cv, err, sizeof err));
    if (check(rc, err)) goto done;
    const char *bm = verified ? NULL : dfl_brittle(2, model, c, c0, g0, stat != 0, method), *wm = dfl_warning();
    res = Py_BuildValue("(NNNNss)", out_bytes(q, n, sizeof(double)), out_bytes(F, n, sizeof(double)),
                        out_bytes(it, n, sizeof(int)), out_bytes(cv, n, sizeof(int)), bm ? bm : "", wm ? wm : "");
done:
    PyMem_Free(q); PyMem_Free(F); PyMem_Free(it); PyMem_Free(cv); PyMem_Free(pc);
    return res;
}

/* moments(pmax, model, c, c0, g0, stat) -> (bytearray m, bytearray err): dfl_moments */
static PyObject *py_moments(PyObject *self, PyObject *args)
{
    int pmax, model, stat, rc = 0;
    double c, c0, g0, *m = NULL, *e = NULL;
    PyObject *res = NULL;
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "iidddi:moments", &pmax, &model, &c, &c0, &g0, &stat)) return NULL;
    if (pmax < 1) { PyErr_SetString(PyExc_ValueError, "p must be a positive integer"); return NULL; }
    if ((m = scratch(pmax, sizeof(double))) == NULL || (e = scratch(pmax, sizeof(double))) == NULL) goto done;
    ENGINE_CALL(rc = dfl_moments(model, c, c0, g0, stat != 0, pmax, m, e, err, sizeof err));
    if (check(rc, err)) goto done;
    res = Py_BuildValue("(NN)", out_bytes(m, pmax, sizeof(double)), out_bytes(e, pmax, sizeof(double)));
done:
    PyMem_Free(m); PyMem_Free(e);
    return res;
}

/* adfmax(kind, x, model, c, stat) -> (value, err, status) as bytearrays (status: C int): dfl_adfmax */
static PyObject *py_adfmax(PyObject *self, PyObject *args)
{
    int kind, model, stat, n = 0, rc = 0, *su = NULL;
    double c, *xc = NULL, *out = NULL, *er = NULL;
    dfl_double_info *info = NULL;
    PyObject *xb, *res = NULL;
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "iOidi:adfmax", &kind, &xb, &model, &c, &stat)) return NULL;
    const double *x = in_doubles(xb, "x", &n, &xc);
    if (x == NULL || (out = scratch(n, sizeof(double))) == NULL || (er = scratch(n, sizeof(double))) == NULL ||
        (su = scratch(n, sizeof(int))) == NULL || (info = scratch(n, sizeof(dfl_double_info))) == NULL) goto done;
    ENGINE_CALL(rc = dfl_adfmax(kind, n, x, model, c, stat != 0, NULL, out, info, err, sizeof err));
    if (check(rc, err)) goto done;
    for (int j = 0; j < n; j++) { er[j] = info[j].err; su[j] = info[j].status; }
    res = Py_BuildValue("(NNN)", out_bytes(out, n, sizeof(double)), out_bytes(er, n, sizeof(double)),
                        out_bytes(su, n, sizeof(int)));
done:
    PyMem_Free(out); PyMem_Free(er); PyMem_Free(su); PyMem_Free(info); PyMem_Free(xc);
    return res;
}

/* region(type, p1, p2, v0, brks, c) -> (P, err, status): dfl_region */
static PyObject *py_region(PyObject *self, PyObject *args)
{
    int type, nb = 0, rc = 0;
    double par[2], v0, c, P = 0, *bc = NULL;
    dfl_double_info info = {0};
    PyObject *bb, *res = NULL;
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "idddOd:region", &type, &par[0], &par[1], &v0, &bb, &c)) return NULL;
    const double *brks = in_doubles(bb, "brks", &nb, &bc);
    if (brks == NULL) goto done;
    ENGINE_CALL(rc = dfl_region(type, par, v0, nb, brks, c, NULL, &P, &info, err, sizeof err));
    if (check(rc, err)) goto done;
    res = Py_BuildValue("(ddi)", P, info.err, info.status);
done:
    PyMem_Free(bc);
    return res;
}

/* lincdf(b, a, model, c, cbar) -> bytearray: P(U - aV <= b), a = NaN: P(V <= b): dfl_lincdf */
static PyObject *py_lincdf(PyObject *self, PyObject *args)
{
    int model, n = 0, rc = 0;
    double a, c, cbar, *bc = NULL, *out = NULL;
    PyObject *bb, *res = NULL;
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "Odidd:lincdf", &bb, &a, &model, &c, &cbar)) return NULL;
    const double *b = in_doubles(bb, "b", &n, &bc);
    if (b == NULL || (out = scratch(n, sizeof(double))) == NULL) goto done;
    ENGINE_CALL(rc = dfl_lincdf(n, b, a, model, c, cbar, NULL, out, err, sizeof err));
    if (check(rc, err)) goto done;
    res = out_bytes(out, n, sizeof(double));
done:
    PyMem_Free(out); PyMem_Free(bc);
    return res;
}

/* envelope(c, model, alpha) -> bytearray: the power envelope at the points c: dfl_power_envelope */
static PyObject *py_envelope(PyObject *self, PyObject *args)
{
    int model, n = 0, rc = 0;
    double alpha, *cc = NULL, *out = NULL;
    PyObject *cb, *res = NULL;
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "Oid:envelope", &cb, &model, &alpha)) return NULL;
    const double *c = in_doubles(cb, "c", &n, &cc);
    if (c == NULL || (out = scratch(n, sizeof(double))) == NULL) goto done;
    ENGINE_CALL(rc = dfl_power_envelope(n, c, model, alpha, NULL, out, err, sizeof err));
    if (check(rc, err)) goto done;
    res = out_bytes(out, n, sizeof(double));
done:
    PyMem_Free(out); PyMem_Free(cc);
    return res;
}

/* threads(n=None) -> the number of worker threads, after setting it to n (None: unchanged) */
static PyObject *py_threads(PyObject *self, PyObject *args)
{
    PyObject *no = Py_None;
    int nw = 0;
    if (!PyArg_ParseTuple(args, "|O:threads", &no)) return NULL;
    if (no != Py_None) {
        long k = PyLong_AsLong(no);
        if (k == -1 && PyErr_Occurred()) return NULL;
        if (k < 1 || k > INT_MAX) { PyErr_SetString(PyExc_ValueError, "n must be a positive integer"); return NULL; }
        ENGINE_CALL(dfl_set_nworkers((int) k));
    }
    ENGINE_CALL(nw = dfl_nworkers());
    return PyLong_FromLong(nw);
}

/* =====================================================================================================================
 * The general engine and the router
 * ===================================================================================================================== */

/* a design of the Python class Design: the tuple (br, C, K, deg, loc, first), br and C as bytes (np + 1 and np (deg + 1) K
 * doubles).  The copies made for the alignment are freed by design_release (also after an error). */
typedef struct { dfl_design D; double *brc, *Cc; } design_arg;
static void design_release(design_arg *a) { PyMem_Free(a->brc); PyMem_Free(a->Cc); a->brc = a->Cc = NULL; }
static int design_in(PyObject *t, design_arg *a)
{
    PyObject *brb, *cb;
    int K, deg, loc, first, nb = 0, nc = 0;
    memset(a, 0, sizeof *a);
    if (!PyTuple_Check(t)) { PyErr_SetString(PyExc_TypeError, "the design must be a tuple (br, C, K, deg, loc, first)"); return -1; }
    if (!PyArg_ParseTuple(t, "OOiiii:design", &brb, &cb, &K, &deg, &loc, &first)) return -1;
    const double *br = in_doubles(brb, "br", &nb, &a->brc);
    if (br == NULL) return -1;
    const double *C = in_doubles(cb, "C", &nc, &a->Cc);
    if (C == NULL) { design_release(a); return -1; }
    if (nb < 2 || K < 0 || deg < 0 || (long long) nc != (long long) (nb - 1) * (deg + 1) * K) {
        design_release(a);
        PyErr_SetString(PyExc_ValueError, "the design's break points and coefficients do not match");
        return -1;
    }
    /* (the engine only reads a design: the break points increase from 0 to 1, the coefficients are finite) */
    int ok = br[0] == 0 && br[nb - 1] == 1;
    for (int j = 1; j < nb && ok; j++) ok = br[j] > br[j - 1];
    for (int j = 0; j < nc && ok; j++) ok = isfinite(C[j]);
    if (!ok) {
        design_release(a);
        PyErr_SetString(PyExc_ValueError, "a design needs break points increasing from 0 to 1 and finite coefficients");
        return -1;
    }
    a->D.K = K; a->D.np = nb - 1; a->D.deg = deg; a->D.br = (double *) br; a->D.C = (double *) C;
    a->D.loc = loc != 0; a->D.first = first != 0;
    return 0;
}
/* the tuple (br, C, K, deg, loc, first) of a design built by the engine (a new reference, NULL with an exception) */
static PyObject *design_out(const dfl_design *D)
{
    return Py_BuildValue("(NNiiii)", PyBytes_FromStringAndSize((const char *) D->br, (Py_ssize_t) (D->np + 1) * (Py_ssize_t) sizeof(double)),
                         PyBytes_FromStringAndSize((const char *) D->C, (Py_ssize_t) D->np * (D->deg + 1) * D->K * (Py_ssize_t) sizeof(double)),
                         D->K, D->deg, D->loc, D->first);
}

/* the options of the general engine: the tuple (c0, start, gamma0, gls, verified) or (c0, start, gamma0, gls, verified,
 * method), gls NaN for least squares, method 0 auto, 1 the line integral with the explicit kernel, 2 the inversion on
 * paths around the roots */
static int detopts_in(PyObject *t, dfl_det_opts *o)
{
    int start, verified, method = DFL_DET_AUTO;
    double c0, g0, gls;
    dfl_det_defaults(o);
    if (!PyTuple_Check(t)) { PyErr_SetString(PyExc_TypeError, "the options must be a tuple (c0, start, gamma0, gls, verified[, method])"); return -1; }
    if (!PyArg_ParseTuple(t, "diddi|i:opts", &c0, &start, &g0, &gls, &verified, &method)) return -1;
    o->c0 = c0; o->start = start; o->gamma0 = g0; o->gls = gls; o->verified = verified != 0; o->method = method;
    return 0;
}

/* a computation of the router: the tuple (design or None, model, start, g0, gls) */
typedef struct { dfl_spec S; design_arg d; } spec_arg;
static void spec_release(spec_arg *s) { design_release(&s->d); }
static int spec_in(PyObject *t, spec_arg *s)
{
    PyObject *dt;
    memset(s, 0, sizeof *s);
    if (!PyTuple_Check(t)) { PyErr_SetString(PyExc_TypeError, "the computation must be a tuple (design, model, start, g0, gls)"); return -1; }
    if (!PyArg_ParseTuple(t, "Oiidd:spec", &dt, &s->S.model, &s->S.start, &s->S.g0, &s->S.gls)) return -1;
    if (dt != Py_None) {
        if (design_in(dt, &s->d)) return -1;
        s->S.D = &s->d.D;
    }
    return 0;
}

/* the warnings of a call of the general engine or the router: dfl_det_warning and dfl_warning, cleared before the call
 * (dfl_det_warning_clear, dfl_warning_clear) and copied on the calling thread before the engine lock is released */
#define DETWARNLEN 2048
#define CLSWARNLEN 1024
typedef struct { char det[DETWARNLEN], cls[CLSWARNLEN]; } warn_t;
static void warn_reset(void)
{
    dfl_det_warning_clear();
    dfl_warning_clear();
}
/* appends the messages of s (lines) that buf does not hold yet */
static void warn_add(char *buf, size_t len, const char *s)
{
    if (s == NULL || !s[0] || strstr(buf, s)) return;
    size_t k = strlen(buf);
    if (k + 2 >= len) return;
    snprintf(buf + k, len - k, "%s%s", k ? "\n" : "", s);
}
static void warn_take(warn_t *w)
{
    warn_add(w->det, sizeof w->det, dfl_det_warning());
    warn_add(w->cls, sizeof w->cls, dfl_warning());
}
/* the call stmt with the warnings reset before and collected after it, under the engine lock without the GIL */
#define WARN_CALL(stmt, w) ENGINE_CALL(warn_reset(); stmt; warn_take(w))

/* the C int values of the bytes object b (ndarray.tobytes() of an intc array) into a new PyMem array */
static int *in_ints(PyObject *b, const char *name, int *n)
{
    if (!PyBytes_Check(b)) { PyErr_Format(PyExc_TypeError, "%s must be bytes", name); return NULL; }
    Py_ssize_t len = PyBytes_Size(b);
    if (len < 0) return NULL;
    if (len % (Py_ssize_t) sizeof(int)) { PyErr_Format(PyExc_ValueError, "%s: the length is not a multiple of %d", name, (int) sizeof(int)); return NULL; }
    if (len / (Py_ssize_t) sizeof(int) > INT_MAX / 2) { PyErr_Format(PyExc_ValueError, "%s is too long", name); return NULL; }
    *n = (int) (len / (Py_ssize_t) sizeof(int));
    int *p = scratch(*n, sizeof(int));
    if (p == NULL) return NULL;
    const char *s = PyBytes_AsString(b);
    if (s == NULL) { PyMem_Free(p); return NULL; }
    memcpy(p, s, (size_t) len);
    return p;
}

/* ---- the designs ---- */
/* design_trend(p), design_ltrend(p) -> design tuple: dfl_design_trend, dfl_design_ltrend */
static PyObject *design_result(int rc, dfl_design *D, const char *err)
{
    /* (D starts zeroed; after an error the constructors have allocated nothing that they keep, but free in any case) */
    if (check(rc, err)) { dfl_design_free(D); return NULL; }
    PyObject *r = design_out(D);
    dfl_design_free(D);
    return r;
}
static PyObject *py_design_trend(PyObject *self, PyObject *args)
{
    int p, legendre = 0, rc = 0;
    dfl_design D; memset(&D, 0, sizeof D);
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "i|i:design_trend", &p, &legendre)) return NULL;
    Py_BEGIN_ALLOW_THREADS
    rc = legendre ? dfl_design_ltrend(p, &D, err, sizeof err) : dfl_design_trend(p, &D, err, sizeof err);
    Py_END_ALLOW_THREADS
    return design_result(rc, &D, err);
}
/* design_perron(model, lam) -> design tuple (model 'A', 'B' or 'C'; lam bytes): dfl_design_perron */
static PyObject *py_design_perron(PyObject *self, PyObject *args)
{
    int model, q = 0, rc = 0;
    double *lc = NULL;
    PyObject *lb;
    dfl_design D; memset(&D, 0, sizeof D);
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "CO:design_perron", &model, &lb)) return NULL;
    const double *lam = in_doubles(lb, "lam", &q, &lc);
    if (lam == NULL) return NULL;
    Py_BEGIN_ALLOW_THREADS
    rc = dfl_design_perron((char) model, q, lam, &D, err, sizeof err);
    Py_END_ALLOW_THREADS
    PyMem_Free(lc);
    return design_result(rc, &D, err);
}
/* design_first(design) -> design tuple: dfl_design_detrend_first */
static PyObject *py_design_first(PyObject *self, PyObject *args)
{
    PyObject *dt;
    design_arg a;
    int rc = 0;
    dfl_design D; memset(&D, 0, sizeof D);
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "O:design_first", &dt)) return NULL;
    if (design_in(dt, &a)) return NULL;
    rc = dfl_design_detrend_first(&a.D, &D, err, sizeof err);
    design_release(&a);
    return design_result(rc, &D, err);
}
/* design_matrix(X, n, K, TB or None, tol, maxdeg, minlen) -> (design tuple, warning): dfl_design_from_matrix (X the bytes
 * of the n x K matrix in column-major order, TB the bytes of C ints) */
static PyObject *py_design_matrix(PyObject *self, PyObject *args)
{
    PyObject *xb, *tb, *res = NULL;
    int n, K, maxdeg, minlen, nx = 0, nb = -1, rc = 0, *TB = NULL;
    double tol, *xc = NULL;
    dfl_design D; memset(&D, 0, sizeof D);
    char err[ERRLEN] = "";
    warn_t w = {"", ""};
    if (!PyArg_ParseTuple(args, "OiiOdii:design_matrix", &xb, &n, &K, &tb, &tol, &maxdeg, &minlen)) return NULL;
    const double *X = in_doubles(xb, "X", &nx, &xc);
    if (X == NULL) return NULL;
    if (n < 0 || K < 0 || (long long) nx != (long long) n * K) { PyErr_SetString(PyExc_ValueError, "X: the size does not match"); goto done; }
    if (tb != Py_None && (TB = in_ints(tb, "TB", &nb)) == NULL) goto done;
    Py_BEGIN_ALLOW_THREADS
    warn_reset();
    rc = dfl_design_from_matrix(n, K, X, nb, TB, tol, maxdeg, minlen, &D, err, sizeof err);
    if (rc == 0) warn_take(&w);
    Py_END_ALLOW_THREADS
    if (check(rc, err)) goto done;
    res = Py_BuildValue("(Ns)", design_out(&D), w.det);
done:
    dfl_design_free(&D); PyMem_Free(xc); PyMem_Free(TB);
    return res;
}
/* the callback of design_fdesign: the Python function returns the bytes of the K regressors at r (the GIL is held) */
typedef struct { PyObject *f; int K, failed; } pyf_ctx;
static void pyf_call(double r, double *out, void *v)
{
    pyf_ctx *p = v;
    for (int i = 0; i < p->K; i++) out[i] = 0;
    if (p->failed) return;
    PyObject *res = PyObject_CallFunction(p->f, "d", r);
    if (res == NULL) { p->failed = 1; return; }
    if (!PyBytes_Check(res) || PyBytes_Size(res) != (Py_ssize_t) p->K * (Py_ssize_t) sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "the function of the design must return K values");
        p->failed = 1;
    } else memcpy(out, PyBytes_AsString(res), (size_t) p->K * sizeof(double));
    Py_DECREF(res);
}
/* design_fdesign(f, K, breaks, tol, maxdeg, sub) -> design tuple: dfl_design_fdesign with a Python function (called with
 * the GIL held, outside the engine lock: a function that calls the package's functions does not deadlock) */
static PyObject *py_design_fdesign(PyObject *self, PyObject *args)
{
    PyObject *f, *bb;
    int K, maxdeg, sub, nb = 0, rc = 0;
    double tol, *bc = NULL;
    dfl_design D; memset(&D, 0, sizeof D);
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "OiOdii:design_fdesign", &f, &K, &bb, &tol, &maxdeg, &sub)) return NULL;
    if (!PyCallable_Check(f)) { PyErr_SetString(PyExc_TypeError, "f must be callable"); return NULL; }
    const double *br = in_doubles(bb, "breaks", &nb, &bc);
    if (br == NULL) return NULL;
    pyf_ctx ctx = {f, K, 0};
    rc = dfl_design_fdesign(pyf_call, &ctx, K, nb, br, tol, maxdeg, sub, &D, err, sizeof err);
    PyMem_Free(bc);
    if (ctx.failed) { dfl_design_free(&D); return NULL; }
    return design_result(rc, &D, err);
}
/* design_value(design, r) -> bytearray of the n x K values (row-major): dfl_design_value; design_regmat(design, n) ->
 * bytearray of the n x K matrix d(t/n), t = 1..n (column-major): dfl_design_regmat */
static PyObject *py_design_value(PyObject *self, PyObject *args)
{
    PyObject *dt, *rb, *res = NULL;
    int n = 0;
    double *rcpy = NULL, *out = NULL;
    design_arg a;
    if (!PyArg_ParseTuple(args, "OO:design_value", &dt, &rb)) return NULL;
    if (design_in(dt, &a)) return NULL;
    const double *r = in_doubles(rb, "r", &n, &rcpy);
    if (r == NULL) goto done;
    if ((long long) n * (a.D.K > 0 ? a.D.K : 1) > INT_MAX / 2) { PyErr_SetString(PyExc_ValueError, "r is too long"); goto done; }
    if ((out = scratch(n * (a.D.K > 0 ? a.D.K : 1), sizeof(double))) == NULL) goto done;
    for (int k = 0; k < n; k++) dfl_design_value(&a.D, r[k], out + (size_t) k * a.D.K);
    res = out_bytes(out, n * a.D.K, sizeof(double));
done:
    PyMem_Free(out); PyMem_Free(rcpy); design_release(&a);
    return res;
}
static PyObject *py_design_regmat(PyObject *self, PyObject *args)
{
    PyObject *dt, *res = NULL;
    int n;
    double *out = NULL;
    design_arg a;
    if (!PyArg_ParseTuple(args, "Oi:design_regmat", &dt, &n)) return NULL;
    if (n < 1) { PyErr_SetString(PyExc_ValueError, "n must be a positive integer"); return NULL; }
    if (design_in(dt, &a)) return NULL;
    if ((long long) n * (a.D.K > 0 ? a.D.K : 1) > INT_MAX / 2) { PyErr_SetString(PyExc_ValueError, "n is too large"); goto done; }
    if ((out = scratch(n * (a.D.K > 0 ? a.D.K : 1), sizeof(double))) == NULL) goto done;
    dfl_design_regmat(&a.D, n, out);
    res = out_bytes(out, n * a.D.K, sizeof(double));
done:
    PyMem_Free(out); design_release(&a);
    return res;
}

/* ---- the user-level functions of the general engine ---- */
/* the operations of det (the values at the points x, or the quantiles at the probabilities x) */
enum {
    DET_CDF = 1, DET_PDF, DET_COEF_CDF, DET_COEF_PDF, DET_COEF_GP, DET_LIN_CDF, DET_LIN_PDF, DET_ADF_CDF, DET_ADF_PDF,
    DET_LPOW, DET_ENV, DET_PT_CDF, DET_PT_PDF, DET_PT_ENV, DET_LM_CDF, DET_MZA_CDF, DET_MSB_CDF,
    DET_Q = 30, DET_COEF_Q, DET_LIN_Q, DET_ADF_Q, DET_PT_Q, DET_LM_Q, DET_MZA_Q, DET_MSB_Q,
    DET_PT_POW = 50, DET_LM_POW
};
/* one operation of the general engine on the points x: the engine call (0 or 1, the message in err); the scalar
 * functions of the engine (the quantiles of the linear form, ADFmax, the point-optimal statistic, the LM statistic,
 * MZ_alpha, MSB) are called point by point, their warnings collected in w */
static int ipar(double v) { return isfinite(v) && fabs(v) < INT_MAX ? (int) v : 0; }
static int det_run(int op, const dfl_design *D, int n, const double *x, double c, const dfl_det_opts *o, double p0, double p1,
                   double p2, double p3, double *out, warn_t *w, char *err, size_t errlen)
{
    int rc = 0;
    switch (op) {
    case DET_CDF: case DET_PDF: return dfl_det_dist(op == DET_CDF ? 2 : 1, D, n, x, c, o, out, err, errlen);
    case DET_COEF_CDF: case DET_COEF_PDF: return dfl_det_coef(op == DET_COEF_CDF ? 2 : 1, D, n, x, c, o, out, err, errlen);
    case DET_COEF_GP: return dfl_det_coef_cdf_gp(D, n, x, c, o, ipar(p0), p1, out, err, errlen);
    case DET_LIN_CDF: case DET_LIN_PDF: return dfl_det_lin(op == DET_LIN_CDF ? 2 : 1, D, n, x, p0, c, o, out, err, errlen);
    case DET_ADF_CDF: case DET_ADF_PDF: return dfl_det_adfmax(op == DET_ADF_CDF ? 2 : 1, D, n, x, c, o, out, err, errlen);
    case DET_LPOW: return dfl_det_local_power(D, n, x, p0, ipar(p1), o, out, err, errlen);
    case DET_ENV: return dfl_det_power_envelope(D, n, x, p0, out, err, errlen);
    case DET_PT_CDF: case DET_PT_PDF: return dfl_det_pt(op == DET_PT_CDF ? 2 : 1, D, n, x, p0, c, ipar(p1), p2, out, err, errlen);
    case DET_PT_ENV: return dfl_det_pt_envelope(D, n, x, p0, ipar(p1), p2, out, err, errlen);
    case DET_LM_CDF: return dfl_det_lm_cdf(D, n, x, c, out, err, errlen);
    case DET_MZA_CDF: return dfl_det_mza_cdf(D, n, x, c, p0, o, out, err, errlen);
    case DET_MSB_CDF: return dfl_det_msb_cdf(D, n, x, c, p0, out, err, errlen);
    case DET_Q: return dfl_det_quantile(D, n, x, c, p0, p1, o, out, err, errlen);
    case DET_COEF_Q: return dfl_det_coef_quantile(D, n, x, c, p0, p1, o, out, err, errlen);
    case DET_PT_POW: return dfl_det_pt_power(D, p0, c, p1, ipar(p2), p3, &out[0], &out[1], err, errlen);
    case DET_LM_POW: return dfl_det_lm_power(D, n, x, p0, &out[0], out + 1, err, errlen);
    default: break;
    }
    for (int k = 0; k < n && !rc; k++) {
        switch (op) {
        case DET_LIN_Q: rc = dfl_det_lin_quantile(D, x[k], p0, c, p1, p2, o, &out[k], err, errlen); break;
        case DET_ADF_Q: rc = dfl_det_adfmax_quantile(D, x[k], c, p0, p1, o, &out[k], err, errlen); break;
        case DET_PT_Q: rc = dfl_det_pt_quantile(D, x[k], p0, ipar(p1), p2, &out[k], err, errlen); break;
        case DET_LM_Q: rc = dfl_det_lm_quantile(D, x[k], c, &out[k], err, errlen); break;
        case DET_MZA_Q: rc = dfl_det_mza_quantile(D, x[k], c, p0, p1, p2, o, &out[k], err, errlen); break;
        case DET_MSB_Q: rc = dfl_det_msb_quantile(D, x[k], c, p0, &out[k], err, errlen); break;
        default: snprintf(err, errlen, "unknown operation %d of the general engine", op); return 1;
        }
        warn_take(w);
    }
    return rc;
}
/* det(op, design, x, c, opts, p0, p1, p2, p3) -> (bytearray out, warning): the general engine (dfl_det_*; the parameters
 * p0 to p3 of each operation in det_run and in the Python layer, _general.py).  out has n values, n + 1 for DET_LM_POW (the critical value
 * first) and 2 for DET_PT_POW (the critical value and the power at c) */
static PyObject *py_det(PyObject *self, PyObject *args)
{
    PyObject *dt, *xb, *ot, *res = NULL;
    int op, n = 0, rc = 0;
    double c, p0 = NAN, p1 = NAN, p2 = NAN, p3 = NAN, *xc = NULL, *out = NULL;
    design_arg a;
    dfl_det_opts o;
    char err[ERRLEN] = "";
    warn_t w = {"", ""};
    if (!PyArg_ParseTuple(args, "iOOdO|dddd:det", &op, &dt, &xb, &c, &ot, &p0, &p1, &p2, &p3)) return NULL;
    if (detopts_in(ot, &o)) return NULL;
    if (design_in(dt, &a)) return NULL;
    const double *x = in_doubles(xb, "x", &n, &xc);
    if (x == NULL || (out = scratch(n + 2, sizeof(double))) == NULL) goto done;
    int nout = op == DET_LM_POW ? n + 1 : op == DET_PT_POW ? 2 : n;
    WARN_CALL(rc = det_run(op, &a.D, n, x, c, &o, p0, p1, p2, p3, out, &w, err, sizeof err), &w);
    if (check(rc, err)) goto done;
    res = Py_BuildValue("(Ns)", out_bytes(out, nout, sizeof(double)), w.det);
done:
    PyMem_Free(out); PyMem_Free(xc); design_release(&a);
    return res;
}
/* det_moments(design, c, pmax, opts) -> (bytearray m, bytearray err, warning): dfl_det_moments */
static PyObject *py_det_moments(PyObject *self, PyObject *args)
{
    PyObject *dt, *ot, *res = NULL;
    int pmax, rc = 0;
    double c, *m = NULL, *e = NULL;
    design_arg a;
    dfl_det_opts o;
    char err[ERRLEN] = "";
    warn_t w = {"", ""};
    if (!PyArg_ParseTuple(args, "OdiO:det_moments", &dt, &c, &pmax, &ot)) return NULL;
    if (pmax < 1) { PyErr_SetString(PyExc_ValueError, "p must be a positive integer"); return NULL; }
    if (detopts_in(ot, &o)) return NULL;
    if (design_in(dt, &a)) return NULL;
    if ((m = scratch(pmax, sizeof(double))) == NULL || (e = scratch(pmax, sizeof(double))) == NULL) goto done;
    WARN_CALL(rc = dfl_det_moments(&a.D, c, pmax, &o, m, e, err, sizeof err), &w);
    if (check(rc, err)) goto done;
    res = Py_BuildValue("(NNs)", out_bytes(m, pmax, sizeof(double)), out_bytes(e, pmax, sizeof(double)), w.det);
done:
    PyMem_Free(m); PyMem_Free(e); design_release(&a);
    return res;
}
/* det_empty() -> None: dfl_det_empty_caches */
static PyObject *py_det_empty(PyObject *self, PyObject *args)
{
    ENGINE_CALL(dfl_det_empty_caches());
    Py_RETURN_NONE;
}

/* ---- the router over the two engines (dfl_engine_*) ---- */
enum { R_CDF = 1, R_PDF, R_ICDF, R_MOM, R_COEF, R_COEF_ICDF, R_ADF_CDF, R_ADF_PDF };
/* route(op, spec, x, c, c0, engine, method, verified, tol) -> (bytearray out, general warning, classical warning): the
 * t-ratio (R_CDF, R_PDF: dfl_engine_dist), its quantiles (R_ICDF), its moments (R_MOM: x has one value, the number of
 * moments), the coefficient statistic (R_COEF, R_COEF_ICDF) and ADFmax (R_ADF_CDF, R_ADF_PDF) */
static PyObject *py_route(PyObject *self, PyObject *args)
{
    PyObject *st, *xb, *res = NULL;
    int op, engine, method, verified, n = 0, rc = 0, nout;
    double c, c0, tol, *xc = NULL, *out = NULL;
    spec_arg s;
    char err[ERRLEN] = "";
    warn_t w = {"", ""};
    if (!PyArg_ParseTuple(args, "iOOddiiid:route", &op, &st, &xb, &c, &c0, &engine, &method, &verified, &tol)) return NULL;
    if (spec_in(st, &s)) return NULL;
    const double *x = in_doubles(xb, "x", &n, &xc);
    if (x == NULL) goto done;
    nout = n;
    if (op == R_MOM) {
        if (n != 1 || !(x[0] >= 1 && x[0] <= INT_MAX / 2 && x[0] == floor(x[0]))) { PyErr_SetString(PyExc_ValueError, "p must be a positive integer"); goto done; }
        nout = (int) x[0];
    }
    if ((out = scratch(nout, sizeof(double))) == NULL) goto done;
    switch (op) {
    case R_CDF: case R_PDF: WARN_CALL(rc = dfl_engine_dist(op == R_CDF ? 2 : 1, n, x, &s.S, c, c0, engine, method, verified, out, err, sizeof err), &w); break;
    case R_ICDF: WARN_CALL(rc = dfl_engine_icdf(n, x, &s.S, c, c0, engine, method, tol, out, err, sizeof err), &w); break;
    case R_MOM: WARN_CALL(rc = dfl_engine_mom(nout, &s.S, c, c0, engine, out, err, sizeof err), &w); break;
    case R_COEF: WARN_CALL(rc = dfl_engine_coef(n, x, &s.S, c, c0, engine, out, err, sizeof err), &w); break;
    case R_COEF_ICDF: WARN_CALL(rc = dfl_engine_coef_icdf(n, x, &s.S, c, c0, engine, tol, out, err, sizeof err), &w); break;
    case R_ADF_CDF: case R_ADF_PDF: WARN_CALL(rc = dfl_engine_adfmax(op == R_ADF_CDF ? 2 : 1, n, x, &s.S, c, engine, out, err, sizeof err), &w); break;
    default: PyErr_Format(PyExc_ValueError, "unknown operation %d of the router", op); goto done;
    }
    if (check(rc, err)) goto done;
    res = Py_BuildValue("(Nss)", out_bytes(out, nout, sizeof(double)), w.det, w.cls);
done:
    PyMem_Free(out); PyMem_Free(xc); spec_release(&s);
    return res;
}
/* route_lin(b, a, model, c, cbar, engine) -> (bytearray, general warning, classical warning): dfl_engine_lincdf */
static PyObject *py_route_lin(PyObject *self, PyObject *args)
{
    PyObject *bb, *res = NULL;
    int model, engine, n = 0, rc = 0;
    double a, c, cbar, *bc = NULL, *out = NULL;
    char err[ERRLEN] = "";
    warn_t w = {"", ""};
    if (!PyArg_ParseTuple(args, "Odiddi:route_lin", &bb, &a, &model, &c, &cbar, &engine)) return NULL;
    const double *b = in_doubles(bb, "b", &n, &bc);
    if (b == NULL || (out = scratch(n, sizeof(double))) == NULL) goto done;
    WARN_CALL(rc = dfl_engine_lincdf(n, b, a, model, c, cbar, engine, out, err, sizeof err), &w);
    if (check(rc, err)) goto done;
    res = Py_BuildValue("(Nss)", out_bytes(out, n, sizeof(double)), w.det, w.cls);
done:
    PyMem_Free(out); PyMem_Free(bc);
    return res;
}
/* route_env(c, model, alpha, engine) -> (bytearray, general warning, classical warning): dfl_engine_envelope */
static PyObject *py_route_env(PyObject *self, PyObject *args)
{
    PyObject *cb, *res = NULL;
    int model, engine, n = 0, rc = 0;
    double alpha, *cc = NULL, *out = NULL;
    char err[ERRLEN] = "";
    warn_t w = {"", ""};
    if (!PyArg_ParseTuple(args, "Oidi:route_env", &cb, &model, &alpha, &engine)) return NULL;
    const double *cs = in_doubles(cb, "c", &n, &cc);
    if (cs == NULL || (out = scratch(n, sizeof(double))) == NULL) goto done;
    WARN_CALL(rc = dfl_engine_envelope(n, cs, model, alpha, engine, out, err, sizeof err), &w);
    if (check(rc, err)) goto done;
    res = Py_BuildValue("(Nss)", out_bytes(out, n, sizeof(double)), w.det, w.cls);
done:
    PyMem_Free(out); PyMem_Free(cc);
    return res;
}
/* ---- the devices of Bao and Kan (dfl_inv_opts: the single inversion, the moments from jets) ---- */
enum { RX_MOM = 1, RX_COEF, RX_COEF_ICDF, RX_COEF_PDF, RX_COEF_MOM, RX_LIN, RX_ENV };
/* route_x(op, spec, x, c, c0, engine, imethod, tol, verified, p0, p1, p2) -> (bytearray out, bytearray errest, general
 * warning, classical warning): the functions of the router with the options of the single inversion (imethod 0 automatic,
 * 5 forces the devices, 6 the engines' computation of before, 1 to 4 the representations of the t-ratio for the
 * coefficient statistic; tol NaN for the default; errest NaN where the engines computed):
 *   RX_MOM, RX_COEF_MOM   the moments (x one value, their number)        RX_COEF, RX_COEF_PDF   the coefficient statistic
 *   RX_COEF_ICDF          its quantiles, p0 the width of the bracket     RX_LIN  P(U - aV <= b), p0 = a, p1 = cbar (case
 *   code: spec without a design, its model; a design with its start)  RX_ENV  the envelope, p0 = model, p1 = alpha */
static PyObject *py_route_x(PyObject *self, PyObject *args)
{
    PyObject *st, *xb, *res = NULL;
    int op, engine, imethod, verified, n = 0, rc = 0, nout;
    double c, c0, tol, p0 = NAN, p1 = NAN, p2 = NAN, *xc = NULL, *out = NULL, *eo = NULL;
    spec_arg s;
    char err[ERRLEN] = "";
    warn_t w = {"", ""};
    if (!PyArg_ParseTuple(args, "iOOddiidi|ddd:route_x", &op, &st, &xb, &c, &c0, &engine, &imethod, &tol, &verified, &p0, &p1, &p2)) return NULL;
    if (spec_in(st, &s)) return NULL;
    const double *x = in_doubles(xb, "x", &n, &xc);
    if (x == NULL) goto done;
    nout = n;
    if (op == RX_MOM || op == RX_COEF_MOM) {
        if (n != 1 || !(x[0] >= 1 && x[0] <= INT_MAX / 2 && x[0] == floor(x[0]))) { PyErr_SetString(PyExc_ValueError, "p must be a positive integer"); goto done; }
        nout = (int) x[0];
    }
    if ((out = scratch(nout, sizeof(double))) == NULL || (eo = scratch(nout, sizeof(double))) == NULL) goto done;
    for (int k = 0; k < nout; k++) eo[k] = NAN;
    dfl_inv_opts io; dfl_inv_defaults(&io); io.method = imethod; io.tol = tol; io.verified = verified != 0;
    switch (op) {
    case RX_MOM: WARN_CALL(rc = dfl_engine_mom_x(nout, &s.S, c, c0, engine, &io, out, eo, err, sizeof err), &w); break;
    case RX_COEF_MOM: WARN_CALL(rc = dfl_engine_coef_mom(nout, &s.S, c, c0, engine, &io, out, eo, err, sizeof err), &w); break;
    case RX_COEF: WARN_CALL(rc = dfl_engine_coef_x(n, x, &s.S, c, c0, engine, &io, out, eo, err, sizeof err), &w); break;
    case RX_COEF_PDF: WARN_CALL(rc = dfl_engine_coef_pdf(n, x, &s.S, c, c0, engine, &io, out, eo, err, sizeof err), &w); break;
    case RX_COEF_ICDF: WARN_CALL(rc = dfl_engine_coef_icdf_x(n, x, &s.S, c, c0, engine, p0, &io, out, eo, err, sizeof err), &w); break;
    case RX_LIN: WARN_CALL(rc = dfl_engine_lincdf_x(n, x, p0, &s.S, c, p1, engine, &io, out, eo, err, sizeof err), &w); break;
    case RX_ENV: WARN_CALL(rc = dfl_engine_envelope_x(n, x, ipar(p0), p1, engine, &io, out, eo, err, sizeof err), &w); break;
    default: PyErr_Format(PyExc_ValueError, "unknown operation %d of the router", op); goto done;
    }
    if (check(rc, err)) goto done;
    res = Py_BuildValue("(NNss)", out_bytes(out, nout, sizeof(double)), out_bytes(eo, nout, sizeof(double)), w.det, w.cls);
done:
    PyMem_Free(out); PyMem_Free(eo); PyMem_Free(xc); spec_release(&s);
    return res;
}
/* det_x(op, design, x, c, opts, method, N, umax, tol, p0, p1) -> (bytearray out, bytearray errest, warning): the
 * point-optimal statistic (DET_PT_CDF, DET_PT_PDF with p0 = cbar; DET_PT_Q with p0 = cbar; DET_PT_POW with p0 = cbar,
 * p1 = level, out the critical value and the power at c; DET_PT_ENV with p0 = level) and DET_COEF_GP, with the method of
 * dfl_det_pt_x (0 automatic, 5 the single inversion of Bao and Kan, 7 the fixed grid; N, umax 0 for the grid's defaults,
 * given > 0 they choose the grid; tol NaN for 1e-12); errest NaN on the grid (DET_PT_POW, DET_PT_ENV: none) */
static PyObject *py_det_x(PyObject *self, PyObject *args)
{
    PyObject *dt, *xb, *ot, *res = NULL;
    int op, method, N, n = 0, rc = 0;
    double c, umax, tol, p0 = NAN, p1 = NAN, *xc = NULL, *out = NULL, *eo = NULL;
    design_arg a;
    dfl_det_opts o;
    char err[ERRLEN] = "";
    warn_t w = {"", ""};
    if (!PyArg_ParseTuple(args, "iOOdOiidd|dd:det_x", &op, &dt, &xb, &c, &ot, &method, &N, &umax, &tol, &p0, &p1)) return NULL;
    if (detopts_in(ot, &o)) return NULL;
    if (design_in(dt, &a)) return NULL;
    const double *x = in_doubles(xb, "x", &n, &xc);
    if (x == NULL || (out = scratch(n + 2, sizeof(double))) == NULL || (eo = scratch(n + 2, sizeof(double))) == NULL) goto done;
    for (int k = 0; k < n + 2; k++) eo[k] = NAN;
    int nout = op == DET_PT_POW ? 2 : n;
    switch (op) {
    case DET_PT_CDF: case DET_PT_PDF: WARN_CALL(rc = dfl_det_pt_x(op == DET_PT_CDF ? 2 : 1, &a.D, n, x, p0, c, method, N, umax, tol, out, eo, err, sizeof err), &w); break;
    case DET_PT_Q:
        ENGINE_CALL(warn_reset(); for (int k = 0; k < n && !rc; k++) { rc = dfl_det_pt_quantile_x(&a.D, x[k], p0, method, N, umax, tol, &out[k], &eo[k], err, sizeof err); warn_take(&w); });
        break;
    case DET_PT_POW: WARN_CALL(rc = dfl_det_pt_power_x(&a.D, p0, c, p1, method, N, umax, tol, &out[0], &out[1], err, sizeof err), &w); break;
    case DET_PT_ENV: WARN_CALL(rc = dfl_det_pt_envelope_x(&a.D, n, x, p0, method, N, umax, tol, out, err, sizeof err), &w); break;
    case DET_COEF_GP: WARN_CALL(rc = dfl_det_coef_cdf_gp_x(&a.D, n, x, c, &o, method, N, umax, tol, out, eo, err, sizeof err), &w); break;
    default: PyErr_Format(PyExc_ValueError, "unknown operation %d of det_x", op); goto done;
    }
    if (check(rc, err)) goto done;
    res = Py_BuildValue("(NNs)", out_bytes(out, nout, sizeof(double)), out_bytes(eo, nout, sizeof(double)), w.det);
done:
    PyMem_Free(out); PyMem_Free(eo); PyMem_Free(xc); design_release(&a);
    return res;
}
/* autoengine(f, model, stat, g0, gls, c, c0, x) -> bytearray of C ints (1: the general engine) at the points x:
 * dfl_autoengine */
static PyObject *py_autoengine(PyObject *self, PyObject *args)
{
    PyObject *xb, *res = NULL;
    int f, model, stat, n = 0, *out = NULL;
    double g0, gls, c, c0, *xc = NULL;
    if (!PyArg_ParseTuple(args, "iiiddddO:autoengine", &f, &model, &stat, &g0, &gls, &c, &c0, &xb)) return NULL;
    const double *x = in_doubles(xb, "x", &n, &xc);
    if (x == NULL || (out = scratch(n, sizeof(int))) == NULL) goto done;
    for (int k = 0; k < n; k++) out[k] = dfl_autoengine(f, model, stat, g0, gls, c, c0, x[k]) == DFL_ENGINE_GENERAL;
    res = out_bytes(out, n, sizeof(int));
done:
    PyMem_Free(out); PyMem_Free(xc);
    return res;
}
/* automethod(model, c, c0, g0, stat, x) -> bytearray of C ints: the classical engine's automatic method at the points x
 * (1 the line, 2 the conditional, 3 the contour integral): dfl_automethod */
static PyObject *py_automethod(PyObject *self, PyObject *args)
{
    PyObject *xb, *res = NULL;
    int model, stat, n = 0, *out = NULL;
    double c, c0, g0, *xc = NULL;
    if (!PyArg_ParseTuple(args, "idddiO:automethod", &model, &c, &c0, &g0, &stat, &xb)) return NULL;
    const double *x = in_doubles(xb, "x", &n, &xc);
    if (x == NULL || (out = scratch(n, sizeof(int))) == NULL) goto done;
    for (int k = 0; k < n; k++) out[k] = dfl_automethod(model, c, c0, g0, stat != 0, x[k]);
    res = out_bytes(out, n, sizeof(int));
done:
    PyMem_Free(out); PyMem_Free(xc);
    return res;
}
/* check_args(model, c, c0, g0, stat) -> "" or the message of the classical engine's checks: dfl_check_args */
static PyObject *py_check_args(PyObject *self, PyObject *args)
{
    int model, stat;
    double c, c0, g0;
    char err[ERRLEN] = "";
    if (!PyArg_ParseTuple(args, "idddi:check_args", &model, &c, &c0, &g0, &stat)) return NULL;
    int rc = dfl_check_args(model, c, c0, g0, stat != 0, err, sizeof err);
    return PyUnicode_FromString(rc ? (err[0] ? err : "the classical engine refuses the arguments") : "");
}
/* brittle(kind, model, c, c0, g0, stat, method) -> "" or the message of a regime in which one evaluation of the classical
 * engine is brittle: dfl_brittle */
static PyObject *py_brittle(PyObject *self, PyObject *args)
{
    int kind, model, stat, method;
    double c, c0, g0;
    if (!PyArg_ParseTuple(args, "iidddii:brittle", &kind, &model, &c, &c0, &g0, &stat, &method)) return NULL;
    const char *bm = dfl_brittle(kind, model, c, c0, g0, stat != 0, method);
    return PyUnicode_FromString(bm ? bm : "");
}

static PyMethodDef METHODS[] = {
    {"dist", py_dist, METH_VARARGS, "dist(kind, x, model, c, c0, g0, stat, method) -> (out, nstatus, brittle, warning): dfl_dist"},
    {"icdf", py_icdf, METH_VARARGS, "icdf(p, model, c, c0, g0, stat, method, tol) -> (q, F, iter, conv, brittle, warning): dfl_icdf"},
    {"moments", py_moments, METH_VARARGS, "moments(pmax, model, c, c0, g0, stat) -> (m, err): dfl_moments"},
    {"adfmax", py_adfmax, METH_VARARGS, "adfmax(kind, x, model, c, stat) -> (value, err, status): dfl_adfmax"},
    {"region", py_region, METH_VARARGS, "region(type, p1, p2, v0, brks, c) -> (P, err, status): dfl_region"},
    {"lincdf", py_lincdf, METH_VARARGS, "lincdf(b, a, model, c, cbar) -> P: dfl_lincdf"},
    {"envelope", py_envelope, METH_VARARGS, "envelope(c, model, alpha) -> the power envelope: dfl_power_envelope"},
    {"threads", py_threads, METH_VARARGS, "threads(n=None) -> the number of worker threads: dfl_nworkers"},
    {"design_trend", py_design_trend, METH_VARARGS, "design_trend(p, legendre=0) -> design: dfl_design_trend, dfl_design_ltrend"},
    {"design_perron", py_design_perron, METH_VARARGS, "design_perron(model, lam) -> design: dfl_design_perron"},
    {"design_first", py_design_first, METH_VARARGS, "design_first(design) -> design: dfl_design_detrend_first"},
    {"design_matrix", py_design_matrix, METH_VARARGS, "design_matrix(X, n, K, TB, tol, maxdeg, minlen) -> (design, warning): dfl_design_from_matrix"},
    {"design_fdesign", py_design_fdesign, METH_VARARGS, "design_fdesign(f, K, breaks, tol, maxdeg, sub) -> design: dfl_design_fdesign"},
    {"design_value", py_design_value, METH_VARARGS, "design_value(design, r) -> the regressors at r: dfl_design_value"},
    {"design_regmat", py_design_regmat, METH_VARARGS, "design_regmat(design, n) -> the regressor matrix: dfl_design_regmat"},
    {"det", py_det, METH_VARARGS, "det(op, design, x, c, opts, p0, p1, p2, p3) -> (out, warning): the general engine"},
    {"det_moments", py_det_moments, METH_VARARGS, "det_moments(design, c, pmax, opts) -> (m, err, warning): dfl_det_moments"},
    {"det_empty", py_det_empty, METH_VARARGS, "det_empty() -> None: dfl_det_empty_caches"},
    {"route", py_route, METH_VARARGS, "route(op, spec, x, c, c0, engine, method, verified, tol) -> (out, warning, warning): dfl_engine_*"},
    {"route_lin", py_route_lin, METH_VARARGS, "route_lin(b, a, model, c, cbar, engine) -> (out, warning, warning): dfl_engine_lincdf"},
    {"route_env", py_route_env, METH_VARARGS, "route_env(c, model, alpha, engine) -> (out, warning, warning): dfl_engine_envelope"},
    {"route_x", py_route_x, METH_VARARGS, "route_x(op, spec, x, c, c0, engine, imethod, tol, verified, p0, p1, p2) -> (out, errest, warning, warning): the devices of Bao and Kan through the router"},
    {"det_x", py_det_x, METH_VARARGS, "det_x(op, design, x, c, opts, method, N, umax, tol, p0, p1) -> (out, errest, warning): the point-optimal statistic and coef_cdf_gp with the single inversion"},
    {"autoengine", py_autoengine, METH_VARARGS, "autoengine(f, model, stat, g0, gls, c, c0, x) -> engines (1 general): dfl_autoengine"},
    {"automethod", py_automethod, METH_VARARGS, "automethod(model, c, c0, g0, stat, x) -> methods: dfl_automethod"},
    {"check_args", py_check_args, METH_VARARGS, "check_args(model, c, c0, g0, stat) -> message or '': dfl_check_args"},
    {"brittle", py_brittle, METH_VARARGS, "brittle(kind, model, c, c0, g0, stat, method) -> message or '': dfl_brittle"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef MODULE = {
    PyModuleDef_HEAD_INIT, "_engine",
    "The C engine of dflimit (private: use the functions of the package dflimit).", -1, METHODS,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__engine(void)
{
    return PyModule_Create(&MODULE);
}
