/* _engine.c: the Python gateway to the C engine (the files of csrc/ are copied from DF-Limit's c/ by
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
 * a line in METHODS, and the user-level function in the Python layer. */
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
 * ""): dfl_dist, dfl_brittle */
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
    const char *bm = verified ? NULL : dfl_brittle(kind, model, c, c0, g0, stat != 0, method);
    res = Py_BuildValue("(Nis)", out_bytes(out, n, sizeof(double)), ns, bm ? bm : "");
done:
    PyMem_Free(out); PyMem_Free(xc);
    return res;
}

/* icdf(p, model, c, c0, g0, stat, method, tol, verified) -> (q, F, iter, conv) as bytearrays (iter, conv: C int) and the
 * brittle-regime message: dfl_icdf */
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
    const char *bm = verified ? NULL : dfl_brittle(2, model, c, c0, g0, stat != 0, method);
    res = Py_BuildValue("(NNNNs)", out_bytes(q, n, sizeof(double)), out_bytes(F, n, sizeof(double)),
                        out_bytes(it, n, sizeof(int)), out_bytes(cv, n, sizeof(int)), bm ? bm : "");
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

static PyMethodDef METHODS[] = {
    {"dist", py_dist, METH_VARARGS, "dist(kind, x, model, c, c0, g0, stat, method) -> (out, nstatus): dfl_dist"},
    {"icdf", py_icdf, METH_VARARGS, "icdf(p, model, c, c0, g0, stat, method, tol) -> (q, F, iter, conv): dfl_icdf"},
    {"moments", py_moments, METH_VARARGS, "moments(pmax, model, c, c0, g0, stat) -> (m, err): dfl_moments"},
    {"adfmax", py_adfmax, METH_VARARGS, "adfmax(kind, x, model, c, stat) -> (value, err, status): dfl_adfmax"},
    {"region", py_region, METH_VARARGS, "region(type, p1, p2, v0, brks, c) -> (P, err, status): dfl_region"},
    {"lincdf", py_lincdf, METH_VARARGS, "lincdf(b, a, model, c, cbar) -> P: dfl_lincdf"},
    {"envelope", py_envelope, METH_VARARGS, "envelope(c, model, alpha) -> the power envelope: dfl_power_envelope"},
    {"threads", py_threads, METH_VARARGS, "threads(n=None) -> the number of worker threads: dfl_nworkers"},
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
