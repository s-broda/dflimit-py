/* dfl_api.c: the user-level entry points of the C engine, with the argument conventions of the MATLAB programs:
 * dfl_single (the single integral of the line integral representation for a vector of x, the general kernel when
 * c0 != 0), dfl_dist (the
 * automatic choice between the single, the double and the contour integral at each x, as ltratio_cdf/ltratio_pdf with
 * method 'auto'), dfl_icdf (quantiles by safeguarded Newton from the normal approximation), and the validation shared
 * by all. */
#include "dflimit.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


int dfl_check_args(int model, double c, double c0, double g0, bool stat, char *err, size_t errlen)
{
    if (model < 1 || model > 4) { snprintf(err, errlen, "model must be 1, 2, 3 or 4 (no deterministic terms, intercept, intercept and trend, GLS-detrended trend)"); return 1; }
    if (model == 4 && stat) { snprintf(err, errlen, "the GLS-detrended case (model 4) has a zero start only"); return 1; }
    if (model == 4 && g0 != 0) { snprintf(err, errlen, "a fixed start g0 applies to models 1, 2 and 3 only"); return 1; }
    if (!isfinite(c)) { snprintf(err, errlen, "c must be a finite real scalar"); return 1; }
    if (!isfinite(c0)) { snprintf(err, errlen, "c0 must be a finite real scalar"); return 1; }
    if (!isfinite(g0)) { snprintf(err, errlen, "g0 must be a finite real scalar"); return 1; }
    if (stat && !(c <= DFL_CSTAT)) { snprintf(err, errlen, "%s", DFL_STAT_MSG); return 1; }
    if (stat && g0 != 0) { snprintf(err, errlen, "g0 (a fixed start) applies to cases DK, DU, DT only"); return 1; }
    if (c < DFL_CMIN) { snprintf(err, errlen, "%s", DFL_CMIN_MSG); return 1; }
    if (g0 != 0 && !stat && !dfl_fixed_ok(model, c, g0)) { snprintf(err, errlen, "%s", DFL_FIXED_MSG); return 1; }
    return 0;
}
static int check_args(int kind, int model, double c, double c0, double g0, bool stat, char *err, size_t errlen)
{
    if (kind != 1 && kind != 2) { snprintf(err, errlen, "kind must be 1 (density) or 2 (distribution function)"); return 1; }
    return dfl_check_args(model, c, c0, g0, stat, err, errlen);
}
/* t_* where the single integral uses it (its line for c > 0 or a stationary start): an error, not NaN, where no zero of
 * D(0, t) is found */
static int tstar_err(int model, double c, bool stat, char *err, size_t errlen)
{
    if ((stat || c > 0) && !isfinite(dfl_tstar(model, c, stat))) { snprintf(err, errlen, "no zero of D(0, t) was found (t_*), c = %g", c); return 1; }
    return 0;
}
/* the value at a non-finite x: 0 at -Inf, 1 (cdf) or 0 (density) at Inf, NaN at NaN */
static double nonfinite(int kind, double x) { return isnan(x) ? NAN : x > 0 ? (kind == 2 ? 1.0 : 0.0) : 0.0; }

int dfl_single(int kind, int n, const double *x, int model, double c, double c0, double g0, bool stat, const dfl_single_opts *o, double *out, char *err, size_t errlen)
{
    dfl_single_opts d; if (o == NULL) { dfl_single_defaults(&d); o = &d; }
    if (check_args(kind, model, c, c0, g0, stat, err, errlen)) return 1;
    if (tstar_err(model, c, stat, err, errlen)) return 1;
    for (int k = 0; k < n; k++) {
        if (!isfinite(x[k])) { out[k] = nonfinite(kind, x[k]); continue; }
        out[k] = (c0 != 0 || g0 != 0 || o->gen) ? dfl_single_value_gen(kind, x[k], model, c, c0, g0, stat, o) : dfl_single_value_c0(kind, x[k], model, c, stat, o);
    }
    return 0;
}

/* the method of method 0 (automatic) with verified = 1 (the rule before November 2026): with a fixed start g0 != 0 in
 * models 2, 3 the contour integral (3); the single integral at c0 = 0 and with a fixed start in model 1; with c0 != 0 and a
 * zero start the double integral, except for c0 > 0 at x >= 0, where X(v) has no positive root and the single integral has
 * no breakpoint; where dfl_autodouble excludes the double integral (a stationary start at c < -50, c0 != 0 with a zero
 * start at c < -100) the single integral alone (dfl_dist then recomputes the tails and replaces flagged values) */
static int automethod_verified(int model, double c, double c0, double g0, bool stat, double x)
{
    if ((model == 2 || model == 3) && g0 != 0 && !stat) return 3;            /* a fixed start in cases DU, DT: the contour integral */
    if (model == 4) return 1;                                                /* the GLS-detrended case: the single integral only */
    if (!dfl_autodouble(c, c0, g0, stat)) return 1;
    if (model == 1 && c0 > 0 && g0 == 0 && fabs(x + sqrt(2 * c0)) <= 1e-12 * fmax(1.0, fabs(x))) return 1;   /* the infinite density of case DK: the single integral returns Inf there without a flag */
    return (c0 != 0 && g0 == 0 && !(c0 > 0 && x >= 0)) ? 2 : 1;
}
/* the rule of the one-evaluation default (automethod of DFExact.jl, November 2026): the fastest representation that one
 * evaluation gives to well below 1e-5 (1e-8 on the grid of the rule).  With a fixed start the contour integral in cases DU,
 * DT, except for c > 1 with c0 < 0, where its far upper tail fails (the conditional integral there), and the line integral
 * in case DK; at c0 = 0 the line integral; with c0 != 0 the conditional integral (one path) where its one evaluation holds,
 * c0 < 0 with a zero start (cases DU, DT; case DK for c <= -5) and a stationary start in case SK for -50 <= c <= -0.5,
 * the line integral elsewhere (here, by the timings of this engine, also with a stationary start in cases SU, ST, where it
 * takes 2.9-3.6 ms against 16-17 ms on ten threads at the rows of c/mex/README.md; the Julia engine takes the conditional integral there) (a zero start with c0 > 0: the conditional integral off by up to 1.5e-2 at x < 0;
 * case DK with c0 < 0 for c > -5; beyond dfl_autodouble).  (x is not used: the rule depends on the parameters alone.) */
/* where the contour integral fails with a fixed start (cases DU, DT): c > 1 with c0 < 0 (the far upper tail) and, in case
 * DU, c > 1.5 with c0 > 3 (outright at c = 2 for c0 >= 5, g0 >= 1); the conditional integral there (ctr_fails of DFExact.jl) */
static bool ctr_fails(int model, double c, double c0) { return (c > 1 && c0 < 0) || (model == 2 && c > 1.5 && c0 > 3); }
static int automethod_one(int model, double c, double c0, double g0, bool stat, double x)
{
    if (g0 != 0) return (model == 2 || model == 3) && !stat ? (ctr_fails(model, c, c0) ? 2 : 3) : 1;
    if (model == 4 || c0 == 0 || !dfl_autodouble(c, c0, g0, stat)) return 1;
    if (stat) return (model == 1 && c <= -0.5) ? 2 : 1;          /* (cases SU, ST: the line integral, here as fast as the conditional or faster) */
    if (model == 1 && c > -5) return 1;
    if (c0 > 0) return (c >= 0 && x < 0) ? 2 : 1;               /* (a zero start with c0 > 0: the conditional integral at x < 0 for c >= 0) */
    return 2;
}
int dfl_automethod(int model, double c, double c0, double g0, bool stat, double x)
{
    return automethod_one(model, c, c0, g0, stat, x);
}

/* the method: 0..3, the double integral not in model 4, the contour integral for a fixed start in models 2, 3 only */
static int check_method(int model, double g0, bool stat, int method, char *err, size_t errlen)
{
    if (method < 0 || method > 3) { snprintf(err, errlen, "method must be 0 (automatic), 1 (single integral), 2 (double integral) or 3 (contour integral)"); return 1; }
    if (model == 4 && method == 2) { snprintf(err, errlen, "the GLS-detrended case (model 4) has the single integral only"); return 1; }
    if (method == 3 && !((model == 2 || model == 3) && g0 != 0 && !stat)) { snprintf(err, errlen, "%s", DFL_CONTOUR_MSG); return 1; }
    return 0;
}
static double single_value(int kind, double x, int model, double c, double c0, double g0, bool stat, const dfl_single_opts *so)
{
    return (c0 != 0 || g0 != 0 || so->gen) ? dfl_single_value_gen(kind, x, model, c, c0, g0, stat, so) : dfl_single_value_c0(kind, x, model, c, stat, so);
}

int dfl_dist(int kind, int n, const double *x, int model, double c, double c0, double g0, bool stat, int method,
             const dfl_single_opts *so, const dfl_double_opts *dopt, double *out, int *nstatus, char *err, size_t errlen)
{
    if (check_args(kind, model, c, c0, g0, stat, err, errlen)) return 1;
    if (check_method(model, g0, stat, method, err, errlen)) return 1;
    if (method != 2 && tstar_err(model, c, stat, err, errlen)) return 1;      /* (dfl_double checks its own) */
    dfl_single_opts sd; if (so == NULL) { dfl_single_defaults(&sd); so = &sd; }
    dfl_double_opts dd; if (dopt == NULL) { dfl_double_defaults(&dd); dopt = &dd; }
    if (nstatus) *nstatus = 0;
    if (!dopt->verified) {
        /* the default: one evaluation by one method at each point (no second evaluation, no refinement driven by an
         * estimate, no fallback, no recomputation of the tails); no status (ltratio_dist of DFExact.jl) */
        int nd1 = 0, nc1 = 0, rc1 = 0;
        double *xd1 = malloc((n + 1) * sizeof(double)), *xc1 = malloc((n + 1) * sizeof(double));
        int *id1 = malloc((n + 1) * sizeof(int)), *ic1 = malloc((n + 1) * sizeof(int));
        for (int k = 0; k < n; k++) {
            if (!isfinite(x[k])) { out[k] = nonfinite(kind, x[k]); continue; }
            int m = method == 0 ? automethod_one(model, c, c0, g0, stat, x[k]) : method;
            if (m == 2) { xd1[nd1] = x[k]; id1[nd1++] = k; }
            else if (m == 3) { xc1[nc1] = x[k]; ic1[nc1++] = k; }
            else out[k] = single_value(kind, x[k], model, c, c0, g0, stat, so);
        }
        if (nc1 > 0) {
            double *F1 = malloc(nc1 * sizeof(double));
            rc1 = dfl_contour(kind, nc1, xc1, model, c, c0, g0, 0, F1, NULL, NULL, NULL, err, errlen);
            if (rc1 == 0) for (int j = 0; j < nc1; j++) out[ic1[j]] = F1[j];
            free(F1);
        }
        if (rc1 == 0 && nd1 > 0) {
            double *F1 = malloc(nd1 * sizeof(double)); dfl_double_info *in1 = malloc(nd1 * sizeof(dfl_double_info));
            rc1 = dfl_double(kind, nd1, xd1, model, c, c0, g0, stat, dopt, F1, in1, err, errlen);
            if (rc1 == 0) for (int j = 0; j < nd1; j++) out[id1[j]] = F1[j];
            free(F1); free(in1);
        }
        if (kind == 1) for (int k = 0; k < n; k++) if (out[k] < 0) out[k] = 0;   /* a density below the error of the method */
        free(xd1); free(xc1); free(id1); free(ic1);
        return rc1;
    }
    /* verified: the error estimates and the status, the fallbacks and the recomputation of the tails (the behaviour before
     * the default became one evaluation, November 2026).  The points for the double integral (xd, id; tosingle: a flagged value is replaced by the single integral's, else the
     * value computed before stays) and for the contour integral (xc, ic); meth the method chosen at each point */
    int nd = 0, nc = 0, rc = 0;
    double *xd = malloc((n + 1) * sizeof(double)), *xc = malloc((n + 1) * sizeof(double));
    int *id = malloc((n + 1) * sizeof(int)), *ic = malloc((n + 1) * sizeof(int)), *tosingle = malloc((n + 1) * sizeof(int)), *meth = malloc((n + 1) * sizeof(int));
    char *flag = calloc(n + 1, 1);
    for (int k = 0; k < n; k++) {
        meth[k] = 0;
        if (!isfinite(x[k])) { out[k] = nonfinite(kind, x[k]); continue; }
        int m = method == 0 ? automethod_verified(model, c, c0, g0, stat, x[k]) : method;
        meth[k] = m;
        if (m == 2) { xd[nd] = x[k]; id[nd] = k; tosingle[nd] = 1; nd++; }
        else if (m == 3) { xc[nc] = x[k]; ic[nc] = k; nc++; }
        else out[k] = single_value(kind, x[k], model, c, c0, g0, stat, so);
    }
    /* the contour integral; with method 0 a point at which it reports that its error estimate is not met gets the double
     * integral's value (the single integral's where that is flagged too), and its values below 1e-3 are recomputed by
     * the double integral as those of the single integral are below (ltratio_dist in DFExact.jl) */
    if (nc > 0) {
        double *Fc = malloc(nc * sizeof(double)); int *sc = malloc(nc * sizeof(int));
        rc = dfl_contour(kind, nc, xc, model, c, c0, g0, 1, Fc, NULL, sc, NULL, err, errlen);
        if (rc == 0) for (int j = 0; j < nc; j++) {
            int k = ic[j]; out[k] = Fc[j];
            if (sc[j]) flag[k] = 1;
            if (method == 0) {
                if (sc[j] != 0) { xd[nd] = x[k]; id[nd] = k; tosingle[nd] = 1; nd++; }
                else if ((kind == 1 ? fabs(Fc[j]) : Fc[j]) < 1e-3) { xd[nd] = x[k]; id[nd] = k; tosingle[nd] = 0; nd++; }
            }
        }
        free(Fc); free(sc);
    }
    /* the tails: the single integral's error is absolute (1e-14 to 1e-21, not reduced by finer rules), so with method 0
     * the values of the single integral below 1e-3 (a density in absolute value, a distribution function in the lower
     * tail) are recomputed by the double integral, which keeps its relative accuracy there (as the MATLAB programs do) */
    if (rc == 0 && method == 0 && model != 4 && dfl_autodouble(c, c0, g0, stat)) for (int k = 0; k < n; k++) {   /* (case DG: the single integral only, no tails) */
        if (meth[k] == 1 && (kind == 1 ? fabs(out[k]) : out[k]) < 1e-3) { xd[nd] = x[k]; id[nd] = k; tosingle[nd] = 0; nd++; }
    }
    if (rc == 0 && nd > 0) {
        double *Fd = malloc(nd * sizeof(double)); dfl_double_info *info = malloc(nd * sizeof(dfl_double_info));
        rc = dfl_double(kind, nd, xd, model, c, c0, g0, stat, dopt, Fd, info, err, errlen);
        if (rc == 0) for (int j = 0; j < nd; j++) {
            int k = id[j];
            if (info[j].status == 0 || method == 2) out[k] = Fd[j];
            else if (tosingle[j]) out[k] = single_value(kind, x[k], model, c, c0, g0, stat, so);   /* a flagged point chosen by the rule or flagged by the contour integral */
            /* (a flagged tail point keeps the value of the single or the contour integral) */
            if (info[j].status) flag[k] = 1;
        }
        free(Fd); free(info);
    }
    if (nstatus) for (int k = 0; k < n; k++) *nstatus += flag[k];
    if (kind == 1) for (int k = 0; k < n; k++) if (out[k] < 0) out[k] = 0;   /* a density below the error of the method */
    free(xd); free(xc); free(id); free(ic); free(tosingle); free(meth); free(flag);
    return rc;
}

/* the brittle regimes of the one-evaluation default (brittle_warn of julia/DFExact.jl, ltratio_brittle.m) */
const char *dfl_brittle(int kind, int model, double c, double c0, double g0, bool stat, int method)
{
    (void) kind;
    static _Thread_local char msg[320];
    /* (x = -1: the rule depends on x only for a zero start with c0 > 0 at c >= 0 in cases DU, DT, where no regime of the
     * conditional integral is brittle) */
    int m = method == 0 ? automethod_one(model, c, c0, g0, stat, -1.0) : method;
    const char *cas = model == 4 ? "DG" : (stat ? (model == 1 ? "SK" : model == 2 ? "SU" : "ST") : (model == 1 ? "DK" : model == 2 ? "DU" : "DT"));
    const char *hint = "; verified = true checks each value";
    if (m == 3 && c0 > 0 && c < -15) {
        snprintf(msg, sizeof msg, "%s, fixed start, c0 > 0 far from the unit root (c < -15): the contour integral by one evaluation may be off by up to about 1e-5%s", cas, hint);
        return msg;
    }
    if (m == 3 && c > 1 && c0 < 0) {
        snprintf(msg, sizeof msg, "%s, fixed start, c0 < 0 at c > 1: the contour integral by one evaluation fails in the far upper tail%s", cas, hint);
        return msg;
    }
    if (m == 3 && model == 2 && c > 1.5 && c0 > 3) {
        snprintf(msg, sizeof msg, "%s, fixed start, c0 > 3 at c > 1.5: the contour integral by one evaluation fails (the automatic method takes the conditional integral there)%s", cas, hint);
        return msg;
    }
    if (m == 2) {
        const char *reg = g0 != 0 ? ((model == 1 && c < -20) ? "fixed start far from the unit root (c < -20)" :
                                     (model != 1 && c < -30) ? "fixed start far from the unit root (c < -30)" : NULL) :
            (!stat && c0 > 0 && c < 0) ? "c0 > 0 below the unit root" :
            (!stat && model == 1 && c0 != 0 && c > -5) ? "case DK with c0 != 0 near or above the unit root (c > -5)" :
            (!stat && c < -100) ? "c < -100" :
            (stat && c < -50) ? "a stationary start with c < -50" :
            (stat && model == 1 && c > -0.5) ? "case SK near the unit root (c > -0.5)" : NULL;
        if (reg) { snprintf(msg, sizeof msg, "%s, %s: the conditional integral by one evaluation may be off by more than 1e-5%s", cas, reg, hint); return msg; }
    }
    return NULL;
}

/* the inverse error function: Giles' approximation refined by two Newton steps on erf */
static double erfinv(double y)
{
    if (!(y > -1 && y < 1)) return y <= -1 ? -INFINITY : y >= 1 ? INFINITY : NAN;
    double w = -log((1 - y) * (1 + y)), p;
    if (w < 5) { w -= 2.5; p = 2.81022636e-08; p = 3.43273939e-07 + p * w; p = -3.5233877e-06 + p * w; p = -4.39150654e-06 + p * w; p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w; p = -0.00417768164 + p * w; p = 0.246640727 + p * w; p = 1.50140941 + p * w; }
    else { w = sqrt(w) - 3; p = -0.000200214257; p = 0.000100950558 + p * w; p = 0.00134934322 + p * w; p = -0.00367342844 + p * w; p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w; p = 0.00943887047 + p * w; p = 1.00167406 + p * w; p = 2.83297682 + p * w; }
    double x = p * y;
    for (int it = 0; it < 2; it++) x -= (erf(x) - y) / (2 / sqrt(M_PI) * exp(-x * x));
    return x;
}

int dfl_icdf(int n, const double *p, int model, double c, double c0, double g0, bool stat, int method, double tol,
             const dfl_single_opts *so, const dfl_double_opts *dopt, double *q, double *Fq, int *iter, int *conv, char *err, size_t errlen)
{
    if (check_args(2, model, c, c0, g0, stat, err, errlen)) return 1;
    if (check_method(model, g0, stat, method, err, errlen)) return 1;
    for (int k = 0; k < n; k++) if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "p must lie in (0, 1)"); return 1; }
    double m[2], me[2], mu, sd;
    if (model == 4) { mu = -1.9 + 0.1 * fmin(c, 0.0); sd = 0.8; }   /* (no moments for the GLS-detrended case: a rough start, the search is safeguarded) */
    else {
        if (dfl_moments(model, c, c0, g0, stat, 2, m, me, err, errlen)) return 1;
        mu = m[0]; sd = sqrt(fmax(m[1] - m[0] * m[0], 1e-6));
    }
    if (!(tol > 0)) tol = 1e-10;
#define CDF(xx, out) do { if (dfl_dist(2, 1, &(xx), model, c, c0, g0, stat, method, so, dopt, &(out), NULL, err, errlen)) return 1; } while (0)
#define PDF(xx, out) do { if (dfl_dist(1, 1, &(xx), model, c, c0, g0, stat, method, so, dopt, &(out), NULL, err, errlen)) return 1; } while (0)
    for (int k = 0; k < n; k++) {
        double pk = p[k], x = mu + sd * sqrt(2) * erfinv(2 * pk - 1), lo = x - sd, hi = x + sd, F, f;
        for (int it = 0; it < 100; it++) { CDF(lo, F); if (!(F > pk)) break; lo -= 2 * (x - lo); }
        for (int it = 0; it < 100; it++) { CDF(hi, F); if (!(F < pk)) break; hi += 2 * (hi - x); }
        int j, cv = 0;
        for (j = 1; j <= 60; j++) {
            CDF(x, F); PDF(x, f);
            if (F > pk) hi = x; else lo = x;
            double xn = x - (F - pk) / f;
            if (!(lo < xn && xn < hi) || !isfinite(xn)) xn = (lo + hi) / 2;
            if (fabs(xn - x) < tol) { x = xn; cv = 1; break; }
            x = xn;
        }
        q[k] = x; if (iter) iter[k] = j > 60 ? 60 : j; if (conv) conv[k] = cv;
        if (Fq) CDF(x, Fq[k]);
    }
#undef CDF
#undef PDF
    return 0;
}
