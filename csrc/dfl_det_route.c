/* dfl_det_route.c: one interface over the two engines (a port of src/merged.jl of DFLimit.jl): the classical engine of
 * this directory (the three classical cases, the line, conditional and contour integrals) and the general engine
 * (dfl_det_*.c, any deterministic terms, the inversion on paths around the roots).  A computation names either a classical
 * case (spec->D NULL: the model with its start, as the classical entry points take it) or a design; the engine is
 * DFL_ENGINE_AUTO (dfl_autoengine decides, for a case code after the classical engine's checks of the arguments, for a
 * design where the classical engine accepts it), DFL_ENGINE_CLASSICAL or DFL_ENGINE_GENERAL.  The rule of dfl_autoengine
 * is that of merged.jl (measured on the Julia engines) adjusted to the timings of the two C engines (c/test/engine_table.c,
 * c/test/engine_table_c.txt): see the comments in dfl_autoengine. */
#include "dfl_det_int.h"

/* ---------------------------------------------------------------------------------------------------------------- */
/* the classical case of a computation: model 1..4 and stat, or 0 */
static int is_trend_p(const dfl_design *D, int p)
{
    int K = p + 1, deg = p > 0 ? p : 0;
    if (D->first || D->loc || D->K != K || D->np != 1 || D->br[0] != 0 || D->br[1] != 1 || D->deg != deg) return 0;
    for (int i = 0; i < K; i++) for (int k = 0; k <= deg; k++) if (D->C[i * (deg + 1) + k] != (k == i ? 1.0 : 0.0)) return 0;
    return 1;
}
typedef struct { int model, stat, start; double g0, gls; const dfl_design *D; dfl_design own; int owned; } spec_t;
static int resolve(const dfl_spec *S, spec_t *R, char *err, size_t errlen)
{
    memset(R, 0, sizeof *R);
    R->g0 = S->g0; R->gls = S->gls; R->start = S->start;
    if (!isfinite(S->g0)) { snprintf(err, errlen, "g0 must be a finite real scalar"); return 1; }
    if (S->start < 0 || S->start > 2) { snprintf(err, errlen, "start must be zero (0), fixed (1) or stationary (2)"); return 1; }
    if (S->start == 2 && S->g0 != 0) { snprintf(err, errlen, "g0 (a fixed start) and start = stationary exclude each other"); return 1; }
    if (S->D == NULL) {
        /* a classical case: the design trend(p) (case DG: trend(1) GLS-detrended at cbar = -13.5) */
        if (S->model < 1 || S->model > 4) { snprintf(err, errlen, "model must be 1, 2, 3 or 4 (no deterministic terms, intercept, intercept and trend, GLS-detrended trend)"); return 1; }
        /* (gls applies to a design; a classical case has its detrending in its model: never a detrending silently dropped) */
        if (!isnan(S->gls) && !(S->model == 4 && S->gls == DFL_CBAR_GLS)) { snprintf(err, errlen, "gls applies to a design (spec->D); a classical case has its detrending in its model (model 4: GLS detrending at cbar = -13.5), and gls must be NaN"); return 1; }
        R->model = S->model; R->stat = S->start == 2;
        if (R->model == 4) { R->gls = DFL_CBAR_GLS; if (R->stat) { snprintf(err, errlen, "the GLS-detrended case (model 4) has a zero start only"); return 1; } }
        else R->gls = NAN;
        R->start = R->stat ? 2 : (S->g0 != 0 ? 1 : 0);
        if (dfl_design_trend(R->model == 4 ? 1 : R->model - 2, &R->own, err, errlen)) return 1;
        R->D = &R->own; R->owned = 1;
        return 0;
    }
    R->D = S->D;
    if (R->start == 1 && R->g0 == 0) R->start = 0;
    if (S->start == 0 && S->g0 != 0) R->start = 1;     /* (as design_spec: a fixed start when g0 != 0 and no start is named) */
    R->stat = R->start == 2;
    for (int p = -1; p <= 1; p++) {
        if (!is_trend_p(S->D, p)) continue;
        if (isnan(S->gls)) R->model = p + 2;
        else if (p == 1 && R->start == 0 && S->gls == DFL_CBAR_GLS) R->model = 4;
        break;
    }
    return 0;
}
static void spec_free(spec_t *R) { if (R->owned) dfl_design_free(&R->own); }
/* the start of a call of the router: both warnings cleared (dfl_warning, dfl_det_warning; the general engine's not under the
 * hold of a composite call, which collects those of its parts) */
static void warn_entry(void) { det_warn_clear(); dfl_warning_clear(); }
/* the classical warning of the parts of a composite call (the router's searches): collected after each part, set at the end */
typedef struct { char m[1024]; } cwarn_t;
static void cwarn_take(cwarn_t *w)
{
    const char *s = dfl_warning();
    if (!s || strstr(w->m, s)) return;
    size_t k = strlen(w->m);
    if (k + 2 < sizeof w->m) snprintf(w->m + k, sizeof w->m - k, "%s%s", k ? "\n" : "", s);
}
static void cwarn_put(const cwarn_t *w) { dfl_warning_clear(); dfl_warning_join(w->m); }
/* the method checks of the classical engine (check_method of dfl_api.c, ltratio_args) */
static int method_ok(int model, double g0, int stat, int method, char *err, size_t errlen)
{
    if (method < 0 || method > 3) { snprintf(err, errlen, "method must be 0 (automatic), 1 (single integral), 2 (double integral) or 3 (contour integral) (4, the general engine's inversion on paths, with a design or engine general)"); return 0; }
    if (model == 4 && method == 2) { snprintf(err, errlen, "case DG has the line integral representation only"); return 0; }
    if (method == 3 && !((model == 2 || model == 3) && g0 != 0 && !stat)) { snprintf(err, errlen, "%s", DFL_CONTOUR_MSG); return 0; }
    return 1;
}
/* whether the classical engine accepts the arguments (classical_valid) */
static int classical_valid(const spec_t *R, double c, double c0, int method)
{
    char e[256];
    if (R->model == 0) return 0;
    if (dfl_check_args(R->model, c, c0, R->g0, R->stat, e, sizeof e)) return 0;
    return method_ok(R->model, R->g0, R->stat, method, e, sizeof e);
}
static void det_opts_of(const spec_t *R, double c0, int verified, dfl_det_opts *o)
{
    dfl_det_defaults(o);
    o->c0 = c0; o->start = R->start; o->gamma0 = R->start == 1 ? R->g0 : 0.0; o->gls = R->gls; o->verified = verified;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* The choice of engine = auto for a computation the classical engine accepts (autoengine of merged.jl).  The cells and
 * their timings are those of dev/engine_table.txt of DFLimit.jl for the Julia engines and of c/test/engine_table_c.txt for
 * the C engines (c/test/engine_table.c); where the C timings point elsewhere the C rule says so. */
int dfl_autoengine(int f, int model, int stat, double g0, double gls, double c, double c0, double x)
{
    int fixed = g0 != 0 && !stat;
    int DK = model == 1 && !stat, DU = model == 2 && !stat, DT = model == 3 && !stat, SU = model == 2 && stat, ST = model == 3 && stat, DG = model == 4;
    switch (f) {
    case DFL_F_DIST:
        /* the t-ratio: the faster engine where the two agree to 1e-10, the more accurate one where they do not */
        if (stat && c0 != 0 && c <= -3e-3) return DFL_ENGINE_GENERAL;
        /* x = 0, the points of the coefficient statistic (dfl_engine_coef: F at x = 0 with the null value c0 + y), one point
         * per call (scratch timings of November 2026, cases DK, DU, DT, c from 1 to -80, y at the 1 to 99% quantiles of the
         * statistic and c0 = -3, 2; the Julia measurements agree): with c0 > 0 the classical line integral, 2 to 17 times
         * faster there (X(v) = c0 v^2 + 1/2 has no root), where the rule below takes the general engine for the t-ratio; with
         * c0 < 0 in case DK for c > -5 the general engine (1.0 to 2.1 times faster; the classical line integral with its
         * general kernel), in case DU the classical engine (over the quantiles at c = 0, -5, -10 1.0 to 1.6 times faster,
         * where the rule below takes the general engine for the t-ratio: coef_cdf of case DU at c = -5 was about three times
         * slower in the R and Python packages at the median); case DT as the t-ratio (the general engine up to 1.7 times
         * faster for -20 <= c <= 0) */
        if (x == 0 && !fixed && !stat && c0 != 0 && -100 < c && c <= 1 && (DK || DU || DT)) {
            if (c0 > 0) return DFL_ENGINE_CLASSICAL;
            if (DK && c > -5) return DFL_ENGINE_GENERAL;
            if (DU) return DFL_ENGINE_CLASSICAL;
        }
        if (!fixed && !stat && c0 > 0 && -100 < c && c <= 1) {
            /* (case DK: the density is infinite at the tangency x* = -sqrt(2 c0); just above it the classical line integral) */
            if (DK) { double xs = -sqrt(2 * c0); if (xs < x && x <= xs + 1e-3 * dmax(1.0, fabs(xs))) return DFL_ENGINE_CLASSICAL; }
            return DFL_ENGINE_GENERAL;
        }
        if (!fixed && !stat && c0 != 0 && c < -100) return DFL_ENGINE_GENERAL;
        if (fixed && DK && c <= 1 && ((c * g0) * (c * g0) / 2 <= 400 || (g0 <= 0.5 && c >= -100))) return DFL_ENGINE_GENERAL;
        /* (at c0 = 0 with a zero or stationary start the classical line integral, as in the Julia rule: with its fixed node
         * layouts (dfl_batch.c, November 2026) it is 1.1 to 1.7 times faster than the general engine at c = -30 to -100 in
         * cases DU, DT, SK, SU, ST and 8 to 11 times in case DK, equal at c = -300 in cases DU, DT (c/test/engine_table_c.txt);
         * without the layouts the general engine had been 1.2 to 2.9 times faster below c = -20, and the C rule took it there.
         * For -20 <= c <= 0 the general engine computes a classical case by the same line integral, with P(t) and the roots
         * from its transform, within 10% of the classical engine's time) */
        /* (the C timings: with a zero start and c0 < 0 the general engine is 1.15 to 1.7 times faster in case DT for
         * -30 <= c <= 0 and in case DU for -10 <= c <= 0; case DK and further out the classical engine.  The tails below 1e-3
         * go back to the classical conditional integral, as for every general value of these cases) */
        if (!fixed && !stat && c0 < 0 && ((DT && c >= -30 && c <= 0) || (DU && c >= -10 && c <= 0))) return DFL_ENGINE_GENERAL;
        return DFL_ENGINE_CLASSICAL;
    case DFL_F_ICDF:
        /* the quantiles at c0 = 0 (the Julia rule: a fixed start, case DT at c <= 0, ST at c <= -1, DU at c = 0; the C timings
         * extend it: with a zero start the general search is 1.2 to 2 times faster in cases DK, DU, DT up to c = 2, with a
         * stationary start at c <= -1, 1.3 to 1.6 times; case DG and c > 2 keep the classical search.  With the fixed node
         * layouts of the classical line integral (November 2026) the general search, whose F and f come from the general
         * engine's line integral for -20 <= c <= 0 and from one pass on its paths beyond, stays 1.1 to 1.6 times faster in cases
         * DU, DT, SU, ST down to c = -100 (case DU at c = -50 1.2 times slower) and 2.6 times in case DK at -20 <= c <= 0, but in
         * case DK below c = -20 the classical search is 2.1 to 3 times faster) */
        if (c0 != 0) return DFL_ENGINE_CLASSICAL;
        if (fixed) return DFL_ENGINE_GENERAL;
        if (DG) return DFL_ENGINE_CLASSICAL;
        if (DK && c < -20) return DFL_ENGINE_CLASSICAL;
        if (!stat && c <= 2) return DFL_ENGINE_GENERAL;
        if (stat && c <= -1) return DFL_ENGINE_GENERAL;
        return DFL_ENGINE_CLASSICAL;
    case DFL_F_MOM:
        /* (the C timings: with c0 != 0 the classical moments are 4 to 9 times faster with a stationary start and 2 to 8 times
         * with a zero start, case DT included, where the Julia rule takes the general engine) */
        if (c0 != 0) return DFL_ENGINE_CLASSICAL;
        if (stat) return (model == 1 ? c <= -3e-3 : c <= -1) ? DFL_ENGINE_GENERAL : DFL_ENGINE_CLASSICAL;
        return DFL_ENGINE_GENERAL;
    case DFL_F_LINCDF:      /* x = a */
        if (DK && isnan(x)) return DFL_ENGINE_GENERAL;
        /* (the C timings: in case DG the classical engine is 1.5 to 16 times faster below the unit root at every cbar; above it
         * the general engine, from c = 3 at cbar = -13.5 (4 to 12 times faster) and for c > 0 at cbar >= -5 (1.2 to 3 times);
         * the Julia rule takes the general engine for cbar <= -2 at every c) */
        if (DG && x != 0 && (c > 2 || (c > 0 && gls >= -5))) return DFL_ENGINE_GENERAL;
        return DFL_ENGINE_CLASSICAL;
    case DFL_F_ADFMAX:
        return (stat && c <= -0.1) ? DFL_ENGINE_GENERAL : DFL_ENGINE_CLASSICAL;
    case DFL_F_ENVELOPE:
        /* (the C timings: the classical engine is 1.6 to 11 times faster for c <= -1 and as fast nearer zero; its critical value
         * is bisected to 1e-9, which moves the power by up to 2e-9; the Julia rule takes the general engine for c <= -0.01 in
         * case DK and c <= -1 in case DG, where its critical value is found to 1e-10 at the same cost) */
        (void) SU; (void) ST;
        return DFL_ENGINE_CLASSICAL;
    default:
        return DFL_ENGINE_CLASSICAL;     /* (the regions) */
    }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the general engine's F (kind 2) or f (kind 1) of the computation (gen_dist: a negative density set to zero) */
/* the method where the general engine computes (engine general, or a design the classical engine does not have): 0 its own
 * rule (DFL_DET_AUTO), 1 the line integral (DFL_DET_LINE: the classical name of the line integral, as for the classical
 * cases), DFL_METHOD_PATHS the inversion on paths; 2 (double) and 3 (contour) name representations of the classical engine */
static int gen_method(int method, int *gm, char *err, size_t errlen)
{
    if (method == 0) { *gm = DFL_DET_AUTO; return 0; }
    if (method == 1) { *gm = DFL_DET_LINE; return 0; }
    if (method == DFL_METHOD_PATHS) { *gm = DFL_DET_PATHS; return 0; }
    snprintf(err, errlen, "method %d names a representation of the classical engine; it does not apply with the general engine (whose methods are 0 automatic, 1 the line integral and %d the inversion on paths)", method, DFL_METHOD_PATHS);
    return 1;
}
static int gen_dist(int kind, int n, const double *x, const spec_t *R, double c, double c0, int verified, int gm, double *out, char *err, size_t errlen)
{
    dfl_det_opts o; det_opts_of(R, c0, verified, &o); o.method = gm;
    if (dfl_det_dist(kind, R->D, n, x, c, &o, out, err, errlen)) return 1;
    if (kind == 1) for (int k = 0; k < n; k++) if (out[k] < 0) out[k] = 0;
    return 0;
}
static int classical_dist(int kind, int n, const double *x, const spec_t *R, double c, double c0, int method, int verified, double *out, char *err, size_t errlen)
{
    dfl_double_opts dv; dfl_double_defaults(&dv); dv.verified = verified;
    return dfl_dist(kind, n, x, R->model, c, c0, R->g0, R->stat, method, NULL, &dv, out, NULL, err, errlen);
}
/* the points of the engine of each (merged_dist): the general engine's values below 1e-3 go back to the classical engine
 * where it would take its conditional integral, whose error is relative */
static int merged_dist(int kind, int n, const double *x, const spec_t *R, double c, double c0, double *out, char *err, size_t errlen)
{
    int *eng = malloc((n + 1) * sizeof(int)), ng = 0, ncl = 0, rc = 0;
    for (int k = 0; k < n; k++) { eng[k] = isfinite(x[k]) ? dfl_autoengine(DFL_F_DIST, R->model, R->stat, R->g0, R->gls, c, c0, x[k]) : DFL_ENGINE_CLASSICAL; if (eng[k] == DFL_ENGINE_GENERAL) ng++; }
    if (ng == 0) { free(eng); return classical_dist(kind, n, x, R, c, c0, 0, 0, out, err, errlen); }
    double *xg = malloc((n + 1) * sizeof(double)), *vg = malloc((n + 1) * sizeof(double)); int *ig = malloc((n + 1) * sizeof(int)), m = 0;
    for (int k = 0; k < n; k++) if (eng[k] == DFL_ENGINE_GENERAL) { xg[m] = x[k]; ig[m++] = k; }
    rc = gen_dist(kind, m, xg, R, c, c0, 0, DFL_DET_AUTO, vg, err, errlen);
    if (!rc) for (int j = 0; j < m; j++) {
        out[ig[j]] = vg[j];
        if (vg[j] < 1e-3 && dfl_automethod(R->model, c, c0, R->g0, R->stat, xg[j]) == 2) eng[ig[j]] = DFL_ENGINE_CLASSICAL;
    }
    if (!rc) {
        double *xc = malloc((n + 1) * sizeof(double)), *vc = malloc((n + 1) * sizeof(double)); int *ic = malloc((n + 1) * sizeof(int));
        for (int k = 0; k < n; k++) if (eng[k] == DFL_ENGINE_CLASSICAL) { xc[ncl] = x[k]; ic[ncl++] = k; }
        if (ncl > 0) rc = classical_dist(kind, ncl, xc, R, c, c0, 0, 0, vc, err, errlen);
        if (!rc) for (int j = 0; j < ncl; j++) out[ic[j]] = vc[j];
        free(xc); free(vc); free(ic);
    }
    free(eng); free(xg); free(vg); free(ig);
    return rc;
}
int dfl_engine_dist(int kind, int n, const double *x, const dfl_spec *S, double c, double c0, int engine, int method, int verified, double *out, char *err, size_t errlen)
{
    warn_entry();
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    int rc;
    if (engine == DFL_ENGINE_GENERAL || (engine == DFL_ENGINE_AUTO && S->D && !classical_valid(&R, c, c0, method))) {
        int gm; if (gen_method(method, &gm, err, errlen)) { spec_free(&R); return 1; }
        rc = gen_dist(kind, n, x, &R, c, c0, verified, gm, out, err, errlen);
    } else if (engine == DFL_ENGINE_CLASSICAL) {
        if (R.model == 0) { snprintf(err, errlen, "engine classical: the classical engine has the cases DK, DU, DT, SK, SU, ST and DG only (trend(-1), trend(0), trend(1), the last GLS-detrended at cbar = -13.5)"); spec_free(&R); return 1; }
        rc = classical_dist(kind, n, x, &R, c, c0, method, verified, out, err, errlen);
    } else {
        /* auto: the classical engine's checks (errors for a case code), its representations named by method, its checked
         * computation (verified) as before the merge; otherwise the engine of each point */
        if (dfl_check_args(R.model, c, c0, R.g0, R.stat, err, errlen) || !method_ok(R.model, R.g0, R.stat, method, err, errlen)) { spec_free(&R); return 1; }
        if (method != 0 || verified) rc = classical_dist(kind, n, x, &R, c, c0, method, verified, out, err, errlen);
        else rc = merged_dist(kind, n, x, &R, c, c0, out, err, errlen);
    }
    spec_free(&R);
    return rc;
}

/* the inverse error function (Giles' approximation refined by Newton on erf) */
static double erfinv_(double y)
{
    if (!(y > -1 && y < 1)) return y <= -1 ? -INFINITY : y >= 1 ? INFINITY : NAN;
    double w = -log((1 - y) * (1 + y)), p;
    if (w < 5) { w -= 2.5; p = 2.81022636e-08; p = 3.43273939e-07 + p * w; p = -3.5233877e-06 + p * w; p = -4.39150654e-06 + p * w; p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w; p = -0.00417768164 + p * w; p = 0.246640727 + p * w; p = 1.50140941 + p * w; }
    else { w = sqrt(w) - 3; p = -0.000200214257; p = 0.000100950558 + p * w; p = 0.00134934322 + p * w; p = -0.00367342844 + p * w; p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w; p = 0.00943887047 + p * w; p = 1.00167406 + p * w; p = 2.83297682 + p * w; }
    double x = p * y;
    for (int it = 0; it < 3; it++) x -= (erf(x) - y) / (2 / sqrt(M_PI) * exp(-x * x));
    return x;
}
/* the general engine's quantiles, started at the normal approximation from the moments (gen_icdf) */
static int gen_icdf(int n, const double *p, const spec_t *R, double c, double c0, double tol, int gm, double *q, char *err, size_t errlen)
{
    double x0 = -2.0, sd = 1.0, m[2], me[2]; int have = 0;
    if (R->model != 0 && R->model != 4 && classical_valid(R, c, c0, 0) && dfl_autoengine(DFL_F_MOM, R->model, R->stat, R->g0, R->gls, c, c0, 0.0) == DFL_ENGINE_CLASSICAL) {
        have = dfl_moments(R->model, c, c0, R->g0, R->stat, 2, m, me, err, errlen) == 0;
    } else if (!R->D->first) {
        dfl_det_opts o; det_opts_of(R, c0, 0, &o);
        have = dfl_det_moments(R->D, c, 2, &o, m, NULL, err, errlen) == 0;
    }
    if (have) { x0 = m[0]; sd = sqrt(dmax(m[1] - m[0] * m[0], 1e-6)); }
    dfl_det_opts o; det_opts_of(R, c0, 0, &o); o.method = gm;
    det_warn_clear(); det_warn_hold(1);
    for (int k = 0; k < n; k++) {
        if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "p must lie in (0, 1)"); det_warn_hold(0); return 1; }
        if (dfl_det_quantile(R->D, 1, &p[k], c, x0 + sd * sqrt(2) * erfinv_(2 * p[k] - 1), tol, &o, &q[k], err, errlen)) { det_warn_hold(0); return 1; }
    }
    det_warn_hold(0);
    return 0;
}
/* the classical search with each evaluation routed (ltratio_icdf_ with engine = auto); the warnings of the evaluations
 * collected (the general engine's under the hold, the classical engine's in cw), the hold released on every return */
static int routed_icdf(int n, const double *p, const spec_t *R, double c, double c0, int method, double tol, double *q, char *err, size_t errlen)
{
    double mu, sd, m[2], me[2];
    for (int k = 0; k < n; k++) if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "p must lie in (0, 1)"); return 1; }
    if (R->model == 4) { mu = -1.9 + 0.1 * dmin(c, 0.0); sd = 0.8; }
    else { if (dfl_moments(R->model, c, c0, R->g0, R->stat, 2, m, me, err, errlen)) return 1; mu = m[0]; sd = sqrt(dmax(m[1] - m[0] * m[0], 1e-6)); }
    dfl_spec S = {NULL, R->model, R->start, R->g0, R->gls};
    cwarn_t cw = {""};
    int rc = 0;
#define CDF(xx, o) do { double x_ = (xx); rc = dfl_engine_dist(2, 1, &x_, &S, c, c0, DFL_ENGINE_AUTO, method, 0, &(o), err, errlen); cwarn_take(&cw); if (rc) goto done; } while (0)
#define PDF(xx, o) do { double x_ = (xx); rc = dfl_engine_dist(1, 1, &x_, &S, c, c0, DFL_ENGINE_AUTO, method, 0, &(o), err, errlen); cwarn_take(&cw); if (rc) goto done; } while (0)
    det_warn_hold(1);
    for (int k = 0; k < n; k++) {
        double pk = p[k];
        double x = mu + sd * sqrt(2) * erfinv_(2 * pk - 1), lo = x - sd, hi = x + sd, F, f;
        for (;;) { CDF(lo, F); if (!(F > pk)) break; lo -= 2 * (x - lo); }
        for (;;) { CDF(hi, F); if (!(F < pk)) break; hi += 2 * (hi - x); }
        int conv = 0;
        for (int it = 0; it < 60; it++) {
            CDF(x, F); PDF(x, f);
            if (F > pk) hi = x; else lo = x;
            double xn = x - (F - pk) / f;
            if (!(lo <= xn && xn <= hi && isfinite(xn))) xn = (lo + hi) / 2;   /* (a step below half an ulp lands on the end of the bracket: converged) */
            if (fabs(xn - x) < tol) { x = xn; conv = 1; break; }
            x = xn;
        }
        if (!conv) det_warn("ltratio_icdf: Newton's method did not converge at p = %g", pk);
        q[k] = x;
    }
#undef CDF
#undef PDF
done:
    det_warn_hold(0);
    cwarn_put(&cw);
    return rc;
}
int dfl_engine_icdf(int n, const double *p, const dfl_spec *S, double c, double c0, int engine, int method, double tol, double *q, char *err, size_t errlen)
{
    warn_entry();
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    if (!(tol > 0)) tol = 1e-10;
    int rc;
    if (engine == DFL_ENGINE_GENERAL || (engine == DFL_ENGINE_AUTO && S->D && !classical_valid(&R, c, c0, method))) {
        int gm; if (gen_method(method, &gm, err, errlen)) { spec_free(&R); return 1; }
        rc = gen_icdf(n, p, &R, c, c0, tol, gm, q, err, errlen);
    } else if (engine == DFL_ENGINE_CLASSICAL) {
        if (R.model == 0) { snprintf(err, errlen, "engine classical: the classical engine has the cases DK, DU, DT, SK, SU, ST and DG only"); spec_free(&R); return 1; }
        rc = dfl_icdf(n, p, R.model, c, c0, R.g0, R.stat, method, tol, NULL, NULL, q, NULL, NULL, NULL, err, errlen);
    } else {
        if (dfl_check_args(R.model, c, c0, R.g0, R.stat, err, errlen) || !method_ok(R.model, R.g0, R.stat, method, err, errlen)) { spec_free(&R); return 1; }
        if (method == 0 && dfl_autoengine(DFL_F_ICDF, R.model, R.stat, R.g0, R.gls, c, c0, 0.0) == DFL_ENGINE_GENERAL) rc = gen_icdf(n, p, &R, c, c0, tol, DFL_DET_AUTO, q, err, errlen);
        else rc = routed_icdf(n, p, &R, c, c0, method, tol, q, err, errlen);
    }
    spec_free(&R);
    return rc;
}

int dfl_engine_mom(int pmax, const dfl_spec *S, double c, double c0, int engine, double *m, char *err, size_t errlen)
{
    warn_entry();
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    dfl_det_opts o; det_opts_of(&R, c0, 0, &o);
    double me[64];
    int rc, gen;
    if (engine == DFL_ENGINE_GENERAL) gen = 1;
    else if (engine == DFL_ENGINE_CLASSICAL) {
        if (R.model == 0 || R.model == 4) { snprintf(err, errlen, R.model == 4 ? "the moments are not available for case DG" : "engine classical: the classical engine has the cases DK, DU, DT, SK, SU, ST only"); spec_free(&R); return 1; }
        gen = 0;
    } else if (S->D) gen = !(classical_valid(&R, c, c0, 0) && R.model != 4) || dfl_autoengine(DFL_F_MOM, R.model, R.stat, R.g0, R.gls, c, c0, 0.0) == DFL_ENGINE_GENERAL;
    else {
        if (dfl_check_args(R.model, c, c0, R.g0, R.stat, err, errlen)) { spec_free(&R); return 1; }
        if (R.model == 4) { snprintf(err, errlen, "the moments are not available for case DG"); spec_free(&R); return 1; }
        gen = dfl_autoengine(DFL_F_MOM, R.model, R.stat, R.g0, R.gls, c, c0, 0.0) == DFL_ENGINE_GENERAL;
    }
    rc = gen ? dfl_det_moments(R.D, c, pmax, &o, m, NULL, err, errlen) : dfl_moments(R.model, c, c0, R.g0, R.stat, pmax, m, pmax <= 64 ? me : NULL, err, errlen);
    spec_free(&R);
    return rc;
}

/* the coefficient statistic: the t-ratio at x = 0 with the null value c0 + y, routed as the t-ratio (the warnings of the
 * points collected) */
int dfl_engine_coef(int n, const double *y, const dfl_spec *S, double c, double c0, int engine, double *out, char *err, size_t errlen)
{
    warn_entry();
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    int gen = engine == DFL_ENGINE_GENERAL || (engine == DFL_ENGINE_AUTO && S->D && !classical_valid(&R, c, c0, 0));
    if (!gen && engine == DFL_ENGINE_CLASSICAL && R.model == 0) { snprintf(err, errlen, "engine classical: the classical engine has the cases DK, DU, DT, SK, SU, ST and DG only"); spec_free(&R); return 1; }
    int rc = 0;
    if (gen) { dfl_det_opts o; det_opts_of(&R, c0, 0, &o); rc = dfl_det_coef(2, R.D, n, y, c, &o, out, err, errlen); }
    else {
        dfl_spec S2 = {NULL, R.model, R.start, R.g0, R.gls};
        cwarn_t cw = {""};
        det_warn_hold(1);
        for (int k = 0; k < n && !rc; k++) {
            if (!isfinite(y[k])) { out[k] = isnan(y[k]) ? NAN : y[k] > 0 ? 1.0 : 0.0; continue; }
            double x0 = 0;
            rc = dfl_engine_dist(2, 1, &x0, &S2, c, c0 + y[k], engine, 0, 0, &out[k], err, errlen);
            cwarn_take(&cw);
        }
        det_warn_hold(0);
        cwarn_put(&cw);
    }
    spec_free(&R);
    return rc;
}
/* its quantiles: the general engine's Newton search, or the Illinois variant of regula falsi of the classical engine on
 * the routed distribution function (coef_icdf) */
int dfl_engine_coef_icdf(int n, const double *p, const dfl_spec *S, double c, double c0, int engine, double tol, double *q, char *err, size_t errlen)
{
    warn_entry();
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    if (!(tol > 0)) tol = 1e-9;
    int gen = engine == DFL_ENGINE_GENERAL || (engine == DFL_ENGINE_AUTO && S->D && !classical_valid(&R, c, c0, 0));
    int rc = 0;
    if (gen) { dfl_det_opts o; det_opts_of(&R, c0, 0, &o); rc = dfl_det_coef_quantile(R.D, n, p, c, NAN, tol, &o, q, err, errlen); spec_free(&R); return rc; }
    for (int k = 0; k < n; k++) if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "coef_icdf: p must lie in (0, 1)"); spec_free(&R); return 1; }
    dfl_spec S2 = {NULL, R.model, R.start, R.g0, R.gls};
    cwarn_t cw = {""};
    det_warn_hold(1);
#define FX(xx, o) do { double y_ = (xx); rc = dfl_engine_coef(1, &y_, &S2, c, c0, engine, &(o), err, errlen); cwarn_take(&cw); if (rc) goto done; (o) -= pk; } while (0)
    for (int k = 0; k < n; k++) {
        double pk = p[k];
        double lo = -40, hi = 10, flo, fhi, x, fx; FX(lo, flo); FX(hi, fhi);
        while (flo > 0) { hi = lo; fhi = flo; lo = 2 * lo - 10; FX(lo, flo); }
        while (fhi < 0) { lo = hi; flo = fhi; hi = 2 * hi + 10; FX(hi, fhi); }
        int side = 0; x = (lo + hi) / 2;
        for (int it = 0; it < 200; it++) {
            x = (lo * fhi - hi * flo) / (fhi - flo); FX(x, fx);
            if (fx == 0) break;
            if (fx < 0) { lo = x; flo = fx; if (side == -1) fhi /= 2; side = -1; }
            else { hi = x; fhi = fx; if (side == 1) flo /= 2; side = 1; }
            if (hi - lo < tol) { x = (lo * fhi - hi * flo) / (fhi - flo); break; }
        }
        q[k] = x;
    }
#undef FX
done:
    det_warn_hold(0);
    cwarn_put(&cw);
    spec_free(&R);
    return rc;
}

/* P(U - aV <= b) in case DK (spec of model 1 or 2) or DG (model 4, detrended at cbar); a = NaN: P(V <= b) */
int dfl_engine_lincdf(int n, const double *b, double a, int model, double c, double cbar, int engine, double *out, char *err, size_t errlen)
{
    warn_entry();
    if (model == 2) model = 1;
    if (model != 1 && model != 4) { snprintf(err, errlen, "ltratio_lincdf: the case must be DK or DG"); return 1; }
    if (isnan(cbar)) cbar = DFL_CBAR_GLS;
    /* (a = +-Inf and an infinite cbar gave 0 in the classical code: refused, as by the general engine; a = NaN is the law
     * of V) */
    if (isinf(a)) { snprintf(err, errlen, "ltratio_lincdf: a must be a finite real scalar (NaN: the law of V)"); return 1; }
    if (model == 4 && !isfinite(cbar)) { snprintf(err, errlen, "ltratio_lincdf: cbar must be a finite real scalar < 0 (NaN: the default -13.5)"); return 1; }
    if (engine == DFL_ENGINE_AUTO && (!isfinite(c) || c < DFL_CMIN)) { snprintf(err, errlen, "ltratio_lincdf: %s", isfinite(c) ? DFL_CMIN_MSG : "c must be a finite real scalar"); return 1; }
    int gen = engine == DFL_ENGINE_GENERAL || (engine == DFL_ENGINE_AUTO && dfl_autoengine(DFL_F_LINCDF, model, 0, 0.0, cbar, c, 0.0, a) == DFL_ENGINE_GENERAL);
    if (!gen) return dfl_lincdf(n, b, a, model, c, cbar, NULL, out, err, errlen);
    dfl_design D; if (dfl_design_trend(model == 4 ? 1 : -1, &D, err, errlen)) return 1;
    dfl_det_opts o; dfl_det_defaults(&o); if (model == 4) o.gls = cbar;
    int rc = dfl_det_lin(2, &D, n, b, a, c, &o, out, err, errlen);
    dfl_design_free(&D);
    return rc;
}
/* the power envelope of case DK (model 1 or 2) or DG (model 4) */
int dfl_engine_envelope(int n, const double *cs, int model, double alpha, int engine, double *out, char *err, size_t errlen)
{
    warn_entry();
    if (model == 2) model = 1;
    if (model != 1 && model != 4) { snprintf(err, errlen, "power_envelope: the case must be DK (or DU) or DG"); return 1; }
    if (!(alpha > 0 && alpha < 1)) { snprintf(err, errlen, "power_envelope: the level must lie in (0, 1)"); return 1; }
    dfl_design D; if (dfl_design_trend(model == 4 ? 1 : -1, &D, err, errlen)) return 1;
    int rc = 0;
    for (int k = 0; k < n && !rc; k++) {
        int gen = engine == DFL_ENGINE_GENERAL || (engine == DFL_ENGINE_AUTO && dfl_autoengine(DFL_F_ENVELOPE, model, 0, 0.0, NAN, cs[k], 0.0, 0.0) == DFL_ENGINE_GENERAL);
        if (!(cs[k] < 0)) { out[k] = isnan(cs[k]) ? NAN : alpha; continue; }
        if (cs[k] == -INFINITY) { out[k] = 1.0; continue; }         /* (the limit; the classical bisection found no bracket) */
        rc = gen ? dfl_det_power_envelope(&D, 1, &cs[k], alpha, &out[k], err, errlen) : dfl_power_envelope(1, &cs[k], model, alpha, NULL, &out[k], err, errlen);
    }
    dfl_design_free(&D);
    return rc;
}
/* ADFmax (kind 2 distribution function, 1 density) for a case DU, DT, SU, ST or a design.  The classical ADFmax has the
 * zero and the stationary start of cases DU, DT only: a fixed start (of a case code or of trend(0), trend(1)) takes the
 * general engine, whose ADFmax is the same functional of the design's (U, V) for every start (it computed a zero start
 * before); GLS detrending is refused (neither engine has the ADFmax of GLS-detrended data: the general engine detrends by
 * least squares, and gls was dropped) */
int dfl_engine_adfmax(int kind, int n, const double *x, const dfl_spec *S, double c, int engine, double *out, char *err, size_t errlen)
{
    warn_entry();
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    int clsmodel = R.model == 2 || R.model == 3, cls = clsmodel && R.start != 1;
    if (S->D == NULL && !clsmodel) { snprintf(err, errlen, "adfmax: the case must be one of DU, DT, SU, ST"); spec_free(&R); return 1; }
    if (!isnan(R.gls)) { snprintf(err, errlen, "adfmax: ADFmax has least-squares detrending only (gls must be NaN)"); spec_free(&R); return 1; }
    if (engine == DFL_ENGINE_CLASSICAL && !cls) { snprintf(err, errlen, "engine classical: ADFmax of the classical engine has the cases DU, DT, SU, ST only, with a zero or a stationary start"); spec_free(&R); return 1; }
    if (engine != DFL_ENGINE_GENERAL && cls && (!isfinite(c) || c < DFL_CMIN || (R.stat && !(c <= DFL_CSTAT)))) {
        snprintf(err, errlen, "%s", !isfinite(c) ? "adfmax: c must be a finite real scalar" : c < DFL_CMIN ? DFL_CMIN_MSG : DFL_STAT_MSG); spec_free(&R); return 1;
    }
    int gen = engine == DFL_ENGINE_GENERAL || !cls || (engine == DFL_ENGINE_AUTO && dfl_autoengine(DFL_F_ADFMAX, R.model, R.stat, 0.0, NAN, c, 0.0, 0.0) == DFL_ENGINE_GENERAL);
    dfl_double_info *info = malloc((n + 1) * sizeof(dfl_double_info));
    int rc;
    if (!gen) rc = dfl_adfmax(kind, n, x, R.model, c, R.stat, NULL, out, info, err, errlen);
    else {
        dfl_det_opts o; det_opts_of(&R, 0.0, 0, &o);
        rc = dfl_det_adfmax(kind, R.D, n, x, c, &o, out, err, errlen);
        if (!rc && kind == 1) for (int k = 0; k < n; k++) if (out[k] < 0) out[k] = 0;
        if (!rc && engine == DFL_ENGINE_AUTO && cls) {
            /* the tails below 1e-3 by the classical conditional integral, whose error is relative */
            for (int k = 0; k < n && !rc; k++) if (isfinite(x[k]) && out[k] < 1e-3) rc = dfl_adfmax(kind, 1, &x[k], R.model, c, R.stat, NULL, &out[k], info, err, errlen);
        }
    }
    free(info); spec_free(&R);
    return rc;
}
/* P{U <= g(V)} in case DK: type 1 the line g(v) = par[1] + par[0] v, 2 the parabola par[0] sqrt(v) + par[1] v; with the
 * classical engine also type 3 (the weighted-average-power boundary) */
int dfl_engine_region(int type, const double *par, const dfl_spec *S, double c, int engine, double *out, char *err, size_t errlen)
{
    warn_entry();
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    int dk = R.model == 1 && !R.stat && R.start == 0;
    int rc;
    /* (the line g(v) = b + a v and the parabola x sqrt(v) + c0 v: a non-finite b or x gives 1, 0, NaN, as the distribution
     * functions do, and the slope a and c0 must be finite; the classical regions gave 0 at b = Inf and at a = +-Inf, and
     * the general engine NaN at a = +-Inf and the law of V at a = NaN) */
    if (type == 1 || type == 2) {
        if (!isfinite(par[type == 1 ? 0 : 1])) { snprintf(err, errlen, "region_prob: %s must be a finite real scalar", type == 1 ? "the slope a of the line" : "c0 of the parabola"); spec_free(&R); return 1; }
        double v = par[type == 1 ? 1 : 0];
        if (!isfinite(v)) { out[0] = isnan(v) ? NAN : v > 0 ? 1.0 : 0.0; spec_free(&R); return 0; }
    }
    if (engine == DFL_ENGINE_CLASSICAL && !dk) { snprintf(err, errlen, "engine classical: the regions of the classical engine have case DK only"); spec_free(&R); return 1; }
    if (engine != DFL_ENGINE_GENERAL && dk) {
        dfl_double_info info;
        rc = dfl_region(type, par, 0.0, 0, NULL, c, NULL, out, &info, err, errlen);
    } else {
        dfl_det_opts o; det_opts_of(&R, 0.0, 0, &o);
        rc = dfl_det_region(R.D, type, par, c, &o, out, err, errlen);
    }
    spec_free(&R);
    return rc;
}
