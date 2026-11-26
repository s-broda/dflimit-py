/* dfl_det_route.c: one interface over the two engines (a port of src/merged.jl of DFLimit.jl): the classical engine of
 * this directory (the three classical cases, the line, conditional and contour integrals) and the general engine
 * (dfl_det_*.c, any deterministic terms, the inversion on paths around the roots).  A computation names either a classical
 * case (spec->D NULL: the model with its start, as the classical entry points take it) or a design; the engine is
 * DFL_ENGINE_AUTO (dfl_autoengine decides, for a case code after the classical engine's checks of the arguments, for a
 * design where the classical engine accepts it), DFL_ENGINE_CLASSICAL or DFL_ENGINE_GENERAL.  The rule of dfl_autoengine
 * is that of merged.jl (measured on the Julia engines) adjusted to the timings of the two C engines (c/test/engine_table.c,
 * c/test/engine_table_c.txt): see the comments in dfl_autoengine. */
#include "dfl_det_int.h"
#include "dfl_kan_int.h"

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
    /* (above the range of the general engine, dfl_det_cmax: 20 without deterministic terms, 16 with them, 10 with them and
     * a fixed start, where it refuses, the classical engine; below, the rule) */
    if (c > (model == 1 ? DFL_DET_CMAX0 : fixed ? DFL_DET_CMAX_FIXED : DFL_DET_CMAX)) return DFL_ENGINE_CLASSICAL;
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
        if (x == 0 && !fixed && !stat && c0 != 0 && -100 < c && c <= 2 && (DK || DU || DT)) {
            if (c0 > 0) return DFL_ENGINE_CLASSICAL;
            if (DK && c > -5) return DFL_ENGINE_GENERAL;
            if (DU) return DFL_ENGINE_CLASSICAL;
        }
        /* (a zero start with c0 > 0 below c = 1, case DT below c = 2: the general engine, 2 to 4.4 times faster at c = -100
         * and 3.6 to 6.8 times in case DT at c = 1.5, 2; re-measured November 26, 2026, as in the Julia rule) */
        if (!fixed && !stat && c0 > 0 && (c <= 1 || (DT && c <= 2))) {
            /* (case DK: the density is infinite at the tangency x* = -sqrt(2 c0); just above it the classical line integral) */
            if (DK) { double xs = -sqrt(2 * c0); if (xs < x && x <= xs + 1e-3 * dmax(1.0, fabs(xs))) return DFL_ENGINE_CLASSICAL; }
            return DFL_ENGINE_GENERAL;
        }
        if (!fixed && !stat && c0 != 0 && c < -100) return DFL_ENGINE_GENERAL;
        /* (the fixed start of case DK, except with c0 < 0 near the unit root (c > -1): there the outer hyperbola of the
         * general engine does not resolve the upper tail, and its one evaluation was off by 2.7e-6 at the 99.99% point (c =
         * 0.5, c0 = -5, g0 = 1), 6.7e-6 at c = 0 with g0 = 1.5 and 2.9e-2 beyond (c0 = -10, where at mean + 4 sd it gave 1.23
         * and the final check of the distribution function refused); the classical line integral, its checked computation
         * and the conditional integral agree to 1e-12 there; autoengine of merged.jl, November 2026) */
        /* (re-measured November 26, 2026, the C timings: beyond (c g0)^2/2 <= 400 the general engine is 1.8 to 3.9 times faster
         * where |g0| <= 1 and |c g0| <= 50 with c >= -100, which holds g0 <= 0.5 with c >= -100; at g0 = 1.5 1.4 to 1.6 times,
         * below the threshold of the C rule, 1.5 (the Julia rule takes |g0| <= 1.5); the law depends on |g0| only, and the
         * rule tested g0 <= 0.5 until then) */
        double ag = fabs(g0);
        if (fixed && DK && c <= 1 && !(c0 < 0 && c > -1) && ((c * ag) * (c * ag) / 2 <= 400 || (c >= -100 && ag <= 1 && fabs(c) * ag <= 50))) return DFL_ENGINE_GENERAL;
        /* (the fixed start of cases DU, DT far from the unit root: with c0 > 0 below c = -20 (and below c = -10 where |c g0| >
         * 20), and with any c0 below c = -20 where |c g0| > 60, the classical contour integral's one evaluation was off, in
         * the density by up to 5e-6 of its peak (case DU at c = -30, g0 = 0.5, c0 = 2), 4.2e-3 (c = -75, c0 = 5) and 2.2e-2
         * (g0 = 1.5, c = -60, c0 = 5), in the distribution function by 3.9e-5 (case DU, g0 = 1.5, c = -60, c0 = -2) and
         * 6.3e-6 (g0 = 0.9, c = -100), and its checked computation in the density by up to 8.4e-6 of the peak; the general
         * engine's density there is the derivative of the distribution function on which the two engines agree (to 1e-10 of
         * the peak), its F within 1e-10 of the classical checked one; autoengine of merged.jl, November 2026) */
        if (fixed && (DU || DT) && c < -10 && ((c0 > 0 && (c < -20 || fabs(c * g0) > 20)) || (c < -20 && fabs(c * g0) > 60))) return DFL_ENGINE_GENERAL;
        /* (at c0 = 0 with a zero or stationary start the classical line integral, as in the Julia rule: with its fixed node
         * layouts (dfl_batch.c, November 2026) it is 1.1 to 1.7 times faster than the general engine at c = -30 to -100 in
         * cases DU, DT, SK, SU, ST and 8 to 11 times in case DK, equal at c = -300 in cases DU, DT (c/test/engine_table_c.txt);
         * without the layouts the general engine had been 1.2 to 2.9 times faster below c = -20, and the C rule took it there.
         * For -20 <= c <= 0 the general engine computes a classical case by the same line integral, with P(t) and the roots
         * from its transform, within 10% of the classical engine's time; since the thinner rules of the paths case DT below
         * c = -20 takes the general engine, below) */
        /* (the C timings: with a zero start and c0 < 0 the general engine is 1.15 to 1.7 times faster in case DT for
         * -30 <= c <= 0 and in case DU for -10 <= c <= 0; case DK and further out the classical engine.  The tails below 1e-3
         * go back to the classical conditional integral, as for every general value of these cases) */
        /* (re-measured November 26, 2026, after the thinner rules of the paths: in case DT the general engine is 1.7 to 3.5
         * times faster down to c = -100 (below, the rule above), and in case DK near the unit root (-5 < c <= 0, where the
         * classical engine takes its line integral with the general kernel) 6.7 to 8.7 times (the Julia rule takes it up to
         * c = 2, measured in Julia); case DU as before) */
        if (!fixed && !stat && c0 < 0 && ((DK && c > -5 && c <= 0) || (DT && c <= 0) || (DU && c >= -10 && c <= 0))) return DFL_ENGINE_GENERAL;
        /* (c0 = 0 with a zero start just above the unit root, 0 < c <= 0.5, where the general engine's method auto takes its
         * paths (det_line_auto): 1.75 to 2.1 times faster than the classical engine in cases DU, DT (November 26, 2026; the Julia
         * rule goes to c = 1.5 in case DU and c = 2 in case DT, where the C gains of the paths fall below 1.5 for some designs) */
        if (!fixed && !stat && c0 == 0 && (DU || DT) && c > 0 && c <= DET_LINE_PATHS_CMAX) return DFL_ENGINE_GENERAL;
        /* (the C timings, November 26, 2026: c0 = 0 far from the unit root, below c = -20, where the general engine takes its
         * paths, is 1.6 to 1.9 times faster than the classical line integral in case DT with a zero start, down to c = -480,
         * the end of the general engine's checks; in cases DU, SU, ST 1.4 to 2.1 times, below the threshold of 1.5 in some
         * cells; in cases DK, SK the classical engine) */
        if (!fixed && !stat && c0 == 0 && DT && c < -20 && c >= -480) return DFL_ENGINE_GENERAL;
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
        /* (case ST from c = -0.5 on: 1.7 times faster, November 26, 2026; case SU 1.5 times, below the threshold of the C rule;
         * next to zero the general engine's search is off, by 2.6e-7 in case ST at c = -1e-3) */
        if (ST && c <= -0.5) return DFL_ENGINE_GENERAL;
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
        /* (re-measured November 26, 2026: in case DG the classical engine is 4.7 to 21 times faster at c = -2 to -20; its
         * power is within 2.1e-9 of the general engine's; the Julia rule takes the classical engine in case DG too) */
        (void) SU; (void) ST;
        return DFL_ENGINE_CLASSICAL;
    case DFL_F_REGION:
        /* (x = 1 for the parabola of the t-ratio, with its c0, 0 for the other boundaries; the parabola at c0 = 0 for
         * -20 <= c <= 0, where the general engine takes its line integral: 3.5 to 45 times faster than the classical region
         * code's conditional integral (November 26, 2026); below c = -20 the general engine takes its paths and is the slower,
         * as with c0 != 0; above the unit root not measured faster, the classical code, as the Julia rule) */
        return (x == 1 && c0 == 0 && c >= -20 && c <= 0) ? DFL_ENGINE_GENERAL : DFL_ENGINE_CLASSICAL;
    default:
        return DFL_ENGINE_CLASSICAL;
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
/* the general engine's quantiles, started at the normal approximation from the moments (gen_icdf): from jets in the six
 * classical cases (dfl_mom_start), otherwise from the engines' moments where they have them */
static int gen_icdf(int n, const double *p, const spec_t *R, double c, double c0, double tol, int gm, double *q, char *err, size_t errlen)
{
    double x0 = -2.0, sd = 1.0, m[2], me[2]; int have = 0;
    int cv = R->model != 0 && R->model != 4 && classical_valid(R, c, c0, 0);
    if (cv) have = dfl_mom_start(2, R->model, R->stat, c, c0, R->start == 1 ? R->g0 : 0.0, m);
    if (have) {
    } else if (cv && dfl_autoengine(DFL_F_MOM, R->model, R->stat, R->g0, R->gls, c, c0, 0.0) == DFL_ENGINE_CLASSICAL) {
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
/* ---- the extensions of the limit through the router (ext_dist, ext_icdf of merged.jl) ---- */
static int ext_method(int method)
{
    switch (method) {
    case 0: return DFL_DET_AUTO;
    case 1: return DFL_DET_LINE;
    case DFL_METHOD_PATHS: return DFL_DET_PATHS;
    case DFL_METHOD_LINES: return DFL_DET_LINES;
    case DFL_METHOD_MIXTURE: return DFL_DET_MIXTURE;
    default: return -1;          /* (a representation of the classical engine: the general engine's checks refuse it) */
    }
}
#define EXT_CLASSICAL_MSG "engine classical: the extensions of the limit (vol and rho) are computed by the general engine"
int dfl_engine_dist_ext(int kind, int n, const double *x, const dfl_spec *S, double c, double c0, int engine, int method, int verified,
                        const dfl_vol *vol, double rho, double *out, char *err, size_t errlen)
{
    if (!vol && rho == 1) return dfl_engine_dist(kind, n, x, S, c, c0, engine, method, verified, out, err, errlen);
    warn_entry();
    if (engine < 0 || engine > 2) { snprintf(err, errlen, "engine must be auto (0), classical (1) or general (2)"); return 1; }
    if (engine == DFL_ENGINE_CLASSICAL) { snprintf(err, errlen, EXT_CLASSICAL_MSG); return 1; }
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    dfl_det_opts o; det_opts_of(&R, c0, verified, &o); o.method = ext_method(method); o.vol = vol; o.rho = rho;
    int rc = dfl_det_dist(kind, R.D, n, x, c, &o, out, err, errlen);
    if (!rc && kind == 1) for (int k = 0; k < n; k++) if (out[k] < 0) out[k] = 0;
    spec_free(&R);
    return rc;
}
int dfl_engine_icdf_ext(int n, const double *p, const dfl_spec *S, double c, double c0, int engine, int method, double tol,
                        const dfl_vol *vol, double rho, double *q, char *err, size_t errlen)
{
    if (!vol && rho == 1) return dfl_engine_icdf(n, p, S, c, c0, engine, method, tol, q, err, errlen);
    warn_entry();
    if (engine < 0 || engine > 2) { snprintf(err, errlen, "engine must be auto (0), classical (1) or general (2)"); return 1; }
    if (engine == DFL_ENGINE_CLASSICAL) { snprintf(err, errlen, EXT_CLASSICAL_MSG); return 1; }
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    if (!(tol > 0)) tol = 1e-10;
    double cm = c;
    if (vol && vol->c && vol->nreg > 0) { cm = 0; for (int i = 0; i < vol->nreg; i++) cm += vol->c[i]; cm /= vol->nreg; }
    double m[2], mu = -2.0 * rho, sd = 1.0; int have = 0;
    if (isfinite(cm) && isfinite(rho) && fabs(rho) <= 1 && isnan(R.gls) && !R.D->first) {
        char e2[512];
        int cv = R.model != 0 && R.model != 4 && classical_valid(&R, cm, c0, 0);
        if (cv) have = dfl_mom_start(2, R.model, R.stat, cm, c0, R.start == 1 ? R.g0 : 0.0, m);
        if (!have) { dfl_det_opts om; det_opts_of(&R, c0, 0, &om); have = dfl_det_moments(R.D, cm, 2, &om, m, NULL, e2, sizeof e2) == 0; }
        det_warn_clear();
    }
    if (have) { mu = rho * m[0]; sd = sqrt(dmax(rho * rho * (m[1] - m[0] * m[0]) + 1 - rho * rho, 1e-6)); }
    dfl_det_opts o; det_opts_of(&R, c0, 0, &o); o.method = ext_method(method); o.vol = vol; o.rho = rho;
    int rc = 0;
    det_warn_hold(1);
    for (int k = 0; k < n && !rc; k++) {
        if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "p must lie in (0, 1)"); rc = 1; break; }
        rc = dfl_det_quantile(R.D, 1, &p[k], c, mu + sd * sqrt(2) * erfinv_(2 * p[k] - 1), tol, &o, &q[k], err, errlen);
    }
    det_warn_hold(0);
    spec_free(&R);
    return rc;
}
/* F and f at a point x of the Newton steps of the classical search in one evaluation (route_dist2 of merged.jl): where the
 * rule gives the point to the general engine (method 0), its paths return both from one pass (dfl_det_dist_both), the
 * values of dfl_engine_dist bit for bit; 0 elsewhere (the classical engine, the tails that go back to it, other methods),
 * and the search takes the two evaluations; 1 with F, f; -1 with an error */
int DFL_DIST2 = 1;
static int route_dist2(double x, const spec_t *R, double c, double c0, int method, double *F, double *f, char *err, size_t errlen)
{
    if (!(DFL_DIST2 && method == 0 && isfinite(x))) return 0;
    if (dfl_autoengine(DFL_F_DIST, R->model, R->stat, R->g0, R->gls, c, c0, x) != DFL_ENGINE_GENERAL) return 0;
    dfl_det_opts o; det_opts_of(R, c0, 0, &o);
    double Fv, fv;
    if (dfl_det_dist_both(R->D, 1, &x, c, &o, &Fv, &fv, err, errlen)) return -1;
    /* (the tails below 1e-3 where the classical engine would take its conditional integral go back to it, as in
     * merged_dist, for F and f each: the search then takes the two evaluations) */
    if ((Fv < 1e-3 || fv < 1e-3) && dfl_automethod(R->model, c, c0, R->g0, R->stat, x) == 2) return 0;
    *F = Fv; *f = fv < 0 ? 0.0 : fv;
    return 1;
}
/* the classical search with each evaluation routed (ltratio_icdf_ with engine = auto); the warnings of the evaluations
 * collected (the general engine's under the hold, the classical engine's in cw), the hold released on every return */
static int routed_icdf(int n, const double *p, const spec_t *R, double c, double c0, int method, double tol, double *q, char *err, size_t errlen)
{
    double mu, sd, m[2], me[2];
    for (int k = 0; k < n; k++) if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "p must lie in (0, 1)"); return 1; }
    if (R->model == 4) { mu = -1.9 + 0.1 * dmin(c, 0.0); sd = 0.8; }
    else {
        /* (the start: the moments from jets, dfl_mom_start, where they hold; otherwise the classical engine's circles) */
        if (!dfl_mom_start(2, R->model, R->stat, c, c0, R->start == 1 ? R->g0 : 0.0, m) && dfl_moments(R->model, c, c0, R->g0, R->stat, 2, m, me, err, errlen)) return 1;
        mu = m[0]; sd = sqrt(dmax(m[1] - m[0] * m[0], 1e-6));
    }
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
            /* (F and f from one evaluation where the rule gives the point to the general engine's paths, route_dist2) */
            int r2 = route_dist2(x, R, c, c0, method, &F, &f, err, errlen);
            if (r2 < 0) { rc = 1; goto done; }
            if (r2 == 0) { CDF(x, F); PDF(x, f); }
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

/* ---------------------------------------------------------------------------------------------------------------- */
/* The single inversion of Bao and Kan for the half-plane statistics and the moments from jets (the functions inv_* of
 * merged.jl): Yong Bao and Raymond Kan, "Exact Finite-Sample Distribution and Moments of the Least Squares Estimator in
 * First-Order Autoregressions: O(1) Algorithms" (dfl_kan.c).  One inversion of the closed-form joint characteristic
 * function along the ray of the half-plane, adaptive quadrature over decades and the tail completed by integration by
 * parts; the default in the six classical cases where it was compared with the engines (the rules below), each value
 * with its error estimate. */
void dfl_inv_defaults(dfl_inv_opts *o) { o->method = 0; o->tol = NAN; o->verified = 0; }
static const dfl_inv_opts *inv_opts_or_default(const dfl_inv_opts *o, dfl_inv_opts *d) { if (o) return o; dfl_inv_defaults(d); return d; }
/* the methods of a function with the single inversion: 0, DFL_METHOD_INVERSION, DFL_METHOD_ENGINE, and the representations
 * of the t-ratio where extra (the coefficient statistic, through the t-ratio's distribution function at zero; inv_method) */
static int inv_method_ok(const char *fn, int method, int extra, char *err, size_t errlen)
{
    if (method == 0 || method == DFL_METHOD_INVERSION || method == DFL_METHOD_ENGINE) return 1;
    if (extra && method >= 1 && method <= DFL_METHOD_PATHS) return 1;
    snprintf(err, errlen, "%s: method must be 0 (automatic), %d (the single inversion of Bao and Kan) or %d (the engines' computation)%s", fn,
             DFL_METHOD_INVERSION, DFL_METHOD_ENGINE, extra ? ", or a representation of the t-ratio (1 the line integral, 2 the conditional integral, 3 the contour integral, 4 the general engine's paths)" : "");
    return 0;
}
/* the tolerance: NaN or 0 the default, otherwise a positive finite number */
static int inv_tol(const char *fn, const dfl_inv_opts *io, double dflt, double *tol, char *err, size_t errlen)
{
    double t = io->tol;
    if (isnan(t) || t == 0) { *tol = dflt; return 0; }
    if (!(t > 0 && isfinite(t))) { snprintf(err, errlen, "%s: tol must be a positive finite number (NaN or 0: the default %g)", fn, dflt); return 1; }
    *tol = t; return 0;
}
/* whether the single inversion computes (inv_take): with method DFL_METHOD_INVERSION always (an error with engine general,
 * or for a case it does not have), with method 0 under engine auto in the cases DK, DU, DT, SK, SU, ST for c up to
 * DFL_INV_CMAX; otherwise the engines (method DFL_METHOD_ENGINE, a representation of the engines, engine classical or
 * general, case DG).  1, 0, or -1 with the error in err */
static int inv_take(const char *fn, int method, int engine, const spec_t *R, double c, char *err, size_t errlen)
{
    int ic = R->model >= 1 && R->model <= 3;
    if (method == DFL_METHOD_INVERSION) {
        if (engine == DFL_ENGINE_GENERAL) { snprintf(err, errlen, "%s: method %d (the devices of Bao and Kan) does not apply with engine general", fn, method); return -1; }
        if (!ic) { snprintf(err, errlen, "%s: method %d (the devices of Bao and Kan) applies to the cases DK, DU, DT, SK, SU, ST", fn, method); return -1; }
        return 1;
    }
    return method == 0 && engine == DFL_ENGINE_AUTO && ic && c <= DFL_INV_CMAX;
}
int dfl_inv_takes(const dfl_spec *S, double c, double c0, int engine, int method)
{
    char e[256]; spec_t R;
    if (resolve(S, &R, e, sizeof e)) return 0;
    int t = (!S->D || classical_valid(&R, c, c0, 0)) && engine != DFL_ENGINE_GENERAL ? inv_take("", method, engine, &R, c, e, sizeof e) : 0;
    spec_free(&R);
    return t > 0;
}
static int lim_of(const spec_t *R, double c, dfl_limcf *F, char *err, size_t errlen) { return dfl_limcf_init(F, R->model, R->stat, c, R->start == 1 ? R->g0 : 0.0, err, errlen); }
/* the values at the points x by the single inversion, on the workers where n > 1 (the transform is immutable): kind 1 the
 * coefficient statistic's distribution function, 2 its density (ftol: the absolute tolerance of each), 3 the linear form
 * (a), 4 the power envelope (a = alpha), 5 the coefficient statistic's quantiles (a = the width) */
typedef struct { int kind; const dfl_limcf *F; const double *x; double a, c0, tol; double *v, *e, *ft; int *rc; } imap_t;
static void imap_body(int k, void *vp)
{
    imap_t *m = vp; double ft = m->tol; char e[256];
    switch (m->kind) {
    case 1: m->rc[k] = dfl_kan_coef_cdf1(m->F, m->x[k], m->c0, m->tol, &m->v[k], &m->e[k]); break;
    case 2: m->rc[k] = dfl_kan_coef_pdf1(m->F, m->x[k], m->c0, m->tol, &m->v[k], &m->e[k], &ft); break;
    case 3: m->rc[k] = dfl_kan_lin_cdf1(m->F, m->x[k], m->a, m->tol, &m->v[k], &m->e[k]); break;
    case 4: m->rc[k] = dfl_kan_envelope1(m->x[k], m->a, m->tol, &m->v[k], &m->e[k], e, sizeof e); break;
    case 5: m->rc[k] = dfl_kan_coef_icdf1(m->F, m->x[k], m->c0, m->a, m->tol, &m->v[k], &m->e[k], e, sizeof e); break;
    }
    if (m->ft) m->ft[k] = ft;
}
static int imap(imap_t *m, int n, const char *fn, char *err, size_t errlen)
{
    m->rc = calloc(n + 1, sizeof(int));
    if (n == 1) imap_body(0, m); else if (n > 1) dfl_parfor(n, imap_body, m);
    int bad = -1; for (int k = 0; k < n && bad < 0; k++) if (m->rc[k]) bad = k;
    if (bad >= 0) snprintf(err, errlen, "%s: the single inversion failed at %g (the quadrature produced a value that is not finite, or no bracket of the quantile was found)", fn, m->x[bad]);
    free(m->rc); m->rc = NULL;
    return bad >= 0;
}
/* the warning of the single inversion where error estimates exceed the tolerance, one per call (inv_warn) */
static void inv_warn(const char *fn, int n, const double *E, const double *T, double tol)
{
    int bad = 0; double mx = 0;
    for (int k = 0; k < n; k++) if (!(E[k] <= (T ? T[k] : tol))) { bad++; mx = isnan(E[k]) || E[k] > mx ? E[k] : mx; }
    if (bad) det_warn("%s: the error estimate of the single inversion exceeds %s %g at %d of %d points (largest %g); verified recomputes them by the engines' checked computation",
                      fn, T ? "its tolerance" : "the tolerance", tol, bad, n, mx);
}
static void nanfill(double *e, int n) { if (e) for (int k = 0; k < n; k++) e[k] = NAN; }

/* ---- the moments of the t-ratio ---- */
static int mom_engine(int pmax, const spec_t *R, const dfl_spec *S, double c, double c0, int engine, double *m, char *err, size_t errlen)
{
    dfl_det_opts o; det_opts_of(R, c0, 0, &o);
    double me[64];
    int gen;
    if (engine == DFL_ENGINE_GENERAL) gen = 1;
    else if (engine == DFL_ENGINE_CLASSICAL) {
        if (R->model == 0 || R->model == 4) { snprintf(err, errlen, R->model == 4 ? "the moments are not available for case DG" : "engine classical: the classical engine has the cases DK, DU, DT, SK, SU, ST only"); return 1; }
        gen = 0;
    } else if (S->D) gen = !(classical_valid(R, c, c0, 0) && R->model != 4) || dfl_autoengine(DFL_F_MOM, R->model, R->stat, R->g0, R->gls, c, c0, 0.0) == DFL_ENGINE_GENERAL;
    else {
        if (dfl_check_args(R->model, c, c0, R->g0, R->stat, err, errlen)) return 1;
        if (R->model == 4) { snprintf(err, errlen, "the moments are not available for case DG"); return 1; }
        gen = dfl_autoengine(DFL_F_MOM, R->model, R->stat, R->g0, R->gls, c, c0, 0.0) == DFL_ENGINE_GENERAL;
    }
    return gen ? dfl_det_moments(R->D, c, pmax, &o, m, NULL, err, errlen) : dfl_moments(R->model, c, c0, R->g0, R->stat, pmax, m, pmax <= 64 ? me : NULL, err, errlen);
}
/* the moments E t^k, k = 1..pmax: by default (method 0 with engine auto, a classical case, c <= DFL_INV_CMAX, pmax up to
 * DFL_KAN_JMAX - 1) from one jet per node of an adaptive quadrature in log s (the moments from jets of Bao and Kan,
 * tratio_mom1); tol (1e-10) the relative tolerance of the quadrature, the estimate of moment j relative to
 * max(|m_j|, m_2^{j/2}); DFL_METHOD_INVERSION (the jets) forces them, DFL_METHOD_ENGINE the engines' circles (inv_mom) */
int dfl_engine_mom_x(int pmax, const dfl_spec *S, double c, double c0, int engine, const dfl_inv_opts *io0, double *m, double *errest, char *err, size_t errlen)
{
    warn_entry();
    dfl_inv_opts d; const dfl_inv_opts *io = inv_opts_or_default(io0, &d);
    if (pmax < 1) { snprintf(err, errlen, "p must be a positive integer"); return 1; }
    if (!(io->method == 0 || io->method == DFL_METHOD_INVERSION || io->method == DFL_METHOD_ENGINE)) {
        snprintf(err, errlen, "ltratio_mom: method must be 0 (automatic), %d (the moments from jets, one jet per node after Bao and Kan) or %d (the engines' circles)", DFL_METHOD_INVERSION, DFL_METHOD_ENGINE); return 1;
    }
    double tol; if (inv_tol("ltratio_mom", io, 1e-10, &tol, err, errlen)) return 1;
    if (engine == DFL_ENGINE_GENERAL && io->method == DFL_METHOD_INVERSION) { snprintf(err, errlen, "ltratio_mom: method %d (the jets of Bao and Kan) does not apply with engine general", DFL_METHOD_INVERSION); return 1; }
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    int rc, t = 0;
    if (engine != DFL_ENGINE_GENERAL && (!S->D || (classical_valid(&R, c, c0, 0) && R.model != 4))) {
        if (dfl_check_args(R.model, c, c0, R.g0, R.stat, err, errlen)) { spec_free(&R); return 1; }
        if (R.model == 4) { snprintf(err, errlen, "the moments are not available for case DG"); spec_free(&R); return 1; }
        t = inv_take("ltratio_mom", io->method, engine, &R, c, err, errlen);
        if (t < 0) { spec_free(&R); return 1; }
        if (t && io->method == 0 && !(dfl_kan_mom_ok(c) && pmax <= DFL_KAN_JMAX - 1)) t = 0;
    } else if (io->method == DFL_METHOD_INVERSION) {
        snprintf(err, errlen, "ltratio_mom: method %d (the jets of Bao and Kan) applies to the cases DK, DU, DT, SK, SU, ST", DFL_METHOD_INVERSION); spec_free(&R); return 1;
    }
    if (!t) { rc = mom_engine(pmax, &R, S, c, c0, engine, m, err, errlen); nanfill(errest, pmax); spec_free(&R); return rc; }
    if (pmax > DFL_KAN_JMAX - 1) { snprintf(err, errlen, "ltratio_mom: the moments from jets go up to p = %d (method %d, the engines' circles, beyond)", DFL_KAN_JMAX - 1, DFL_METHOD_ENGINE); spec_free(&R); return 1; }
    dfl_limcf F; int k = pmax < 2 ? 2 : pmax;
    double mm[DFL_KAN_JMAX], ee[DFL_KAN_JMAX];
    rc = lim_of(&R, c, &F, err, errlen) || dfl_kan_ratio_mom(&F, k, c0 - c, 0.5, tol, mm, ee, err, errlen);
    if (!rc) {
        /* (the tolerance relative to max(|m_j|, m_2^{j/2}): a mean near zero, as with c0 = c far above the unit root, has no
         * relative accuracy to speak of) */
        char list[256] = ""; int nbad = 0;
        for (int j = 0; j < pmax; j++) {
            double sc = fmax(fabs(mm[j]), pow(fabs(mm[1]), (j + 1) / 2.0));
            if (!(ee[j] <= tol * sc)) { size_t l = strlen(list); if (l < 200) snprintf(list + l, sizeof list - l, "%s%d", nbad ? ", " : "", j + 1); nbad++; }
        }
        for (int j = 0; j < pmax; j++) { m[j] = mm[j]; if (errest) errest[j] = ee[j]; }
        if (nbad) {
            if (io->verified) { rc = mom_engine(pmax, &R, S, c, c0, DFL_ENGINE_AUTO, m, err, errlen); }
            else det_warn("ltratio_mom: the error estimate of the moments from jets exceeds the tolerance %g (relative) for the moments %s; verified recomputes them by the engines", tol, list);
        }
    }
    spec_free(&R);
    return rc;
}
int dfl_engine_mom(int pmax, const dfl_spec *S, double c, double c0, int engine, double *m, char *err, size_t errlen) { return dfl_engine_mom_x(pmax, S, c, c0, engine, NULL, m, NULL, err, errlen); }

/* ---- the coefficient statistic ---- */
/* the engines' computation: the t-ratio at x = 0 with the null value c0 + y, routed as the t-ratio with the method (0, or a
 * representation) and verified (the warnings of the points collected) */
static int coef_engine(int n, const double *y, const spec_t *R, double c, double c0, int engine, int method, int verified, double *out, char *err, size_t errlen)
{
    dfl_spec S2 = {NULL, R->model, R->start, R->g0, R->gls};
    cwarn_t cw = {""};
    int rc = 0;
    det_warn_hold(1);
    for (int k = 0; k < n && !rc; k++) {
        if (!isfinite(y[k])) { out[k] = isnan(y[k]) ? NAN : y[k] > 0 ? 1.0 : 0.0; continue; }
        double x0 = 0;
        rc = dfl_engine_dist(2, 1, &x0, &S2, c, c0 + y[k], engine, method, verified, &out[k], err, errlen);
        cwarn_take(&cw);
    }
    det_warn_hold(0);
    cwarn_put(&cw);
    return rc;
}
/* the distribution function of the coefficient statistic (U - c0 V)/V at y: by default (method 0 with engine auto, the cases
 * DK, DU, DT with a zero or fixed start and SK, SU, ST, c <= DFL_INV_CMAX) the single inversion of Bao and Kan
 * (coef_cdf1), tol (1e-12) its absolute tolerance; DFL_METHOD_INVERSION forces it, DFL_METHOD_ENGINE and the
 * representations of the t-ratio take the computation of before (the t-ratio at 0 with the null value c0 + y), as do
 * engine classical or general and case DG (inv_coef_cdf) */
int dfl_engine_coef_x(int n, const double *y, const dfl_spec *S, double c, double c0, int engine, const dfl_inv_opts *io0, double *out, double *errest, char *err, size_t errlen)
{
    warn_entry();
    dfl_inv_opts d; const dfl_inv_opts *io = inv_opts_or_default(io0, &d);
    if (!inv_method_ok("coef_cdf", io->method, 1, err, errlen)) return 1;
    double tol; if (inv_tol("coef_cdf", io, DFL_KAN_TOL, &tol, err, errlen)) return 1;
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    int gen = engine == DFL_ENGINE_GENERAL || (engine == DFL_ENGINE_AUTO && S->D && !classical_valid(&R, c, c0, 0));
    if (!gen && engine == DFL_ENGINE_CLASSICAL && R.model == 0) { snprintf(err, errlen, "engine classical: the classical engine has the cases DK, DU, DT, SK, SU, ST and DG only"); spec_free(&R); return 1; }
    int rc = 0, t = 0;
    if (gen) {
        if (io->method == DFL_METHOD_INVERSION) { inv_take("coef_cdf", io->method, engine == DFL_ENGINE_GENERAL ? engine : DFL_ENGINE_AUTO, &R, c, err, errlen); spec_free(&R); return 1; }
        dfl_det_opts o; det_opts_of(&R, c0, io->verified, &o); rc = dfl_det_coef(2, R.D, n, y, c, &o, out, err, errlen);
        nanfill(errest, n); spec_free(&R); return rc;
    }
    if (dfl_check_args(R.model, c, c0, R.g0, R.stat, err, errlen)) { spec_free(&R); return 1; }
    t = inv_take("coef_cdf", io->method, engine, &R, c, err, errlen);
    if (t < 0) { spec_free(&R); return 1; }
    if (!t) {
        int m = io->method == DFL_METHOD_ENGINE ? 0 : io->method;
        rc = coef_engine(n, y, &R, c, c0, engine, m, io->verified, out, err, errlen);
        nanfill(errest, n); spec_free(&R); return rc;
    }
    dfl_limcf F;
    if (lim_of(&R, c, &F, err, errlen)) { spec_free(&R); return 1; }
    double *E = malloc((n + 1) * sizeof(double));
    imap_t mp = {1, &F, y, 0, c0, tol, out, E, NULL, NULL};
    rc = imap(&mp, n, "coef_cdf", err, errlen);
    if (!rc) {
        if (io->verified) {
            cwarn_t cw = {""};
            for (int k = 0; k < n && !rc; k++) if (!(E[k] <= tol)) { rc = coef_engine(1, &y[k], &R, c, c0, DFL_ENGINE_AUTO, 0, 1, &out[k], err, errlen); cwarn_take(&cw); }
            cwarn_put(&cw);
        } else inv_warn("coef_cdf", n, E, NULL, tol);
        if (errest) memcpy(errest, E, n * sizeof(double));
    }
    free(E); spec_free(&R);
    return rc;
}
int dfl_engine_coef(int n, const double *y, const dfl_spec *S, double c, double c0, int engine, double *out, char *err, size_t errlen) { return dfl_engine_coef_x(n, y, S, c, c0, engine, NULL, out, NULL, err, errlen); }
/* its quantiles: by default (as dfl_engine_coef_x) the Illinois variant of regula falsi on the single inversion's distribution
 * function (tolerance DFL_KAN_TOL) from the bracket E[U - cV]/E[V] -/+ twice the scale of the ratio, to the width tol (1e-9);
 * errest the estimate of the distribution function at the quantile divided by the density there.  Otherwise the general
 * engine's Newton search, or the Illinois variant of regula falsi from [-40, 10] on the engines' distribution function
 * (coef_icdf, inv_coef_icdf) */
static int coef_icdf_engine(int n, const double *p, const spec_t *R, double c, double c0, int engine, int method, int verified, double tol, double *q, char *err, size_t errlen)
{
    dfl_spec S2 = {NULL, R->model, R->start, R->g0, R->gls};
    dfl_inv_opts eo = {method == 0 ? DFL_METHOD_ENGINE : method, NAN, verified};
    cwarn_t cw = {""};
    int rc = 0;
    det_warn_hold(1);
#define FX(xx, o) do { double y_ = (xx); rc = dfl_engine_coef_x(1, &y_, &S2, c, c0, engine, &eo, &(o), NULL, err, errlen); cwarn_take(&cw); if (rc) goto done; (o) -= pk; } while (0)
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
    return rc;
}
int dfl_engine_coef_icdf_x(int n, const double *p, const dfl_spec *S, double c, double c0, int engine, double tol, const dfl_inv_opts *io0, double *q, double *errest, char *err, size_t errlen)
{
    warn_entry();
    dfl_inv_opts d; const dfl_inv_opts *io = inv_opts_or_default(io0, &d);
    if (!inv_method_ok("coef_icdf", io->method, 1, err, errlen)) return 1;
    if (isnan(tol) || tol <= 0) tol = 1e-9;
    if (!isfinite(tol)) { snprintf(err, errlen, "coef_icdf: tol must be a positive finite number (NaN or <= 0: the default 1e-9)"); return 1; }
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    int gen = engine == DFL_ENGINE_GENERAL || (engine == DFL_ENGINE_AUTO && S->D && !classical_valid(&R, c, c0, 0));
    int rc = 0, t;
    if (gen) {
        if (io->method == DFL_METHOD_INVERSION) { inv_take("coef_icdf", io->method, engine == DFL_ENGINE_GENERAL ? engine : DFL_ENGINE_AUTO, &R, c, err, errlen); spec_free(&R); return 1; }
        dfl_det_opts o; det_opts_of(&R, c0, 0, &o); rc = dfl_det_coef_quantile(R.D, n, p, c, NAN, tol, &o, q, err, errlen);
        nanfill(errest, n); spec_free(&R); return rc;
    }
    for (int k = 0; k < n; k++) if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "coef_icdf: p must lie in (0, 1)"); spec_free(&R); return 1; }
    if (engine == DFL_ENGINE_CLASSICAL && R.model == 0) { snprintf(err, errlen, "engine classical: the classical engine has the cases DK, DU, DT, SK, SU, ST and DG only"); spec_free(&R); return 1; }
    if (dfl_check_args(R.model, c, c0, R.g0, R.stat, err, errlen)) { spec_free(&R); return 1; }
    t = inv_take("coef_icdf", io->method, engine, &R, c, err, errlen);
    if (t < 0) { spec_free(&R); return 1; }
    if (!t) {
        rc = coef_icdf_engine(n, p, &R, c, c0, engine, io->method == DFL_METHOD_ENGINE ? 0 : io->method, io->verified, tol, q, err, errlen);
        nanfill(errest, n); spec_free(&R); return rc;
    }
    dfl_limcf F;
    if (lim_of(&R, c, &F, err, errlen)) { spec_free(&R); return 1; }
    double *E = malloc((n + 1) * sizeof(double));
    imap_t mp = {5, &F, p, tol, c0, DFL_KAN_TOL, q, E, NULL, NULL};
    rc = imap(&mp, n, "coef_icdf", err, errlen);
    if (!rc) {
        if (io->verified) {
            cwarn_t cw = {""};
            for (int k = 0; k < n && !rc; k++) if (!(E[k] <= DFL_KAN_TOL)) { rc = coef_icdf_engine(1, &p[k], &R, c, c0, DFL_ENGINE_AUTO, 0, 1, tol, &q[k], err, errlen); cwarn_take(&cw); E[k] = NAN; }
            cwarn_put(&cw);
        } else inv_warn("coef_icdf", n, E, NULL, DFL_KAN_TOL);
        if (errest && !rc) {
            /* (in y: the estimate of the distribution function divided by the density) */
            for (int k = 0; k < n; k++) { double f, e, ft; if (isnan(E[k]) || dfl_kan_coef_pdf1(&F, q[k], c0, DFL_KAN_TOL, &f, &e, &ft)) errest[k] = NAN; else errest[k] = E[k] / f; }
        }
    }
    free(E); spec_free(&R);
    return rc;
}
int dfl_engine_coef_icdf(int n, const double *p, const dfl_spec *S, double c, double c0, int engine, double tol, double *q, char *err, size_t errlen)
{
    return dfl_engine_coef_icdf_x(n, p, S, c, c0, engine, tol, NULL, q, NULL, err, errlen);
}
/* the density of the coefficient statistic: by default (as dfl_engine_coef_x; also with engine classical, which has no
 * density of the coefficient statistic) the single inversion of Bao and Kan, the density of a ratio by the formula of
 * Geary from jets (coef_pdf1), tol its tolerance relative to the scale max(1, E[V]/sd(U)) of the density; otherwise
 * (DFL_METHOD_ENGINE, engine general, case DG, a design beyond the classical cases) the general engine's density, the kernel
 * 2 v^3 of its contour integral (dfl_det_coef), with verified its checked computation */
int dfl_engine_coef_pdf(int n, const double *y, const dfl_spec *S, double c, double c0, int engine, const dfl_inv_opts *io0, double *out, double *errest, char *err, size_t errlen)
{
    warn_entry();
    dfl_inv_opts d; const dfl_inv_opts *io = inv_opts_or_default(io0, &d);
    if (!inv_method_ok("coef_pdf", io->method, 0, err, errlen)) return 1;
    if (engine < 0 || engine > 2) { snprintf(err, errlen, "engine must be auto (0), classical (1) or general (2)"); return 1; }
    double tol; if (inv_tol("coef_pdf", io, DFL_KAN_TOL, &tol, err, errlen)) return 1;
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    int rc = 0, t = 0;
    if (S->D && engine != DFL_ENGINE_GENERAL && R.model == 0 && engine == DFL_ENGINE_CLASSICAL) {
        snprintf(err, errlen, "engine classical: the classical engine has the cases DK, DU, DT, SK, SU, ST and DG only"); spec_free(&R); return 1;
    }
    int cls = engine != DFL_ENGINE_GENERAL && (!S->D || engine == DFL_ENGINE_CLASSICAL || classical_valid(&R, c, c0, 0));
    if (cls) {
        if (dfl_check_args(R.model, c, c0, R.g0, R.stat, err, errlen)) { spec_free(&R); return 1; }
        t = inv_take("coef_pdf", io->method, engine == DFL_ENGINE_CLASSICAL ? DFL_ENGINE_AUTO : engine, &R, c, err, errlen);
        if (t < 0) { spec_free(&R); return 1; }
    } else if (io->method == DFL_METHOD_INVERSION) {
        inv_take("coef_pdf", io->method, engine, &R, c, err, errlen);
        if (engine != DFL_ENGINE_GENERAL) snprintf(err, errlen, "coef_pdf: method %d applies to the cases DK, DU, DT, SK, SU, ST", DFL_METHOD_INVERSION);
        spec_free(&R); return 1;
    }
    dfl_det_opts o; det_opts_of(&R, c0, io->verified, &o);
    if (!t) {
        rc = dfl_det_coef(1, R.D, n, y, c, &o, out, err, errlen);
        if (!rc) for (int k = 0; k < n; k++) if (out[k] < 0) out[k] = 0;
        nanfill(errest, n); spec_free(&R); return rc;
    }
    dfl_limcf F;
    if (lim_of(&R, c, &F, err, errlen)) { spec_free(&R); return 1; }
    double *E = malloc((n + 1) * sizeof(double)), *T = malloc((n + 1) * sizeof(double));
    imap_t mp = {2, &F, y, 0, c0, tol, out, E, T, NULL};
    rc = imap(&mp, n, "coef_pdf", err, errlen);
    if (!rc) {
        if (io->verified) {
            o.verified = 1;
            for (int k = 0; k < n && !rc; k++) if (!(E[k] <= T[k])) { rc = dfl_det_coef(1, R.D, 1, &y[k], c, &o, &out[k], err, errlen); if (!rc && out[k] < 0) out[k] = 0; }
        } else inv_warn("coef_pdf", n, E, T, tol);
        if (errest) memcpy(errest, E, n * sizeof(double));
    }
    free(E); free(T); spec_free(&R);
    return rc;
}
/* the moments E[{(U - c0 V)/V}^j], j = 1..pmax, of the coefficient statistic, the cases DK, DU, DT (zero or fixed start) and
 * SK, SU, ST: Sawa's representation with the derivative from one jet per node (coef_mom1), tol (1e-10) the relative
 * tolerance of the quadrature; up to c = DFL_INV_CMAX with method 0 (DFL_METHOD_INVERSION beyond); the engines have no
 * moments of the coefficient statistic (coef_mom) */
int dfl_engine_coef_mom(int pmax, const dfl_spec *S, double c, double c0, int engine, const dfl_inv_opts *io0, double *m, double *errest, char *err, size_t errlen)
{
    warn_entry();
    dfl_inv_opts d; const dfl_inv_opts *io = inv_opts_or_default(io0, &d);
    if (pmax < 1) { snprintf(err, errlen, "p must be a positive integer"); return 1; }
    if (!(io->method == 0 || io->method == DFL_METHOD_INVERSION)) { snprintf(err, errlen, "coef_mom: method must be 0 (automatic: the jets of Bao and Kan) or %d", DFL_METHOD_INVERSION); return 1; }
    if (engine == DFL_ENGINE_GENERAL) { snprintf(err, errlen, "coef_mom: the general engine has no moments of the coefficient statistic"); return 1; }
    if (engine != DFL_ENGINE_AUTO && engine != DFL_ENGINE_CLASSICAL) { snprintf(err, errlen, "engine must be auto (0), classical (1) or general (2)"); return 1; }
    double tol; if (inv_tol("coef_mom", io, 1e-10, &tol, err, errlen)) return 1;
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    int ic = R.model >= 1 && R.model <= 3;
    if (S->D && !ic) { snprintf(err, errlen, "coef_mom: the moments of the coefficient statistic are available for the cases DK, DU, DT, SK, SU, ST (trend(-1), trend(0), trend(1) with a zero, fixed or stationary start)"); spec_free(&R); return 1; }
    if (dfl_check_args(R.model, c, c0, R.g0, R.stat, err, errlen)) { spec_free(&R); return 1; }
    if (!ic) { snprintf(err, errlen, "coef_mom: the moments of the coefficient statistic are available for the cases DK, DU, DT, SK, SU, ST"); spec_free(&R); return 1; }
    if (!(c <= DFL_INV_CMAX) && io->method == 0) { snprintf(err, errlen, "coef_mom: c must be at most %g (method %d beyond)", DFL_INV_CMAX, DFL_METHOD_INVERSION); spec_free(&R); return 1; }
    if (pmax > DFL_KAN_JMAX - 1) { snprintf(err, errlen, "coef_mom: the moments from jets go up to p = %d", DFL_KAN_JMAX - 1); spec_free(&R); return 1; }
    dfl_limcf F; double ee[DFL_KAN_JMAX];
    int rc = lim_of(&R, c, &F, err, errlen) || dfl_kan_ratio_mom(&F, pmax, c0 - c, 1.0, tol, m, ee, err, errlen);
    if (!rc && errest) memcpy(errest, ee, pmax * sizeof(double));
    spec_free(&R);
    return rc;
}

/* ---- the linear form U - aV ---- */
/* the engines' computation for a case DK (model 1, zero start) or DG (model 4, detrended at cbar): the rule of the C router */
static int lin_engine_case(int n, const double *b, double a, int model, double c, double cbar, int engine, double *out, char *err, size_t errlen)
{
    int gen = engine == DFL_ENGINE_GENERAL || (engine == DFL_ENGINE_AUTO && dfl_autoengine(DFL_F_LINCDF, model, 0, 0.0, cbar, c, 0.0, a) == DFL_ENGINE_GENERAL);
    if (!gen) return dfl_lincdf(n, b, a, model, c, cbar, NULL, out, err, errlen);
    dfl_design D; if (dfl_design_trend(model == 4 ? 1 : -1, &D, err, errlen)) return 1;
    dfl_det_opts o; dfl_det_defaults(&o); if (model == 4) o.gls = cbar;
    int rc = dfl_det_lin(2, &D, n, b, a, c, &o, out, err, errlen);
    dfl_design_free(&D);
    return rc;
}
/* the engines' computation of a classical case through its design (lin_engine): case DK with a zero start by the case code,
 * the others by the general engine */
static int lin_engine_spec(int n, const double *b, double a, const spec_t *R, double c, double *out, char *err, size_t errlen)
{
    if (R->model == 1 && !R->stat && R->start == 0) return lin_engine_case(n, b, a, 1, c, DFL_CBAR_GLS, DFL_ENGINE_AUTO, out, err, errlen);
    dfl_det_opts o; det_opts_of(R, 0.0, 0, &o);
    return dfl_det_lin(2, R->D, n, b, a, c, &o, out, err, errlen);
}
/* the single inversion where dfl_kan_lin_ok takes the points (all with method DFL_METHOD_INVERSION), the engines at the
 * others; 0 or 1 (an error), *done 0 where no point is taken (inv_lin) */
static int lin_inv(int n, const double *b, double a, const spec_t *R, double c, int engine, const dfl_inv_opts *io, double tol, double *out, double *errest, int *done, char *err, size_t errlen)
{
    *done = 0;
    int t = inv_take("ltratio_lincdf", io->method, engine, R, c, err, errlen);
    if (t < 0) return 1;
    if (!t) return 0;
    double g0 = R->start == 1 ? R->g0 : 0.0;
    int *ok = malloc((n + 1) * sizeof(int)), nok = 0;
    for (int k = 0; k < n; k++) { ok[k] = io->method == DFL_METHOD_INVERSION || dfl_kan_lin_ok(R->model, R->stat, c, g0, a); nok += ok[k]; }
    if (nok == 0) { free(ok); return 0; }
    dfl_limcf F;
    if (lim_of(R, c, &F, err, errlen)) { free(ok); return 1; }
    double *bx = malloc((n + 1) * sizeof(double)), *v = malloc((n + 1) * sizeof(double)), *E = malloc((n + 1) * sizeof(double)); int *ix = malloc((n + 1) * sizeof(int)), m = 0;
    for (int k = 0; k < n; k++) if (ok[k]) { bx[m] = b[k]; ix[m++] = k; }
    imap_t mp = {3, &F, bx, a, 0, tol, v, E, NULL, NULL};
    int rc = imap(&mp, m, "ltratio_lincdf", err, errlen);
    if (!rc) {
        nanfill(errest, n);
        for (int j = 0; j < m; j++) { out[ix[j]] = v[j]; if (errest) errest[ix[j]] = E[j]; }
        if (m < n) {
            int q = 0; for (int k = 0; k < n; k++) if (!ok[k]) bx[q++] = b[k];
            rc = lin_engine_spec(q, bx, a, R, c, v, err, errlen);
            q = 0; for (int k = 0; k < n && !rc; k++) if (!ok[k]) out[k] = v[q++];
        }
        if (!rc) {
            if (io->verified) {
                for (int j = 0; j < m && !rc; j++) if (!(E[j] <= tol)) rc = lin_engine_spec(1, &b[ix[j]], a, R, c, &out[ix[j]], err, errlen);
            } else inv_warn("ltratio_lincdf", m, E, NULL, tol);
        }
    }
    free(ok); free(bx); free(v); free(E); free(ix);
    *done = !rc;
    return rc;
}
/* P(U - aV <= b): a case code (S->D NULL: model 1 (or 2) case DK with the zero start, model 4 case DG detrended at cbar,
 * NaN for -13.5) or a design (with its start, g0 and gls; cbar unused).  By default (method 0 with engine auto, a finite
 * a != 0) the single inversion of Bao and Kan in case DK with a zero start for -100 <= c <= 4 and in the six classical
 * cases through their designs (fixed and stationary starts included) for -100 <= c <= 2 (dfl_kan_lin_ok), tol (1e-12) its
 * absolute tolerance; elsewhere, with DFL_METHOD_ENGINE, engine classical or general, case DG and a = NaN (the law of V),
 * the engines as before (ltratio_lincdf of DFExact.jl and merged.jl) */
int dfl_engine_lincdf_x(int n, const double *b, double a, const dfl_spec *S, double c, double cbar, int engine, const dfl_inv_opts *io0, double *out, double *errest, char *err, size_t errlen)
{
    warn_entry();
    dfl_inv_opts d; const dfl_inv_opts *io = inv_opts_or_default(io0, &d);
    if (!inv_method_ok("ltratio_lincdf", io->method, 0, err, errlen)) return 1;
    double tol; if (inv_tol("ltratio_lincdf", io, DFL_KAN_TOL, &tol, err, errlen)) return 1;
    /* (a = +-Inf and an infinite cbar gave 0 in the classical code: refused, as by the general engine; a = NaN is the law
     * of V) */
    if (isinf(a)) { snprintf(err, errlen, "ltratio_lincdf: a must be a finite real scalar (NaN: the law of V)"); return 1; }
    int rc, done = 0;
    if (S->D == NULL) {
        int model = S->model == 2 ? 1 : S->model;
        if (model != 1 && model != 4) { snprintf(err, errlen, "ltratio_lincdf: the case must be DK or DG"); return 1; }
        if (S->start != 0 || S->g0 != 0) { snprintf(err, errlen, "ltratio_lincdf: the case %s has the zero start here; a fixed or stationary start takes a design (trend(-1) with g0 or start)", model == 1 ? "DK" : "DG"); return 1; }
        if (isnan(cbar)) cbar = DFL_CBAR_GLS;
        if (model == 4 && !isfinite(cbar)) { snprintf(err, errlen, "ltratio_lincdf: cbar must be a finite real scalar < 0 (NaN: the default -13.5)"); return 1; }
        if (engine == DFL_ENGINE_GENERAL) {
            if (io->method == DFL_METHOD_INVERSION) { snprintf(err, errlen, "ltratio_lincdf: method %d (the single inversion of Bao and Kan) does not apply with engine general", DFL_METHOD_INVERSION); return 1; }
            nanfill(errest, n);
            return lin_engine_case(n, b, a, model, c, cbar, engine, out, err, errlen);
        }
        if (!isfinite(c) || c < DFL_CMIN) { snprintf(err, errlen, "ltratio_lincdf: %s", isfinite(c) ? DFL_CMIN_MSG : "c must be a finite real scalar"); return 1; }
        if (io->method == DFL_METHOD_INVERSION) {
            if (isnan(a)) { snprintf(err, errlen, "ltratio_lincdf: method %d needs a finite a (a = NaN, the law of V, is the general engine's)", DFL_METHOD_INVERSION); return 1; }
            if (model == 4) { snprintf(err, errlen, "ltratio_lincdf: method %d applies to case DK (the designs of the six classical cases), not to case DG", DFL_METHOD_INVERSION); return 1; }
        }
        if (model == 1 && !isnan(a)) {
            spec_t R; dfl_spec S1 = {NULL, 1, 0, 0.0, NAN};
            if (resolve(&S1, &R, err, errlen)) return 1;
            rc = lin_inv(n, b, a, &R, c, engine, io, tol, out, errest, &done, err, errlen);
            spec_free(&R);
            if (rc || done) return rc;
        }
        nanfill(errest, n);
        return lin_engine_case(n, b, a, model, c, cbar, engine, out, err, errlen);
    }
    spec_t R; if (resolve(S, &R, err, errlen)) return 1;
    int inv = engine != DFL_ENGINE_GENERAL && R.model >= 1 && R.model <= 3 && isfinite(a) && classical_valid(&R, c, 0.0, 0);
    if (io->method == DFL_METHOD_INVERSION && !inv) {
        snprintf(err, errlen, "ltratio_lincdf: method %d applies to the cases DK, DU, DT, SK, SU, ST (trend(-1), trend(0), trend(1)) within the classical engine's checks, a finite a and engine auto or classical", DFL_METHOD_INVERSION);
        spec_free(&R); return 1;
    }
    if (inv) { rc = lin_inv(n, b, a, &R, c, engine, io, tol, out, errest, &done, err, errlen); if (rc || done) { spec_free(&R); return rc; } }
    nanfill(errest, n);
    /* (the classical linear form has the zero start only: no deterministic terms with a fixed start take the general engine) */
    int cm = R.start != 0 ? 0 : R.model == 1 ? 1 : R.model == 4 ? 4 : 0;
    if (engine != DFL_ENGINE_GENERAL && cm) { rc = lin_engine_case(n, b, a, cm, c, R.gls, engine, out, err, errlen); spec_free(&R); return rc; }
    if (engine == DFL_ENGINE_CLASSICAL) { snprintf(err, errlen, "engine classical: the linear form of the classical engine has the cases DK and DG only"); spec_free(&R); return 1; }
    dfl_det_opts o; det_opts_of(&R, 0.0, 0, &o);
    rc = dfl_det_lin(2, R.D, n, b, a, c, &o, out, err, errlen);
    spec_free(&R);
    return rc;
}
int dfl_engine_lincdf(int n, const double *b, double a, int model, double c, double cbar, int engine, double *out, char *err, size_t errlen)
{
    dfl_spec S = {NULL, model, 0, 0.0, NAN};
    return dfl_engine_lincdf_x(n, b, a, &S, c, cbar, engine, NULL, out, NULL, err, errlen);
}
/* the power envelope of case DK (model 1 or 2) or DG (model 4): by default (method 0 with engine auto) in case DK for
 * c <= -1e-4 both the critical value (regula falsi to 1e-12) and the power by the single inversion of Bao and Kan
 * (envelope1), errest the sum of the two estimates; elsewhere (DFL_METHOD_ENGINE, engine classical or general, case DG,
 * -1e-4 < c < 0) the engines as before (inv_envelope) */
static int env_engine(int n, const double *cs, int model, double alpha, int engine, double *out, char *err, size_t errlen)
{
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
int dfl_engine_envelope_x(int n, const double *cs, int model, double alpha, int engine, const dfl_inv_opts *io0, double *out, double *errest, char *err, size_t errlen)
{
    warn_entry();
    dfl_inv_opts d; const dfl_inv_opts *io = inv_opts_or_default(io0, &d);
    if (!inv_method_ok("power_envelope", io->method, 0, err, errlen)) return 1;
    double tol; if (inv_tol("power_envelope", io, DFL_KAN_TOL, &tol, err, errlen)) return 1;
    if (engine == DFL_ENGINE_GENERAL && io->method == DFL_METHOD_INVERSION) { snprintf(err, errlen, "power_envelope: method %d (the single inversion of Bao and Kan) does not apply with engine general", DFL_METHOD_INVERSION); return 1; }
    if (model == 2) model = 1;
    if (model != 1 && model != 4) { snprintf(err, errlen, "power_envelope: the case must be DK (or DU) or DG"); return 1; }
    if (!(alpha > 0 && alpha < 1)) { snprintf(err, errlen, "power_envelope: the level must lie in (0, 1)"); return 1; }
    if (io->method == DFL_METHOD_INVERSION && model == 4) { snprintf(err, errlen, "power_envelope: method %d applies to case DK (DU), not to case DG", DFL_METHOD_INVERSION); return 1; }
    int take = model == 1 && (io->method == DFL_METHOD_INVERSION || (io->method == 0 && engine == DFL_ENGINE_AUTO));
    nanfill(errest, n);
    if (!take) return env_engine(n, cs, model, alpha, engine, out, err, errlen);
    double *cx = malloc((n + 1) * sizeof(double)), *v = malloc((n + 1) * sizeof(double)), *E = malloc((n + 1) * sizeof(double)); int *ix = malloc((n + 1) * sizeof(int)), m = 0, q = 0, rc = 0;
    for (int k = 0; k < n; k++) if (io->method == DFL_METHOD_INVERSION || dfl_kan_envelope_ok(cs[k])) { cx[m] = cs[k]; ix[m++] = k; }
    imap_t mp = {4, NULL, cx, alpha, 0, tol, v, E, NULL, NULL};
    rc = imap(&mp, m, "power_envelope", err, errlen);
    for (int j = 0; j < m && !rc; j++) { out[ix[j]] = v[j]; if (errest) errest[ix[j]] = E[j]; }
    if (!rc && m < n) {
        for (int k = 0, j = 0; k < n; k++) { if (j < m && ix[j] == k) { j++; continue; } cx[q++] = cs[k]; }
        rc = env_engine(q, cx, 1, alpha, DFL_ENGINE_AUTO, v, err, errlen);
        for (int k = 0, j = 0, l = 0; k < n && !rc; k++) { if (j < m && ix[j] == k) { j++; continue; } out[k] = v[l++]; }
    }
    if (!rc) {
        if (io->verified) {
            for (int j = 0; j < m && !rc; j++) if (!(E[j] <= tol)) rc = env_engine(1, &cs[ix[j]], 1, alpha, DFL_ENGINE_AUTO, &out[ix[j]], err, errlen);
        } else inv_warn("power_envelope", m, E, NULL, tol);
    }
    free(cx); free(v); free(E); free(ix);
    return rc;
}
int dfl_engine_envelope(int n, const double *cs, int model, double alpha, int engine, double *out, char *err, size_t errlen)
{
    return dfl_engine_envelope_x(n, cs, model, alpha, engine, NULL, out, NULL, err, errlen);
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
    /* (engine auto: the rule, dfl_autoengine with x = 1 for the parabola, c0 = par[1]) */
    int gen = engine == DFL_ENGINE_GENERAL || (engine == DFL_ENGINE_AUTO && dk && type == 2 && dfl_autoengine(DFL_F_REGION, 1, 0, 0.0, NAN, c, par[1], 1.0) == DFL_ENGINE_GENERAL);
    if (!gen && dk) {
        dfl_double_info info;
        rc = dfl_region(type, par, 0.0, 0, NULL, c, NULL, out, &info, err, errlen);
    } else {
        dfl_det_opts o; det_opts_of(&R, 0.0, 0, &o);
        rc = dfl_det_region(R.D, type, par, c, &o, out, err, errlen);
    }
    spec_free(&R);
    return rc;
}
