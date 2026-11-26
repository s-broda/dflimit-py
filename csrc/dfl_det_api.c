/* dfl_det_api.c: the user-level functions of the general engine (a port of DFDet.jl): the distribution function, density
 * and quantiles of the limiting t-ratio (dist_cdf, dist_pdf, dist_quantile) and of the coefficient statistic (coef_cdf,
 * coef_pdf, coef_quantile, coef_cdf_gp), the checked computation (verify_eval), the moments (dist_moments), local power,
 * the linear form U - aV and the law of V (lin_cdf, lin_pdf, lin_quantile, v_cdf, v_pdf), the power envelope, ADFmax and
 * the regions below a line or a parabola (region_prob). */
#include "dfl_det_int.h"

void dfl_det_defaults(dfl_det_opts *o)
{
    o->c0 = 0; o->start = DFL_START_ZERO; o->gamma0 = 0; o->gls = NAN; o->verified = 0; o->vtol = 1e-9; o->phi = NAN; o->hfac = NAN;
    o->method = DFL_DET_AUTO; o->rule = DFL_RULE_AUTO; o->n = 0; o->ntail = 0;
}
static const dfl_det_opts *opts_or_default(const dfl_det_opts *o, dfl_det_opts *d) { if (o) return o; dfl_det_defaults(d); return d; }
/* the largest c at which the general engine computes (gen_cmax of DFDet.jl): DFL_DET_CMAX0 without deterministic terms,
 * DFL_DET_CMAX with them, DFL_DET_CMAX_FIXED with them and a fixed start (dflimit.h) */
double dfl_det_cmax(const dfl_design *D, int start, double gamma0)
{
    if (D && D->K == 0) return DFL_DET_CMAX0;
    return start == DFL_START_FIXED && gamma0 != 0 ? DFL_DET_CMAX_FIXED : DFL_DET_CMAX;
}
/* the common checks of the arguments; what names the statistic in the refusal above dfl_det_cmax */
static int check_common(const dfl_design *D, double c, const dfl_det_opts *o, const char *what, char *err, size_t errlen)
{
    if (!D) { snprintf(err, errlen, "no design"); return 1; }
    if (!isfinite(c)) { snprintf(err, errlen, "c must be a finite real scalar"); return 1; }
    double cb = dfl_det_cmax(D, o->start, o->gamma0);
    if (c > cb) {
        snprintf(err, errlen, "%s at c = %g: the general engine computes up to c = %g for this design and start (%g without deterministic terms; with them %g, and %g with a fixed start; above, the roots of its transform are no longer resolved in double precision); the classical cases DK, DU, DT are available with the engine \"classical\" or the default engine \"auto\"",
                 what, c, cb, DFL_DET_CMAX0, DFL_DET_CMAX, DFL_DET_CMAX_FIXED);
        return 1;
    }
    if (!isfinite(o->c0)) { snprintf(err, errlen, "c0 must be a finite real scalar"); return 1; }
    if (o->start < 0 || o->start > 2) { snprintf(err, errlen, "start must be zero (0), fixed (1) or stationary (2)"); return 1; }
    if (!isfinite(o->gamma0)) { snprintf(err, errlen, "gamma0 must be a finite real scalar"); return 1; }
    if (o->start == 2 && !(c < 0)) { snprintf(err, errlen, "the stationary start needs c < 0"); return 1; }
    return 0;
}
/* the start value of a search (NaN: the default, set before) must be finite, the tolerance (NaN or <= 0: the default) not
 * +Inf (Newton's method from +-Inf did not return, and a tolerance +Inf stopped it after one step) */
static int start_tol_check(const char *fn, const char *name, double x0, double tol, char *err, size_t errlen)
{
    if (!isfinite(x0)) { snprintf(err, errlen, "%s: %s must be a finite real scalar (NaN: the default)", fn, name); return 1; }
    if (tol == INFINITY) { snprintf(err, errlen, "%s: tol must be a positive finite number (NaN: the default)", fn); return 1; }
    return 0;
}
static double nonfinite1(int cdf, double x) { return isnan(x) ? NAN : x > 0 ? (cdf ? 1.0 : 0.0) : 0.0; }
/* the finite points of x (into xf, their indices into ix): their number; the others set in out */
static int split_finite(int n, const double *x, int cdf, double *out, double *xf, int *ix)
{
    int m = 0;
    for (int k = 0; k < n; k++) { if (isfinite(x[k])) { xf[m] = x[k]; ix[m++] = k; } else out[k] = nonfinite1(cdf, x[k]); }
    return m;
}

/* ---------- the checked computation (verify_eval) ---------- */
typedef int (*evalf_t)(det_path *CP, const det_eval_opts *o, void *ctx, double *v, char *err, size_t errlen);
/* one evaluation; n, ntail > 0 the inner nodes of the rule of one evaluation (dfl_det_dist), otherwise those of the present
 * rule */
static int one_eval_n(det_path *CP, evalf_t ev, void *ctx, int n, int ntail, double *v, char *err, size_t errlen)
{
    det_eval_opts o; det_eval_defaults(CP, &o);
    if (n > 0) o.n = n;
    if (ntail > 0) o.ntail = ntail;
    return ev(CP, &o, ctx, v, err, errlen);
}
static int one_eval(det_path *CP, evalf_t ev, void *ctx, int nv, double *v, char *err, size_t errlen)
{
    (void) nv;
    return one_eval_n(CP, ev, ctx, 0, 0, v, err, errlen);
}
static int verify_eval(det_path *CP, evalf_t ev, void *ctx, int nv, double vtol, double *v, char *err, size_t errlen)
{
    int fixed = CP->Mo->start == 1, n0 = fixed ? 48 : 24, nt0 = fixed ? 64 : 40;
    det_eval_opts o; det_eval_defaults(CP, &o);
    o.alpha = 0.5; o.n = n0; o.ntail = nt0;
    if (ev(CP, &o, ctx, v, err, errlen)) return 1;
    double *v2 = malloc(nv * sizeof(double)), *v3 = malloc(nv * sizeof(double));
    det_eval_opts o2 = o; o2.alpha = 0.3; o2.n = n0 + 16; o2.ntail = nt0 + 24;
    det_path *C1 = det_variant_path(CP, 1, err, errlen);
    int rc = C1 ? ev(C1, &o2, ctx, v2, err, errlen) : 1;
    det_path_release(C1);
    if (rc) {
        /* (any error of the variant, also one of the workers of its node loops, which det_fail keeps: the groups not
         * separated on the steeper hyperbola, the guard on the nodes of its U-shapes; the variant at the default angle
         * takes its place.  An exhausted memory reaches the caller, as in the Julia engine) */
        if (strstr(err, "out of memory")) { free(v2); free(v3); return 1; }
        det_path *C0 = det_variant_path(CP, 0, err, errlen);
        rc = C0 ? ev(C0, &o2, ctx, v2, err, errlen) : 1;
        det_path_release(C0);
        if (rc) { free(v2); free(v3); return 1; }
    }
    double d = 0; int fin = 1;
    for (int i = 0; i < nv; i++) { d = dmax(d, fabs(v[i] - v2[i])); if (!isfinite(v[i])) fin = 0; }
    if (d <= vtol || !fin) { free(v2); free(v3); return 0; }
    det_eval_opts o3 = o; o3.alpha = 0.5; o3.n = 2 * n0; o3.ntail = 2 * nt0;
    det_path *C2 = det_variant_path(CP, 2, err, errlen);
    rc = C2 ? ev(C2, &o3, ctx, v3, err, errlen) : 1;
    det_path_release(C2);
    if (rc) { free(v2); free(v3); return 1; }
    double d3 = 0; for (int i = 0; i < nv; i++) d3 = dmax(d3, fabs(v3[i] - v2[i]));
    if (!(d3 <= vtol)) det_warn("verified: two evaluations by different rules differ by %.2g (above %g) (start %d, c = %g)", d3, vtol, CP->Mo->start, CP->c);
    memcpy(v, v3, nv * sizeof(double));
    free(v2); free(v3);
    return 0;
}
static int run_eval(det_path *CP, evalf_t ev, void *ctx, int nv, int verified, double vtol, double *v, char *err, size_t errlen)
{
    return verified ? verify_eval(CP, ev, ctx, nv, vtol, v, err, errlen) : one_eval(CP, ev, ctx, nv, v, err, errlen);
}

/* ---------- the t-ratio ---------- */
typedef struct { int nx; const double *xs; double c0; int which, vpow; } tr_ctx;   /* which 1: F, 2: f */
static int tr_eval(det_path *CP, const det_eval_opts *o, void *v, double *out, char *err, size_t errlen)
{
    tr_ctx *t = v;
    det_path *CPr = det_line_refined(CP, t->nx, t->xs, 1, &CP->Mo->a, 1, &t->c0, err, errlen);
    if (!CPr) return 1;
    double *F = malloc(t->nx * sizeof(double)), *f = malloc(t->nx * sizeof(double));
    int rc = det_contour_eval(CPr, t->nx, t->xs, t->c0, t->vpow, o, F, f, NULL, NULL, err, errlen);
    if (!rc) memcpy(out, t->which == 1 ? F : f, t->nx * sizeof(double));
    free(F); free(f); det_path_release(CPr);
    return rc;
}
/* the outer path of the statistics other than the t-ratio (hfac NaN: 1) */
static det_path *user_path(const dfl_design *D, double c, const dfl_det_opts *o, char *err, size_t errlen)
{
    return det_path_cached(D, o->gls, o->start, o->gamma0, c, o->phi, isnan(o->hfac) ? 1.0 : o->hfac, err, errlen);
}

/* ---------- the rules of one evaluation of the t-ratio's inversion on paths (dist_cdf, dist_pdf, dist_quantile) ----------
 * PATH_RULES of DFDet.jl: (hfac, n, ntail, n with a fixed start, ntail with a fixed start), indexed by DFL_RULE_* (the
 * README, "The rules of one evaluation"; dflimit.h) */
typedef struct { double hfac; int n, ntail, nf, ntf; } det_rule_t;
static const det_rule_t DET_PATH_RULES[7] = {
    {1.0, 24, 40, 48, 64},      /* (DFL_RULE_AUTO: resolved by path_region) */
    {1.0, 24, 40, 48, 64},      /* DFL_RULE_PRESENT */
    {1.5, 16, 32, 32, 48},      /* DFL_RULE_THIN */
    {1.0, 16, 32, 32, 48},      /* DFL_RULE_INNER */
    {1.5, 24, 40, 48, 64},      /* DFL_RULE_OUTER */
    {0.5, 16, 32, 32, 48},      /* DFL_RULE_FINE */
    {1.0, 24, 40, 48, 64}};     /* DFL_RULE_FINESTEP */
#define DET_RULE_GIVEN (-1)     /* (hfac given: the rule's inner nodes on the hyperbola of that step) */
/* the borders of the regions of path_region in c (PATH_UNITROOT_CMIN, PATH_THIN_CMAX_FIXED, PATH_THIN_CMAX of DFDet.jl) */
#define DET_PATH_UNITROOT_CMIN (-1.0)
#define DET_PATH_THIN_CMAX_FIXED 2.0
#define DET_PATH_THIN_CMAX 5.0
/* the angle below which the step of DFL_RULE_FINESTEP shrinks (FINESTEP_PHI) */
#define DET_FINESTEP_PHI (70 * M_PI / 180)
#define DET_RULE_MSG "rule must be 0 (auto), 1 (present), 2 (thin), 3 (inner), 4 (outer), 5 (fine) or 6 (finestep) (the rules of one evaluation of the inversion on paths, DFL_RULE_*)"
/* The rule of one evaluation for a design (rank1: without deterministic terms) with its start at c and the local null value
 * c0, by regions measured in November 2026 against a reference with half the outer step and twice the inner nodes (the
 * criterion: within max(1e-9, twice the present rule's own error) in F and max(1e-8, twice its own) of the largest density
 * in f; path_region of DFDet.jl).  Near and above the unit root (c > -1) the outer integrand of the upper tail needs the
 * outer step of the present rule or a finer one: with c0 < 0, where the parabola X(v) has its root b ~ x/|c0| far out, the
 * thinner outer step lost up to 1e-4 with a fixed start (case DK at c = 0, c0 = -5 by 3.9e-5, case DU at c = 2, c0 = -5 by
 * 3.1e-6) and 1.3e-8 of the density's peak with a zero start (case DK at c = 0, c0 = -5); without deterministic terms and
 * with a fixed start the present step itself was off by up to 0.86 in the upper tail (case DK at c = 0.5, g0 = 1, c0 =
 * -20; 1.8e-3 at c0 = -5), where the steps of DFL_RULE_FINESTEP are within 1e-11 of the classical engine, and with
 * deterministic terms DFL_RULE_FINE within 1e-12; there at c0 = 0 too the thinner step lost 1.4e-8 of the peak at mean +
 * 5.75 sd (c = 0.5).  Beyond the range of the sweep (a fixed start above c = 2, any start above c = 5; at c = 5 with a
 * fixed start the thin rule differed from the present one by up to 5.6e-6 of the peak of the density, Perron's model C with
 * two breaks) and far from the unit root with c0 other than 0 and c (below DET_FAR_CMIN; in case DK at c = -480, c0 = 2 the
 * thin rule differed from the present one by 7.7e-9) the present rule; the thin rule elsewhere. */
int dfl_det_path_region(int fixed, int rank1, double c, double c0)
{
    if (c > DET_PATH_UNITROOT_CMIN) {
        if ((c > DET_PATH_THIN_CMAX_FIXED && fixed) || c > DET_PATH_THIN_CMAX) return DFL_RULE_PRESENT;
        if (c0 < 0) return !fixed ? DFL_RULE_INNER : rank1 ? DFL_RULE_FINESTEP : DFL_RULE_FINE;
        if (c0 == 0 && fixed && rank1) return DFL_RULE_INNER;
    }
    if (c < DET_FAR_CMIN && c0 != 0 && c0 != c) return DFL_RULE_PRESENT;
    return DFL_RULE_THIN;
}
/* The outer path and the inner nodes of the rule of one evaluation (path_rule and rule_path of DFDet.jl): the rule of o->rule
 * (DFL_RULE_AUTO: dfl_det_path_region; DFL_RULE_PRESENT with verified), with o->hfac (a number), o->n, o->ntail (> 0) in
 * its place where given.  The rules with a finer step keep the angle of the present rule's hyperbola (where fixed_phi
 * chooses it, its criterion on the largest term weighs the step, and a finer step took a flatter hyperbola with larger
 * terms: case DK at c = -0.5, g0 = 2, c0 = -20 was off by 1.7e-4 on it, against 4.5e-12 on the present one): DFL_RULE_FINE
 * half the present step, DFL_RULE_FINESTEP the step min(1/2, 2.5/(g0 |c0|)) times the present one where the present
 * hyperbola is flatter than 70 degrees (at c = 0.5, g0 = 1, c0 = -20 the present step was off by 0.86, half of it by 0.31,
 * this one by 7e-12; on the steep hyperbolas of g0 = 2, 88 degrees, the present step is within 1e-12 and the thin inner rule
 * is not); on the vertical line the line itself.  *rl the rule taken (DET_RULE_GIVEN where hfac is given) */
static det_path *rule_path(const dfl_design *D, double c, const dfl_det_opts *o, int verified, int *n, int *ntail, int *rl, char *err, size_t errlen)
{
    if (o->rule < DFL_RULE_AUTO || o->rule > DFL_RULE_FINESTEP) { snprintf(err, errlen, "%s", DET_RULE_MSG); return NULL; }
    int fixed = o->start == DFL_START_FIXED && o->gamma0 != 0, r = o->rule;
    if (verified) r = DFL_RULE_PRESENT;
    else if (r == DFL_RULE_AUTO) {
        int rank1 = 0;
        if (fixed) {
            /* (the rank of the numerator and the negative roots: those of the prepared design with a fixed start) */
            det_prep *P = det_prep_cached(D, o->gls, err, errlen);
            if (!P) return NULL;
            rank1 = P->r == 1 && P->rm == 0;
            det_prep_release(P);
        }
        r = dfl_det_path_region(fixed, rank1, c, o->c0);
    }
    const det_rule_t *R = &DET_PATH_RULES[r];
    double hf = R->hfac, phi = o->phi;
    if (!isnan(o->hfac)) { hf = o->hfac; r = DET_RULE_GIVEN; }
    *n = o->n > 0 ? o->n : fixed ? R->nf : R->n;
    *ntail = o->ntail > 0 ? o->ntail : fixed ? R->ntf : R->ntail;
    if (rl) *rl = r;
    if (isnan(phi) && (r == DFL_RULE_FINE || r == DFL_RULE_FINESTEP)) {
        det_path *CP1 = det_path_cached(D, o->gls, o->start, o->gamma0, c, NAN, 1.0, err, errlen);
        if (!CP1) return NULL;
        if (isnan(CP1->h)) return CP1;
        phi = CP1->phi;
        if (r == DFL_RULE_FINESTEP) hf = CP1->phi < DET_FINESTEP_PHI ? dmin(0.5, 2.5 / (fabs(CP1->Mo->g0) * fabs(o->c0))) : 1.0;
        det_path_release(CP1);
    }
    return det_path_cached(D, o->gls, o->start, o->gamma0, c, phi, hf, err, errlen);
}

/* the values at the finite points xf by the inversion on paths (one evaluation with the inner nodes n, ntail, or the checked
 * computation), with the infinite density at the tangency of a rank-one design with c0 > 0 (tangency!) */
static int paths_values(int kind, det_path *CP, int m, const double *xf, const dfl_det_opts *o, int verified, int n, int ntail, double *v, char *err, size_t errlen)
{
    tr_ctx t = {m, xf, o->c0, kind == 2 ? 1 : 2, 2};
    int rc = verified ? run_eval(CP, tr_eval, &t, m, 1, o->vtol, v, err, errlen) : one_eval_n(CP, tr_eval, &t, n, ntail, v, err, errlen);
    if (!rc && kind == 1 && o->c0 > 0 && CP->Mo->r == 1 && CP->Mo->rm == 0) {
        double xs_ = -sqrt(4 * o->c0 * CP->Mo->a);
        for (int j = 0; j < m; j++) if (fabs(xf[j] - xs_) <= 1e-12 * dmax(1.0, fabs(xf[j]))) v[j] = INFINITY;
    }
    return rc;
}
/* the values by the paths at the points the line integral leaves to them, by the rule of one evaluation (with o->verified:
 * the present rule, as the check of line_values); verified 1 the checked computation on the present rule's hyperbola (or
 * that of a given hfac) */
static int paths_subset(int kind, const dfl_design *D, double c, int m, const double *xf, const dfl_det_opts *o, int verified, double *v, char *err, size_t errlen)
{
    int n = 0, nt = 0;
    det_path *CP = verified ? user_path(D, c, o, err, errlen) : rule_path(D, c, o, o->verified, &n, &nt, NULL, err, errlen);
    if (!CP) return 1;
    int rc = paths_values(kind, CP, m, xf, o, verified, n, nt, v, err, errlen);
    det_path_release(CP);
    return rc;
}
/* the values by the line integral with the explicit kernel (line_dist): one evaluation at each point; at a point where the
 * classical integrator's paths would exceed the bounds of the engine (refused before anything is evaluated) or the
 * continuation of log P fails, method auto takes the inversion on paths and method line returns an error.  verified = 1:
 * the values checked against one evaluation of the inversion on paths; where the two differ by more than vtol, the checked
 * inversion on paths is returned, with a warning */
static int line_values(int kind, const dfl_design *D, dfl_prov *pv, double c, int m, const double *xf, const dfl_det_opts *o, double *v, char *err, size_t errlen)
{
    int *fb = malloc((m + 1) * sizeof(int)), nfb = 0, rc = 0;
    /* (where the line integral was taken only for c0 > 0 on the vertical line, det_line_vline, it computes the points x < 0,
     * whose parabola has two positive roots, and the paths the others, as before: the line integral grows slow at large x) */
    int vonly = o->method == DFL_DET_AUTO && !det_line_auto(D, c, o->c0);
    for (int j = 0; j < m && !rc; j++) {
        if (vonly && !(xf[j] < 0)) { fb[nfb++] = j; continue; }
        int r = det_line_value(pv, kind, xf[j], o->c0, &v[j]);
        if (r == 0) continue;
        if (o->method == DFL_DET_LINE) { det_line_point_error(err, errlen, kind == 2 ? "dist_cdf" : "dist_pdf", pv, xf[j], o->c0, r); rc = 1; break; }
        fb[nfb++] = j;
    }
    if (!rc && nfb > 0) {
        double *xb = malloc(nfb * sizeof(double)), *vb = malloc(nfb * sizeof(double));
        for (int i = 0; i < nfb; i++) xb[i] = xf[fb[i]];
        rc = paths_subset(kind, D, c, nfb, xb, o, 0, vb, err, errlen);
        if (!rc) for (int i = 0; i < nfb; i++) v[fb[i]] = vb[i];
        free(xb); free(vb);
    }
    if (!rc && o->verified) {
        double *vp = malloc(m * sizeof(double)), d = 0;
        rc = paths_subset(kind, D, c, m, xf, o, 0, vp, err, errlen);
        for (int j = 0; j < m && !rc; j++) if (isfinite(v[j]) || isfinite(vp[j])) { double dj = fabs(v[j] - vp[j]); if (!(dj <= d)) d = dj; }
        if (!rc && !(d <= o->vtol)) {
            rc = paths_subset(kind, D, c, m, xf, o, 1, v, err, errlen);
            det_warn("verified: the line integral and the inversion on paths differ by %.2g (above %g); the checked inversion on paths is returned (c = %g, c0 = %g)", d, o->vtol, c, o->c0);
        }
        free(vp);
    }
    free(fb);
    return rc;
}
/* the final check of the general engine's distribution functions (dfl_cdf_range_check; cdf_range_check of DFDet.jl): the
 * design by its regressors, c, c0 and the start */
static int range_check(const char *fn, const dfl_design *D, double c, const dfl_det_opts *o, const char *xname, int m, const double *xs, const double *v, char *err, size_t errlen)
{
    char args[200];
    snprintf(args, sizeof args, "the general engine, a design of %d regressors on %d pieces%s, c = %g, c0 = %g, %s start%s", D->K, D->np, D->first ? " (detrended first)" : "",
             c, o->c0, o->start == 2 ? "stationary" : o->start == 1 ? "fixed" : "zero", isnan(o->gls) ? "" : ", GLS-detrended");
    return dfl_cdf_range_check(fn, args, xname, m, xs, v, err, errlen);
}
/* the brittle region far above the unit root (far_above_warn of DFDet.jl): above c = DET_FAR_ABOVE_CMIN (5.5), beyond the range
 * in which the rules of one evaluation were swept, the inversion on paths by one evaluation was off by more than 1e-5 in the
 * upper tail for designs beyond the classical cases and for a fixed start with deterministic terms (November 26, 2026:
 * Perron's model C at 0.3 by 1.5e-5 in F at c = 7, the linear trend with a fixed start by 8.7e-5 at c = 7); a warning where the
 * paths compute.  The classical designs: trend(-1), trend(0), trend(1) in the one-step regression with least squares
 * detrending, and trend(1) GLS-detrended at cbar = -13.5 with a zero start */
static int is_trend_design(const dfl_design *D, int p)
{
    int K = p + 1, deg = p > 0 ? p : 0;
    if (D->first || D->loc || D->K != K || D->np != 1 || D->br[0] != 0 || D->br[1] != 1 || D->deg != deg) return 0;
    for (int i = 0; i < K; i++) for (int k = 0; k <= deg; k++) if (D->C[i * (deg + 1) + k] != (k == i ? 1.0 : 0.0)) return 0;
    return 1;
}
static void far_above_warn(const char *fn, const dfl_design *D, double c, const dfl_det_opts *o)
{
    if (!(c > DET_FAR_ABOVE_CMIN)) return;
    int fixed = o->start == DFL_START_FIXED && o->gamma0 != 0, classical = 0;
    for (int p = -1; p <= 1 && !classical; p++)
        if (is_trend_design(D, p)) classical = isnan(o->gls) || (p == 1 && o->gls == DFL_CBAR_GLS && o->start == DFL_START_ZERO);
    if (classical && !(fixed && D->K >= 1)) return;
    det_warn("%s: far above the unit root (c = %g > %g) the inversion on paths by one evaluation may be off by more than 1e-5 in the upper tail for a design beyond the classical cases or a fixed start with deterministic terms (Perron's model C at c = 7 by 1.5e-5 in F, the linear trend with a fixed start at c = 7 by 8.7e-5); verified = 1 checks it",
             fn, c, DET_FAR_ABOVE_CMIN);
}
int dfl_det_dist(int kind, const dfl_design *D, int n, const double *x, double c, const dfl_det_opts *o0, double *out, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = opts_or_default(o0, &od);
    det_warn_clear();
    if (kind != 1 && kind != 2) { snprintf(err, errlen, "kind must be 1 (density) or 2 (distribution function)"); return 1; }
    if (check_common(D, c, o, kind == 2 ? "the t-ratio (dist_cdf)" : "the t-ratio (dist_pdf)", err, errlen)) return 1;
    if (o->rule < DFL_RULE_AUTO || o->rule > DFL_RULE_FINESTEP) { snprintf(err, errlen, "%s", DET_RULE_MSG); return 1; }
    dfl_prov *pv;
    if (det_line_select(D, c, o, &pv, err, errlen)) return 1;
    det_path *CP = NULL; int nr = 0, ntr = 0;
    if (!pv) { CP = rule_path(D, c, o, o->verified, &nr, &ntr, NULL, err, errlen); if (!CP) return 1; }
    if (!pv && !o->verified) far_above_warn(kind == 2 ? "dist_cdf" : "dist_pdf", D, c, o);
    double *xf = malloc((n + 1) * sizeof(double)); int *ix = malloc((n + 1) * sizeof(int));
    int m = split_finite(n, x, kind == 2, out, xf, ix), rc = 0;
    if (m > 0) {
        double *v = malloc(m * sizeof(double));
        rc = pv ? line_values(kind, D, pv, c, m, xf, o, v, err, errlen) : paths_values(kind, CP, m, xf, o, o->verified, nr, ntr, v, err, errlen);
        if (!rc) for (int j = 0; j < m; j++) out[ix[j]] = v[j];
        if (!rc && kind == 2) rc = range_check("dist_cdf", D, c, o, "x", m, xf, v, err, errlen);
        free(v);
    }
    free(xf); free(ix); det_path_release(CP); det_line_release(pv);
    return rc;
}
/* F and f of the inversion on paths from one pass (dist_both of DFDet.jl): the values of dfl_det_dist with method paths bit
 * for bit (the same path, rule and points; det_contour_eval returns both), for the Newton steps of the router's quantile
 * search (route_dist2 in dfl_det_route.c) */
int dfl_det_dist_both(const dfl_design *D, int n, const double *x, double c, const dfl_det_opts *o0, double *F, double *f, char *err, size_t errlen)
{
    dfl_det_opts o; dfl_det_defaults(&o);
    if (o0) { o.c0 = o0->c0; o.start = o0->start; o.gamma0 = o0->gamma0; o.gls = o0->gls; }
    det_warn_clear();
    if (check_common(D, c, &o, "the t-ratio (dist_cdf)", err, errlen)) return 1;
    for (int k = 0; k < n; k++) if (!isfinite(x[k])) { snprintf(err, errlen, "dist_both: finite points only"); return 1; }
    int nr = 0, ntr = 0;
    det_path *CP = rule_path(D, c, &o, 0, &nr, &ntr, NULL, err, errlen);
    if (!CP) return 1;
    far_above_warn("dist_cdf", D, c, &o);
    int rc = 0;
    if (n > 0) {
        det_path *CPr = det_line_refined(CP, n, x, 1, &CP->Mo->a, 1, &o.c0, err, errlen);
        if (!CPr) rc = 1;
        else {
            det_eval_opts eo; det_eval_defaults(CPr, &eo); eo.n = nr; eo.ntail = ntr;
            rc = det_contour_eval(CPr, n, x, o.c0, 2, &eo, F, f, NULL, NULL, err, errlen);
            det_path_release(CPr);
        }
        if (!rc && o.c0 > 0 && CP->Mo->r == 1 && CP->Mo->rm == 0) {
            double xs_ = -sqrt(4 * o.c0 * CP->Mo->a);
            for (int j = 0; j < n; j++) if (fabs(x[j] - xs_) <= 1e-12 * dmax(1.0, fabs(x[j]))) f[j] = INFINITY;
        }
        if (!rc) rc = range_check("dist_cdf", D, c, &o, "x", n, x, F, err, errlen);
    }
    det_path_release(CP);
    return rc;
}

/* newton_safe: F(x) = p for an increasing F, ff(x) -> (F, f) */
typedef int (*ff_t)(double x, void *ctx, double *F, double *f, char *err, size_t errlen);
static int newton_safe(ff_t ff, void *ctx, double p, double x0, double step, double tol, double *xout, char *err, size_t errlen)
{
    double x = x0, lo = x - step, hi = x + step, w = step, F, f;
    for (int i = 0; i < 60; i++) { if (ff(lo, ctx, &F, &f, err, errlen)) return 1; if (!(F > p)) break; lo -= w; w *= 2; }
    w = step;
    for (int i = 0; i < 60; i++) { if (ff(hi, ctx, &F, &f, err, errlen)) return 1; if (!(F < p)) break; hi += w; w *= 2; }
    for (int it = 0; it < 100; it++) {
        if (ff(x, ctx, &F, &f, err, errlen)) return 1;
        if (F > p) hi = x; else lo = x;
        double xn = x - (F - p) / f;
        if (!(lo <= xn && xn <= hi && isfinite(xn))) xn = (lo + hi) / 2;   /* (a step below half an ulp lands on the end of the bracket: converged) */
        if (fabs(xn - x) < tol) { *xout = xn; return 0; }
        x = xn;
    }
    det_warn("newton_safe: no convergence (p = %g, x = %g)", p, x);
    *xout = x;
    return 0;
}
/* (n, ntail > 0: the inner nodes of the rule of one evaluation of the t-ratio; 0: the present rule) */
typedef struct { det_path *CP; double c0; int vpow; int refine; int n, ntail; } q_ctx;
static int q_ff(double x, void *v, double *F, double *f, char *err, size_t errlen)
{
    q_ctx *q = v;
    double xx = q->vpow == 3 ? 0.0 : x, c0 = q->vpow == 3 ? q->c0 + x : q->c0;
    det_path *CPr = q->refine ? det_line_refined(q->CP, 1, &xx, 1, &q->CP->Mo->a, 1, &c0, err, errlen) : q->CP;
    if (!CPr) return 1;
    det_eval_opts eo; det_eval_defaults(CPr, &eo);
    if (q->n > 0) eo.n = q->n;
    if (q->ntail > 0) eo.ntail = q->ntail;
    int rc = det_contour_eval(CPr, 1, &xx, c0, q->vpow, &eo, F, f, NULL, NULL, err, errlen);
    if (q->refine) det_path_release(CPr);
    return rc;
}
/* Newton's method from x0 with the step limited to lim (maxit steps), the safeguarded search where it does not converge */
static int quantile_search(q_ctx *q, double p, double x0, double tol, int maxit, double lim, double *out, char *err, size_t errlen)
{
    double x = x0, F, f;
    for (int it = 0; it < maxit; it++) {
        if (q_ff(x, q, &F, &f, err, errlen)) return 1;
        double dx = dclamp((F - p) / f, -lim, lim);
        if ((F - p) / f != (F - p) / f) dx = NAN;
        if (!isfinite(dx)) break;
        x -= dx;
        if (fabs(dx) < tol) { *out = x; return 0; }
    }
    return newton_safe(q_ff, q, p, x0, lim, tol, out, err, errlen);
}
/* the quantile by the line integral (line_quantile): Newton's method on F(x) - p with the density, from x0, the step limited
 * to 1, then the safeguarded search; rc 2 where a point of the search is beyond the line integral (the bounds, the
 * continuation): method auto then takes the inversion on paths for this quantile, method line returns an error */
typedef struct { dfl_prov *pv; double c0; int bad; double xbad; } lineq_ctx;
static int lineq_ff(double x, void *v, double *F, double *f, char *err, size_t errlen)
{
    lineq_ctx *q = v; (void) err; (void) errlen;
    int r = det_line_value(q->pv, 2, x, q->c0, F);
    if (r == 0) r = det_line_value(q->pv, 1, x, q->c0, f);
    if (r) { q->bad = r; q->xbad = x; return 1; }
    return 0;
}
static int line_quantile(dfl_prov *pv, double c0, double p, double x0, double tol, double *out, char *err, size_t errlen)
{
    lineq_ctx qc = {pv, c0, 0, NAN};
    double x = x0, F, f;
    for (int it = 0; it < 30; it++) {
        if (lineq_ff(x, &qc, &F, &f, err, errlen)) return 2;
        double dx = dclamp((F - p) / f, -1.0, 1.0);
        if ((F - p) / f != (F - p) / f) dx = NAN;
        if (!isfinite(dx)) break;
        x -= dx;
        if (fabs(dx) < tol) { *out = x; return 0; }
    }
    int rc = newton_safe(lineq_ff, &qc, p, x0, 1.0, tol, out, err, errlen);
    return qc.bad ? 2 : rc;
}
int dfl_det_quantile(const dfl_design *D, int n, const double *p, double c, double x0, double tol, const dfl_det_opts *o0, double *q, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = opts_or_default(o0, &od);
    det_warn_clear();
    if (check_common(D, c, o, "the quantiles of the t-ratio (dist_quantile)", err, errlen)) return 1;
    for (int k = 0; k < n; k++) if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "dist_quantile: p must lie in (0, 1)"); return 1; }
    if (isnan(x0)) x0 = -2.0;
    if (start_tol_check("dist_quantile", "x0", x0, tol, err, errlen)) return 1;
    if (!(tol > 0)) tol = 1e-9;
    if (o->rule < DFL_RULE_AUTO || o->rule > DFL_RULE_FINESTEP) { snprintf(err, errlen, "%s", DET_RULE_MSG); return 1; }
    dfl_prov *pv;
    if (det_line_select(D, c, o, &pv, err, errlen)) return 1;
    /* (the quantiles keep the paths where the line integral was taken only for c0 > 0 on the vertical line, det_line_vline:
     * the search may reach x > 0, where the line integral grows slow; the paths warn there) */
    if (pv && o->method == DFL_DET_AUTO && !det_line_auto(D, c, o->c0)) { det_line_release(pv); pv = NULL; }
    int rc = 0, *todo = malloc((n + 1) * sizeof(int)), nt = 0;
    for (int k = 0; k < n && !rc; k++) {
        if (!pv) { todo[nt++] = k; continue; }
        int r = line_quantile(pv, o->c0, p[k], x0, tol, &q[k], err, errlen);
        if (r == 2) {
            if (o->method == DFL_DET_LINE) { snprintf(err, errlen, "dist_quantile: method line: the line integral cannot take a point of the search (c = %g, c0 = %g, p = %g: the bounds of the engine or the continuation of log P); method auto takes the inversion on paths", c, o->c0, p[k]); rc = 1; }
            else todo[nt++] = k;
        } else rc = r;
    }
    det_line_release(pv);
    if (!rc && nt > 0) {
        /* (the paths by the rule of one evaluation, dist_quantile of DFDet.jl) */
        int nr = 0, ntr = 0;
        det_path *CP = rule_path(D, c, o, 0, &nr, &ntr, NULL, err, errlen);
        if (!CP) { free(todo); return 1; }
        far_above_warn("dist_quantile", D, c, o);
        q_ctx qc = {CP, o->c0, 2, 1, nr, ntr};
        for (int i = 0; i < nt && !rc; i++) rc = quantile_search(&qc, p[todo[i]], x0, tol, 30, 1.0, &q[todo[i]], err, errlen);
        det_path_release(CP);
    }
    free(todo);
    return rc;
}

/* ---------- the coefficient statistic ---------- */
typedef struct { int ny; const double *ys; double c0; int which, vpow; } coef_ctx;
static int coef_eval(det_path *CP, const det_eval_opts *o, void *v, double *out, char *err, size_t errlen)
{
    coef_ctx *t = v; double x0 = 0, F, f;
    for (int j = 0; j < t->ny; j++) {
        if (det_contour_eval(CP, 1, &x0, t->c0 + t->ys[j], t->vpow, o, &F, &f, NULL, NULL, err, errlen)) return 1;
        out[j] = t->which == 1 ? F : f;
    }
    return 0;
}
int dfl_det_coef(int kind, const dfl_design *D, int n, const double *y, double c, const dfl_det_opts *o0, double *out, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = opts_or_default(o0, &od);
    det_warn_clear();
    if (kind != 1 && kind != 2) { snprintf(err, errlen, "kind must be 1 (density) or 2 (distribution function)"); return 1; }
    if (check_common(D, c, o, kind == 2 ? "the coefficient statistic (coef_cdf)" : "the coefficient statistic (coef_pdf)", err, errlen)) return 1;
    det_path *CP = user_path(D, c, o, err, errlen);
    if (!CP) return 1;
    double *yf = malloc((n + 1) * sizeof(double)); int *ix = malloc((n + 1) * sizeof(int));
    int m = split_finite(n, y, kind == 2, out, yf, ix), rc = 0;
    if (m > 0) {
        double *v = malloc(m * sizeof(double));
        coef_ctx t = {m, yf, o->c0, kind == 2 ? 1 : 2, kind == 2 ? 2 : 3};
        rc = run_eval(CP, coef_eval, &t, m, o->verified, o->vtol, v, err, errlen);
        if (!rc) for (int j = 0; j < m; j++) out[ix[j]] = v[j];
        if (!rc && kind == 2) rc = range_check("coef_cdf", D, c, o, "y", m, yf, v, err, errlen);
        free(v);
    }
    free(yf); free(ix); det_path_release(CP);
    return rc;
}
int dfl_det_coef_quantile(const dfl_design *D, int n, const double *p, double c, double y0, double tol, const dfl_det_opts *o0, double *q, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = opts_or_default(o0, &od);
    det_warn_clear();
    if (check_common(D, c, o, "the coefficient statistic (coef_quantile)", err, errlen)) return 1;
    for (int k = 0; k < n; k++) if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "coef_quantile: p must lie in (0, 1)"); return 1; }
    if (isnan(y0)) y0 = -10.0;
    if (start_tol_check("coef_quantile", "y0", y0, tol, err, errlen)) return 1;
    if (!(tol > 0)) tol = 1e-9;
    det_path *CP = user_path(D, c, o, err, errlen);
    if (!CP) return 1;
    q_ctx qc = {CP, o->c0, 3, 0, 0, 0};
    int rc = 0;
    for (int k = 0; k < n && !rc; k++) rc = quantile_search(&qc, p[k], y0, tol, 40, 5.0, &q[k], err, errlen);
    det_path_release(CP);
    return rc;
}

/* the inversion formula of Gil-Pelaez for the coefficient statistic (coef_cdf_gp) */
typedef struct { const det_model *Mo; double c, a; const double *u; cdouble *L0, *LD; det_status st; } gp_ctx;
static void gp_body(int k, void *v)
{
    gp_ctx *o = v;
    if (det_failed(&o->st)) return;
    char buf[DET_WORK_STACK]; det_work W;
    if (det_work_init(&W, o->Mo, buf, sizeof buf)) { det_fail(&o->st, "out of memory"); return; }
    det_fixt *fx = o->Mo->start == 1 ? det_fixt_alloc(o->Mo->r) : NULL;
    cdouble s = CPX(0, o->u[k]), t = CPX(0, -o->a * o->u[k]), th;
    if (det_logmgf_tpart(&W, o->c, t, &th, fx)) det_fail(&o->st, "the eigenvalues of the reciprocal roots did not converge");
    det_logmgf_spart(&W, o->c, s, t, th, fx, &o->L0[k], &o->LD[k]);
    free(fx); det_work_free(&W);
}
static void gpgrid(int N, double umax, double *u, double *w)
{
    double h = asinh(umax) / N;
    for (int k = 0; k <= N; k++) { double v = k * h; u[k] = sinh(v); w[k] = cosh(v) * h; }
    u[0] = 1e-7; w[0] /= 2;
}
/* the distance from the origin of the nearest singularity of phi(u) = E exp(iu (U - aV)) (gp_edge): U - aV is a quadratic
 * form, whose transform M(s, -a s) is singular on the real line only, at the first zeros s_+ > 0 and s_- < 0 of the entire
 * function e^{c + theta} det(...) along s, scanned by factors of 1.1 from 1e-10 to 1e4 and bisected; the smaller of s_+
 * and |s_-| (-1: out of memory).  (The phase of the determinant alone is -Im theta on the real line where -a s > c^2/2, so
 * that the sign change is read off the entire function.) */
static int gp_bad(det_work *W, det_fixt *fx, double c, double a, double s)
{
    cdouble t = CPX(-a * s, 0), th, l0, ld;
    if (det_logmgf_tpart(W, c, t, &th, fx)) return 1;
    det_logmgf_spart(W, c, CPX(s, 0), t, th, fx, &l0, &ld);
    cdouble L = cadd(ld, caddr(th, c));
    return !(isfinite(creal(L)) && fabs(cimag(L)) < 1.0);
}
static double gp_edge(const det_model *Mo, double a, double c)
{
    char buf[DET_WORK_STACK]; det_work W;
    if (det_work_init(&W, Mo, buf, sizeof buf)) return -1;
    det_fixt *fx = Mo->start == 1 ? det_fixt_alloc(Mo->r) : NULL;
    double e = INFINITY;
    for (int side = 0; side < 2; side++) {
        double sg = side ? -1.0 : 1.0, s = 1e-10;
        while (s < 1e4 && !gp_bad(&W, fx, c, a, sg * 1.1 * s)) s *= 1.1;
        if (!(s < 1e4)) continue;
        double lo = s, hi = 1.1 * s;
        for (int it = 0; it < 40; it++) { double m = sqrt(lo * hi); if (gp_bad(&W, fx, c, a, sg * m)) hi = m; else lo = m; }
        e = dmin(e, lo);
    }
    free(fx); det_work_free(&W);
    return e;
}
/* the smallest ratio of the distance d of the singularity of phi (gp_edge) to the step h = asinh(umax)/N of the grid near
 * the origin, below which coef_cdf_gp refuses (GP_EDGE of DFDet.jl): the error of the trapezoidal rule is about
 * 0.16 exp(-2 pi d/h) (case DK against the classical engine: 4.5e-4 at d/h = 0.98, 8.4e-6 at 1.57, below 6e-8 from 3.06
 * on); above the unit root d shrinks like 2c e^{-c}, so that with the default grid the 30% point of case DK has d/h = 3.1
 * at c = 10 and 0.5 at c = 12 (off by 1.1e-2) */
#define GP_EDGE 3.3
static int coef_cdf_grid(const dfl_design *D, int n, const double *y, double c, const dfl_det_opts *o, int N, double umax, double *out, char *err, size_t errlen)
{
    if (check_common(D, c, o, "the coefficient statistic (coef_cdf_gp)", err, errlen)) return 1;
    if (N <= 0) N = 60000;
    if (!(umax > 0)) umax = 4000;
    if (umax == INFINITY) { snprintf(err, errlen, "coef_cdf_gp: umax must be a positive finite number (NaN or <= 0: the default 4000)"); return 1; }
    det_prep *P = det_prep_cached(D, o->gls, err, errlen);
    if (!P) return 1;
    det_model Mo;
    if (det_model_make(P, o->start, o->gamma0, &Mo, err, errlen)) { det_prep_release(P); return 1; }
    double *u = malloc((N + 1) * sizeof(double)), *w = malloc((N + 1) * sizeof(double));
    cdouble *L0 = malloc((N + 1) * sizeof(cdouble)), *LD = malloc((N + 1) * sizeof(cdouble));
    gpgrid(N, umax, u, w);
    int rc = 0;
    for (int j = 0; j < n && !rc; j++) {
        /* (1, 0, NaN at Inf, -Inf, NaN, without the sum: at Inf the transform at t = -i Inf u did not return) */
        if (!isfinite(y[j])) { out[j] = nonfinite1(1, y[j]); continue; }
        double a = o->c0 + y[j], d = gp_edge(&Mo, a, c), hgp = asinh(umax) / N;
        if (d < 0) { snprintf(err, errlen, "out of memory"); rc = 1; break; }
        if (!(d / hgp >= GP_EDGE)) {
            snprintf(err, errlen, "coef_cdf_gp at c = %g, y = %g: the transform of U - (c0 + y) V has a singularity at %.3g from the real line, which the grid of the inversion (step %.3g near the origin) does not resolve (above the unit root the law widens like e^{2c}); it needs N >= %.0f, or coef_cdf (the contour integral)",
                     c, y[j], d, hgp, ceil(GP_EDGE * asinh(umax) / d));
            rc = 1; break;
        }
        gp_ctx g = {&Mo, c, a, u, L0, LD, {0, ""}};
        dfl_parfor(N + 1, gp_body, &g);
        if (det_failed(&g.st)) { snprintf(err, errlen, "%s", g.st.msg); rc = 1; break; }
        double ph = cimag(LD[0]), acc = 0;
        for (int k = 0; k <= N; k++) {
            if (k > 0) { double d = cimag(LD[k]) - cimag(LD[k - 1]); ph += atan2(sin(d), cos(d)); }
            cdouble lM = csub(L0[k], cscale(0.5, CPX(creal(LD[k]), ph)));
            acc += cimag(cexp(lM)) / u[k] * w[k];
        }
        out[j] = 0.5 - acc / M_PI;
    }
    free(u); free(w); free(L0); free(LD); det_model_free(&Mo); det_prep_release(P);
    return rc;
}

/* coef_cdf_gp: by default (a zero or stationary start, any detrending, c up to DFL_GP_INV_CMAX) the single inversion of Bao
 * and Kan with the branch of the square root of the determinant from the pivots at each point (det_gcoef_cdf1 of
 * dfl_det_kan.c), tol (1e-12) its absolute tolerance; DFL_METHOD_GRID (or N, umax given (> 0), and a fixed start gamma0 !=
 * 0) the trapezoidal rule of before on the fixed grid in u = sinh(v) with its refusal (coef_cdf_grid).  (GP_INV_CMAX:
 * compared in November 2026 with the contour integral for an intercept, linear and quadratic trends, Perron's model C with
 * one and two breaks, a stationary start and GLS detrending, c from -30 to 5: within 9.6e-13, 2.5e-11 with GLS detrending;
 * at c = 10 one value takes seconds, and the grid with its refusal stays the default above c = 5) */
int dfl_det_coef_cdf_gp_x(const dfl_design *D, int n, const double *y, double c, const dfl_det_opts *o0, int method, int N, double umax, double tol, double *out, double *errest, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = opts_or_default(o0, &od);
    det_warn_clear();
    if (!(method == 0 || method == DFL_METHOD_INVERSION || method == DFL_METHOD_GRID)) {
        snprintf(err, errlen, "coef_cdf_gp: method must be 0 (automatic), %d (the single inversion of Bao and Kan) or %d (the fixed grid)", DFL_METHOD_INVERSION, DFL_METHOD_GRID); return 1;
    }
    int fixed = o->start == DFL_START_FIXED && o->gamma0 != 0, given = N > 0 || umax > 0;
    if (method == DFL_METHOD_INVERSION && (fixed || given)) { snprintf(err, errlen, "coef_cdf_gp: method %d takes a zero or stationary start and no N, umax", DFL_METHOD_INVERSION); return 1; }
    if (errest) for (int k = 0; k < n; k++) errest[k] = NAN;
    if (method == DFL_METHOD_GRID || given || fixed || (method == 0 && !(c <= DFL_GP_INV_CMAX))) return coef_cdf_grid(D, n, y, c, o, N, umax, out, err, errlen);
    double tl = tol;
    if (isnan(tl) || tl == 0) tl = DFL_KAN_TOL;
    else if (!(tl > 0 && isfinite(tl))) { snprintf(err, errlen, "coef_cdf_gp: tol must be a positive finite number (NaN or 0: the default %g)", DFL_KAN_TOL); return 1; }
    if (check_common(D, c, o, "the coefficient statistic (coef_cdf_gp)", err, errlen)) return 1;
    det_prep *P = det_prep_cached(D, o->gls, err, errlen);
    if (!P) return 1;
    det_model Mo;
    if (det_model_make(P, o->start, o->gamma0, &Mo, err, errlen)) { det_prep_release(P); return 1; }
    det_cf *F = det_cf_new(&Mo, c, 0);
    double *E = malloc((n + 1) * sizeof(double));
    int rc = F ? 0 : 1;
    if (!F) snprintf(err, errlen, "out of memory");
    for (int k = 0; k < n && !rc; k++) {
        rc = det_gcoef_cdf1(F, y[k], o->c0, tl, &out[k], &E[k]);
        if (rc) snprintf(err, errlen, "coef_cdf_gp: the single inversion failed at y = %g", y[k]);
    }
    if (!rc) {
        int bad = 0; double mx = 0;
        for (int k = 0; k < n; k++) if (!(E[k] <= tl)) { bad++; mx = isnan(E[k]) || E[k] > mx ? E[k] : mx; }
        if (bad) det_warn("coef_cdf_gp: the error estimate of the single inversion exceeds the tolerance %g at %d of %d points (largest %g); method %d gives the fixed grid", tl, bad, n, mx, DFL_METHOD_GRID);
        if (errest) memcpy(errest, E, n * sizeof(double));
    }
    free(E); det_cf_free(F); det_model_free(&Mo); det_prep_release(P);
    return rc;
}
int dfl_det_coef_cdf_gp(const dfl_design *D, int n, const double *y, double c, const dfl_det_opts *o0, int N, double umax, double *out, char *err, size_t errlen)
{
    return dfl_det_coef_cdf_gp_x(D, n, y, c, o0, 0, N, umax, NAN, out, NULL, err, errlen);
}

/* ---------- the moments ---------- */
/* the logarithm along a closed circle, continued from its first point (circlog) */
static int circlog(int n, const cdouble *Z, cdouble *out)
{
    double ph0 = cimag(clog(Z[0])), ph = ph0, prev = ph0; int ok = fabs(ph0) < 1e-8;
    out[0] = clog(Z[0]);
    for (int k = 1; k < n; k++) {
        cdouble lz = clog(Z[k]);
        double dd = cimag(lz) - prev, d = atan2(sin(dd), cos(dd));
        if (!(fabs(d) < M_PI / 3)) ok = 0;
        ph += d; prev = cimag(lz);
        out[k] = CPX(creal(lz), ph);
    }
    if (!(fabs(ph + carg(cdiv(Z[0], Z[n - 1])) - ph0) < 1e-6)) ok = 0;
    return ok;
}
typedef struct { const det_model *Mo; double c, c0, smax; int p, Nu; const double *sv; double *acc; det_status st; } mom_ctx;
static void mom_body(int i, void *v)
{
    mom_ctx *o = v;
    if (det_failed(&o->st)) return;
    const det_model *Mo = o->Mo;
    char buf[DET_WORK_STACK]; det_work W;
    if (det_work_init(&W, Mo, buf, sizeof buf)) { det_fail(&o->st, "out of memory"); return; }
    int Nu = o->Nu, r = Mo->r;
    double s = o->sv[i], c = o->c, c0 = o->c0;
    cdouble th, ld, mu[64], *muv = r <= 64 ? mu : malloc(r * sizeof(cdouble));
    det_fixt *fx = Mo->start == 1 ? det_fixt_alloc(r) : NULL;
    if (det_mgfparts(&W, c, CPX(-s, 0), &th, &ld, muv, NULL)) { det_fail(&o->st, "the eigenvalues of the reciprocal roots did not converge"); goto out; }
    double mx = 0; for (int k = 0; k < r; k++) mx = dmax(mx, cabs(muv[k]));
    double rad = 0.4 / mx;
    if (c0 != 0) rad = dmin(rad, 0.4 * (s + o->smax) / fabs(c0));
    if (Mo->start != 0) rad = dmin(rad, 0.25);
    if (c < -20) rad = dmin(rad, 2.0);
    if (!isnan(Mo->P->cbar)) rad = dmin(rad, 2.0);
    cdouble *L0 = malloc(Nu * sizeof(cdouble)), *LD = malloc(Nu * sizeof(cdouble)), *Z = malloc(Nu * sizeof(cdouble)), *lz = malloc(Nu * sizeof(cdouble)), *Lv = malloc(Nu * sizeof(cdouble));
    int ok = 0;
    for (int att = 0; att < 8; att++) {
        cdouble thk; cdouble U0 = CPX(rad, 0);
        if (det_logmgf_tpart(&W, c, csub(CPX(-s, 0), cscale(c0, U0)), &thk, fx)) { det_fail(&o->st, "the eigenvalues of the reciprocal roots did not converge"); break; }
        int cut = c0 != 0 && -s + fabs(c0) * rad > c * c / 2;   /* (whether the circle in t = -s - c0 u reaches the cut of theta) */
        for (int k = 0; k < Nu; k++) {
            cdouble Uk = cscale(rad, cis_(2 * M_PI * k / Nu)), tk = csub(CPX(-s, 0), cscale(c0, Uk));
            if (c0 != 0 && k > 0) det_logmgf_tpart(&W, c, tk, &thk, fx);
            det_logmgf_spart(&W, c, Uk, tk, thk, fx, &L0[k], &LD[k]);
            /* (where the circle reaches the cut, the logarithm continued around it is that of the entire function
             * e^{c + theta} det(...), real and positive at the real point u = r, and not that of the determinant alone,
             * whose phase jumps with theta across the cut: where -s - c0 r > c^2/2, as at c = 0 with c0 < 0 for every s
             * below |c0| r, theta is imaginary at the real point, the check of the real start failed on every circle, and
             * the moments refused (case DT at c = 0, c0 = -2); elsewhere the logarithm of the determinant, as before, bit
             * for bit, as dist_moments of DFDet.jl) */
            if (cut) { cdouble e = caddr(thk, c); LD[k] = cadd(LD[k], e); L0[k] = cadd(L0[k], cscale(0.5, e)); }
        }
        double r0 = creal(LD[0]);
        for (int k = 0; k < Nu; k++) Z[k] = cexp(caddr(LD[k], -r0));
        ok = circlog(Nu, Z, lz);
        for (int k = 0; k < Nu; k++) Lv[k] = csub(L0[k], cscale(0.5, caddr(lz[k], r0)));
        if (ok) break;
        rad /= 2;
    }
    if (!det_failed(&o->st)) {
        if (!ok) det_fail(&o->st, "dist_moments: no circle in u free of zeros of the determinant at s = %g", s);
        else for (int j = 1; j <= o->p; j++) {
            cdouble acc = DZERO;
            for (int k = 0; k < Nu; k++) {
                cdouble wi = cinv(cis_(2 * M_PI * k / Nu)), wp = DONE;
                for (int l = 0; l < j; l++) wp = cmul(wp, wi);
                acc = cadd(acc, cmul(cexp(Lv[k]), wp));
            }
            double fac = 1; for (int l = 2; l <= j; l++) fac *= l;
            double Mj = fac / (Nu * pow(rad, j)) * creal(acc);
            o->acc[(size_t) (j - 1) * 0 + (size_t) i * o->p + (j - 1)] = pow(s, j / 2.0) * Mj;
        }
    }
    free(L0); free(LD); free(Z); free(lz); free(Lv);
out:
    if (muv != mu) free(muv);
    free(fx); det_work_free(&W);
}
int dfl_det_moments(const dfl_design *D, double c, int pmax, const dfl_det_opts *o0, double *m, double *errest, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = opts_or_default(o0, &od);
    det_warn_clear();
    if (check_common(D, c, o, "the moments of the t-ratio (dist_moments)", err, errlen)) return 1;
    if (D->first) { snprintf(err, errlen, "the moments of the statistic that detrends first (detrend_first) are not implemented (not validated)"); return 1; }
    if (pmax < 1) { snprintf(err, errlen, "pmax >= 1"); return 1; }
    det_prep *P = det_prep_cached(D, o->gls, err, errlen);
    if (!P) return 1;
    det_model Mo;
    if (det_model_make(P, o->start, o->gamma0, &Mo, err, errlen)) { det_prep_release(P); return 1; }
    const double h = 0.05;
    double tlo = dmin(-60.0, -2 * c - 40), thi = c < -20 ? dmax(8.0, log(80 * fabs(c))) : 8.0;
    /* (the grid grows with c above the unit root: 2c/h points, 4e5 at c = 1e4; beyond 1e6 an error) */
    if (!((thi - tlo) / h <= 1e6)) { snprintf(err, errlen, "dist_moments: the grid of the integral in s would have %.3g points at c = %g (more than 1e6): c is beyond the range of the moments", (thi - tlo) / h, c); det_model_free(&Mo); det_prep_release(P); return 1; }
    int ns = (int) floor((thi - tlo) / h + 1e-10) + 1;
    double *sv = malloc(ns * sizeof(double)), *acc = calloc((size_t) ns * pmax, sizeof(double));
    for (int i = 0; i < ns; i++) sv[i] = exp(tlo + i * h);
    double smax;
    int rc = det_tstar(&Mo, c, &smax, NULL, err, errlen);
    if (!rc) {
        mom_ctx g = {&Mo, c, o->c0, smax, pmax, 64, sv, acc, {0, ""}};
        dfl_parfor(ns, mom_body, &g);
        if (det_failed(&g.st)) { snprintf(err, errlen, "%s", g.st.msg); rc = 1; }
    }
    if (!rc) for (int j = 1; j <= pmax; j++) {
        double s1 = 0, s2 = 0;
        for (int i = 0; i < ns; i++) s1 += acc[(size_t) i * pmax + (j - 1)];
        for (int i = 0; i < ns; i += 2) s2 += acc[(size_t) i * pmax + (j - 1)];
        double g = tgamma(j / 2.0);
        m[j - 1] = h * s1 / g;
        if (errest) errest[j - 1] = fabs(h * s1 - 2 * h * s2) / g;
    }
    free(sv); free(acc); det_model_free(&Mo); det_prep_release(P);
    return rc;
}

/* ---------- local power ---------- */
int dfl_det_local_power(const dfl_design *D, int n, const double *cs, double level, int statistic, const dfl_det_opts *o0, double *out, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = opts_or_default(o0, &od);
    if (statistic != 0 && statistic != 1) { snprintf(err, errlen, "statistic must be 0 (the t-ratio) or 1 (the coefficient statistic)"); return 1; }
    if (!(level > 0 && level < 1)) { det_warn_clear(); snprintf(err, errlen, "local_power: the level must lie in (0, 1)"); return 1; }
    if (!isfinite(o->c0)) { det_warn_clear(); snprintf(err, errlen, "c0 must be a finite real scalar"); return 1; }
    double q; int rc = 0;
    det_warn_clear(); det_warn_hold(1);
    if (statistic == 0) {
        rc = dfl_det_quantile(D, 1, &level, o->c0, isnan(o->gls) ? -3.5 : -2.5, 1e-11, o, &q, err, errlen);
        for (int k = 0; k < n && !rc; k++) rc = dfl_det_dist(2, D, 1, &q, cs[k], o, &out[k], err, errlen);
    } else {
        rc = dfl_det_coef_quantile(D, 1, &level, o->c0, -15.0, 1e-11, o, &q, err, errlen);
        for (int k = 0; k < n && !rc; k++) rc = dfl_det_coef(2, D, 1, &q, cs[k], o, &out[k], err, errlen);
    }
    det_warn_hold(0);
    return rc;
}

/* ---------- events bounded by parabolas: the line U - aV <= b, the law of V, ADFmax ---------- */
typedef struct { int ne; const int *nt; const det_xspec *sp; int which; } spec_ctx;
static int spec_eval_one(det_path *CP, const det_eval_opts *o, void *v, double *out, char *err, size_t errlen)
{
    spec_ctx *s = v;
    int ntot = 0; for (int i = 0; i < s->ne; i++) ntot += s->nt[i];
    double *qm = malloc((s->ne + 1) * sizeof(double)), *am = malloc((ntot + 1) * sizeof(double)), *cz = malloc((ntot + 1) * sizeof(double));
    int it = 0, ncz = 0;
    for (int i = 0; i < s->ne; i++) {
        double q = 0;
        for (int l = 0; l < s->nt[i]; l++, it++) { q = dmax(q, fabs(s->sp[it].q)); am[it] = fabs(s->sp[it].A); if (s->sp[it].q != 0) cz[ncz++] = s->sp[it].c0; }
        qm[i] = q;
    }
    det_path *CPr = det_line_refined(CP, s->ne, qm, ntot, am, ncz, cz, err, errlen);   /* (the parabolas with a slope: their c0) */
    free(qm); free(am); free(cz);
    if (!CPr) return 1;
    double *F = malloc(s->ne * sizeof(double)), *f = malloc(s->ne * sizeof(double));
    int rc = det_contour_eval_specs(CPr, s->ne, s->nt, s->sp, o, F, f, NULL, NULL, err, errlen);
    if (!rc) memcpy(out, s->which == 1 ? F : f, s->ne * sizeof(double));
    free(F); free(f); det_path_release(CPr);
    return rc;
}
/* the law of V at ys: 0 at y <= 0 and at -Inf (V > 0), 1 (cdf) or 0 (density) at Inf, NaN at NaN; the inversion at y > 0
 * (at y < 0 the factor e^{-t y} grows along the path, and the sum gave NaN) */
static int vlaw(int kind, const dfl_design *D, int n, const double *ys, double c, const dfl_det_opts *o, double *out, char *err, size_t errlen)
{
    det_path *CP = det_v_path(D, o->gls, o->start, o->gamma0, c, o->phi, isnan(o->hfac) ? 1.0 : o->hfac, err, errlen);
    if (!CP) return 1;
    for (int k = 0; k < n; k++) {
        double y = ys[k];
        out[k] = isnan(y) ? NAN : !(y > 0) ? 0.0 : isinf(y) ? (kind == 2 ? 1.0 : 0.0) : kind == 2 ? 1 - det_vsum(CP, y, 1) : det_vsum(CP, y, 0);
    }
    det_path_release(CP);
    return 0;
}
int dfl_det_lin(int kind, const dfl_design *D, int n, const double *b, double a, double c, const dfl_det_opts *o0, double *out, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = opts_or_default(o0, &od);
    det_warn_clear();
    if (kind != 1 && kind != 2) { snprintf(err, errlen, "kind must be 1 (density) or 2 (distribution function)"); return 1; }
    if (check_common(D, c, o, isnan(a) ? "the law of V (v_cdf, v_pdf)" : kind == 2 ? "the linear form U - aV (lin_cdf)" : "the linear form U - aV (lin_pdf)", err, errlen)) return 1;
    if (isnan(a)) return vlaw(kind, D, n, b, c, o, out, err, errlen);
    if (!isfinite(a)) { snprintf(err, errlen, "%s: a must be a finite real scalar (NaN: the law of V)", kind == 2 ? "lin_cdf" : "lin_pdf"); return 1; }
    det_path *CP = user_path(D, c, o, err, errlen);
    if (!CP) return 1;
    double *bf = malloc((n + 1) * sizeof(double)); int *ix = malloc((n + 1) * sizeof(int));
    int m = split_finite(n, b, kind == 2, out, bf, ix), rc = 0;
    if (m > 0) {
        det_xspec *sp = malloc(m * sizeof(det_xspec)); int *nt = malloc(m * sizeof(int)); double *v = malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) { det_xspec s = {a, 0, CP->Mo->a + bf[j], INFINITY, 1, 0, 0, 1, 1}; sp[j] = s; nt[j] = 1; }
        spec_ctx sc = {m, nt, sp, kind == 2 ? 1 : 2};
        rc = run_eval(CP, spec_eval_one, &sc, m, o->verified, o->vtol, v, err, errlen);
        if (!rc) for (int j = 0; j < m; j++) out[ix[j]] = v[j];
        free(sp); free(nt); free(v);
    }
    free(bf); free(ix); det_path_release(CP);
    return rc;
}
typedef struct { det_path *CP; double a; int adf; } lq_ctx;
static int lq_ff(double b, void *v, double *F, double *f, char *err, size_t errlen)
{
    lq_ctx *q = v; det_path *CP = q->CP; double A = CP->Mo->a;
    det_xspec sp[2]; int nt; det_path *CPr;
    if (q->adf) {
        double vm = b < 0 ? -0.5 / b : INFINITY;
        det_xspec s1 = {0, b, A, vm, 1, 0, 1, 0, 0}, s2 = {0, -b, A - 1, vm, -1, 0, -1, 0, 0};
        sp[0] = s1; sp[1] = s2; nt = 2;
        CPr = det_line_refined(CP, 1, &b, 1, &A, 0, NULL, err, errlen);
    } else {
        det_xspec s = {q->a, 0, A + b, INFINITY, 1, 0, 0, 1, 1}; sp[0] = s; nt = 1;
        double am = fabs(A + b);
        CPr = det_line_refined(CP, 0, NULL, 1, &am, 0, NULL, err, errlen);
    }
    if (!CPr) return 1;
    int rc = det_contour_eval_specs(CPr, 1, &nt, sp, NULL, F, f, NULL, NULL, err, errlen);
    det_path_release(CPr);
    return rc;
}
int dfl_det_lin_quantile(const dfl_design *D, double p, double a, double c, double b0, double tol, const dfl_det_opts *o0, double *q, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = opts_or_default(o0, &od);
    det_warn_clear();
    if (check_common(D, c, o, "the quantiles of the linear form U - aV (lin_quantile)", err, errlen)) return 1;
    if (!(p > 0 && p < 1)) { snprintf(err, errlen, "lin_quantile: p must lie in (0, 1)"); return 1; }
    if (!isfinite(a)) { snprintf(err, errlen, "lin_quantile: a must be a finite real scalar"); return 1; }
    if (isnan(b0)) b0 = 0;
    if (start_tol_check("lin_quantile", "b0", b0, tol, err, errlen)) return 1;
    if (!(tol > 0)) tol = 1e-10;
    det_path *CP = user_path(D, c, o, err, errlen);
    if (!CP) return 1;
    lq_ctx lc = {CP, a, 0};
    int rc = newton_safe(lq_ff, &lc, p, b0, 1.0, tol, q, err, errlen);
    det_path_release(CP);
    return rc;
}

/* whether D is trend(p) (p = 0, 1: one piece, the monomial coefficients; the cases of the power envelope) */
static int is_trend(const dfl_design *D, int p)
{
    int K = p + 1, deg = p > 0 ? p : 0;
    if (D->K != K || D->np != 1 || D->br[0] != 0 || D->br[1] != 1 || D->deg != deg) return 0;
    for (int i = 0; i < K; i++) for (int k = 0; k <= deg; k++) if (D->C[i * (deg + 1) + k] != (k == i ? 1.0 : 0.0)) return 0;
    return 1;
}
int dfl_det_power_envelope(const dfl_design *D, int n, const double *cs, double level, double *out, char *err, size_t errlen)
{
    det_warn_clear();
    if (!D) { snprintf(err, errlen, "no design"); return 1; }
    if (!(level > 0 && level < 1)) { snprintf(err, errlen, "power_envelope: the level must lie in (0, 1)"); return 1; }
    int trendcase = is_trend(D, 1), nodet = D->K == 0 || is_trend(D, 0);
    if (!(trendcase || nodet)) { snprintf(err, errlen, "power_envelope: the design must be trend(-1), trend(0) or trend(1) (the limit of the point-optimal statistic of other designs was not derived)"); return 1; }
    dfl_design T; if (dfl_design_trend(-1, &T, err, errlen)) return 1;
    int rc = 0;
    det_warn_hold(1);
    for (int k = 0; k < n && !rc; k++) {
        double c = cs[k];
        if (!(c < 0)) { out[k] = isnan(c) ? NAN : level; continue; }   /* (NaN at NaN, as the classical envelope) */
        if (c == -INFINITY) { out[k] = 1.0; continue; }                /* (the limit) */
        dfl_det_opts o; dfl_det_defaults(&o);
        double b, a;
        if (nodet) {
            a = c / 2;
            rc = dfl_det_lin_quantile(&T, level, a, 0.0, 0.0, 1e-10, &o, &b, err, errlen);
            if (!rc) rc = dfl_det_lin(2, &T, 1, &b, a, c, &o, &out[k], err, errlen);
        } else {
            a = c * c / (2 * (c - 1)); o.gls = c;
            rc = dfl_det_lin_quantile(D, level, a, 0.0, 0.0, 1e-10, &o, &b, err, errlen);
            if (!rc) rc = dfl_det_lin(2, D, 1, &b, a, c, &o, &out[k], err, errlen);
        }
    }
    det_warn_hold(0);
    dfl_design_free(&T);
    return rc;
}

/* ADFmax: least-squares detrending at the null value phi = 1 (the general engine has no ADFmax of GLS-detrended data and
 * no local null value; both were dropped without notice before) */
static int adf_opts_check(const dfl_det_opts *o, char *err, size_t errlen)
{
    if (!isnan(o->gls)) { snprintf(err, errlen, "adfmax: ADFmax has least-squares detrending only (gls must be NaN)"); return 1; }
    if (o->c0 != 0) { snprintf(err, errlen, "adfmax: ADFmax is the statistic of the null value phi = 1 (c0 must be 0)"); return 1; }
    return 0;
}
int dfl_det_adfmax(int kind, const dfl_design *D, int n, const double *x, double c, const dfl_det_opts *o0, double *out, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o1 = opts_or_default(o0, &od);
    dfl_det_opts o = *o1;
    det_warn_clear();
    if (kind != 1 && kind != 2) { snprintf(err, errlen, "kind must be 1 (density) or 2 (distribution function)"); return 1; }
    if (adf_opts_check(&o, err, errlen) || check_common(D, c, &o, kind == 2 ? "ADFmax (adfmax_cdf)" : "ADFmax (adfmax_pdf)", err, errlen)) return 1;
    double *xf = malloc((n + 1) * sizeof(double)); int *ix = malloc((n + 1) * sizeof(int));
    int m = split_finite(n, x, kind == 2, out, xf, ix), rc = 0;
    if (m > 0) {
        det_path *CP = user_path(D, c, &o, err, errlen);
        if (!CP) { free(xf); free(ix); return 1; }
        double A = CP->Mo->a;
        det_xspec *sp = malloc(2 * m * sizeof(det_xspec)); int *nt = malloc(m * sizeof(int)); double *v = malloc(m * sizeof(double));
        for (int j = 0; j < m; j++) {
            double xx = xf[j], vm = xx < 0 ? -0.5 / xx : INFINITY;
            det_xspec s1 = {0, xx, A, vm, 1, 0, 1, 0, 0}, s2 = {0, -xx, A - 1, vm, -1, 0, -1, 0, 0};
            sp[2 * j] = s1; sp[2 * j + 1] = s2; nt[j] = 2;
        }
        spec_ctx sc = {m, nt, sp, kind == 2 ? 1 : 2};
        rc = run_eval(CP, spec_eval_one, &sc, m, o.verified, o.vtol, v, err, errlen);
        if (!rc) for (int j = 0; j < m; j++) out[ix[j]] = v[j];
        free(sp); free(nt); free(v); det_path_release(CP);
    }
    free(xf); free(ix);
    return rc;
}
int dfl_det_adfmax_quantile(const dfl_design *D, double p, double c, double x0, double tol, const dfl_det_opts *o0, double *q, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o1 = opts_or_default(o0, &od);
    dfl_det_opts o = *o1;
    det_warn_clear();
    if (adf_opts_check(&o, err, errlen) || check_common(D, c, &o, "the quantiles of ADFmax (adfmax_quantile)", err, errlen)) return 1;
    if (!(p > 0 && p < 1)) { snprintf(err, errlen, "adfmax_quantile: p must lie in (0, 1)"); return 1; }
    if (isnan(x0)) x0 = -2.0;
    if (start_tol_check("adfmax_quantile", "x0", x0, tol, err, errlen)) return 1;
    if (!(tol > 0)) tol = 1e-10;
    det_path *CP = user_path(D, c, &o, err, errlen);
    if (!CP) return 1;
    lq_ctx lc = {CP, 0, 1};
    int rc = newton_safe(lq_ff, &lc, p, x0, 1.0, tol, q, err, errlen);
    det_path_release(CP);
    return rc;
}
int dfl_det_region(const dfl_design *D, int type, const double *par, double c, const dfl_det_opts *o0, double *out, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o1 = opts_or_default(o0, &od);
    if (type == 1) {
        if (!isfinite(par[0])) { det_warn_clear(); snprintf(err, errlen, "region_prob: the slope a of the line must be a finite real scalar"); return 1; }
        return dfl_det_lin(2, D, 1, &par[1], par[0], c, o1, out, err, errlen);
    }
    if (type == 2) { dfl_det_opts o = *o1; o.c0 = par[1]; return dfl_det_dist(2, D, 1, &par[0], c, &o, out, err, errlen); }
    if (type == 3 || type == 4) { snprintf(err, errlen, "region_prob: the boundary type %s is not available by the contour integral (its kernels are in closed form for boundaries quadratic in sqrt(V) only)", type == 3 ? "wap" : "fun"); return 1; }
    snprintf(err, errlen, "region_prob: unknown boundary type %d", type);
    return 1;
}
