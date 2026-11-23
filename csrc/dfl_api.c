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
static const char *case_name(int model, bool stat)
{
    return model == 4 ? "DG" : stat ? (model == 1 ? "SK" : model == 2 ? "SU" : "ST") : (model == 1 ? "DK" : model == 2 ? "DU" : "DT");
}
/* one value of the line integral (the general kernel for c0 != 0, a fixed start or so->gen): 0, or the bound of the engine
 * that its paths would exceed (bd) */
static int single_point(int kind, double x, int model, double c, double c0, double g0, bool stat, const dfl_single_opts *so, double *out, dfl_bound_t *bd)
{
    return (c0 != 0 || g0 != 0 || so->gen) ? dfl_single_value_gen_b(kind, x, model, c, c0, g0, stat, so, out, bd) : dfl_single_value_c0_b(kind, x, model, c, stat, so, out, bd);
}

/* The error message of a point at which the line integral's paths exceed the bounds of the engine, with the arguments
 * (path_error of DFExact.jl).  Two regions, measured in November 2026 on the Julia engine, whose paths are those of this
 * one: (a) c0 < 0 near zero with x > 0, where the root b = (x + (x^2 + 2|c0|)^(1/2))/(2|c0|) ~ x/|c0| of
 * X(v) = 1/2 + x v + c0 v^2 lies far out and the paths of the general kernel's inner integrals follow the phase of e^{-t v^2}
 * up to it: more than DFL_NODEMAX nodes on one path at c0 = -1e-4 from x = 0.05, at c0 = -1e-3 at x = 1.5 (at x = 0.5 too in
 * case SK and at c = 2), at c0 = -1e-2 in case SK for -0.4 <= c < 0 and at c = 2 in cases DK, DU, and not at c0 = -0.1
 * (a scan of c from -60 to 2 and x up to 1.5; far above the unit root c0 = -0.1 is affected too: case DK at c = 5,
 * x = 14.17); the size is set by c0 and x much more than by c, the case or the start, and c0 > 0 and x <= 0 are not
 * affected; (b) above
 * the unit root at large x, where the segment to the saddle of the central kernel's Gaussian factor needs a rule of more
 * than DFL_RULEMAX nodes: at c = 7 from x ~ 750 (case DK), 780 (DU), 660 (DT), at c = 10 from 815, 715, 650, not up to
 * x = 5000 for c <= 5. */
void dfl_bound_msg(char *err, size_t errlen, int kind, int model, double c, double c0, double g0, bool stat, double x, const dfl_bound_t *bd)
{
    char args[160], why[200];
    int k = snprintf(args, sizeof args, "case %s, c = %g", case_name(model, stat), c);
    if (c0 != 0 && k < (int) sizeof args) k += snprintf(args + k, sizeof args - k, ", c0 = %g", c0);
    if (g0 != 0 && k < (int) sizeof args) k += snprintf(args + k, sizeof args - k, ", g0 = %g", g0);
    if (k < (int) sizeof args) snprintf(args + k, sizeof args - k, ", x = %g", x);
    if (bd->bound == DFL_BOUND_RULE) snprintf(why, sizeof why, "at large x above the unit root");
    else if (c0 < 0 && x > 0) snprintf(why, sizeof why, "c0 < 0 near zero with x > 0: the root %.3g of X(v) = 1/2 + x v + c0 v^2 lies far out", (x + sqrt(x * x + 2 * fabs(c0))) / (2 * fabs(c0)));
    else snprintf(why, sizeof why, "a root of X(v) far out");
    snprintf(err, errlen, "%s: %s: the line integral needs more nodes than the engine allows here (%s; %s needs %.3g nodes, at most %d); %s",
             kind == 2 ? "ltratio_cdf" : "ltratio_pdf", args, why, bd->what ? bd->what : "a path of the line integral", bd->need,
             bd->bound == DFL_BOUND_RULE ? DFL_RULEMAX : DFL_NODEMAX,
             model == 4 ? "case DG has the line integral representation only" : "the automatic method takes the conditional integral at such points");
}

int dfl_single(int kind, int n, const double *x, int model, double c, double c0, double g0, bool stat, const dfl_single_opts *o, double *out, char *err, size_t errlen)
{
    dfl_single_opts d; if (o == NULL) { dfl_single_defaults(&d); o = &d; }
    if (check_args(kind, model, c, c0, g0, stat, err, errlen)) return 1;
    if (tstar_err(model, c, stat, err, errlen)) return 1;
    for (int k = 0; k < n; k++) {
        if (!isfinite(x[k])) { out[k] = nonfinite(kind, x[k]); continue; }
        dfl_bound_t bd;
        if (single_point(kind, x[k], model, c, c0, g0, stat, o, &out[k], &bd)) { dfl_bound_msg(err, errlen, kind, model, c, c0, g0, stat, x[k], &bd); return 10 + bd.bound; }
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
/* The conditional integral at a point where the line integral's paths exceed the bounds of the engine, with method 0
 * (route_double of DFExact.jl), by the region of dfl_bound_msg.  (a) A far root (DFL_BOUND_PATH): with its checks (verified,
 * the agreement of two paths and the refinement of their rules), whatever its status; *flag where the value is flagged.
 * Against the contour integral of a newer engine (November 2026): case SK, stationary start, c = c0 in {-1e-4, -1e-3, -1e-2,
 * -0.1}, x in {0.05, 0.5, 1, 1.5}, within 5e-7 with the status flagged at most of the points, where one path is off by up
 * to 5e-5, but the density at c = c0 = -1e-3, x = 0.05 off by 1.2e-3 (flagged): dfl_warning names the call where a value
 * returned is flagged; case DK with a zero start (c in {-1, 0, 2}, c0 in {-1e-3, -1e-2, -0.1}) and a fixed start (g0 = 1,
 * c0 = -1e-2) within 1e-13.  (b) Large x above the unit root (DFL_BOUND_RULE): in case DK the checked value whatever its
 * status, in cases DU, DT the checked value where its status is 0 and otherwise one path.  Against the newer engine at
 * c = 7 (x up to 1000, cases DK, DU, DT) and c = 10 (case DT, x up to 1500) within 1e-14, where in case DT the checked value
 * is off by up to 3e-2 with the status 2 and one path is right.  In case DK the status 2 is a false alarm and one path is
 * far off: against an independent computation (the law of V given J(1) from Girsanov's theorem, inverted on a line
 * through its saddle point and integrated over J(1)), at c = 7, 10 and 13 with c0 in {-2, 0, 2, c - 1, c + 2} from the 24%
 * to the 99.7% point the checked value is within 1.2e-9 at every point (32 of 36 with the status 2), one path off by up to
 * 8; before November 23, 2026 the flagged points of case DK got one path too, and the automatic method returned values
 * outside [0, 1] there (route_double of DFExact.jl).  The options of the conditional integral are those of dopt, with its
 * checks as stated. */
static int route_double(int kind, double x, int model, double c, double c0, double g0, bool stat, const dfl_double_opts *dopt, const dfl_bound_t *bd,
                        double *out, int *flag, char *err, size_t errlen)
{
    dfl_double_opts dv = *dopt; dv.verified = 1;
    dfl_double_info info; double F;
    *flag = 0;
    if (dfl_double(kind, 1, &x, model, c, c0, g0, stat, &dv, &F, &info, err, errlen)) return 1;
    if (bd->bound == DFL_BOUND_PATH) { *out = F; *flag = info.status != 0; return 0; }
    if (info.status == 0 || model == 1) { *out = F; return 0; }
    dv.verified = 0;
    if (dfl_double(kind, 1, &x, model, c, c0, g0, stat, &dv, &F, &info, err, errlen)) return 1;
    *out = F;
    return 0;
}
/* Far above the unit root at large |x| (method 0, a zero start, c > DFL_FAR_C): at x > DFL_FAR_X in models 1, 2, 3, and in
 * model 1 with c0 > c at x < -DFL_FAR_XNEG, the conditional integral as route_double gives it (in model 1 the checked value,
 * in models 2, 3 the checked value where its status is 0 and one path otherwise), as at the points where the line integral
 * exceeds the bounds of the engine; these points are taken before the rule's representations (dist_impl).  At large x the
 * line integral's cost grows like x^4 (the phase of the saddle factor of its inner integrals grows like x^2, and the panels
 * of the outer rule follow it): in the Julia engine, case DK at c = 10, c0 = -2, 0.9 s at x = 200, 3.2 s at 400, 10 s at
 * 700, 35 s at 1300, more than three minutes at 2350, and cases DU, DT at c = 16, c0 = 0, 10 to 20 s and some 2 GB at
 * x = 1.3e6 and more than 8 GB at x = 4.4e6 (the median; here the size of the outer rule overflowed an int), where the
 * conditional integral takes 0.1 to 0.4 s at every x.  With c0 > c the bulk of case DK lies at x < 0, where the line
 * integral is off by up to 9.8e-6 at c = 5, 5.5e-5 at c = 7 (a negative value at c0 = 9, x = -627), 1.5e-3 at c = 10 and
 * 2.9e-4 at c = 13 without a warning, and within 5e-7 at x >= -20.  The values: in case DK within 1.3e-8 of an independent
 * computation up to c = 14; in cases DU, DT within 1e-14 of the general engine up to c = 13 (x up to 3e4) and within 5e-7
 * of the large-c law at c = 13 and 14 (far_tail of DFExact.jl). */
static bool far_tail(int model, bool stat, double g0, double c, double c0, double x)
{
    return model >= 1 && model <= 3 && !stat && g0 == 0 && c > DFL_FAR_C && (x > DFL_FAR_X || (model == 1 && c0 > c && x < -DFL_FAR_XNEG));
}
/* the largest c at which the classical engine computes far above the unit root at large |x| (far_tail, or where the line
 * integral exceeds its bounds): DFL_FAR_CMAX = 14.  In case DK the checked conditional integral is off by up to 7.7e-5 at
 * c = 16, 5.5e-3 at c = 18 and by orders of magnitude at c = 20 against an independent computation (within 1.3e-8 at c = 13
 * and 14); in case DT at c = 16 one path gives values of 1e44 and more at the 95% and 99.7% points; in case DU its line
 * integral at c0 = 0 exhausts the memory there.  Above, such a point is an error, before anything is computed (far_check of
 * DFExact.jl) */
static int far_check(int kind, int model, double c, double c0, double x, char *err, size_t errlen)
{
    if (!(c > DFL_FAR_CMAX)) return 0;
    char cc0[48] = ""; if (c0 != 0) snprintf(cc0, sizeof cc0, ", c0 = %g", c0);
    snprintf(err, errlen, "%s: case %s, c = %g%s, x = %g: far above the unit root at large |x| (for c > %g at x > %g, in case DK with c0 > c at x < -%g, and where the line integral exceeds the bounds of the engine) the classical engine computes for c <= %g only: beyond, the conditional integral that takes these points is off by up to 7.7e-5 in case DK at c = 16 and gives values of 1e44 and more in case DT, the line integral exhausts the memory, and the general engine's line exceeds its bounds",
             kind == 2 ? "ltratio_cdf" : "ltratio_pdf", model == 1 ? "DK" : model == 2 ? "DU" : "DT", c, cc0, x, DFL_FAR_C, DFL_FAR_X, DFL_FAR_XNEG, DFL_FAR_CMAX);
    return 1;
}
/* a point of far_tail: the conditional integral as route_double gives it */
static int far_point(int kind, double x, int model, double c, double c0, double g0, bool stat, const dfl_double_opts *dopt, double *out, char *err, size_t errlen)
{
    dfl_bound_t bd; bd.bound = DFL_BOUND_RULE; bd.need = 0; bd.what = "the line integral far above the unit root";
    int fl = 0;
    return route_double(kind, x, model, c, c0, g0, stat, dopt, &bd, out, &fl, err, errlen);
}
/* the line integral at a point chosen for it: where its paths exceed the bounds of the engine (refused before its inner
 * integrals are evaluated), with method 0 the conditional integral (route_double; *nflag counts its flagged values), with
 * method 1 and in model 4 (which has the line integral only) an error that names the arguments (dfl_bound_msg) */
static int single_route(int kind, double x, int model, double c, double c0, double g0, bool stat, int method, const dfl_single_opts *so,
                        const dfl_double_opts *dopt, double *out, int *nflag, char *err, size_t errlen)
{
    dfl_bound_t bd;
    if (single_point(kind, x, model, c, c0, g0, stat, so, out, &bd) == 0) return 0;
    if (method != 0 || model == 4) { dfl_bound_msg(err, errlen, kind, model, c, c0, g0, stat, x, &bd); return 10 + bd.bound; }
    if (bd.bound == DFL_BOUND_RULE && !stat && g0 == 0 && far_check(kind, model, c, c0, x, err, errlen)) return 1;
    int fl = 0;
    if (route_double(kind, x, model, c, c0, g0, stat, dopt, &bd, out, &fl, err, errlen)) return 1;
    *nflag += fl;
    return 0;
}

/* the warning of the last dfl_dist or dfl_icdf on this thread (route_warn of DFExact.jl, and the flagged values that
 * verified = 1 keeps where the line integral refuses the point) */
static _Thread_local char WARN[1024];
const char *dfl_warning(void) { return WARN[0] ? WARN : NULL; }
void dfl_warning_clear(void) { WARN[0] = 0; }
/* (the router's composite calls, dfl_det_route.c: the warnings of their parts joined, each message once) */
void dfl_warning_join(const char *msg)
{
    if (msg == NULL || !msg[0] || strstr(WARN, msg)) return;
    size_t k = strlen(WARN);
    if (k + 2 >= sizeof WARN) return;
    snprintf(WARN + k, sizeof WARN - k, "%s%s", k ? "\n" : "", msg);
}
static void set_warning(int model, double c, double c0, double g0, bool stat, int nflag, int nkeep)
{
    WARN[0] = 0;
    if (nflag == 0 && nkeep == 0) return;
    char head[160]; int k = snprintf(head, sizeof head, "%s, c = %g, c0 = %g", case_name(model, stat), c, c0);
    if (g0 != 0 && k < (int) sizeof head) snprintf(head + k, sizeof head - k, ", g0 = %g", g0);
    int w = 0;
    if (nflag > 0) w = snprintf(WARN, sizeof WARN, "%s: the line integral's paths exceed the bounds of the engine at x > 0 (c0 < 0 near zero), and the "
                                "conditional integral that takes these points flags some of them; such values may be off by more than 1e-5 (the "
                                "density of case SK at c = c0 = -1e-3, x = 0.05 by 1.2e-3)", head);
    if (nkeep > 0 && w >= 0 && w < (int) sizeof WARN)
        snprintf(WARN + w, sizeof WARN - w, "%s%s: at some points the value of the conditional (or contour) integral is flagged and the line integral's "
                 "paths exceed the bounds of the engine; the flagged values are returned", w > 0 ? "\n" : "", head);
}

static int dist_impl(int kind, int n, const double *x, int model, double c, double c0, double g0, bool stat, int method,
                     const dfl_single_opts *so, const dfl_double_opts *dopt, double *out, int *nstatus, int *nflag, int *nkeep, char *err, size_t errlen)
{
    if (check_args(kind, model, c, c0, g0, stat, err, errlen)) return 1;
    if (check_method(model, g0, stat, method, err, errlen)) return 1;
    if (method != 2 && tstar_err(model, c, stat, err, errlen)) return 1;      /* (dfl_double checks its own) */
    dfl_single_opts sd; if (so == NULL) { dfl_single_defaults(&sd); so = &sd; }
    dfl_double_opts dd; if (dopt == NULL) { dfl_double_defaults(&dd); dopt = &dd; }
    if (nstatus) *nstatus = 0;
    /* (far above the unit root at large |x|, far_tail: refused above c = 14 before anything is computed, with verified too) */
    if (method == 0 && c > DFL_FAR_CMAX)
        for (int k = 0; k < n; k++) if (isfinite(x[k]) && far_tail(model, stat, g0, c, c0, x[k]) && far_check(kind, model, c, c0, x[k], err, errlen)) return 1;
    if (!dopt->verified) {
        /* the default: one evaluation by one method at each point (no second evaluation, no refinement driven by an
         * estimate, no fallback, no recomputation of the tails); no status (ltratio_dist of DFExact.jl).  A point that the
         * line integral refuses before evaluating it (its paths would exceed the bounds of the engine) goes to the
         * conditional integral with method 0 (single_route) */
        int nd1 = 0, nc1 = 0, rc1 = 0;
        double *xd1 = malloc((n + 1) * sizeof(double)), *xc1 = malloc((n + 1) * sizeof(double));
        int *id1 = malloc((n + 1) * sizeof(int)), *ic1 = malloc((n + 1) * sizeof(int));
        for (int k = 0; k < n && rc1 == 0; k++) {
            if (!isfinite(x[k])) { out[k] = nonfinite(kind, x[k]); continue; }
            if (method == 0 && far_tail(model, stat, g0, c, c0, x[k])) { rc1 = far_point(kind, x[k], model, c, c0, g0, stat, dopt, &out[k], err, errlen); continue; }
            int m = method == 0 ? automethod_one(model, c, c0, g0, stat, x[k]) : method;
            if (m == 2) { xd1[nd1] = x[k]; id1[nd1++] = k; }
            else if (m == 3) { xc1[nc1] = x[k]; ic1[nc1++] = k; }
            else rc1 = single_route(kind, x[k], model, c, c0, g0, stat, method, so, dopt, &out[k], nflag, err, errlen);
        }
        if (rc1 == 0 && nc1 > 0) {
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
        if (method == 0 && far_tail(model, stat, g0, c, c0, x[k])) { meth[k] = -1; if (rc == 0) rc = far_point(kind, x[k], model, c, c0, g0, stat, dopt, &out[k], err, errlen); continue; }
        int m = method == 0 ? automethod_verified(model, c, c0, g0, stat, x[k]) : method;
        meth[k] = m;
        if (m == 2) { xd[nd] = x[k]; id[nd] = k; tosingle[nd] = 1; nd++; }
        else if (m == 3) { xc[nc] = x[k]; ic[nc] = k; nc++; }
        else if (rc == 0) {
            int nf0 = *nflag;
            rc = single_route(kind, x[k], model, c, c0, g0, stat, method, so, dopt, &out[k], nflag, err, errlen);
            if (*nflag > nf0) flag[k] = 1;
        }
    }
    /* the contour integral; with method 0 a point at which it reports that its error estimate is not met gets the double
     * integral's value (the single integral's where that is flagged too), and its values below 1e-3 are recomputed by
     * the double integral as those of the single integral are below (ltratio_dist in DFExact.jl) */
    if (rc == 0 && nc > 0) {
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
            else if (tosingle[j]) {
                /* a flagged point chosen by the rule or flagged by the contour integral: the single integral's value, or, where
                 * its paths exceed the bounds of the engine, the flagged value (the contour integral's in out[k], the double
                 * integral's otherwise), with a warning */
                double v; dfl_bound_t bd;
                if (single_point(kind, x[k], model, c, c0, g0, stat, so, &v, &bd) == 0) out[k] = v;
                else { if (meth[k] == 2) out[k] = Fd[j]; (*nkeep)++; }
            }
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
int dfl_cdf_range_check(const char *fn, const char *args, const char *xname, int n, const double *x, const double *F, char *err, size_t errlen)
{
    for (int k = 0; k < n; k++) {
        if (!(F[k] < -DFL_CDF_RANGE_TOL || F[k] > 1 + DFL_CDF_RANGE_TOL)) continue;
        snprintf(err, errlen, "%s: %s, %s = %g: the computed distribution function is %.12g, outside [0, 1] by more than %g; the computation fails here and no value is returned",
                 fn, args, xname, x[k], F[k], DFL_CDF_RANGE_TOL);
        return 1;
    }
    return 0;
}
int dfl_dist(int kind, int n, const double *x, int model, double c, double c0, double g0, bool stat, int method,
             const dfl_single_opts *so, const dfl_double_opts *dopt, double *out, int *nstatus, char *err, size_t errlen)
{
    int nflag = 0, nkeep = 0;
    WARN[0] = 0;
    int rc = dist_impl(kind, n, x, model, c, c0, g0, stat, method, so, dopt, out, nstatus, &nflag, &nkeep, err, errlen);
    if (rc == 0 && kind == 2) {
        /* the final check of the distribution function (dfl_cdf_range_check) */
        char args[160]; int k = snprintf(args, sizeof args, "case %s, c = %g", case_name(model, stat), c);
        if (c0 != 0 && k < (int) sizeof args) k += snprintf(args + k, sizeof args - k, ", c0 = %g", c0);
        if (g0 != 0 && k < (int) sizeof args) snprintf(args + k, sizeof args - k, ", g0 = %g", g0);
        rc = dfl_cdf_range_check("ltratio_cdf", args, "x", n, x, out, err, errlen);
    }
    if (rc == 0) set_warning(model, c, c0, g0, stat, nflag, nkeep);
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
    /* (the contour integral with c0 > 0 below c = -15: 6e-6 at c = -27, c0 = 2.3, (c g0)^2/2 = 146; in the density up to 2e-2
     * at c = -60, g0 = 1.5, c0 = 5 in case DU; and with any c0 below c = -20 where |c g0| > 60: in case DU 3.9e-5 at c = -60,
     * g0 = 1.5, c0 = -2, and 6.3e-6 at c = -100, g0 = 0.9; brittle_msg of DFExact.jl, November 2026) */
    if (m == 3 && c0 > 0 && c < -15) {
        snprintf(msg, sizeof msg, "%s, fixed start, c0 > 0 far from the unit root (c < -15): the contour integral by one evaluation may be off by more than 1e-5%s", cas, hint);
        return msg;
    }
    if (m == 3 && c < -20 && fabs(c * g0) > 60) {
        snprintf(msg, sizeof msg, "%s, fixed start far from the unit root with |c g0| > 60: the contour integral by one evaluation may be off by more than 1e-5%s", cas, hint);
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
    WARN[0] = 0;
    if (check_args(2, model, c, c0, g0, stat, err, errlen)) return 1;
    if (check_method(model, g0, stat, method, err, errlen)) return 1;
    for (int k = 0; k < n; k++) if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "p must lie in (0, 1)"); return 1; }
    int nflag = 0, nkeep = 0;   /* (the warning of the points the line integral refused, once for the call: dfl_warning) */
    double m[2], me[2], mu, sd;
    if (model == 4) { mu = -1.9 + 0.1 * fmin(c, 0.0); sd = 0.8; }   /* (no moments for the GLS-detrended case: a rough start, the search is safeguarded) */
    else {
        /* (the start: the moments from jets of Bao and Kan, dfl_mom_start, 0.2 to 1 ms; otherwise this engine's circles, 10 to
         * 150 ms) */
        if (!dfl_mom_start(2, model, stat, c, c0, g0, m) && dfl_moments(model, c, c0, g0, stat, 2, m, me, err, errlen)) return 1;
        mu = m[0]; sd = sqrt(fmax(m[1] - m[0] * m[0], 1e-6));
    }
    if (!(tol > 0)) tol = 1e-10;
#define CDF(xx, out) do { if (dist_impl(2, 1, &(xx), model, c, c0, g0, stat, method, so, dopt, &(out), NULL, &nflag, &nkeep, err, errlen)) return 1; } while (0)
#define PDF(xx, out) do { if (dist_impl(1, 1, &(xx), model, c, c0, g0, stat, method, so, dopt, &(out), NULL, &nflag, &nkeep, err, errlen)) return 1; } while (0)
    for (int k = 0; k < n; k++) {
        double pk = p[k], x = mu + sd * sqrt(2) * erfinv(2 * pk - 1), lo = x - sd, hi = x + sd, F, f;
        for (int it = 0; it < 100; it++) { CDF(lo, F); if (!(F > pk)) break; lo -= 2 * (x - lo); }
        for (int it = 0; it < 100; it++) { CDF(hi, F); if (!(F < pk)) break; hi += 2 * (hi - x); }
        int j, cv = 0;
        for (j = 1; j <= 60; j++) {
            CDF(x, F); PDF(x, f);
            if (F > pk) hi = x; else lo = x;
            double xn = x - (F - pk) / f;
            if (!(lo <= xn && xn <= hi) || !isfinite(xn)) xn = (lo + hi) / 2;   /* (a step below half an ulp lands on the end of the bracket: converged) */
            if (fabs(xn - x) < tol) { x = xn; cv = 1; break; }
            x = xn;
        }
        q[k] = x; if (iter) iter[k] = j > 60 ? 60 : j; if (conv) conv[k] = cv;
        if (Fq) CDF(x, Fq[k]);
    }
#undef CDF
#undef PDF
    set_warning(model, c, c0, g0, stat, nflag, nkeep);
    return 0;
}
