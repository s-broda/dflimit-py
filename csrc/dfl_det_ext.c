/* dfl_det_ext.c: the extensions of the limit of the general engine (a port of src/DFDet_vol.jl, src/DFDet_cov.jl and
 * src/DFDet_seas.jl of DFLimit.jl): the checks of a volatility path; a normal component, T = rho tau_c0 + sqrt(1 - rho^2) Z,
 * the limit of the covariate-augmented t-ratio and of the t-ratio of the slope in a predictive regression with a regressor
 * local to unity (the inversion on two lines with the normal component in the kernel, the normal average of the t-ratio's
 * distribution function, the local null values of the two applications); the first seasonal t-ratio of the regression of
 * Hylleberg, Engle, Granger and Yoo at a harmonic frequency (the sum of the residues on the outer contour).  The transform
 * and the paths of the volatility regimes are in dfl_det_mgf.c and dfl_det_path.c, their user-level functions in
 * dfl_det_api.c (dfl_det_dist and dfl_det_quantile with o->vol). */
#include "dfl_det_int.h"

int dfl_vol_check(const dfl_vol *v, char *err, size_t errlen)
{
    if (!v || v->nreg < 1) { snprintf(err, errlen, "volpath: a volatility path has at least one regime (nreg >= 1)"); return 1; }
    for (int i = 0; i < v->nreg - 1; i++) if (!(isfinite(v->breaks[i]) && v->breaks[i] > 0 && v->breaks[i] < 1)) { snprintf(err, errlen, "volpath: the break dates must lie in (0, 1)"); return 1; }
    for (int i = 0; i + 1 < v->nreg - 1; i++) if (!(v->breaks[i] < v->breaks[i + 1])) { snprintf(err, errlen, "volpath: the break dates must be strictly increasing"); return 1; }
    if (v->omega) for (int i = 0; i < v->nreg; i++) if (!(isfinite(v->omega[i]) && v->omega[i] > 0)) { snprintf(err, errlen, "volpath: the standard deviations omega must be finite and positive"); return 1; }
    return 0;
}
/* ---------- the local null values of the two applications of a normal component ---------- */
int dfl_cadf_c0(double c, double rho, double R, double *c0, char *err, size_t errlen)
{
    if (isnan(R)) R = fabs(rho);
    if (!(isfinite(rho) && rho >= -1 && rho <= 1 && rho != 0)) { snprintf(err, errlen, "cadf_c0: rho must lie in [-1, 1] and not be 0"); return 1; }
    if (!(isfinite(R) && R > 0)) { snprintf(err, errlen, "cadf_c0: R must be finite and positive"); return 1; }
    if (!isfinite(c)) { snprintf(err, errlen, "cadf_c0: c must be finite"); return 1; }
    *c0 = c == 0 ? 0.0 : c * (1 - 1 / (rho * R));
    return 0;
}
int dfl_predictive_c0(double c, double rho, double b, double *c0, char *err, size_t errlen)
{
    if (!(isfinite(rho) && rho >= -1 && rho <= 1 && rho != 0)) { snprintf(err, errlen, "predictive_c0: rho must lie in [-1, 1] and not be 0"); return 1; }
    if (!(isfinite(c) && isfinite(b))) { snprintf(err, errlen, "predictive_c0: c and b must be finite"); return 1; }
    *c0 = c - b / rho;
    return 0;
}

/* ---------- a normal component: T = rho tau_c0 + sqrt(1 - rho^2) Z (src/DFDet_cov.jl) ---------- */
static double normcdf_(double x) { return erfc(-x / sqrt(2)) / 2; }
static double normpdf_(double x) { return exp(-x * x / 2) / sqrt(2 * M_PI); }
/* the inverse error function (Giles' approximation refined by Newton on erf; as in dfl_det_route.c) */
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
#define COV_METHOD_MSG "with rho != 1 the method must be 'auto', 'lines' (the inversion on two lines with the normal component in the kernel) or 'mixture' (the normal average of the t-ratio's distribution function)"
/* the checks of the arguments with rho (cov_args and the options that do not apply) */
static int cov_args(const char *fn, const dfl_design *D, const dfl_det_opts *o, char *err, size_t errlen)
{
    if (!(isfinite(o->rho) && o->rho >= -1 && o->rho <= 1)) { snprintf(err, errlen, "%s: rho must lie in [-1, 1]", fn); return 1; }
    if (!(o->method == DFL_DET_AUTO || o->method == DFL_DET_LINES || o->method == DFL_DET_MIXTURE)) { snprintf(err, errlen, "%s: " COV_METHOD_MSG, fn); return 1; }
    if (!isnan(o->gls)) { snprintf(err, errlen, "%s: rho != 1 (a normal component: the covariate-augmented t-ratio, the predictive regression) is implemented for least squares detrending; GLS detrending (gls) is not supported with it", fn); return 1; }
    if (D->first) { snprintf(err, errlen, "%s: rho != 1 is implemented for the one-step regression; the statistic that detrends first (detrend_first) is not supported with it", fn); return 1; }
    if (o->n > 0 || o->ntail > 0) { snprintf(err, errlen, "%s: with rho != 1 the options %s do not apply", fn, o->n > 0 && o->ntail > 0 ? "n, ntail" : o->n > 0 ? "n" : "ntail"); return 1; }
    if (!isfinite(o->c0)) { snprintf(err, errlen, "%s: c0 must be a finite real scalar", fn); return 1; }
    if (o->start < 0 || o->start > 2) { snprintf(err, errlen, "start must be zero (0), fixed (1) or stationary (2)"); return 1; }
    return 0;
}
/* the computation of method auto (cov_method): the lines for 0.1 <= rho^2 <= 0.99 with a zero or stationary start at c <= 0,
 * the normal average otherwise (near |rho| = 1 the kernel of the lines oscillates, its phase reaching x^2/(2 (1 - rho^2)),
 * and the mixture is the cheaper; the lines do not cover a fixed start or the vertical line above the unit root) */
static int cov_method(const char *fn, int method, double rho, double c, int fixed, int *m, char *err, size_t errlen)
{
    if (method == DFL_DET_LINES) {
        if (fixed) { snprintf(err, errlen, "%s: method = 'lines' (the inversion on two lines) needs a zero or a stationary start; method = 'mixture' covers a fixed start", fn); return 1; }
        if (c > 0) { snprintf(err, errlen, "%s: method = 'lines' (the inversion on two lines) is implemented below and at the unit root (c <= 0); method = 'mixture' covers c > 0", fn); return 1; }
        *m = DFL_DET_LINES; return 0;
    }
    if (method == DFL_DET_MIXTURE) { *m = DFL_DET_MIXTURE; return 0; }
    *m = (fixed || c > 0 || !(0.1 <= rho * rho && rho * rho <= 0.99)) ? DFL_DET_MIXTURE : DFL_DET_LINES;
    return 0;
}
/* a generous bound on the bulk of tau_c0 = tau_c + (c - c0) sqrt(V): m -+ 40 s, m = (c - c0) sqrt(E int J^2), s = 1 + |c -
 * c0| sqrt(E int J^2), E int J^2 of the process with its start (tau_bulk) */
static void tau_bulk(double c, double c0, int start, double g0, double *lo, double *hi)
{
    double ej = fabs(c) < 1e-6 ? 0.5 : (expm1(2 * c) - 2 * c) / (4 * c * c);
    if (start == DFL_START_STAT) ej = -1 / (2 * c);
    if (start == DFL_START_FIXED) ej += g0 * g0 * (fabs(c) < 1e-6 ? 1.0 : expm1(2 * c) / (2 * c));
    double r = sqrt(ej), m = (c - c0) * r, s = 1 + fabs(c - c0) * r;
    *lo = m - 40 * s; *hi = m + 40 * s;
}
/* the rank of the numerator and the shift a of the model of D with its start (startmodel) */
static int start_model_ra(const dfl_design *D, int start, double g0, int *r, double *a, char *err, size_t errlen)
{
    det_prep *P = det_prep_cached(D, NAN, err, errlen);
    if (!P) return 1;
    det_model Mo;
    int rc = det_model_make(P, start, g0, &Mo, err, errlen);
    if (!rc) { *r = Mo.r; *a = Mo.a; det_model_free(&Mo); }
    det_prep_release(P);
    return rc;
}
/* P(rho tau_c0 + rb Z <= x) and its density for 0 < rho < 1 at the finite points x from the t-ratio's distribution function
 * (mixture_both): P(T <= x) = int F((x - rb z)/rho) phi(z) dz, f_T(x) = -(1/rb) int F((x - rb z)/rho) z phi(z) dz, the
 * trapezoidal rule with step h in z on |z| <= 8.5 (default 0.25/(1 + rb/rho), half that for a numerator of rank one), for
 * c0 > 0 Gauss-Legendre panels of 12 nodes graded toward the kink of the distribution function at q_* = -2 sqrt(c0 a), no
 * longer than 4h; F 0 or 1 outside the bulk of tau; the engine's distribution function in one call for all points (the
 * inversion on paths where it is not finite) */
static int mixture_both(const dfl_design *D, int nx, const double *x, double c, double rho, double c0, int start, double g0, int verified, double h,
                        double *F, double *f, char *err, size_t errlen)
{
    const double zmax = 8.5; const int n = 12;
    double rb = sqrt(1 - rho * rho), a; int r;
    if (start_model_ra(D, start, g0, &r, &a, err, errlen)) return 1;
    double hz = isnan(h) ? (r == 1 ? 0.125 : 0.25) / (1 + rb / rho) : h;
    double qstar = c0 > 0 ? -2 * sqrt(c0 * a) : NAN;
    const double *gw, *gz = det_gl(n, &gw);
    dvec zs = {0}, ws = {0};
    int *first = malloc((nx + 1) * sizeof(int));
    for (int i = 0; i < nx; i++) {
        first[i] = zs.n;
        double zstar = (x[i] - rho * qstar) / rb;
        if (c0 > 0 && fabs(zstar) < zmax) {
            double *ab, hmin = 1e-8;
            int np = det_graded_panels(-zmax, zmax, 1, &zstar, &hmin, &ab);
            for (int k = 0; k < np; k++) {
                double a0 = ab[2 * k], b0 = ab[2 * k + 1];
                int m = imax(1, (int) ceil((b0 - a0) / (4 * hz)));
                for (int l = 0; l < m; l++) {
                    double a1 = a0 + (b0 - a0) * l / m, b1 = a0 + (b0 - a0) * (l + 1) / m;
                    for (int j = 0; j < n; j++) { dvec_push(&zs, a1 + (b1 - a1) * gz[j]); dvec_push(&ws, (b1 - a1) * gw[j]); }
                }
            }
            free(ab);
        } else {
            int nz = (int) ceil(zmax / hz);
            for (int k = -nz; k <= nz; k++) { dvec_push(&zs, k * hz); dvec_push(&ws, hz); }
        }
    }
    first[nx] = zs.n;
    int N = zs.n, rc = 0;
    double *q = malloc((N + 1) * sizeof(double)), *Fq = malloc((N + 1) * sizeof(double)), *qb = malloc((N + 1) * sizeof(double)), *vb = malloc((N + 1) * sizeof(double));
    int *ib = malloc((N + 1) * sizeof(int)), nb = 0;
    for (int i = 0; i < nx; i++) for (int k = first[i]; k < first[i + 1]; k++) q[k] = (x[i] - rb * zs.v[k]) / rho;
    double lo, hi; tau_bulk(c, c0, start, g0, &lo, &hi);
    for (int k = 0; k < N; k++) { Fq[k] = q[k] < lo ? 0.0 : 1.0; if (lo <= q[k] && q[k] <= hi) { ib[nb] = k; qb[nb++] = q[k]; } }
    if (nb > 0) {
        dfl_det_opts o; dfl_det_defaults(&o); o.c0 = c0; o.start = start; o.gamma0 = g0; o.verified = verified;
        rc = dfl_det_dist(2, D, nb, qb, c, &o, vb, err, errlen);
        if (!rc) {
            /* (where the line integral, the engine's default for these designs at c0 = 0, returns a value that is not
             * finite, the inversion on paths: case DK at c = 0 gives NaN at x = -7.8125 and below x = 1e-150) */
            int nn = 0;
            for (int j = 0; j < nb; j++) { Fq[ib[j]] = vb[j]; if (!isfinite(vb[j])) { ib[nn] = ib[j]; qb[nn++] = qb[j]; } }
            if (nn > 0) {
                o.method = DFL_DET_PATHS;
                rc = dfl_det_dist(2, D, nn, qb, c, &o, vb, err, errlen);
                if (!rc) for (int j = 0; j < nn; j++) Fq[ib[j]] = vb[j];
            }
        }
    }
    if (!rc) for (int i = 0; i < nx; i++) {
        double sF = 0, sf = 0;
        for (int k = first[i]; k < first[i + 1]; k++) {
            double w = ws.v[k] * normpdf_(zs.v[k]) * dclamp(Fq[k], 0.0, 1.0);
            sF += w; sf -= w * zs.v[k] / rb;
        }
        F[i] = sF; f[i] = sf;
    }
    free(first); dvec_free(&zs); dvec_free(&ws); free(q); free(Fq); free(qb); free(vb); free(ib);
    return rc;
}
/* F and f of T at the finite points x by the method m (cov_both) */
static int cov_both(const dfl_design *D, int nx, const double *x, double c, double rho, double c0, int start, double g0, int m, int verified, double h,
                    double *F, double *f, char *err, size_t errlen)
{
    if (m == DFL_DET_LINES) {
        det_path *CP = det_path_cached(D, NAN, start, g0, c, NAN, 1.0, err, errlen);
        if (!CP) return 1;
        int rc = det_lines_both(CP, nx, x, rho, c0, F, f, err, errlen);
        det_path_release(CP);
        return rc;
    }
    return mixture_both(D, nx, x, c, rho, c0, start, g0, verified, h, F, f, err, errlen);
}
/* the checked computation with rho (cov_verified): the other route as well; where the two differ by more than vtol the
 * normal average with half its step on the checked distribution function of tau, with a warning */
static int cov_verified(const dfl_design *D, int nx, const double *x, double c, double rho, double c0, int start, double g0, int m, double vtol,
                        double *F, double *f, char *err, size_t errlen)
{
    if (cov_both(D, nx, x, c, rho, c0, start, g0, m, 0, NAN, F, f, err, errlen)) return 1;
    double rb = sqrt(1 - rho * rho), a; int r;
    if (start_model_ra(D, start, g0, &r, &a, err, errlen)) return 1;
    double hh = (r == 1 ? 0.0625 : 0.125) / (1 + rb / rho);
    double *F2 = malloc((nx + 1) * sizeof(double)), *f2 = malloc((nx + 1) * sizeof(double));
    int rc = m == DFL_DET_LINES ? cov_both(D, nx, x, c, rho, c0, start, g0, DFL_DET_MIXTURE, 0, NAN, F2, f2, err, errlen)
                                : cov_both(D, nx, x, c, rho, c0, start, g0, DFL_DET_MIXTURE, 1, hh, F2, f2, err, errlen);
    double d = 0;
    for (int i = 0; i < nx && !rc; i++) d = dmax(d, dmax(fabs(F[i] - F2[i]), fabs(f[i] - f2[i])));
    if (!rc && !(d <= vtol)) {
        if (m == DFL_DET_LINES) rc = cov_both(D, nx, x, c, rho, c0, start, g0, DFL_DET_MIXTURE, 1, hh, F2, f2, err, errlen);
        if (!rc) {
            memcpy(F, F2, nx * sizeof(double)); memcpy(f, f2, nx * sizeof(double));
            det_warn("verified: the two computations of the normal component differ by %.2g (above %g); the normal average on the checked distribution function is returned (c = %g, c0 = %g, rho = %g)", d, vtol, c, c0, rho);
        }
    }
    free(F2); free(f2);
    return rc;
}
static int cov_dist_(int kind, const dfl_design *D, int n, const double *x, double c, const dfl_det_opts *o, double *out, char *err, size_t errlen)
{
    const char *fn = kind == 2 ? "dist_cdf" : "dist_pdf";
    if (cov_args(fn, D, o, err, errlen)) return 1;
    double rho = o->rho;
    if (rho == 0) { for (int k = 0; k < n; k++) out[k] = kind == 2 ? normcdf_(x[k]) : normpdf_(x[k]); return 0; }
    if (rho < 0) {
        /* the reflection P(T <= x) = 1 - P(|rho| tau + rb Z <= -x), f_T(x) = f(-x) (rho = -1: the t-ratio's own functions) */
        double *mx = malloc((n + 1) * sizeof(double));
        for (int k = 0; k < n; k++) mx[k] = -x[k];
        int rc;
        if (rho == -1) {
            dfl_det_opts o1; dfl_det_defaults(&o1); o1.c0 = o->c0; o1.start = o->start; o1.gamma0 = o->gamma0; o1.verified = o->verified;
            rc = dfl_det_dist(kind, D, n, mx, c, &o1, out, err, errlen);
        } else { dfl_det_opts o1 = *o; o1.rho = -rho; rc = cov_dist_(kind, D, n, mx, c, &o1, out, err, errlen); }
        free(mx);
        if (!rc && kind == 2) for (int k = 0; k < n; k++) out[k] = isnan(out[k]) ? NAN : 1 - out[k];
        return rc;
    }
    if (!isfinite(c)) { snprintf(err, errlen, "c must be a finite real scalar"); return 1; }
    int fixed = o->start == DFL_START_FIXED && o->gamma0 != 0, m;
    if (cov_method(fn, o->method, rho, c, fixed, &m, err, errlen)) return 1;
    double *xf = malloc((n + 1) * sizeof(double)), *F = malloc((n + 1) * sizeof(double)), *f = malloc((n + 1) * sizeof(double));
    int *ix = malloc((n + 1) * sizeof(int)), nf = 0, rc = 0;
    for (int k = 0; k < n; k++) {
        if (isfinite(x[k])) { xf[nf] = x[k]; ix[nf++] = k; }
        else out[k] = isnan(x[k]) ? NAN : x[k] > 0 ? (kind == 2 ? 1.0 : 0.0) : 0.0;
    }
    if (nf > 0) {
        rc = o->verified ? cov_verified(D, nf, xf, c, rho, o->c0, o->start, o->gamma0, m, o->vtol, F, f, err, errlen)
                         : cov_both(D, nf, xf, c, rho, o->c0, o->start, o->gamma0, m, 0, NAN, F, f, err, errlen);
        if (!rc) for (int j = 0; j < nf; j++) out[ix[j]] = kind == 2 ? F[j] : f[j];
    }
    free(xf); free(F); free(f); free(ix);
    return rc;
}
int det_cov_dist(int kind, const dfl_design *D, int n, const double *x, double c, const dfl_det_opts *o, double *out, char *err, size_t errlen)
{
    det_warn_hold(1);
    int rc = cov_dist_(kind, D, n, x, c, o, out, err, errlen);
    det_warn_hold(0);
    return rc;
}
/* the p-quantile with rho != 1 (cov_quantile): Newton's method with the density, the step limited to 1, then the
 * safeguarded search; rho = 0 the normal quantile, rho < 0 by the reflection */
typedef struct { const dfl_design *D; double c, rho, c0; int start; double g0; int m; } covq_ctx;
static int covq_ff(double y, void *v, double *F, double *f, char *err, size_t errlen)
{
    covq_ctx *q = v;
    return cov_both(q->D, 1, &y, q->c, q->rho, q->c0, q->start, q->g0, q->m, 0, NAN, F, f, err, errlen);
}
static int cov_quantile1(const dfl_design *D, double p, double c, double x0, double tol, const dfl_det_opts *o, double *q, char *err, size_t errlen)
{
    const char *fn = "dist_quantile";
    if (cov_args(fn, D, o, err, errlen)) return 1;
    double rho = o->rho;
    if (rho == 0) { *q = sqrt(2) * erfinv_(2 * p - 1); return 0; }
    if (rho < 0) {
        double pp = 1 - p, qq; int rc;
        if (rho == -1) {
            dfl_det_opts o1; dfl_det_defaults(&o1); o1.c0 = o->c0; o1.start = o->start; o1.gamma0 = o->gamma0;
            rc = dfl_det_quantile(D, 1, &pp, c, -x0, tol, &o1, &qq, err, errlen);
        } else { dfl_det_opts o1 = *o; o1.rho = -rho; rc = cov_quantile1(D, pp, c, -x0, tol, &o1, &qq, err, errlen); }
        if (!rc) *q = -qq;
        return rc;
    }
    if (!isfinite(c)) { snprintf(err, errlen, "c must be a finite real scalar"); return 1; }
    int m;
    if (cov_method(fn, o->method, rho, c, o->start == DFL_START_FIXED && o->gamma0 != 0, &m, err, errlen)) return 1;
    covq_ctx qc = {D, c, rho, o->c0, o->start, o->gamma0, m};
    double x = x0, F, f;
    for (int it = 0; it < 30; it++) {
        if (covq_ff(x, &qc, &F, &f, err, errlen)) return 1;
        double dx = dclamp((F - p) / f, -1.0, 1.0);
        if ((F - p) / f != (F - p) / f) dx = NAN;
        if (!isfinite(dx)) break;
        x -= dx;
        if (fabs(dx) < tol) { *q = x; return 0; }
    }
    return det_newton_safe(covq_ff, &qc, p, x0, 1.0, tol, q, err, errlen);
}
int det_cov_quantile(const dfl_design *D, int n, const double *p, double c, double x0, double tol, const dfl_det_opts *o, double *q, char *err, size_t errlen)
{
    det_warn_hold(1);
    int rc = 0;
    for (int k = 0; k < n && !rc; k++) rc = cov_quantile1(D, p[k], c, x0, tol, o, &q[k], err, errlen);
    det_warn_hold(0);
    return rc;
}

/* ---------- the first seasonal t-ratio of the regression of Hylleberg, Engle, Granger and Yoo (src/DFDet_seas.jl) ---------- */
/* the smallest distance of two roots of one group along the path, relative to their size (the residues divide by it) */
static double t3_gap(const det_path *CP)
{
    int r = CP->r, rp = CP->Mo->rp; double g = INFINITY;
    for (int k = 0; k < CP->K; k++) {
        const cdouble *R = CP->R + (size_t) k * r;
        for (int grp = 0; grp < 2; grp++) {
            int lo = grp ? rp : 0, hi = grp ? r : rp;
            for (int i = lo; i < hi; i++) for (int j = i + 1; j < hi; j++) g = dmin(g, cfabs(csub(R[i], R[j])) / dmax(cfabs(R[i]), cfabs(R[j])));
        }
    }
    return g;
}
/* the relative distance of two roots of a group below which the doubled transform on the paths computes */
#define T3_GAP 1e-4
/* the transform e^{-s}/D(s,t) in the form the engine's inversion on paths integrates (t3_doubled): every root of the design
 * twice, log P twice, the shift a = 1; the path and its model in one block (free) */
typedef struct { det_path CP; det_model Mo; } t3dbl_t;
static t3dbl_t *t3_doubled(const det_path *CP)
{
    int r = CP->r, rp = CP->Mo->rp, K = CP->K;
    t3dbl_t *T = calloc(1, sizeof *T);
    T->Mo = *CP->Mo; T->Mo.r = 2 * r; T->Mo.rp = 2 * rp; T->Mo.rm = 2 * CP->Mo->rm; T->Mo.a = 1.0; T->Mo.kappa = 0.5; T->Mo.own = 0;
    T->CP = *CP; T->CP.r = 2 * r; T->CP.Mo = &T->Mo; T->CP.fx = NULL; T->CP.mref = NULL; T->CP.vm = NULL; T->CP.refs = 1;
    T->CP.R = malloc((size_t) 2 * r * imax(K, 1) * sizeof(cdouble)); T->CP.lP = malloc(imax(K, 1) * sizeof(cdouble));
    for (int k = 0; k < K; k++) {
        const cdouble *R = CP->R + (size_t) k * r; cdouble *R2 = T->CP.R + (size_t) k * 2 * r;
        for (int j = 0; j < rp; j++) { R2[2 * j] = R[j]; R2[2 * j + 1] = R[j]; }
        for (int j = rp; j < r; j++) { R2[2 * j] = R[j]; R2[2 * j + 1] = R[j]; }
        T->CP.lP[k] = cscale(2, CP->lP[k]);
    }
    return T;
}
static void t3_doubled_free(t3dbl_t *T) { if (!T) return; free(T->CP.R); free(T->CP.lP); free(T); }
/* F and f of t3 at the finite points x on the path: the residues, or where two roots of a group come within T3_GAP the
 * engine's inversion on paths of the doubled transform (with n, ntail > 0 its inner nodes) */
static int t3_eval(const det_path *CP, int gap, int nx, const double *x, double c0, int n, int ntail, double *F, double *f, char *err, size_t errlen)
{
    if (!gap && n <= 0) return det_t3_residues(CP, nx, x, c0, F, f, err, errlen);
    t3dbl_t *T = t3_doubled(CP);
    det_eval_opts o; det_eval_defaults(&T->CP, &o);
    if (n > 0) o.n = n;
    if (ntail > 0) o.ntail = ntail;
    int rc = det_contour_eval(&T->CP, nx, x, c0, 2, &o, F, f, NULL, NULL, err, errlen);
    t3_doubled_free(T);
    return rc;
}
/* the outer contour of t3 (t3_path): the engine's (the present rule) for the design at c with a zero start, or the
 * volatility path */
static det_path *t3_path(const char *fn, const dfl_design *D, double c, const dfl_det_opts *o, char *err, size_t errlen)
{
    if (o->vol) {
        dfl_det_opts ov; dfl_det_defaults(&ov); ov.vol = o->vol;
        det_vmodel *VM; double c1 = c;
        if (det_vol_args(fn, D, c, &ov, &VM, &c1, err, errlen)) return NULL;
        if (VM) { det_path *CP = det_vol_path(VM, NAN, 1.0, err, errlen); det_vol_release(VM); return CP; }
        c = c1;
    }
    if (!isfinite(c)) { snprintf(err, errlen, "%s: c must be a finite real scalar", fn); return NULL; }
    if (!(c <= 0)) { snprintf(err, errlen, "%s: c must be at most 0 (the seasonal statistics are implemented below and at the unit root)", fn); return NULL; }
    if (!(c >= -480)) { snprintf(err, errlen, "%s: c must be at least -480 (the range in which the computation is checked)", fn); return NULL; }
    det_path *CP = det_path_cached(D, NAN, DFL_START_ZERO, 0.0, c, NAN, 1.0, err, errlen);
    if (CP && isnan(CP->h)) {
        det_path_release(CP);
        snprintf(err, errlen, "%s: the outer contour of this design is the vertical line (no hyperbola keeps its positive roots right of the origin); the seasonal statistics are not implemented there", fn);
        return NULL;
    }
    return CP;
}
static int t3_opts(const char *fn, const dfl_design *D, const dfl_det_opts *o, char *err, size_t errlen)
{
    if (!D) { snprintf(err, errlen, "no design"); return 1; }
    if (!isfinite(o->c0)) { snprintf(err, errlen, "%s: c0 must be a finite real scalar", fn); return 1; }
    if (o->start != DFL_START_ZERO || !isnan(o->gls) || o->rho != 1) { snprintf(err, errlen, "%s: the seasonal t-ratio has the zero start, least squares detrending and no normal component (start, gls and rho at their defaults)", fn); return 1; }
    return 0;
}
int dfl_hegy_t3(int kind, const dfl_design *D, int n, const double *x, double c, const dfl_det_opts *o0, double *out, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = o0; if (!o) { dfl_det_defaults(&od); o = &od; }
    const char *fn = kind == 2 ? "hegy_t3_cdf" : "hegy_t3_pdf";
    det_warn_clear();
    if (kind != 1 && kind != 2) { snprintf(err, errlen, "kind must be 1 (density) or 2 (distribution function)"); return 1; }
    if (t3_opts(fn, D, o, err, errlen)) return 1;
    det_path *CP = t3_path(fn, D, c, o, err, errlen);
    if (!CP) return 1;
    double *xf = malloc((n + 1) * sizeof(double)), *F = malloc((n + 1) * sizeof(double)), *f = malloc((n + 1) * sizeof(double));
    int *ix = malloc((n + 1) * sizeof(int)), m = 0, rc = 0;
    for (int k = 0; k < n; k++) {
        if (isfinite(x[k])) { xf[m] = x[k]; ix[m++] = k; }
        else out[k] = isnan(x[k]) ? NAN : x[k] > 0 ? (kind == 2 ? 1.0 : 0.0) : 0.0;
    }
    if (m > 0) {
        int gap = t3_gap(CP) < T3_GAP;
        rc = t3_eval(CP, gap, m, xf, o->c0, 0, 0, F, f, err, errlen);
        if (!rc && o->verified) {
            /* the residues against the inversion on paths of the doubled transform with 40/64 nodes; the latter where they differ */
            double *F2 = malloc((m + 1) * sizeof(double)), *f2 = malloc((m + 1) * sizeof(double)), d = 0;
            rc = t3_eval(CP, 1, m, xf, o->c0, 40, 64, F2, f2, err, errlen);
            for (int j = 0; j < m && !rc; j++) d = dmax(d, fabs(kind == 2 ? F[j] - F2[j] : f[j] - f2[j]));
            if (!rc && !(d <= o->vtol)) {
                det_warn("verified: the residues and the inversion on paths of the doubled transform differ by %.2g (above %g); the latter is returned (c = %g, c0 = %g)", d, o->vtol, c, o->c0);
                memcpy(F, F2, m * sizeof(double)); memcpy(f, f2, m * sizeof(double));
            }
            free(F2); free(f2);
        }
        if (!rc) for (int j = 0; j < m; j++) out[ix[j]] = kind == 2 ? F[j] : f[j];
        if (!rc && kind == 2) {
            char args[200];
            snprintf(args, sizeof args, "the first seasonal t-ratio, a design of %d regressors on %d pieces, c = %g, c0 = %g", D->K, D->np, o->vol ? CP->c : c, o->c0);
            double *vf = malloc((m + 1) * sizeof(double));
            for (int j = 0; j < m; j++) vf[j] = F[j];
            rc = dfl_cdf_range_check(fn, args, "x", m, xf, vf, err, errlen);
            free(vf);
        }
    }
    free(xf); free(F); free(f); free(ix); det_path_release(CP);
    return rc;
}
typedef struct { const det_path *CP; int gap; double c0; } t3q_ctx;
static int t3q_ff(double y, void *v, double *F, double *f, char *err, size_t errlen)
{
    t3q_ctx *q = v;
    return t3_eval(q->CP, q->gap, 1, &y, q->c0, 0, 0, F, f, err, errlen);
}
int dfl_hegy_t3_quantile(const dfl_design *D, int n, const double *p, double c, double x0, double tol, const dfl_det_opts *o0, double *q, char *err, size_t errlen)
{
    dfl_det_opts od; const dfl_det_opts *o = o0; if (!o) { dfl_det_defaults(&od); o = &od; }
    const char *fn = "hegy_t3_quantile";
    det_warn_clear();
    for (int k = 0; k < n; k++) if (!(p[k] > 0 && p[k] < 1)) { snprintf(err, errlen, "%s: p must lie in (0, 1)", fn); return 1; }
    if (isnan(x0)) x0 = -2.0;
    if (!isfinite(x0)) { snprintf(err, errlen, "%s: x0 must be a finite real scalar (NaN: the default)", fn); return 1; }
    if (tol == INFINITY) { snprintf(err, errlen, "%s: tol must be a positive finite number (NaN: the default)", fn); return 1; }
    if (!(tol > 0)) tol = 1e-10;
    if (t3_opts(fn, D, o, err, errlen)) return 1;
    det_path *CP = t3_path(fn, D, c, o, err, errlen);
    if (!CP) return 1;
    t3q_ctx qc = {CP, t3_gap(CP) < T3_GAP, o->c0};
    int rc = 0;
    for (int k = 0; k < n && !rc; k++) {
        double x = x0, F, f; int done = 0;
        for (int it = 0; it < 30 && !rc; it++) {
            rc = t3q_ff(x, &qc, &F, &f, err, errlen);
            if (rc) break;
            double dx = dclamp((F - p[k]) / f, -1.0, 1.0);
            if ((F - p[k]) / f != (F - p[k]) / f) dx = NAN;
            if (!isfinite(dx)) break;
            x -= dx;
            if (fabs(dx) < tol) { q[k] = x; done = 1; break; }
        }
        if (!rc && !done) rc = det_newton_safe(t3q_ff, &qc, p[k], x0, 1.0, tol, &q[k], err, errlen);
    }
    det_path_release(CP);
    return rc;
}
