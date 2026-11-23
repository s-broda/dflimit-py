/* dfl_det_kan.c: the devices of Bao and Kan for the general engine (a port of src/DFKan_det.jl of DFLimit.jl): the joint
 * transform of the pair (U, V) of a model of the general engine (a design with a zero or stationary start, any detrending)
 * with the branch of the square root of its determinant fixed at each point from the pivots of the precision matrix
 * (det_detlog_piv in dfl_det_mgf.c), and the single inversion of the statistics that reject on a half-plane: the
 * point-optimal statistic (its distribution function, density and quantiles) and the coefficient statistic by the formula
 * of Gil-Pelaez (coef_cdf_gp), by the generic inversion of dfl_kan.c (adaptive quadrature over decades, the tail completed
 * by integration by parts).  The method is that of Yong Bao and Raymond Kan, "Exact Finite-Sample Distribution and Moments
 * of the Least Squares Estimator in First-Order Autoregressions: O(1) Algorithms".
 *
 * The general engine writes M(s, t) = e^{-s/2} [e^{c + theta} det(I - 2 Q Sigma)]^{-1/2}, Q = s A + t B + gamma E11, Sigma =
 * Sigma(theta) the covariance of the N functionals under a reference law, theta = (c^2 - 2t)^{1/2}; the grid of the fixed
 * inversion of Gil-Pelaez continued the phase of the determinant along u, here it is fixed at the point. */
#include "dfl_det_int.h"
#include "dfl_kan_int.h"
#include <stdint.h>

typedef struct { double t1, t2; cdouble v; int used; } cf_ent;
struct det_cf {
    const det_model *Mo;
    double c;
    det_work W;
    cdouble *K, *Sl, *Sc;
    int *piv;
    int memo;                  /* the values along the ray kept (MemoCF: the quantile searches evaluate the same ray at many b) */
    cf_ent *tab; size_t cap, n;
    int have_scale; double scale[4];
    dfl_kcf k;
};
/* log E exp(i t1 A + i t2 B) = log M(i t1, i (t2 - c t1)) at one point (logmgf of DetCF) */
static cdouble cf_raw(det_cf *F, double t1, double t2)
{
    cdouble s = cmul(CPX(0, 1), CPX(t1, 0)), t = cmul(CPX(0, 1), csub(CPX(t2, 0), cscale(F->c, CPX(t1, 0)))), th;
    if (det_logmgf_tpart(&F->W, F->c, t, &th, NULL)) return CPX(NAN, NAN);
    cdouble ld = det_detlog_piv(&F->W, F->c, s, t, th, F->K, F->Sl, F->Sc, F->piv);
    return csub(csub(cdivr(cneg(s), 2), cdivr(caddr(th, F->c), 2)), cdivr(ld, 2));
}
static size_t cf_hash(double a, double b, size_t cap)
{
    uint64_t x, y; memcpy(&x, &a, 8); memcpy(&y, &b, 8);
    uint64_t h = x * 0x9E3779B97F4A7C15ULL ^ (y + 0x632BE59BD9B4E019ULL + (x << 6) + (x >> 2));
    h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 29;
    return (size_t) h & (cap - 1);
}
static cf_ent *cf_find(cf_ent *tab, size_t cap, double t1, double t2)
{
    size_t i = cf_hash(t1, t2, cap);
    while (tab[i].used && !(tab[i].t1 == t1 && tab[i].t2 == t2)) i = (i + 1) & (cap - 1);
    return &tab[i];
}
static cdouble cf_lcf(dfl_kcf *K, double t1, double t2)
{
    det_cf *F = K->ctx;
    if (!F->memo) return cf_raw(F, t1, t2);
    if (2 * (F->n + 1) > F->cap) {
        size_t nc = F->cap ? 2 * F->cap : 4096; cf_ent *nt = calloc(nc, sizeof(cf_ent));
        if (!nt) return cf_raw(F, t1, t2);
        for (size_t i = 0; i < F->cap; i++) if (F->tab[i].used) *cf_find(nt, nc, F->tab[i].t1, F->tab[i].t2) = F->tab[i];
        free(F->tab); F->tab = nt; F->cap = nc;
    }
    cf_ent *e = cf_find(F->tab, F->cap, t1, t2);
    if (!e->used) { e->used = 1; e->t1 = t1; e->t2 = t2; e->v = cf_raw(F, t1, t2); F->n++; }
    return e->v;
}
/* means and variances of (A, B) from the log characteristic function next to the origin (no jets for this transform; the
 * same at every call, so computed once) */
static void cf_scale(dfl_kcf *K, double *out)
{
    det_cf *F = K->ctx;
    if (!F->have_scale) {
        double m[2][2];
        for (int q = 0; q < 2; q++) {
            double a = q == 0 ? 1.0 : 0.0, b = 1 - a, h = 1e-2;
            cdouble L = cf_raw(F, a * h, b * h);
            while (cabs(L) > 0.05 && h > 1e-12) { h /= 4; L = cf_raw(F, a * h, b * h); }
            m[q][0] = cimag(L) / h; m[q][1] = -2 * creal(L) / (h * h);
        }
        F->scale[0] = m[0][0]; F->scale[1] = m[1][0]; F->scale[2] = sqrt(fmax(m[1][1], 0.0)); F->scale[3] = m[0][1];
        F->have_scale = 1;
    }
    memcpy(out, F->scale, 4 * sizeof(double));
}
/* the tail int_S^inf psi(u) e^{-i w u} du for a transform without jets, psi(u) = phi(u alpha, u beta) e^{-i b u} e^{i w u}/u
 * (cdf) or without the 1/u (the density): the derivatives of the slowly varying factor at S from the polynomial through 9
 * Chebyshev points of the real axis about S (the arguments stay imaginary), four terms of the expansion -> (the integral,
 * the first omitted term) (tail_int of DFKan_det.jl) */
static double fact_(int k) { return exp(lgamma(k + 1.0)); }
static void cf_tail(dfl_kcf *K, double alpha, double beta, double b, double w, double S, int cdf, cdouble *T, double *e)
{
    det_cf *F = K->ctx;
    enum { m = 9 };
    double L = S / 16, xk[m], V[m][m], gr[m], gi[m];
    for (int k = 0; k < m; k++) xk[k] = cos(M_PI * k / (m - 1));
    for (int k = 0; k < m; k++) {
        double u = S + L * xk[k];
        cdouble lc = cf_raw(F, alpha * u, beta * u);
        cdouble z = cadd(csub(lc, cscale(u, CPX(0, b))), cscale(u - S, CPX(0, w)));
        cdouble g = cexp(z);
        if (cdf) g = cdivr(g, u);
        gr[k] = creal(g); gi[k] = cimag(g);
        for (int j = 0; j < m; j++) V[k][j] = pow(xk[k], j);
    }
    /* the Vandermonde system by Gaussian elimination with partial pivoting, for the real and the imaginary parts */
    for (int col = 0; col < m; col++) {
        int pv = col; for (int i = col + 1; i < m; i++) if (fabs(V[i][col]) > fabs(V[pv][col])) pv = i;
        if (pv != col) { for (int j = 0; j < m; j++) { double t = V[col][j]; V[col][j] = V[pv][j]; V[pv][j] = t; } double t = gr[col]; gr[col] = gr[pv]; gr[pv] = t; t = gi[col]; gi[col] = gi[pv]; gi[pv] = t; }
        for (int i = col + 1; i < m; i++) {
            double l = V[i][col] / V[col][col];
            for (int j = col; j < m; j++) V[i][j] -= l * V[col][j];
            gr[i] -= l * gr[col]; gi[i] -= l * gi[col];
        }
    }
    cdouble cf[m];
    for (int i = m - 1; i >= 0; i--) {
        double sr = gr[i], si = gi[i];
        for (int j = i + 1; j < m; j++) { sr -= V[i][j] * creal(cf[j]); si -= V[i][j] * cimag(cf[j]); }
        cf[i] = CPX(sr / V[i][i], si / V[i][i]);
    }
    cdouble acc = DZERO, iw = cinv(CPX(0, w)), p = iw; enum { KT = 4 };
    for (int k = 0; k < KT; k++) { acc = cadd(acc, cmul(cdivr(cscale(fact_(k), cf[k]), pow(L, k)), p)); p = cmul(p, iw); }
    *T = acc; *e = cabs(cmul(cdivr(cscale(fact_(KT), cf[KT]), pow(L, KT)), p));
}
det_cf *det_cf_new(const det_model *Mo, double c, int memo)
{
    det_cf *F = calloc(1, sizeof(det_cf));
    if (!F) return NULL;
    int N = Mo->N, N0 = Mo->P->N;
    F->Mo = Mo; F->c = c; F->memo = memo;
    if (det_work_init(&F->W, Mo, NULL, 0)) { free(F); return NULL; }
    F->K = malloc((size_t) N * N * sizeof(cdouble)); F->Sl = malloc((size_t) N * N * sizeof(cdouble)); F->Sc = malloc((size_t) N0 * N0 * sizeof(cdouble));
    F->piv = malloc((N + 1) * sizeof(int));
    if (!F->K || !F->Sl || !F->Sc || !F->piv) { det_cf_free(F); return NULL; }
    F->k.c = c; F->k.w0 = Mo->a; F->k.lcf = cf_lcf; F->k.scale = cf_scale; F->k.tail = cf_tail; F->k.ctx = F;
    return F;
}
void det_cf_free(det_cf *F)
{
    if (!F) return;
    det_work_free(&F->W); free(F->K); free(F->Sl); free(F->Sc); free(F->piv); free(F->tab); free(F);
}
void det_cf_memo(det_cf *F, int on) { F->memo = on; if (!on) { free(F->tab); F->tab = NULL; F->cap = F->n = 0; } }
cdouble det_cf_logcf(det_cf *F, double t1, double t2) { return cf_raw(F, t1, t2); }
void det_cf_scale(det_cf *F, double *out) { cf_scale(&F->k, out); }
dfl_kcf *det_cf_kcf(det_cf *F) { return &F->k; }

/* ---------------------------------------------------------------------------------------------------------------- */
/* the density of alpha A + beta B at b: (1/pi) int_0^inf Re[phi(u alpha, u beta) e^{-i b u}] du over [0, S] by adaptive
 * quadrature over decades and beyond S = 400/|w| (w = w0 alpha + b) by the expansion of the tail (S doubled at most five
 * times while its first omitted term exceeds tol/10) (half_pdf of DFKan_det.jl) */
typedef struct { det_cf *F; double alpha, beta, b, sc; } hp_ctx;
static double hp_f(double z, void *v)
{
    hp_ctx *h = v; double u = z * h->sc;
    cdouble L = cf_raw(h->F, h->alpha * u, h->beta * u);
    double y = h->sc * exp(creal(L)) * cos(cimag(L) - h->b * u);
    return isfinite(y) ? y : 0.0;
}
static int half_pdf(det_cf *F, double alpha, double beta, double b, double tol, double w0, double *f, double *err)
{
    double sc4[4]; cf_scale(&F->k, sc4);
    hp_ctx h = {F, alpha, beta, b, dfl_kan_ray_scale(sc4, alpha, beta)};
    double w = w0 * alpha + b, T = 0, et = 0, S = INFINITY, Iv, e;
    if (fabs(w) > 1e-7) {
        S = 400 / fabs(w);
        for (int dbl = 0; dbl <= 5; dbl++) {
            cdouble Tc; cf_tail(&F->k, alpha, beta, b, w, S, 0, &Tc, &et); T = creal(Tc);
            if (et <= tol / 10 || dbl == 5) break;
            S *= 2;
        }
    }
    long nev = 0;
    if (dfl_kan_decades(hp_f, &h, S / h.sc, tol, fmax(tol, 1e-14), &Iv, &e, &nev)) return 1;
    *f = (Iv + T) / M_PI; *err = (e + et) / M_PI;
    return 0;
}
static double clamp01(double x) { return x < 0 ? 0.0 : x > 1 ? 1.0 : x; }
/* P(Q <= q) for the point-optimal statistic, Q = U + 1/2 + cbar^2 V = A + (cbar^2 + c) B + 1/2 (F the transform of its
 * quadratic form), by the single inversion: no grid, so that neither the distance of the singularity of the transform
 * from the real line (far above the unit root) nor the size of q limits it (pt_cdf1) */
int det_pt_cdf1(det_cf *F, double cbar, double q, double tol, double *P, double *e)
{
    if (isnan(q)) { *P = NAN; *e = 0; return 0; }
    if (isinf(q)) { *P = q > 0 ? 1.0 : 0.0; *e = 0; return 0; }
    if (dfl_kan_half_cdf(&F->k, 1.0, cbar * cbar + F->c, q - 0.5, tol, 0.5, P, e)) return 1;
    *P = clamp01(*P);
    return 0;
}
/* its density (pt_pdf1; the values are not kept along the ray) */
int det_pt_pdf1(det_cf *F, double cbar, double q, double tol, double *f, double *e)
{
    if (!isfinite(q)) { *f = isnan(q) ? NAN : 0.0; *e = 0; return 0; }
    int memo = F->memo; F->memo = 0;
    int rc = half_pdf(F, 1.0, cbar * cbar + F->c, q - 0.5, tol, 0.5, f, e);
    F->memo = memo;
    if (rc) return 1;
    *f = fmax(*f, 0.0);
    return 0;
}
/* the p-quantile: the Illinois variant of regula falsi on det_pt_cdf1 to the width 1e-12 times the scale of Q, from the
 * bracket E Q -/+ its standard deviation (grown as needed), with the transform values along the ray kept across the search
 * (pt_quantile1) */
typedef struct { det_cf *F; double cbar, p, tol, elast; } pq_ctx;
static double pq_g(double x, void *v)
{
    pq_ctx *h = v; double P, e;
    if (det_pt_cdf1(h->F, h->cbar, x, h->tol, &P, &e)) return NAN;
    h->elast = e;
    return P - h->p;
}
int det_pt_quantile1(det_cf *F, double cbar, double p, double tol, double *q, double *e, char *err, size_t errlen)
{
    int memo = F->memo; det_cf_memo(F, 1);
    double sc4[4]; cf_scale(&F->k, sc4);
    double bq = cbar * cbar + F->c, m = sc4[0] + bq * sc4[1] + 0.5, s = sqrt(sc4[3] + bq * bq * (sc4[2] * sc4[2]));
    pq_ctx h = {F, cbar, p, tol, 0};
    int rc = dfl_kan_illinois(pq_g, &h, fmax(m - s, 0.0), m + s, 1e-12 * s, s, q, err, errlen);
    det_cf_memo(F, memo);
    *e = h.elast;
    return rc;
}
/* P[(U - c0 V)/V <= y] = P[A + (c - c0 - y) B <= 0] for a design of the general engine by the single inversion (gcoef_cdf1) */
int det_gcoef_cdf1(det_cf *F, double y, double c0, double tol, double *P, double *e)
{
    if (isnan(y)) { *P = NAN; *e = 0; return 0; }
    if (isinf(y)) { *P = y > 0 ? 1.0 : 0.0; *e = 0; return 0; }
    if (dfl_kan_half_cdf(&F->k, 1.0, F->c - (c0 + y), 0.0, tol, F->k.w0, P, e)) return 1;
    *P = clamp01(*P);
    return 0;
}
