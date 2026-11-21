/* dfl_gilpelaez.c: the linear form U - aV of the limiting functionals with a zero start, cases DK and DG, by the inversion
 * formula of Gil-Pelaez, and the asymptotic power envelope of Elliott, Rothenberg and Stock (1996) (tr_gilpelaez,
 * ltratio_lincdf and power_envelope of julia/DFExact.jl; TR_GILPELAEZ, LTRATIO_LINCDF and POWER_ENVELOPE of the MATLAB
 * programs).  P(X < b) = 1/2 - (1/pi) int_0^inf Im[phi(u) e^{-iub}]/u du by the trapezoidal rule in u = sinh(v), N steps up
 * to umax, from u = 1e-7; the characteristic function phi from the joint moment generating function, its phase unwrapped
 * along the grid.  The grid is cut into blocks of a fixed size, so that the sums do not depend on the number of threads. */
#include "dflimit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GP_BLOCK 1024
static const cdouble ZERO = {0}, ONE = {1};

void dfl_gp_defaults(dfl_gp_opts *o) { o->N = 60000; o->umax = 4000.0; }

/* e^z - 1 without the cancellation near z = 0 (as expm1 of a complex number in Julia) */
static cdouble cexpm1(cdouble z)
{
    double x = creal(z), y = cimag(z), em = expm1(x), s = sin(0.5 * y);
    if (y == 0) return CPX(em, y);
    double er = em + 1;
    return CPX(em - 2 * er * s * s, er * sin(y));
}

typedef struct {
    int model, np, nblk;      /* model 1 (case DK) or 4 (case DG); np = N + 1 points in nblk blocks */
    bool vonly;               /* a = NaN: the distribution of V (case DK) */
    double a, c, cbar, h;
    double *u, *w, *p1, *p2;  /* the grid and its weights; the arguments to unwrap */
    cdouble *A, *B, *phi;     /* the pieces of phi along the grid, and phi */
} gp_t;

static int gp_init(gp_t *G, int model, double a, double c, double cbar, const dfl_gp_opts *o)
{
    memset(G, 0, sizeof *G);
    G->model = model; G->vonly = isnan(a); G->a = a; G->c = c; G->cbar = cbar;
    G->np = o->N + 1; G->nblk = (G->np + GP_BLOCK - 1) / GP_BLOCK; G->h = asinh(o->umax) / o->N;
    size_t n = (size_t) G->np;
    G->u = malloc(n * sizeof(double)); G->w = malloc(n * sizeof(double)); G->p1 = malloc(n * sizeof(double)); G->p2 = malloc(n * sizeof(double));
    G->A = malloc(n * sizeof(cdouble)); G->B = malloc(n * sizeof(cdouble)); G->phi = malloc(n * sizeof(cdouble));
    return !(G->u && G->w && G->p1 && G->p2 && G->A && G->B && G->phi);
}
static void gp_free(gp_t *G) { free(G->u); free(G->w); free(G->p1); free(G->p2); free(G->A); free(G->B); free(G->phi); }
static void gp_range(const gp_t *G, int j, int *k0, int *k1) { *k0 = j * GP_BLOCK; *k1 = *k0 + GP_BLOCK < G->np ? *k0 + GP_BLOCK : G->np; }
static void gp_run(int n, void (*body)(int, void *), void *ctx, bool par) { if (par) dfl_parfor(n, body, ctx); else for (int j = 0; j < n; j++) body(j, ctx); }

/* case DG at t: A = cosh th |I - Q Sig| and s1 (the operations of cf_gls; gp_pass1 and the conditional integral given R) */
static void dg_parts(double c, double cbar, cdouble t, cdouble *Aout, cdouble *s1out)
{
    double lam = dfl_lam_gls(cbar), l2 = lam * lam, f = (1 - lam) * (1 - lam);
    dfl_girs_t Gs; dfl_girsanov(4, c, t, &Gs);          /* (Sig and log cosh th: those of case T) */
    const cdouble *Sg = Gs.Sg;
    cdouble Q[9] = {cadd(CPX(c, 0), cscale(2 * l2 / 3, t)), ZERO, cscale(-2 * l2, t), ZERO, ZERO, ZERO, cscale(-2 * l2, t), ZERO, cscale(-6 * (1 - l2), t)};
    cdouble N[9], aN[9];
    for (int r = 0; r < 3; r++) for (int q = 0; q < 3; q++) {
        cdouble acc = ZERO;
        for (int m = 0; m < 3; m++) acc = cadd(acc, cmul(Q[3 * r + m], Sg[3 * m + q]));
        N[3 * r + q] = csub(r == q ? ONE : ZERO, acc);
    }
#define E_(i, jj) N[3 * (i) + (jj)]
    aN[0] = csub(cmul(E_(1,1), E_(2,2)), cmul(E_(1,2), E_(2,1))); aN[1] = cneg(csub(cmul(E_(0,1), E_(2,2)), cmul(E_(0,2), E_(2,1)))); aN[2] = csub(cmul(E_(0,1), E_(1,2)), cmul(E_(0,2), E_(1,1)));
    aN[3] = cneg(csub(cmul(E_(1,0), E_(2,2)), cmul(E_(1,2), E_(2,0)))); aN[4] = csub(cmul(E_(0,0), E_(2,2)), cmul(E_(0,2), E_(2,0))); aN[5] = cneg(csub(cmul(E_(0,0), E_(1,2)), cmul(E_(0,2), E_(1,0))));
    aN[6] = csub(cmul(E_(1,0), E_(2,1)), cmul(E_(1,1), E_(2,0))); aN[7] = cneg(csub(cmul(E_(0,0), E_(2,1)), cmul(E_(0,1), E_(2,0)))); aN[8] = csub(cmul(E_(0,0), E_(1,1)), cmul(E_(0,1), E_(1,0)));
    cdouble dN = cadd(cadd(cmul(E_(0,0), aN[0]), cmul(E_(0,1), aN[3])), cmul(E_(0,2), aN[6]));
#undef E_
    /* x = N^{-1} v = adj(N) v / det N, then v' Sig x */
    cdouble x[3], y[3];
    for (int r = 0; r < 3; r++) x[r] = cdiv(csub(aN[3 * r], cscale(3, aN[3 * r + 2])), dN);
    for (int r = 0; r < 3; r++) y[r] = cadd(cadd(cmul(Sg[3 * r], x[0]), cmul(Sg[3 * r + 1], x[1])), cmul(Sg[3 * r + 2], x[2]));
    *s1out = cdiv(ONE, cscale(f, csub(y[0], cscale(3, y[2]))));
    *Aout = cmul(cexp(Gs.lch), dN);
}

/* the pieces at the points of block j: the grid; case DK (cf_from of DFExact.jl): log D(s,t) = mu - log 2 + log br,
 * br = (1 + e^{-2mu}) - (c + s)(1 - e^{-2mu})/mu, mu^2 = c^2 - 2t, with (s, t) = (iu, -a iu) for U - aV and (0, iu) for V;
 * case DG (cf_gls): D = cosh th |I - Q Sig| and F = 1 - s/s_1 from the Girsanov forms of case T, s = iu, t = -a s, with the
 * GLS detrending at cbar, lam = (1 - cbar)/(1 - cbar + cbar^2/3), Q = diag(c, 0, 0) - 2t Gam, Gam = [-lam^2/3 0 lam^2;
 * 0 0 0; lam^2 0 3 (1 - lam^2)], s_1 = 1/((1 - lam)^2 v' Sig (I - Q Sig)^{-1} v), v = (1, 0, -3) */
static void gp_pass1(int j, void *v)
{
    gp_t *G = v; int k0, k1; gp_range(G, j, &k0, &k1);
    double c = G->c;
    for (int k = k0; k < k1; k++) {
        double vk = k * G->h, u = k == 0 ? 1e-7 : sinh(vk);
        G->u[k] = u; G->w[k] = k == 0 ? cosh(vk) * G->h / 2 : cosh(vk) * G->h;
        if (G->model == 1) {
            cdouble S = G->vonly ? ZERO : CPX(0, u), T = G->vonly ? CPX(0, u) : CPX(0, -G->a * u);
            cdouble mu = csqrt(CPX(c * c - 2 * creal(T), -2 * cimag(T))), E = cexp(cscale(-2, mu));
            cdouble q = cabs(mu) < 1e-6 ? csub(CPX(2, 0), cscale(2, mu)) : cdiv(cneg(cexpm1(cscale(-2, mu))), mu);
            G->A[k] = csub(mu, CPX(log(2.0), 0));
            G->B[k] = csub(cadd(ONE, E), cmul(cadd(CPX(c, 0), S), q));
            G->p1[k] = carg(G->B[k]);
        } else {
            cdouble s = CPX(0, u), t = cscale(-G->a, s), s1;
            dg_parts(c, G->cbar, t, &G->A[k], &s1);
            G->B[k] = csub(ONE, cdiv(s, s1));
            G->p1[k] = carg(G->A[k]); G->p2[k] = carg(G->B[k]);
        }
    }
}
/* phi at the points of block j, from the pieces and the unwrapped arguments */
static void gp_pass2(int j, void *v)
{
    gp_t *G = v; int k0, k1; gp_range(G, j, &k0, &k1);
    double c = G->c;
    for (int k = k0; k < k1; k++) {
        if (G->model == 1) {
            cdouble S = G->vonly ? ZERO : CPX(0, G->u[k]);
            G->phi[k] = cexp(csub(cscale(-0.5, cadd(S, CPX(c, 0))), cscale(0.5, cadd(G->A[k], CPX(log(cabs(G->B[k])), G->p1[k])))));
        } else {
            cdouble E = cexp(cscale(-0.5, CPX(c, G->u[k])));
            cdouble z = cmul(cscale(pow(cabs(G->A[k]), -0.5), E), cexp(CPX(0, -G->p1[k] / 2)));
            G->phi[k] = cmul(cscale(pow(cabs(G->B[k]), -0.5), z), cexp(CPX(0, -G->p2[k] / 2)));
        }
    }
}
static void unwrap(double *ph, int n) { for (int j = 1; j < n; j++) ph[j] -= 2 * M_PI * round((ph[j] - ph[j - 1]) / (2 * M_PI)); }
/* the characteristic function on the grid, at G->c */
static void gp_phi(gp_t *G, bool par)
{
    gp_run(G->nblk, gp_pass1, G, par);
    unwrap(G->p1, G->np); if (G->model == 4) unwrap(G->p2, G->np);
    gp_run(G->nblk, gp_pass2, G, par);
}

/* P(X < b): the sum of the terms of each block, then of the blocks in order */
typedef struct { const gp_t *G; double b, *part; } sumjob;
static double gp_block(const gp_t *G, double b, int j)
{
    int k0, k1; gp_range(G, j, &k0, &k1); double s = 0;
    for (int k = k0; k < k1; k++) { double ub = G->u[k] * b; s += (cimag(G->phi[k]) * cos(ub) - creal(G->phi[k]) * sin(ub)) / G->u[k] * G->w[k]; }
    return s;
}
static void sum_body(int j, void *v) { sumjob *s = v; s->part[j] = gp_block(s->G, s->b, j); }
static double gp_P(const gp_t *G, double b, bool par)
{
    double s = 0;
    if (par && G->nblk > 1) {
        double *part = malloc(G->nblk * sizeof(double));
        if (part) {
            sumjob sj = {G, b, part}; dfl_parfor(G->nblk, sum_body, &sj);
            for (int j = 0; j < G->nblk; j++) s += part[j];
            free(part); return 0.5 - s / M_PI;
        }
    }
    for (int j = 0; j < G->nblk; j++) s += gp_block(G, b, j);
    return 0.5 - s / M_PI;
}

/* ---- the linear form where the inversion of Gil-Pelaez is not accurate (gp_trusted, lincond, lin_exact of DFExact.jl) ----
 * The transform of U - aV decays like a power of u for a = 0 (R = U + 1/2 is a gamma(1/2) variable, whose density is
 * infinite at its edge) and like exp(-(|a| u)^{1/2}) only from u ~ 1/|a| on, so that the inversion truncated at
 * umax = 4000 is off by up to 7e-2 near the edge b = -1/2 and 1e-3 in the bulk for |a| up to about 0.3 (1e-4 at |a| = 0.3
 * for c = -50, where R and V are concentrated), and for c > 0 where U - aV is wide (up to 1e-2 in case DK at c = 5).
 * Where the transform at the end of the grid exceeds GPTAIL, or c > 0, the probability comes from a conditional integral
 * instead: in case DK the region {U <= aV + b} of the plane by the conditional integral given V (dfl_region), in case DG
 * the conditional integral given R (lincond). */
#define GPTAIL 5e-9                /* the error of the inversion is about 0.02 |phi(umax)| */
static bool gp_trusted(const gp_t *G)
{
    if (!(G->c <= 0)) return false;
    double m = 0; for (int k = G->np - 101 > 0 ? G->np - 101 : 0; k < G->np; k++) m = fmax(m, cabs(G->phi[k]));
    return m <= GPTAIL;
}

/* The conditional integral given R = U + 1/2.  With a zero start E[e^{sR + tV}] = B(t) (1 - s/s1(t))^{-1/2} in cases DK
 * and DG: R is a gamma(1/2) variable of rate kappa = s1(0), and the conditional moment generating function of V given
 * R = r is M(t | r) = B(t) (s1(t)/kappa)^{1/2} e^{-(s1(t) - kappa) r}.  With x = b + 1/2,
 *   a > 0:  P(U - aV <= b) = P(R <= x) + int_{r > x} f_R(r) P(V >= (r - x)/a | R = r) dr,
 *   a < 0:  P(U - aV <= b) = P(R <= x) - int_{r < x} f_R(r) P(V > (x - r)/|a| | R = r) dr     (0 for x <= 0),
 * P(R <= x) = erf(sqrt(kappa x)), exactly so at a = 0.  P(V > y | r) by Gil-Pelaez, the midpoint rule in w with the step
 * 2 pi/L, L = 2.2 ymax beyond the range ymax = E V + 42/t_* of y, up to where the transform has fallen below 1e-17 w; the
 * outer integral on panels graded from its lower end in y (the first E V/16 long, doubling) and at the scale of f_R
 * (rho = k/sqrt(kappa)), Gauss-Legendre in rho = sqrt(r) on each.  At most LCNMAX nodes in w (case DG for c up to about
 * 3.5; an error beyond). */
#define LCNMAX 2000000
#define LCTOL 1e-13                /* the outer integral's panels: |G24 - G12| at most this (absolute), else halved */
#define LCDEPTH 30
#define LCPMAX 4096
typedef struct { int model; double kap, ev, ymax, dw; int nw; double *w; cdouble *lB, *s1k; } lc_t;
/* log(B s1^{1/2}) and s1 at t = i w[0..n-1], the phases continued along w */
static void lc_parts(int model, double c, double cbar, const double *w, int n, cdouble *lB, cdouble *s1)
{
    double *ph = malloc(n * sizeof(double)), *ps = malloc(n * sizeof(double)), *lr = malloc(n * sizeof(double));
    for (int k = 0; k < n; k++) {
        if (model == 1) {
            cdouble mu = csqrt(CPX(c * c, -2 * w[k]));
            cdouble q = cabs(mu) < 1e-6 ? csub(CPX(2, 0), cscale(2, mu)) : cdiv(cneg(cexpm1(cscale(-2, mu))), mu);
            s1[k] = cdiv(csub(cadd(ONE, cexp(cscale(-2, mu))), cscale(c, q)), q);
            lr[k] = creal(mu) - log(2.0) + log(cabs(q)); ph[k] = cimag(mu) + carg(q);
        } else {
            cdouble A; dg_parts(c, cbar, CPX(0, w[k]), &A, &s1[k]);
            lr[k] = log(cabs(A)) - log(cabs(s1[k])); ph[k] = carg(A); ps[k] = carg(s1[k]);
        }
    }
    unwrap(ph, n); if (model == 4) unwrap(ps, n);
    for (int k = 0; k < n; k++)
        lB[k] = model == 1 ? CPX(-c / 2 - lr[k] / 2, -ph[k] / 2) : CPX(-c / 2 - lr[k] / 2, -ph[k] / 2 + ps[k] / 2);
    free(ph); free(ps); free(lr);
}
/* the abscissa t_* of the moment generating function of V in case DG detrended at cbar: the first zero of
 * cosh th |I - Q Sig| at real t > 0 (dfl_tstar(4, c) holds for cbar = DFL_CBAR_GLS only; tstar_gls of DFExact.jl) */
static double dg_g(double c, double cbar, double t) { cdouble A, s1; dg_parts(c, cbar, CPX(t, 0), &A, &s1); return creal(A); }
static double dg_tstar(double c, double cbar)
{
    double a = 1e-6, b;
    while (dg_g(c, cbar, a) <= 0 && a > 1e-300) a /= 1e3;
    b = a;
    while (dg_g(c, cbar, b) > 0) { a = b; b *= 1.05; if (b > fmax(1e4, 2 * c * c)) return NAN; }
    for (int it = 0; it < 100; it++) { double m = (a + b) / 2; if (dg_g(c, cbar, m) > 0) a = m; else b = m; if (b - a < 1e-15 * b) break; }
    return (a + b) / 2;
}
static void lc_free(lc_t *L) { free(L->w); free(L->lB); free(L->s1k); L->w = NULL; L->lB = L->s1k = NULL; }
static int lc_build(lc_t *L, int model, double c, double cbar, char *err, size_t errlen)
{
    memset(L, 0, sizeof *L); L->model = model;
    double w0 = 0, w1 = 1e-6; cdouble lb, s1;
    lc_parts(model, c, cbar, &w0, 1, &lb, &s1); L->kap = creal(s1);
    lc_parts(model, c, cbar, &w1, 1, &lb, &s1); L->ev = cimag(csub(lb, cscale(0.5, clog(s1)))) / 1e-6;   /* E V: log B(iw) ~ i w E V */
    double ts = model == 1 ? dfl_tstar(1, c, false) : dg_tstar(c, cbar);
    if (!(ts > 0)) { snprintf(err, errlen, "ltratio_lincdf: case %s, c = %g: no zero of the moment generating function of V was found (t_*)", model == 1 ? "DK" : "DG", c); return 1; }
    L->ymax = L->ev + 42 / ts; L->dw = 2 * M_PI / (2.2 * L->ymax);
    double W = 1;
    for (;;) {
        lc_parts(model, c, cbar, &W, 1, &lb, &s1);
        if (creal(lb) - log(L->kap) / 2 - log(W) < log(1e-17)) break;
        W *= 1.3;
        if (W / L->dw > LCNMAX) {
            snprintf(err, errlen, "ltratio_lincdf: case %s, c = %g: the conditional integral of the linear form needs more than %d nodes in its inner inversion (the law of V spreads like 1/t_* = %.3g as c grows); case DG is available for c up to about 3.5 at cbar = -13.5, less for cbar nearer 0",
                     model == 1 ? "DK" : "DG", c, LCNMAX, 1 / ts);
            return 1;
        }
    }
    L->nw = (int) ceil(W / L->dw);
    L->w = malloc(L->nw * sizeof(double)); L->lB = malloc(L->nw * sizeof(cdouble)); L->s1k = malloc(L->nw * sizeof(cdouble));
    if (!L->w || !L->lB || !L->s1k) { lc_free(L); snprintf(err, errlen, "out of memory"); return 1; }
    for (int k = 0; k < L->nw; k++) L->w[k] = (k + 0.5) * L->dw;
    lc_parts(model, c, cbar, L->w, L->nw, L->lB, L->s1k);
    for (int k = 0; k < L->nw; k++) { L->lB[k] = csub(L->lB[k], CPX(log(L->kap) / 2, 0)); L->s1k[k] = csub(L->s1k[k], CPX(L->kap, 0)); }
    return 0;
}
/* P(V > y | R = r) */
static double lc_sv(const lc_t *L, double y, double r)
{
    if (y <= 0) return 1.0;
    double s = 0;
    for (int k = 0; k < L->nw; k++) s += cimag(cexp(csub(csub(L->lB[k], cscale(r, L->s1k[k])), CPX(0, L->w[k] * y)))) / L->w[k];
    return 0.5 + s * L->dw / M_PI;
}
static int cmp_dbl(const void *p, const void *q) { double a = *(const double *) p, b = *(const double *) q; return a < b ? -1 : a > b; }
/* the outer integral of |a| f_R(x + a y) P(V > y | R = x + a y) over [y0, y1]; zlo, zhi: r = 0 at y0, at y1 (taken
 * exactly: the root of x + a y0 rounded off leaves r ~ 1e-18 there, and sqrt(r) ~ 1e-9 would cut off f_R(r) dr ~ 1e-9) */
typedef struct { const lc_t *L; double x, a; const double *rho; double *val; } ocjob;
static void oc_body(int j, void *v)
{
    ocjob *o = v; double rho = o->rho[j], r = rho * rho;
    o->val[j] = 2 * sqrt(o->L->kap / M_PI) * exp(-o->L->kap * r) * lc_sv(o->L, (r - o->x) / o->a, r);
}
static double lc_outer(const lc_t *L, double x, double a, double y0, double y1, bool zlo, bool zhi)
{
    if (y1 <= y0) return 0;
    double br[256]; int nb = 0; br[nb++] = y0; double h = L->ev / 16;
    while (br[nb - 1] < y1 && nb < 200) { br[nb] = fmin(y1, br[nb - 1] + h); nb++; h *= 2; }
    double rh[272]; int nr = 0;
    for (int j = 0; j < nb; j++) rh[nr++] = (j == 0 && zlo) ? 0.0 : (j == nb - 1 && zhi) ? 0.0 : sqrt(fmax(x + a * br[j], 0.0));
    double lo = rh[0], hi = rh[0]; for (int j = 1; j < nr; j++) { lo = fmin(lo, rh[j]); hi = fmax(hi, rh[j]); }
    const double ms[8] = {0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}; double sk = 1 / sqrt(L->kap);
    for (int m = 0; m < 8; m++) if (lo < ms[m] * sk && ms[m] * sk < hi) rh[nr++] = ms[m] * sk;
    qsort(rh, nr, sizeof(double), cmp_dbl);
    int nu = 0; for (int j = 0; j < nr; j++) if (j == 0 || rh[j] != rh[nu - 1]) rh[nu++] = rh[j];
    /* on each panel Gauss-Legendre with 24 points, checked against 12; a panel where the two differ by more than LCTOL is
     * halved (level by level, at most LCDEPTH times, at most LCPMAX panels a level): the conditional law of V given R = r
     * can turn within a fraction of the scale 1/kappa of r (case DG for c > 0 detrended at cbar near 0, where
     * P(V > y | R = r) rises from 0.03 to 1 between r = 0.06/kappa and 1/kappa), which fixed panels do not resolve */
    const double *g24, *u24 = dfl_rule(24, &g24), *g12, *u12 = dfl_rule(12, &g12);
    int np = nu - 1;
    double *pa = malloc((np + 1) * sizeof(double)), *pb = malloc((np + 1) * sizeof(double)), acc = 0;
    for (int j = 0; j < np; j++) { pa[j] = rh[j]; pb[j] = rh[j + 1]; }
    for (int depth = 0; depth <= LCDEPTH && np > 0; depth++) {
        int nn = 36 * np; double *rho = malloc(nn * sizeof(double)), *val = malloc(nn * sizeof(double));
        double *qa = malloc(2 * np * sizeof(double)), *qb = malloc(2 * np * sizeof(double)); int nq = 0;
        for (int j = 0; j < np; j++) {
            for (int k = 0; k < 24; k++) rho[36 * j + k] = pa[j] + (pb[j] - pa[j]) * u24[k];
            for (int k = 0; k < 12; k++) rho[36 * j + 24 + k] = pa[j] + (pb[j] - pa[j]) * u12[k];
        }
        ocjob oj = {L, x, a, rho, val};
        dfl_parfor(nn, oc_body, &oj);
        for (int j = 0; j < np; j++) {
            double s24 = 0, s12 = 0;
            for (int k = 0; k < 24; k++) s24 += g24[k] * val[36 * j + k];
            for (int k = 0; k < 12; k++) s12 += g12[k] * val[36 * j + 24 + k];
            s24 *= pb[j] - pa[j]; s12 *= pb[j] - pa[j];
            if (fabs(s24 - s12) <= LCTOL || depth == LCDEPTH || nq >= LCPMAX) acc += s24;
            else { double m = (pa[j] + pb[j]) / 2; qa[nq] = pa[j]; qb[nq] = m; nq++; qa[nq] = m; qb[nq] = pb[j]; nq++; }
        }
        free(rho); free(val); free(pa); free(pb);
        pa = qa; pb = qb; np = nq;
    }
    free(pa); free(pb);
    return acc;
}
/* P(U - aV <= x - 1/2) by the conditional integral given R, clamped to [0, 1] */
static double lc_x(const lc_t *L, double x, double a)
{
    double F0 = x > 0 ? erf(sqrt(L->kap * x)) : 0.0, Y = L->ymax, F;
    if (a == 0) return F0;
    if (a > 0) { double y0 = fmax(0.0, -x / a); F = y0 >= Y ? F0 : F0 + lc_outer(L, x, a, y0, Y, x <= 0, false); }
    else { double y1 = x / (-a); F = x <= 0 ? 0.0 : F0 - lc_outer(L, x, a, 0.0, fmin(Y, y1), false, y1 <= Y); }
    return fmin(1.0, fmax(0.0, F));
}
/* P(U - aV <= x - 1/2) where the inversion is not trusted: case DK by the region {U <= aV + b} (dfl_region, the line
 * g(v) = (x - 1/2) + a v), case DG by lc_x (L built for the case DG at c); bx is b (isb) or x = b + 1/2 (the envelope,
 * where x carries digits that b - 1/2 would lose) */
static int lin_exact(int model, double c, double a, double bx, bool isb, const lc_t *L, double *P, char *err, size_t errlen)
{
    if (model == 1) {
        double par[2] = {a, isb ? bx : bx - 0.5}; dfl_double_info info;
        if (dfl_region(1, par, 0, 0, NULL, c, NULL, P, &info, err, errlen)) return 1;
        *P = fmin(1.0, fmax(0.0, *P)); return 0;
    }
    *P = lc_x(L, isb ? bx + 0.5 : bx, a);
    return 0;
}

/* The law of V alone (a = NaN, case DK) where the grid does not serve: its transform decays only from u ~ 15 |c|^{3/2}
 * on for c < 0 (off by 8e-8 at c = -50, 4e-2 at c = -200 with umax = 4000), and for c > 2, where V spreads like 1/t_*,
 * the step must shrink like t_*^{1/2} (off by 1.2e-5 at c = 5).  There the grid goes on with the same step until |phi| <
 * 1e-16, the step h0 min(1, (t_* / t_*(2))^{1/2}), h0 = asinh(4000)/60000; at most VNMAX nodes (an error beyond: c up to
 * about 7).  Returns 0 (the default grid serves: c <= 2 and |phi| < 1e-9 at its end), 1 (N, umax set) or -1 (an error) */
#define VNMAX 4000000
static double absphiV(double c, double u)
{
    cdouble mu = csqrt(CPX(c * c, -2 * u)), E = cexp(cscale(-2, mu));
    cdouble q = cabs(mu) < 1e-6 ? csub(CPX(2, 0), cscale(2, mu)) : cdiv(cneg(cexpm1(cscale(-2, mu))), mu);
    cdouble br = csub(cadd(ONE, E), cscale(c, q));
    return exp(-c / 2 - 0.5 * (creal(mu) - log(2.0) + log(cabs(br))));
}
int dfl_vgrid(double c, double tail, int *N, double *umax, char *err, size_t errlen)
{
    if (c <= 2 && tail <= 1e-9) return 0;
    double h0 = asinh(4000.0) / 60000, h = c > 2 ? h0 * fmin(1.0, sqrt(dfl_tstar(1, c, false) / dfl_tstar(1, 2.0, false))) : h0;
    double U = 4000;
    while (absphiV(c, U) > 1e-16 && U < 1e9) U *= 1.5;
    double n = ceil(asinh(U) / h);
    if (n > VNMAX) { snprintf(err, errlen, "ltratio_lincdf: P(V <= b) at c = %g needs more than %d nodes in the inversion of Gil-Pelaez (case DK is available for c up to about 7)", c, VNMAX); return -1; }
    *N = (int) n; *umax = sinh(*N * h);
    return 1;
}

static int gp_check_opts(const dfl_gp_opts *o, char *err, size_t errlen)
{
    if (!(o->N >= 1 && o->umax > 0 && isfinite(o->umax))) { snprintf(err, errlen, "N must be a positive integer and umax a positive number"); return 1; }
    return 0;
}

/* (clamped to [0, 1]: the inversion's error of order 1e-14 leaves probabilities such as -1e-14 below the support) */
static double clamp01(double P) { return fmin(1.0, fmax(0.0, P)); }
typedef struct { const gp_t *G; const double *b; const int *id; double *out; } bjob;
static void b_body(int j, void *v) { bjob *x = v; x->out[x->id[j]] = clamp01(gp_P(x->G, x->b[x->id[j]], false)); }
static double gp_tail(const gp_t *G) { double m = 0; for (int k = G->np - 101 > 0 ? G->np - 101 : 0; k < G->np; k++) m = fmax(m, cabs(G->phi[k])); return m; }
int dfl_lincdf(int n, const double *b, double a, int model, double c, double cbar, const dfl_gp_opts *o, double *out, char *err, size_t errlen)
{
    if (model != 1 && model != 4) { snprintf(err, errlen, "the case must be DK (model 1) or DG (model 4)"); return 1; }
    if (!isfinite(c)) { snprintf(err, errlen, "c must be a finite real scalar"); return 1; }
    if (c < DFL_CMIN) { snprintf(err, errlen, "ltratio_lincdf: %s", DFL_CMIN_MSG); return 1; }
    if (model == 4 && isnan(a)) { snprintf(err, errlen, "P(V < b) is available in case DK only"); return 1; }
    dfl_gp_opts d; if (o == NULL) { dfl_gp_defaults(&d); o = &d; }
    if (gp_check_opts(o, err, errlen)) return 1;
    int nf = 0; int *id = malloc((n + 1) * sizeof(int));
    if (!id) { snprintf(err, errlen, "out of memory"); return 1; }
    for (int k = 0; k < n; k++) { if (isfinite(b[k])) id[nf++] = k; else out[k] = isnan(b[k]) ? NAN : b[k] > 0 ? 1.0 : 0.0; }
    if (nf == 0) { free(id); return 0; }
    gp_t G;
    if (gp_init(&G, model, a, c, cbar, o)) { gp_free(&G); free(id); snprintf(err, errlen, "out of memory"); return 1; }
    gp_phi(&G, true);
    if (isnan(a)) {
        /* the law of V: a longer or finer grid where the default does not serve (dfl_vgrid) */
        dfl_gp_opts ov; int r = dfl_vgrid(c, gp_tail(&G), &ov.N, &ov.umax, err, errlen);
        if (r < 0) { gp_free(&G); free(id); return 1; }
        if (r > 0) {
            gp_free(&G);
            if (gp_init(&G, model, a, c, cbar, &ov)) { gp_free(&G); free(id); snprintf(err, errlen, "out of memory"); return 1; }
            gp_phi(&G, true);
        }
    } else if (!gp_trusted(&G)) {
        /* the inversion is not trusted (a transform that has not decayed by umax, or c > 0): a conditional integral */
        gp_free(&G);
        lc_t L; memset(&L, 0, sizeof L);
        if (model == 4 && lc_build(&L, 4, c, cbar, err, errlen)) { free(id); return 1; }
        for (int j = 0; j < nf; j++)
            if (lin_exact(model, c, a, b[id[j]], true, &L, &out[id[j]], err, errlen)) { lc_free(&L); free(id); return 1; }
        lc_free(&L); free(id);
        return 0;
    }
    if (nf == 1) out[id[0]] = clamp01(gp_P(&G, b[id[0]], true));
    else { bjob bj = {&G, b, id, out}; dfl_parfor(nf, b_body, &bj); }
    gp_free(&G); free(id);
    return 0;
}

/* the envelope at one c: the critical value of the test against cbar = c, which rejects for small U - aV, at c = 0 by
 * bisection on [-50, 50] to 1e-9 (ltratio_bisect of DFExact.jl), then its power at c; the weight a = c/2 in case DK (log
 * likelihood ratio c U - c^2 V/2) and a = c^2/(2 (c - 1)) in case DG (the invariant statistic c^2 V + (1 - c) J(1)^2 of
 * the data GLS-detrended at c, J(1)^2 = 2U + 1; po_weight of DFExact.jl); 1: no sign change (flo, fhi), 2: out of memory */
static int envelope_one(double ck, int model, double alpha, const dfl_gp_opts *o, bool par, double *P, double *flo_out, double *fhi_out)
{
    if (isnan(ck)) { *P = NAN; return 0; }
    if (!(ck < 0)) { *P = alpha; return 0; }
    double a = model == 4 ? ck * ck / (2 * (ck - 1)) : ck / 2;
    gp_t G;
    if (gp_init(&G, model, a, 0.0, ck, o)) { gp_free(&G); return 2; }
    gp_phi(&G, par);
    if (!gp_trusted(&G)) {
        /* near c = 0, where a is small and the inversion is not trusted: the conditional integral, its critical value by
         * bisection in x = b + 1/2 to 1e-12 of the scale s = E(U + 1/2) + |a| E V of U + 1/2 - aV (in case DG detrended at
         * cbar = c the law of U + 1/2 shrinks like c^4, so that a tolerance in b of 1e-9 does not resolve it) */
        gp_free(&G);
        lc_t L0, L1; memset(&L0, 0, sizeof L0); memset(&L1, 0, sizeof L1);
        char e2[512];
        if (model == 4 && (lc_build(&L0, 4, 0.0, ck, e2, sizeof e2) || lc_build(&L1, 4, ck, ck, e2, sizeof e2))) { lc_free(&L0); lc_free(&L1); return 3; }
        double kap = model == 1 ? 1.0 : L0.kap, ev = model == 1 ? 0.5 : L0.ev, sc = 1 / (2 * kap) + fabs(a) * ev, F;
        double lo = a < 0 ? 0.0 : -sc, hi = sc, flo, fhi;
#define FX(xx, out) do { if (lin_exact(model, 0.0, a, (xx), false, &L0, &F, e2, sizeof e2)) { lc_free(&L0); lc_free(&L1); return 3; } (out) = F - alpha; } while (0)
        FX(hi, fhi); for (int it = 0; it < 200 && fhi < 0; it++) { hi *= 2; FX(hi, fhi); }
        FX(lo, flo); for (int it = 0; it < 200 && a >= 0 && flo > 0; it++) { lo *= 2; FX(lo, flo); }
        if (!(flo * fhi < 0)) { *flo_out = flo; *fhi_out = fhi; lc_free(&L0); lc_free(&L1); return 1; }
        for (int it = 0; it < 200; it++) {
            double m = (lo + hi) / 2, fm; FX(m, fm);
            if (fm * flo < 0) { hi = m; fhi = fm; } else { lo = m; flo = fm; }
            if (hi - lo < 1e-12 * sc) break;
        }
#undef FX
        int r = lin_exact(model, ck, a, (lo + hi) / 2, false, &L1, P, e2, sizeof e2);
        lc_free(&L0); lc_free(&L1);
        return r ? 3 : 0;
    }
    double lo = -50, hi = 50, flo = gp_P(&G, lo, par) - alpha, fhi = gp_P(&G, hi, par) - alpha;
    if (!(flo * fhi < 0)) { *flo_out = flo; *fhi_out = fhi; gp_free(&G); return 1; }
    for (int it = 0; it < 200; it++) {
        double m = (lo + hi) / 2, fm = gp_P(&G, m, par) - alpha;
        if (fm * flo < 0) { hi = m; fhi = fm; } else { lo = m; flo = fm; }
        if (hi - lo < 1e-9) break;
    }
    double bc = (lo + hi) / 2;
    G.c = ck; gp_phi(&G, par);
    *P = gp_P(&G, bc, par);
    gp_free(&G);
    return 0;
}
typedef struct { const double *c; int model; double alpha; const dfl_gp_opts *o; double *out, *flo, *fhi; int *rc; } ejob;
static void e_body(int j, void *v) { ejob *x = v; x->rc[j] = envelope_one(x->c[j], x->model, x->alpha, x->o, false, &x->out[j], &x->flo[j], &x->fhi[j]); }
int dfl_power_envelope(int n, const double *c, int model, double alpha, const dfl_gp_opts *o, double *out, char *err, size_t errlen)
{
    if (model != 1 && model != 4) { snprintf(err, errlen, "the case must be DK (or DU; model 1) or DG (model 4)"); return 1; }
    dfl_gp_opts d; if (o == NULL) { dfl_gp_defaults(&d); o = &d; }
    if (gp_check_opts(o, err, errlen)) return 1;
    if (n <= 0) return 0;
    int *rc = malloc(n * sizeof(int)); double *flo = malloc(n * sizeof(double)), *fhi = malloc(n * sizeof(double));
    if (!rc || !flo || !fhi) { free(rc); free(flo); free(fhi); snprintf(err, errlen, "out of memory"); return 1; }
    if (n == 1) rc[0] = envelope_one(c[0], model, alpha, o, true, &out[0], &flo[0], &fhi[0]);
    else { ejob ej = {c, model, alpha, o, out, flo, fhi, rc}; dfl_parfor(n, e_body, &ej); }
    int r = 0;
    for (int k = 0; k < n; k++) {
        if (rc[k] == 1) { snprintf(err, errlen, "no sign change on [-50, 50] (%g, %g) in the bisection for the critical value at c = %g", flo[k], fhi[k], c[k]); r = 1; break; }
        if (rc[k] == 2) { snprintf(err, errlen, "out of memory"); r = 1; break; }
        if (rc[k] == 3) { snprintf(err, errlen, "the conditional integral of the linear form failed at c = %g", c[k]); r = 1; break; }
    }
    free(rc); free(flo); free(fhi);
    return r;
}
