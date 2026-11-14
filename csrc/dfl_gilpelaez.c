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
            double lam = dfl_lam_gls(G->cbar), l2 = lam * lam, f = (1 - lam) * (1 - lam);
            cdouble s = CPX(0, u), t = cscale(-G->a, s);
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
            cdouble s1 = cdiv(ONE, cscale(f, csub(y[0], cscale(3, y[2]))));
            G->A[k] = cmul(cexp(Gs.lch), dN);
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

static int gp_check_opts(const dfl_gp_opts *o, char *err, size_t errlen)
{
    if (!(o->N >= 1 && o->umax > 0 && isfinite(o->umax))) { snprintf(err, errlen, "N must be a positive integer and umax a positive number"); return 1; }
    return 0;
}

typedef struct { const gp_t *G; const double *b; const int *id; double *out; } bjob;
static void b_body(int j, void *v) { bjob *x = v; x->out[x->id[j]] = gp_P(x->G, x->b[x->id[j]], false); }
int dfl_lincdf(int n, const double *b, double a, int model, double c, double cbar, const dfl_gp_opts *o, double *out, char *err, size_t errlen)
{
    if (model != 1 && model != 4) { snprintf(err, errlen, "the case must be DK (model 1) or DG (model 4)"); return 1; }
    if (!isfinite(c)) { snprintf(err, errlen, "c must be a finite real scalar"); return 1; }
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
    if (nf == 1) out[id[0]] = gp_P(&G, b[id[0]], true);
    else { bjob bj = {&G, b, id, out}; dfl_parfor(nf, b_body, &bj); }
    gp_free(&G); free(id);
    return 0;
}

/* the envelope at one c: the critical value of the test against cbar = c (a = c/2) at c = 0 by bisection on [-50, 50] to
 * 1e-9 (ltratio_bisect of DFExact.jl), then its power at c; 1: no sign change (flo, fhi), 2: out of memory */
static int envelope_one(double ck, int model, double alpha, const dfl_gp_opts *o, bool par, double *P, double *flo_out, double *fhi_out)
{
    if (isnan(ck)) { *P = NAN; return 0; }
    if (!(ck < 0)) { *P = alpha; return 0; }
    gp_t G;
    if (gp_init(&G, model, ck / 2, 0.0, ck, o)) { gp_free(&G); return 2; }
    gp_phi(&G, par);
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
    }
    free(rc); free(flo); free(fhi);
    return r;
}
