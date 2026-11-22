/* dfl_batch.c: the fixed node layouts of the line integral's inner integrals (the batched evaluation of outer / density /
 * cdf of DFExact.jl).  With node layouts that do not depend on t, every piece of the inner transforms is a sum over nodes of
 * a cached weight C_k times one complex exponential exp(A_k t + B_k s(t)) (and, for the K_0 kernel, K_0(b X_k)): the
 * Gaussian factor and the exponential of the kernel combine, and the algebraic factors, the partition and the Jacobians are
 * cached.  The layouts are geometric panels (ratio 4) that resolve every scale that occurs along the line, with
 * Gauss-Legendre nodes per panel (32 for the chi-square kernel, 48 for the K_0 kernel's bump and edge segment, 32 for its
 * rays); nodes whose exponent is below -45 are skipped (the nodes are ordered from the dominant end).  The exponentials of
 * all nodes of a point are taken at once (dfl_vmath.h).  The layouts serve c <= 0 and |q| above about 0.034 (usebatch: for
 * smaller |q| the bulge of the segment path lets the Gaussian factor grow and the segment integral cancels; the adaptive
 * rules then take the ray from X = 1/2), the chi-square kernel (no deterministic terms, zero start) at tau = 1, the K_0
 * kernel's density at any tau and its distribution function at tau = 1 (its constant part, M_0 less the ray over X > 1/2
 * on a fixed layout, assumes |s_2| >= 2.5 at the real point).  For c < -20 the first panels of the edge segment and of
 * the ray X < 0 are at most 1/b and 1e-2/|s_1| long (b = (s_2 - s_1)/2 and s_1 at tau: dfl_k0first), which agrees with
 * the general kernel and with layouts of twice the nodes to 1e-11 on [-480, -20] (DFExact.jl). */
#include "dflimit.h"
#include "dfl_lock.h"
#include "dfl_vmath.h"
#include <stdlib.h>
#include <string.h>

#define NPAN 32          /* nodes per panel of the chi-square layouts */
#define NPANK 48         /* ... of the K_0 layouts (bump and edge segment) */
#define NPANR 32         /* ... of the K_0 rays (X < 0 and the constant part's ray over X > 1/2) */
#define RATIO 4.0        /* the growth factor of the geometric panels */
#define CUTOFF 45.0      /* nodes whose exponent is below -CUTOFF are skipped (e^{-45} = 3e-20) */
#define EULERGAMMA 0.5772156649015329
int DFL_BATCHK0 = 1;     /* the K_0 layouts (1); 0 leaves the K_0 kernel to the adaptive rules */

typedef struct { int n; cdouble *A, *B, *C, *X; int neg; } layout_t;   /* X: K_0 at b X_k (NULL for the chi-square kernel) */
struct dfl_batch {
    int kern, dens; double q, tau, e1, r1;
    layout_t L[6]; int nl;      /* kern 1: smooth, edge; kern 2: smooth, edge segment, ray X < 0, constant part's ray */
    int is, ie0, ie1, ic;       /* the index ranges of the parts: smooth [is], edge [ie0, ie1), constant part [ic] (-1: none) */
    int nmax, refs;
};

static const cdouble DZERO = {0}, DONE = {1};
static inline cdouble caddr(cdouble a, double b) { return CPX(creal(a) + b, cimag(a)); }
static cdouble E0v(void) { return CPX(cos(-M_PI / 8), sin(-M_PI / 8)); }   /* the fixed direction of the rays from X = 1/2 */
static cdouble E1v(void) { return CPX(cos(M_PI / 8), sin(M_PI / 8)); }     /* ... of the constant part's ray (Re(s_2 e1) > 0, c <= 0) */
static double capf(double x) { return x / sqrt(1 + x * x); }

/* ---------- the panels and the layouts ---------- */
typedef struct { double *x, *w; int n, cap; } pts_t;
static void pts_push(pts_t *p, double x, double w)
{
    if (p->n == p->cap) { p->cap = p->cap ? 2 * p->cap : 256; p->x = realloc(p->x, p->cap * sizeof(double)); p->w = realloc(p->w, p->cap * sizeof(double)); }
    p->x[p->n] = x; p->w[p->n] = w; p->n++;
}
/* nodes and weights of geometric panels on [0, L]: the first panel [0, a1] with the substitution x = a1 u^p, then panels
 * of ratio RATIO up to L, n nodes each (geopanels) */
static pts_t geopanels(double a1, double L, int n, int p)
{
    pts_t P = {0};
    const double *gw; const double *u = dfl_rule(n, &gw);
    for (int k = 0; k < n; k++) {
        if (p == 1) pts_push(&P, a1 * u[k], a1 * gw[k]);
        else pts_push(&P, a1 * pow(u[k], p), a1 * p * pow(u[k], p - 1) * gw[k]);
    }
    double a = a1;
    while (a < L) {
        double b = fmin(L, RATIO * a);
        for (int k = 0; k < n; k++) pts_push(&P, a + (b - a) * u[k], (b - a) * gw[k]);
        a = b;
    }
    return P;
}
static void pts_free(pts_t *p) { free(p->x); free(p->w); }
static void lay_init(layout_t *L, int n, int withX, int neg)
{
    L->n = 0; L->neg = neg;
    L->A = malloc((n + 1) * sizeof(cdouble)); L->B = malloc((n + 1) * sizeof(cdouble)); L->C = malloc((n + 1) * sizeof(cdouble));
    L->X = withX ? malloc((n + 1) * sizeof(cdouble)) : NULL;
}
static void lay_push(layout_t *L, cdouble A, cdouble B, cdouble C, cdouble X)
{
    L->A[L->n] = A; L->B[L->n] = B; L->C[L->n] = C; if (L->X) L->X[L->n] = X; L->n++;
}
static void lay_free(layout_t *L) { free(L->A); free(L->B); free(L->C); free(L->X); }

/* the bulged segment of the distribution function, X = Xr + i g(Xr) on (0, 1/2), and its partition of unity */
typedef struct { double kap, H, q; } seg_t;
static void seg_path(const seg_t *S, double Xr, cdouble *X, cdouble *dX)
{
    double H = S->H, a = capf(Xr / H), b = capf((0.5 - Xr) / H), g = H * a * b;
    double dg = b / pow(sqrt(1 + (Xr / H) * (Xr / H)), 3) - a / pow(sqrt(1 + ((0.5 - Xr) / H) * ((0.5 - Xr) / H)), 3);
    *X = CPX(Xr, g); *dX = CPX(1, dg);
}
static cdouble seg_chi(const seg_t *S, double Xr)
{
    return cscale(0.5, cadd(CPX(1, 0), dfl_erf(cscale(S->kap, CPX(0.25 - Xr, -S->H * capf(Xr / S->H) * capf((0.5 - Xr) / S->H))))));
}
/* the bulged segment of the density in v, parametrized by d = v* - r, and its partition */
typedef struct { double vs, kap, H, aq; } segd_t;
static void segd_path(const segd_t *S, double d, cdouble *v, cdouble *dv, cdouble *X)
{
    double H = S->H, r = S->vs - d, a = capf(r / H), b = capf(d / H), g = H * a * b;
    double dg = b / pow(sqrt(1 + (r / H) * (r / H)), 3) - a / pow(sqrt(1 + (d / H) * (d / H)), 3);
    *v = CPX(r, -g); *dv = CPX(1, -dg); *X = CPX(S->aq * d, S->aq * g);
}
static cdouble segd_chi(const segd_t *S, double d)
{
    return cscale(0.5, cadd(CPX(1, 0), dfl_erf(cscale(S->kap, CPX(S->vs / 2 - d, -S->H * capf((S->vs - d) / S->H) * capf(d / S->H))))));
}

/* the chi-square kernel (layouts_m1): the smooth part and the edge part of the transforms at q */
static void layouts_m1(double q, int dens, layout_t *Ls, layout_t *Le)
{
    double aq = fabs(q); cdouble e0 = E0v();
    Le->n = 0; Le->A = Le->B = Le->C = Le->X = NULL;
    if (q > 0) {
        pts_t P = geopanels(0.001 * aq, aq * sqrt(85.0), NPAN, 1);      /* the Gaussian is spent beyond r = |q| sqrt(85/tau), tau = 1 */
        lay_init(Ls, P.n, 0, 0);
        for (int k = 0; k < P.n; k++) {
            double r = P.x[k], w = P.w[k];
            if (dens) {
                cdouble v = cscale(r / aq, e0), X = caddr(cscale(q, v), 0.5);
                lay_push(Ls, cneg(cmul(v, v)), cneg(X), cmul(cdiv(cscale(2, cmul(v, v)), csqrt(cscale(M_PI, X))), cscale(w / aq, e0)), DZERO);
            } else {
                cdouble re = cscale(r, e0), X = caddr(re, 0.5);
                lay_push(Ls, cscale(-1 / (q * q), cmul(re, re)), cneg(X), cmul(cdiv(DONE, csqrt(cscale(M_PI, X))), cscale(w, e0)), DZERO);
            }
        }
        pts_free(&P); return;
    }
    if (!dens) {
        seg_t S = {DFL_KAPPA / 0.25, 1.5 / (DFL_KAPPA / 0.25), q};
        pts_t P = geopanels(0.001 * aq, 0.5, NPAN, 1);                    /* smooth: y = 1/2 - Xr from 0, weight 1 - chi */
        lay_init(Ls, P.n, 0, 0);
        for (int k = 0; k < P.n; k++) {
            cdouble X, dX; seg_path(&S, 0.5 - P.x[k], &X, &dX); cdouble x = caddr(X, -0.5);
            lay_push(Ls, cscale(-1 / (q * q), cmul(x, x)), cneg(X), cscale(-P.w[k], cmul(cdiv(dX, csqrt(cscale(M_PI, X))), csub(DONE, seg_chi(&S, 0.5 - P.x[k])))), DZERO);
        }
        pts_free(&P);
        P = geopanels(0.001 * aq, 0.5, NPAN, 4);                          /* edge: Xr from 0 with the X^{-1/2} singularity, weight chi */
        lay_init(Le, P.n, 0, 0);
        for (int k = 0; k < P.n; k++) {
            cdouble X, dX; seg_path(&S, P.x[k], &X, &dX); cdouble x = caddr(X, -0.5);
            lay_push(Le, cscale(-1 / (q * q), cmul(x, x)), cneg(X), cscale(-P.w[k], cmul(cdiv(dX, csqrt(cscale(M_PI, X))), seg_chi(&S, P.x[k]))), DZERO);
        }
        pts_free(&P); return;
    }
    double vs = 1 / (2 * aq), kap = DFL_KAPPA / (vs / 2);
    segd_t S = {vs, kap, 1.5 / kap, aq};
    pts_t P = geopanels(0.001, vs, NPAN, 1);                              /* smooth: r = vs - d from the bump at v = 0 */
    lay_init(Ls, P.n, 0, 0);
    for (int k = 0; k < P.n; k++) {
        cdouble v, dv, X; segd_path(&S, vs - P.x[k], &v, &dv, &X);
        cdouble C = cmul(cdiv(cscale(2, cmul(v, v)), csqrt(cscale(M_PI, X))), cmul(dv, cscale(P.w[k], csub(DONE, segd_chi(&S, vs - P.x[k])))));
        lay_push(Ls, cneg(cmul(v, v)), cneg(X), C, DZERO);
    }
    pts_free(&P);
    P = geopanels(0.001, vs, NPAN, 4);                                    /* edge: d from the support edge */
    lay_init(Le, P.n, 0, 0);
    for (int k = 0; k < P.n; k++) {
        cdouble v, dv, X; segd_path(&S, P.x[k], &v, &dv, &X);
        cdouble C = cmul(cdiv(cscale(2, cmul(v, v)), csqrt(cscale(M_PI, X))), cmul(dv, cscale(P.w[k], segd_chi(&S, P.x[k]))));
        lay_push(Le, cneg(cmul(v, v)), cneg(X), C, DZERO);
    }
    pts_free(&P);
}

/* the K_0 kernel (layouts_k0) at q and tau: the smooth part (L[0]), the edge part (the segment near X = 0, L[1], and the ray
 * X < 0, L[2]) and the ray X > 1/2 of the constant part (L[3], distribution function only); c <= 0.  The first panels of the
 * edge segment and of the ray X < 0 are at most (e1, r1) long in X (dfl_k0first) */
static void layouts_k0(double q, int dens, double tau, double e1, double r1, struct dfl_batch *B)
{
    double aq = fabs(q); cdouble e0 = E0v(), e1v = E1v();
    layout_t *Ls = &B->L[0], *Le = &B->L[1], *Ln = &B->L[2], *Lc = &B->L[3];
    memset(B->L, 0, sizeof B->L);
    B->nl = 4; B->is = 0; B->ie0 = 1; B->ie1 = 3; B->ic = dens ? -1 : 3;
    if (!dens) {
        pts_t P = geopanels(0.002, 36.0 / 2.5, NPANR, 1);                 /* |s_2| >= 2.5 at the real point for c <= 0 */
        lay_init(Lc, P.n, 1, 0);
        for (int k = 0; k < P.n; k++) { cdouble X = caddr(cscale(P.x[k], e1v), 0.5); lay_push(Lc, DZERO, cneg(X), cscale(P.w[k] / M_PI, e1v), X); }
        pts_free(&P);
    }
    if (q > 0) {
        pts_t P = geopanels(0.001 * aq, aq * sqrt(85.0 / tau), NPANK, 1);
        lay_init(Ls, P.n, 1, 0);
        for (int k = 0; k < P.n; k++) {
            double r = P.x[k], w = P.w[k];
            if (dens) {
                cdouble v = cscale(r / aq, e0), X = caddr(cscale(q, v), 0.5);
                lay_push(Ls, cneg(cmul(v, v)), cneg(X), cmul(cscale(2 * w / (aq * M_PI), cmul(v, v)), e0), X);
            } else {
                cdouble re = cscale(r, e0), X = caddr(re, 0.5);
                lay_push(Ls, cscale(-1 / (q * q), cmul(re, re)), cneg(X), cscale(w / M_PI, e0), X);
            }
        }
        pts_free(&P);
        B->ie0 = B->ie1 = 1;           /* (no edge part) */
        return;
    }
    if (!dens) {
        seg_t S = {DFL_KAPPA / 0.25, 1.5 / (DFL_KAPPA / 0.25), q};
        pts_t P = geopanels(0.001 * aq, 0.5, NPANK, 1);
        lay_init(Ls, P.n, 1, 0);
        for (int k = 0; k < P.n; k++) {
            cdouble X, dX; seg_path(&S, 0.5 - P.x[k], &X, &dX); cdouble x = caddr(X, -0.5);
            lay_push(Ls, cscale(-1 / (q * q), cmul(x, x)), cneg(X), cscale(-P.w[k] / M_PI, cmul(dX, csub(DONE, seg_chi(&S, 0.5 - P.x[k])))), X);
        }
        pts_free(&P);
        P = geopanels(fmin(0.001 * aq, e1), 0.5, NPANK, 4);
        lay_init(Le, P.n, 1, 0);
        for (int k = 0; k < P.n; k++) {
            cdouble X, dX; seg_path(&S, P.x[k], &X, &dX); cdouble x = caddr(X, -0.5);
            lay_push(Le, cscale(-1 / (q * q), cmul(x, x)), cneg(X), cscale(-P.w[k] / M_PI, cmul(dX, seg_chi(&S, P.x[k]))), X);
        }
        pts_free(&P);
        /* the ray X = -r e0, X < 0: -phi_neg(X) e^{-t x^2/q^2} e0, x = X - 1/2 */
        P = geopanels(fmin(1e-4 * q * q, r1), aq * sqrt(85.0 / tau), NPANR, 4);
        lay_init(Ln, P.n, 1, 1);
        for (int k = 0; k < P.n; k++) {
            cdouble X = cscale(-P.x[k], e0), x = caddr(X, -0.5);
            lay_push(Ln, cscale(-1 / (q * q), cmul(x, x)), cneg(X), cscale(-P.w[k] / M_PI, e0), X);
        }
        pts_free(&P);
        return;
    }
    double vs = 1 / (2 * aq), kap = DFL_KAPPA / (vs / 2);
    segd_t S = {vs, kap, 1.5 / kap, aq};
    pts_t P = geopanels(0.001, vs, NPANK, 1);
    lay_init(Ls, P.n, 1, 0);
    for (int k = 0; k < P.n; k++) {
        cdouble v, dv, X; segd_path(&S, vs - P.x[k], &v, &dv, &X);
        lay_push(Ls, cneg(cmul(v, v)), cneg(X), cmul(cscale(2 * P.w[k] / M_PI, cmul(v, v)), cmul(dv, csub(DONE, segd_chi(&S, vs - P.x[k])))), X);
    }
    pts_free(&P);
    P = geopanels(fmin(0.001, e1 / aq), vs, NPANK, 4);
    lay_init(Le, P.n, 1, 0);
    for (int k = 0; k < P.n; k++) {
        cdouble v, dv, X; segd_path(&S, P.x[k], &v, &dv, &X);
        lay_push(Le, cneg(cmul(v, v)), cneg(X), cmul(cscale(2 * P.w[k] / M_PI, cmul(v, v)), cmul(dv, segd_chi(&S, P.x[k]))), X);
    }
    pts_free(&P);
    /* the ray beyond v*: v = vs + r e0, X = -|q| r e0: 2 v^2 e^{-t v^2} phi_neg(X) e0 */
    P = geopanels(fmin(1e-4, r1 / aq), sqrt(85.0 / tau), NPANR, 4);
    lay_init(Ln, P.n, 1, 1);
    for (int k = 0; k < P.n; k++) {
        cdouble v = caddr(cscale(P.x[k], e0), vs), X = cscale(-aq * P.x[k], e0);
        lay_push(Ln, cneg(cmul(v, v)), cneg(X), cmul(cscale(2 * P.w[k] / M_PI, cmul(v, v)), e0), X);
    }
    pts_free(&P);
}

/* ---------- the cache of the layouts (at most 64, emptied when full; references held while in use) ---------- */
#define BCACHE_MAX 64
static struct dfl_batch *BC[BCACHE_MAX]; static int NBC = 0;
static dfl_lock_t BLOCK = DFL_LOCK_INIT;
static void batch_free(struct dfl_batch *B) { for (int i = 0; i < B->nl; i++) lay_free(&B->L[i]); free(B); }
static void batch_unref_locked(struct dfl_batch *B) { if (--B->refs == 0) batch_free(B); }
void dfl_batch_release(const dfl_batch *B) { if (!B) return; dfl_lock(&BLOCK); batch_unref_locked((struct dfl_batch *) B); dfl_unlock(&BLOCK); }
void dfl_batch_empty_cache(void) { dfl_lock(&BLOCK); for (int i = 0; i < NBC; i++) batch_unref_locked(BC[i]); NBC = 0; dfl_unlock(&BLOCK); }
static int same(const struct dfl_batch *B, int kern, int dens, double q, double tau, double e1, double r1)
{
    return B->kern == kern && B->dens == dens && B->q == q && (kern == 1 || (B->tau == tau && B->e1 == e1 && B->r1 == r1));
}
/* the layouts of the kernel kern (1 chi-square, 2 K_0) for the density (kind 1) or the distribution function (kind 2) at q
 * (and tau, e1, r1 for the K_0 kernel): a reference (dfl_batch_release) */
const dfl_batch *dfl_batch_get(int kern, int kind, double q, double tau, double e1, double r1)
{
    int dens = kind == 1;
    dfl_lock(&BLOCK);
    for (int i = 0; i < NBC; i++) if (same(BC[i], kern, dens, q, tau, e1, r1)) { BC[i]->refs++; dfl_unlock(&BLOCK); return BC[i]; }
    dfl_unlock(&BLOCK);
    struct dfl_batch *B = calloc(1, sizeof *B);
    B->kern = kern; B->dens = dens; B->q = q; B->tau = tau; B->e1 = e1; B->r1 = r1;
    if (kern == 1) { layouts_m1(q, dens, &B->L[0], &B->L[1]); B->nl = 2; B->is = 0; B->ie0 = 1; B->ie1 = 2; B->ic = -1; }
    else layouts_k0(q, dens, tau, e1, r1, B);
    for (int i = 0; i < B->nl; i++) if (B->L[i].n > B->nmax) B->nmax = B->L[i].n;
    B->refs = 2;     /* the caller's and the cache's */
    dfl_lock(&BLOCK);
    if (NBC == BCACHE_MAX) { for (int i = 0; i < NBC; i++) batch_unref_locked(BC[i]); NBC = 0; }
    BC[NBC++] = B;
    dfl_unlock(&BLOCK);
    return B;
}
/* the rule: the fixed layouts serve c <= 0 and |q| above about 0.034 (3 tau / (64 q^2) <= 40) */
int dfl_usebatch(int batched, double c, double q, double tau)
{
#ifdef DFL_NO_BATCH
    (void) batched; (void) c; (void) q; (void) tau; return 0;      /* (a build without the layouts, for checking) */
#else
    return batched && c <= 0 && !(3 * tau / (64 * q * q) > 40);
#endif
}

/* ---------- the batch evaluation ---------- */
typedef struct { double *a, *b, *e, *sn, *cs; cdouble *k; } buf_t;
static void buf_alloc(buf_t *w, int n)
{
    w->a = malloc(5 * (size_t) (n + 1) * sizeof(double)); w->b = w->a + n + 1; w->e = w->b + n + 1; w->sn = w->e + n + 1; w->cs = w->sn + n + 1;
    w->k = malloc((size_t) (n + 1) * sizeof(cdouble));
}
static void buf_free(buf_t *w) { free(w->a); free(w->k); }
/* the exponents z_k = A_k t + B_k s into a, b; the number of nodes up to the last one above the cutoff */
static int exponents(const layout_t *L, cdouble t, cdouble s, buf_t *w)
{
    int n = L->n, neff = 0;
    double tr = creal(t), ti = cimag(t), sr = creal(s), si = cimag(s);
    for (int k = 0; k < n; k++) {
        double Ar = creal(L->A[k]), Ai = cimag(L->A[k]), Br = creal(L->B[k]), Bi = cimag(L->B[k]);
        w->a[k] = Ar * tr - Ai * ti + Br * sr - Bi * si; w->b[k] = Ar * ti + Ai * tr + Br * si + Bi * sr;
    }
    for (int k = 0; k < n; k++) if (w->a[k] >= -CUTOFF) neff = k + 1;
    return neff;
}
/* K_0(z) e^{shift} = e^z K_0(z) without the exponential: K_0(z) and shift = z for |z| < 25 (the power series below |z| = 1, the
 * fine Taylor cells up to 2, the cells of side 1/4 beyond), e^z K_0(z) and shift 0 from 25 on (the asymptotic series, 17
 * terms); Re z < 0 (not met on the layouts) by dfl_k0x */
static cdouble k0_shift(cdouble z, cdouble *shift)
{
    double x = creal(z), y = cimag(z), a2 = x * x + y * y;
    *shift = z;
    if (x < 0) { *shift = DZERO; return dfl_k0x(z); }
    if (a2 < 1) {
        cdouble q4 = cscale(0.25, cmul(z, z)), t = DONE, I0 = DONE, S = DZERO; double H = 0;
        for (int j = 1; j <= 10; j++) { t = cscale(1.0 / (j * j), cmul(t, q4)); H += 1.0 / j; I0 = cadd(I0, t); S = cadd(S, cscale(H, t)); }
        cdouble lg = CPX(0.5 * log(a2) - 0.6931471805599453 + EULERGAMMA, atan2(y, x));
        return csub(S, cmul(lg, I0));
    }
    if (a2 < DFL_K0R * DFL_K0R) {
        int lower = y < 0; cdouble zu = lower ? conj(z) : z;
        double h = a2 < 4 ? DFL_K0HF : DFL_K0H; int ni = a2 < 4 ? 16 : DFL_K0NI, nn = a2 < 4 ? K0F_N : K0C_N;
        const double *T = a2 < 4 ? K0F : K0C;
        int i = (int) floor(creal(zu) / h), j = (int) floor(cimag(zu) / h);
        cdouble u = csub(zu, CPX((i + 0.5) * h, (j + 0.5) * h));
        const double *cf = T + 2 * nn * (i + ni * j);
        cdouble acc = CPX(cf[2 * (nn - 1)], cf[2 * (nn - 1) + 1]);
        for (int n = nn - 2; n >= 0; n--) acc = cadd(cmul(acc, u), CPX(cf[2 * n], cf[2 * n + 1]));
        return lower ? conj(acc) : acc;
    }
    *shift = DZERO;
    cdouble wz = cdiv(DONE, z), acc = CPX(K0A[17], 0);
    for (int j = 16; j >= 0; j--) acc = cadd(cmul(acc, wz), CPX(K0A[j], 0));
    double r = sqrt(a2), p = sqrt((r + x) / 2), qq = y / (2 * p), f = 1.2533141373155003 / r;
    return cmul(acc, CPX(f * p, -f * qq));
}
/* sum_k C_k exp(A_k t + B_k s) (bK = 0), or sum_k C_k K_0(bK X_k) exp(A_k t + B_k s) */
static cdouble expsum(const layout_t *L, cdouble t, cdouble s, cdouble bK, buf_t *w)
{
    int n = exponents(L, t, s, w);
    if (L->X) for (int k = 0; k < n; k++) { cdouble sh; w->k[k] = k0_shift(cmul(bK, L->X[k]), &sh); w->a[k] += creal(sh); w->b[k] += cimag(sh); }
    dfl_vexp(n, w->a, w->e); dfl_vsincos(n, w->b, w->sn, w->cs);
    double ar = 0, ai = 0;
    if (L->X) {
        for (int k = 0; k < n; k++) {
            double er = w->e[k] * w->cs[k], ei = w->e[k] * w->sn[k];
            double kr = creal(w->k[k]), ki = cimag(w->k[k]), Cr = creal(L->C[k]), Ci = cimag(L->C[k]);
            double mr = Cr * kr - Ci * ki, mi = Cr * ki + Ci * kr;
            ar += mr * er - mi * ei; ai += mr * ei + mi * er;
        }
    } else {
        for (int k = 0; k < n; k++) {
            double er = w->e[k] * w->cs[k], ei = w->e[k] * w->sn[k], Cr = creal(L->C[k]), Ci = cimag(L->C[k]);
            ar += Cr * er - Ci * ei; ai += Cr * ei + Ci * er;
        }
    }
    return CPX(ar, ai);
}
/* the inner value of one node: part 1 the smooth part (with the constant part of the distribution function: erf(sqrt(s_1/2))/
 * sqrt(s_1) for the chi-square kernel, M_0 less the ray over X > 1/2 for the K_0 kernel), part 2 the edge part */
static cdouble node_value(const struct dfl_batch *B, int part, cdouble t, cdouble s1, cdouble s2, cdouble M0, buf_t *w)
{
    if (B->kern == 1) {
        if (part == 1) {
            cdouble v = B->L[0].n ? expsum(&B->L[0], t, s1, DZERO, w) : DZERO;
            if (!B->dens) v = cadd(v, cdiv(dfl_erf(csqrt(cscale(0.5, s1))), csqrt(s1)));
            return v;
        }
        return B->L[1].n ? expsum(&B->L[1], t, s1, DZERO, w) : DZERO;
    }
    cdouble b = cscale(0.5, csub(s2, s1)), v = DZERO;
    int i0 = part == 1 ? B->is : B->ie0, i1 = part == 1 ? B->is + 1 : B->ie1;
    for (int i = i0; i < i1; i++) {
        const layout_t *L = &B->L[i];
        if (L->n) v = cadd(v, L->neg ? expsum(L, t, s1, cneg(b), w) : expsum(L, t, s2, b, w));
    }
    if (part == 1 && B->ic >= 0) v = cadd(v, csub(M0, B->L[B->ic].n ? expsum(&B->L[B->ic], t, s2, b, w) : DZERO));
    return v;
}
typedef struct { const struct dfl_batch *B; int part, n; const cdouble *t, *s1, *s2, *M0; cdouble *out; } bctx;
#define BCHUNK 8
static void chunk_body(int c, void *v)
{
    bctx *x = v; buf_t w; buf_alloc(&w, x->B->nmax);
    for (int j = c * BCHUNK; j < x->n && j < (c + 1) * BCHUNK; j++) x->out[j] = node_value(x->B, x->part, x->t[j], x->s1[j], x->s2[j], x->M0[j], &w);
    buf_free(&w);
}
/* the inner values of the nodes t[0..n-1] of the outer integral (part 1 smooth, 2 edge), on the workers */
void dfl_batch_values(const dfl_batch *B, int part, int n, const cdouble *t, const cdouble *s1, const cdouble *s2, const cdouble *M0, cdouble *out)
{
    bctx x = {B, part, n, t, s1, s2, M0, out};
    dfl_parfor((n + BCHUNK - 1) / BCHUNK, chunk_body, &x);
}
