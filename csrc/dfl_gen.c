/* dfl_gen.c: the single integral with a local null value c0 != 0 or a fixed start g0 != 0: the kernel runs along the
 * parabola X(v) = c0 v^2 + x v + x0 (x0 = 1/2, or (1 + g0^2)/2 in case DK with a fixed start), with one Filon part of
 * the outer integral per positive root of X (and the vertex as a pseudo-edge for c0 > 0), the segments between
 * breakpoints on three-piece steepest-descent paths.  With a fixed start the density of R under the tilt is the
 * noncentral one (nckernel, phialg): elementary in case DK, a one-dimensional l-integral (ncG) in cases DU, DT.  A port
 * of breakpts / inner_gen / segint / pieces_nodes / phase_nodes / descent / outer_gen / nckernel / ncG of DFExact.jl
 * (Raymond Kan's extension of the MATLAB port).  At the end of the file the contour integral representation of a fixed
 * start in cases DU, DT (dfl_contour; ctr_* and tr_contour of DFExact.jl). */
#include "dflimit.h"
#include "dfl_vmath.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef DFL_QUIET
/* builds in which the engine may not write to stdout (the R package; Writing R Extensions, 1.6.4): the debugging
 * output of this file (DFL_DEBUG* in the environment, dfl_debug_inner_gen) is compiled out */
static inline int dfl_quiet_printf(const char *fmt, ...) { (void) fmt; return 0; }
#define printf dfl_quiet_printf
#endif

static const cdouble ZERO = {0}, ONE = {1};
#define EXPO DFL_EXPO
#define KAPPA DFL_KAPPA
#define WMAX DFL_WMAX
extern int DFL_NODEFAC;
extern double DFL_FILONW;
int DFL_NLNC = 40;                              /* (40 agrees with 384 to 1e-14; 32 loses 3e-13) */
/* shared with dfl_single.c */
cdouble dfl_phi_pos(int kern, cdouble X, cdouble s1, cdouble s2);
cdouble dfl_phi_neg(int kern, cdouble X, cdouble s1, cdouble s2);
cdouble dfl_k0x(cdouble z);
typedef cdouble (*cfn_t)(cdouble x, const void *ctx);
cdouble dfl_gaussexp(cfn_t A, const void *Actx, cdouble t, double q, cdouble sd, double lA, int force, int p);
bool dfl_ray_ok(cdouble sd, cdouble e, double q, double at);
cdouble dfl_psi_half(int kern, cdouble s1, cdouble s2, cdouble M0);
cdouble dfl_filon_panel(const cdouble *G, double a, double b, double y, int N);
double dfl_default_tau(int model, double c, bool stat);

/* ---- the breakpoints ---- */
typedef struct {
    double q, c0, x0, lead;
    cdouble rr[2]; int nr;                     /* the roots of X(v) */
    double b[2], y[2], ell[2]; int kindb[2]; int ne; bool negl[2];
    double ell0; int sgn[3]; bool dbl; double vsad;
} breaks_t;
static cdouble xat(const breaks_t *R, cdouble dv, double b)
{
    cdouble X = CPX(R->lead, 0);
    for (int k = 0; k < R->nr; k++) X = cmul(X, cadd(dv, csub(CPX(b, 0), R->rr[k])));
    return X;
}
static double dXv(const breaks_t *R, double v) { return 2 * R->c0 * v + R->q; }
static void breakpts(double q, double c0, double tau, double x0, breaks_t *R)
{
    memset(R, 0, sizeof *R);
    R->q = q; R->c0 = c0; R->x0 = x0;
    double r[2]; int nrr = 0;
    if (c0 == 0) { R->rr[0] = CPX(-x0 / q, 0); R->nr = 1; R->lead = q; r[0] = -x0 / q; nrr = 1; }
    else {
        double d = q * q - 4 * c0 * x0;
        cdouble sq = d >= 0 ? CPX(sqrt(d), 0) : CPX(0, sqrt(-d));
        R->rr[0] = cdiv(csub(CPX(-q, 0), sq), CPX(2 * c0, 0)); R->rr[1] = cdiv(cadd(CPX(-q, 0), sq), CPX(2 * c0, 0)); R->nr = 2; R->lead = c0;
        if (d >= 0) { r[0] = creal(R->rr[0]); r[1] = creal(R->rr[1]); if (r[0] > r[1]) { double t = r[0]; r[0] = r[1]; r[1] = t; } nrr = 2; }
    }
    double b[2]; int nb = 0;
    for (int k = 0; k < nrr; k++) if (r[k] > 0) b[nb++] = r[k];
    bool noroot = nb == 0;
    /* a far root is dropped for c0 = 0 only: with c0 != 0 X changes sign at the root and the segment beyond it has the other
     * form with the other exponential factor; dropping it ran the near side's form out to infinity (Raymond Kan, Nov 2026) */
    if (c0 == 0) { int nb2 = 0; for (int k = 0; k < nb; k++) if (3 * tau * b[k] * b[k] / 4 <= 40) b[nb2++] = b[k]; nb = nb2; }
    int kindb[2] = {1, 1}; double ell[2] = {INFINITY, INFINITY};
    if (nb == 2 && b[1] - b[0] <= 1e-9 * b[1]) { b[0] = (b[0] + b[1]) / 2; nb = 1; R->dbl = true; }
    else if (nb == 2) ell[0] = ell[1] = b[1] - b[0];
    if (c0 > 0 && q < 0 && noroot) { b[0] = -q / (2 * c0); nb = 1; kindb[0] = 0; ell[0] = sqrt(fmax(x0 - q * q / (4 * c0), 0) / c0); }
    for (int k = 0; k < nb; k++) {
        for (int m = 0; m < R->nr; m++) { double dr = cabs(csub(CPX(b[k], 0), R->rr[m])); if (dr > 1e-9 * fmax(b[k], 1)) ell[k] = fmin(ell[k], dr); }
        ell[k] = fmax(ell[k], 1e-8 * fmax(b[k], 1));
    }
    R->ell0 = INFINITY; for (int m = 0; m < R->nr; m++) R->ell0 = fmin(R->ell0, cabs(R->rr[m]));
    R->ne = nb;
    for (int k = 0; k < nb; k++) { R->b[k] = b[k]; R->y[k] = b[k] * b[k]; R->kindb[k] = kindb[k]; R->ell[k] = ell[k]; }
    double ed[4] = {0, nb > 0 ? b[0] : INFINITY, nb > 1 ? b[1] : INFINITY, INFINITY};
    for (int j = 0; j <= nb; j++) {
        double vm = isfinite(ed[j + 1]) ? (ed[j] + ed[j + 1]) / 2 : ed[j] + 1;
        double sv = creal(xat(R, CPX(vm, 0), 0)); R->sgn[j] = sv < 0 ? -1 : 1;
    }
    for (int k = 0; k < nb; k++) { double mid = (ed[k] + ed[k + 1]) / 2; R->negl[k] = 3 * tau * mid * mid / 4 > 40; }
    R->vsad = NAN;
}

/* ---- the noncentral kernel of a fixed start (see DFExact.jl, "the fixed start g0 != 0 through the single integral") ----
 * Case DK: R + g0^2/2 = J(1)^2/2 with J(1) ~ N(m, 1/s1); phi_t(X~) = pref e^{-s1 m^2/2} e^{-s1 X~} cosh(m s1 sqrt(2 X~))/sqrt(pi X~).
 * Cases DU, DT: R = (X1^2 - X2^2)/2, X1 ~ N(m1, 1/s2), X2 ~ N(m2, -1/s1);
 *   phi_t(X) = pref (2/pi) e^{(s1 m2^2 - s2 m1^2)/2} e^{-s2 X} G(X; a, m2 s1, m1 s2)   (X > 0), with -X, e^{-s1 X} and
 *   the b's swapped for X < 0; G(Y; a, b1, b2) = int_0^inf e^{-a l^2/2} cosh(b1 l) cosh(b2 sqrt(2Y + l^2)) (2Y + l^2)^{-1/2} dl.
 * pref = exp(g0^2 Q(0,t)) multiplies the transform. */
typedef struct { bool nc; int kern; cdouble m1, m2, pref, lpref; double g0; } nck_t;   /* lpref = log pref = g0^2 Q(0,t) */
static void nckernel(int model, double c, cdouble t, double g0, cdouble s1, cdouble s2, nck_t *K)
{
    K->nc = true; K->g0 = g0;
    if (model == 1) {
        dfl_girs_t G; dfl_girsanov(1, c, t, &G);
        K->kern = 1; K->m1 = cscale(g0, cdiv(cexp(cneg(G.lsh)), G.eta)); K->m2 = ZERO; K->lpref = cscale(g0 * g0, cdiv(t, G.eta)); K->pref = cexp(K->lpref);
        return;
    }
    cdouble Q0, m1s, m2s;
    dfl_qpieces(model, c, t, g0, cdiv(ONE, s2), cneg(cdiv(ONE, s1)), c > DFL_CGIRS, &Q0, &m1s, &m2s);
    K->kern = 2; K->m1 = csqrt(m1s); K->m2 = csqrt(m2s); K->pref = cexp(Q0); K->lpref = Q0;
}
/* G(Y; a, b1, b2): l = al sinh T, al = sqrt|2Y|, Gauss-Legendre in T up to where the Gaussian, against the growth of the
 * cosh factors, is below e^{-41.4}; sqrt(2Y + l^2) = al sqrt(2Y/al^2 + sinh^2 T) (principal branch) */
long DFL_NCG_CALLS = 0, DFL_NCG_SEG = 0, DFL_NCG_RAY = 0;   /* counters, for profiling (DFL_NCG_SEG, DFL_NCG_RAY: the nodes of the segments and of the rays) */
int DFL_NCG_LIBM = 0;                           /* 1: the libm version of ncG (checking) */
/* the libm version: complex exp, cosh, sqrt at each node */
/* the nodes and weights of the l-rule for int_0^lm e^{-a l^2/2 + bb l + i(...)} (...) dl with the edge scale al (the
 * factor (2Y + l^2)^{-1/2} varies on it): the Gaussian times the growth peaks at l* = bb/Re a with the width sigma =
 * (Re a)^{-1/2}.  When the peak is separated from the edge scale (l* - 8 sigma > 2 al, large noncentrality), l = al sinh T
 * with Nl Gauss-Legendre nodes in T on [0, l* - 8 sigma] and Gauss-Legendre in l on [l* - 8 sigma, l* + 8 sigma], with the
 * node count from the phase change (ai = Im a, bi = |Im b_1| + |Im b_2|) across it; otherwise l = al sinh T on [0, lm] with
 * the node count raised where the peak is narrow in T.  Returns the count, at most NLMAX. */
#define NLMAX 256
static int clampn(double n, int lo) { int k = (int) ceil(n / 8) * 8; return k < lo ? lo : k > NLMAX ? NLMAX : k; }
static int lrule(double al, double lm, double bb, double ra, double ai, double bi, int Nl, double *ls, double *ws)
{
    double lstar = bb / ra, sig = 1 / sqrt(ra);
    if (Nl > NLMAX / 2) Nl = NLMAX / 2;
    if (lstar - 8 * sig > 2 * al) {
        double l1 = lstar - 8 * sig, lhi = fmin(lm, lstar + 8 * sig);
        double dphi = fabs(ai) * (lhi * lhi - l1 * l1) / 2 + bi * (lhi - l1);
        int nB = clampn(1.2 * dphi + 24, Nl); if (nB > NLMAX - Nl) nB = NLMAX - Nl;
        const double *wl; const double *zl = dfl_rule(Nl, &wl); const double *wB; const double *zB = dfl_rule(nB, &wB);
        double tm = asinh(l1 / al);
        for (int k = 0; k < Nl; k++) { double T = zl[k] * tm; ls[k] = al * sinh(T); ws[k] = al * cosh(T) * wl[k] * tm; }
        for (int k = 0; k < nB; k++) { ls[Nl + k] = l1 + (lhi - l1) * zB[k]; ws[Nl + k] = (lhi - l1) * wB[k]; }
        return Nl + nB;
    }
    double tm = asinh(lm / al), dT = sig / sqrt(al * al + lstar * lstar);
    int n = clampn(3 * tm / dT, Nl);
    const double *wl; const double *zl = dfl_rule(n, &wl);
    for (int k = 0; k < n; k++) { double T = zl[k] * tm; ls[k] = al * sinh(T); ws[k] = al * cosh(T) * wl[k] * tm; }
    return n;
}
static cdouble ncG_libm(cdouble Y, cdouble a, cdouble b1, cdouble b2, double *E)
{
    int Nl = DFL_NLNC;
    double ra = fmax(creal(a), 1e-300), bb = cabs(b1) + cabs(b2);
    double lm = (bb + sqrt(bb * bb + 2 * 41.4 * ra)) / ra;
    double al = fmax(sqrt(cabs(cscale(2, Y))), 1e-8 * lm);
    double ls[NLMAX], ws[NLMAX]; int n = lrule(al, lm, fabs(creal(b1)) + fabs(creal(b2)), ra, cimag(a), fabs(cimag(b1)) + fabs(cimag(b2)), Nl, ls, ws);
    cdouble Y2 = cscale(2 / (al * al), Y), acc = ZERO;
    cdouble Es[NLMAX], rs[NLMAX], fs[NLMAX]; double Eref = -INFINITY;
    for (int k = 0; k < n; k++) {
        double l = ls[k], sh = l / al;
        cdouble r = cscale(al, csqrt(cadd(Y2, CPX(sh * sh, 0))));
        cdouble z1 = cscale(l, b1), z2 = cmul(b2, r);
        if (creal(z1) < 0) z1 = cneg(z1);
        if (creal(z2) < 0) z2 = cneg(z2);
        /* cosh z = e^{z} (1 + e^{-2z})/2 for Re z >= 0: one exponential of the combined exponent */
        Es[k] = cadd(cadd(cscale(-l * l / 2, a), z1), z2); rs[k] = r;
        fs[k] = cscale(0.25, cmul(cadd(ONE, cexp(cscale(-2, z1))), cadd(ONE, cexp(cscale(-2, z2)))));
        if (creal(Es[k]) > Eref) Eref = creal(Es[k]);
    }
    /* the terms as e^{E_k - Eref}: G = S e^{Eref} with S bounded (the exponents reach 2 lam at large noncentrality) */
    for (int k = 0; k < n; k++) acc = cadd(acc, cscale(ws[k], cdiv(cmul(cexp(csub(Es[k], CPX(Eref, 0))), fs[k]), rs[k])));
    *E = Eref;
    return acc;
}
/* the vectorizable version: the same sum in real arithmetic on arrays of length Nl, exp and sincos from dfl_vmath.h, so
 * that the compiler vectorizes the loops (NEON, SSE2; AVX2+FMA on Linux x86 through target_clones) */
#if defined(__linux__) && defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
#define DFL_CLONES __attribute__((target_clones("avx2,fma", "default")))
#else
#define DFL_CLONES
#endif
static inline void csqrt_arr(int n, const double *p, const double *q, double *re, double *im)
{
    for (int i = 0; i < n; i++) {
        double pp = p[i], qq = q[i], mag = sqrt(pp * pp + qq * qq), u = sqrt((mag + fabs(pp)) / 2);
        u = u > 1e-300 ? u : 1e-300;
        double aq = fabs(qq), sq = qq < 0 ? -1.0 : 1.0;
        re[i] = pp >= 0 ? u : aq / (2 * u);
        im[i] = pp >= 0 ? qq / (2 * u) : sq * u;
    }
}
/* the factor (1 + e^{-2 z})/2 of cosh z = e^{z} (1 + e^{-2 z})/2, Re z >= 0, from e = e^{-2 Re z}, s = sin(-2 Im z), c = cos(-2 Im z) */
static inline void coshfac_arr(int n, const double *e, const double *s, const double *c, double *re, double *im)
{
    for (int i = 0; i < n; i++) { re[i] = (1 + e[i] * c[i]) / 2; im[i] = e[i] * s[i] / 2; }
}
DFL_CLONES static cdouble ncG_vec(cdouble Y, cdouble a, cdouble b1, cdouble b2, double *E)
{
    int Nl = DFL_NLNC;
    double ar = creal(a), ai = cimag(a), ra = fmax(ar, 1e-300), bb = cabs(b1) + cabs(b2);
    double lm = (bb + sqrt(bb * bb + 2 * 41.4 * ra)) / ra;
    double al = fmax(sqrt(cabs(cscale(2, Y))), 1e-8 * lm);
    double Y2r = 2 * creal(Y) / (al * al), Y2i = 2 * cimag(Y) / (al * al);
    double b1r = creal(b1), b1i = cimag(b1), b2r = creal(b2), b2i = cimag(b2);
    enum { NM = NLMAX };
    double l[NM], wk[NM], l2[NM], sh[NM];
    double pr[NM], qi[NM], rr[NM], ri[NM];                   /* r = al sqrt(Y2 + sh^2) */
    double x1[NM], y1[NM], x2[NM], y2[NM];                   /* z1 = b1 l, z2 = b2 r, with Re >= 0 */
    double xe[NM], ye[NM], ee[NM], se[NM], ce[NM];           /* the combined exponent E = -a l^2/2 + z1 + z2 */
    double m1[NM], n1[NM], e1[NM], s1[NM], c1[NM], f1r[NM], f1i[NM];   /* (1 + e^{-2 z1})/2 */
    double m2[NM], n2[NM], e2[NM], s2[NM], c2[NM], f2r[NM], f2i[NM];   /* (1 + e^{-2 z2})/2 */
    double accr[NM], acci[NM];
    Nl = lrule(al, lm, fabs(b1r) + fabs(b2r), ra, ai, fabs(b1i) + fabs(b2i), Nl, l, wk);     /* now the count of nodes */
    for (int k = 0; k < Nl; k++) { sh[k] = l[k] / al; l2[k] = l[k] * l[k]; }
    for (int k = 0; k < Nl; k++) { pr[k] = Y2r + sh[k] * sh[k]; qi[k] = Y2i; }
    csqrt_arr(Nl, pr, qi, rr, ri);
    for (int k = 0; k < Nl; k++) {
        rr[k] *= al; ri[k] *= al;
        double zr = b1r * l[k], zi = b1i * l[k]; double sg = zr < 0 ? -1.0 : 1.0; x1[k] = sg * zr; y1[k] = sg * zi;
        zr = b2r * rr[k] - b2i * ri[k]; zi = b2r * ri[k] + b2i * rr[k]; sg = zr < 0 ? -1.0 : 1.0; x2[k] = sg * zr; y2[k] = sg * zi;
        xe[k] = -ar * l2[k] / 2 + x1[k] + x2[k]; ye[k] = -ai * l2[k] / 2 + y1[k] + y2[k];
        m1[k] = -2 * x1[k]; n1[k] = -2 * y1[k]; m2[k] = -2 * x2[k]; n2[k] = -2 * y2[k];
    }
    double Eref = -INFINITY; for (int k = 0; k < Nl; k++) if (xe[k] > Eref) Eref = xe[k];
    for (int k = 0; k < Nl; k++) xe[k] -= Eref;                       /* the terms as e^{E_k - Eref}: G = S e^{Eref} */
    *E = Eref;
    dfl_vexp(Nl, xe, ee); dfl_vsincos(Nl, ye, se, ce);
    dfl_vexp(Nl, m1, e1); dfl_vsincos(Nl, n1, s1, c1); coshfac_arr(Nl, e1, s1, c1, f1r, f1i);
    dfl_vexp(Nl, m2, e2); dfl_vsincos(Nl, n2, s2, c2); coshfac_arr(Nl, e2, s2, c2, f2r, f2i);
    for (int k = 0; k < Nl; k++) {
        double er = ee[k] * ce[k], ei = ee[k] * se[k];                    /* e^E */
        double tr = er * f1r[k] - ei * f1i[k], ti = er * f1i[k] + ei * f1r[k];
        double ur = tr * f2r[k] - ti * f2i[k], ui = tr * f2i[k] + ti * f2r[k];
        double d = rr[k] * rr[k] + ri[k] * ri[k], f = wk[k] / d;                 /* / r */
        accr[k] = (ur * rr[k] + ui * ri[k]) * f; acci[k] = (ui * rr[k] - ur * ri[k]) * f;
    }
    double sr = 0, si = 0; for (int k = 0; k < Nl; k++) { sr += accr[k]; si += acci[k]; }
    return CPX(sr, si);
}
/* G = S e^{E}, S bounded.  Along the ray l = e s, e = e^{-i arg(a)/2}, the Gaussian is e^{-|a| s^2/2} (no oscillation) and
 * r = e (2Y e^{i arg a} + s^2)^{1/2}: the same integral with the arguments (Y e^{i arg a}, |a|, b1 e, b2 e), whose cut in Y
 * lies where a Y is negative real, as the cut of K_0((s2 - s1) X/2) of the central kernel does, so the paths of inner_gen,
 * which avoid that cut, serve (with the real-axis rule the cut would lie at negative real Y) */
static cdouble ncG_rot(cdouble Y, cdouble a, cdouble b1, cdouble b2, double *E, int libm)
{
    double ph = carg(a); cdouble e = cexp(CPX(0, -ph / 2));
    Y = cmul(Y, cexp(CPX(0, ph))); a = CPX(cabs(a), 0); b1 = cmul(b1, e); b2 = cmul(b2, e);
    return libm ? ncG_libm(Y, a, b1, b2, E) : ncG_vec(Y, a, b1, b2, E);
}
static cdouble ncG(cdouble Y, cdouble a, cdouble b1, cdouble b2, double *E)
{
#if defined(__GNUC__) || defined(__clang__)
    __atomic_add_fetch(&DFL_NCG_CALLS, 1, __ATOMIC_RELAXED);
#else
    DFL_NCG_CALLS++;
#endif
    return ncG_rot(Y, a, b1, b2, E, DFL_NCG_LIBM);
}
cdouble dfl_ncG(cdouble Y, cdouble a, cdouble b1, cdouble b2, int libm) { double E; cdouble S = ncG_rot(Y, a, b1, b2, &E, libm); return cmul(S, cexp(CPX(E, 0))); }   /* exported for the tests */
cdouble dfl_ncG_S(cdouble Y, cdouble a, cdouble b1, cdouble b2, int libm, double *E) { return ncG_rot(Y, a, b1, b2, E, libm); }   /* G = S e^E: S, and E */
/* phialg of cases DU, DT with a fixed start as S e^{E}, S bounded.  The constant e^{(s1 m2^2 - s2 m1^2)/2} is of the size
 * e^{-lam} and the sum of ncG of the size e^{lam}, lam ~ (c g0)^2/2 in case DU and (c g0)^2/8 in case DT: its modulus goes
 * into the exponent E with that of ncG (as a factor of S it underflows for lam > 745, e.g. case DU at c = -18, g0 = 3, and
 * 0 e^{E} with E > 709 is NaN), its phase stays with S (phialg_scaled in DFExact.jl) */
static cdouble phialg_scaled(const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2, double *E)
{
    cdouble a = csub(s2, s1);
    cdouble ex = cscale(0.5, csub(cmul(s1, cmul(nc->m2, nc->m2)), cmul(s2, cmul(nc->m1, nc->m1))));
    cdouble cst = cmul(cscale(2 / M_PI, nc->pref), cexp(CPX(0, cimag(ex))));
    cdouble S = sgn > 0 ? ncG(X, a, cmul(nc->m2, s1), cmul(nc->m1, s2), E) : ncG(cneg(X), a, cmul(nc->m1, s2), cmul(nc->m2, s1), E);
    *E += creal(ex);
    return cmul(cst, S);
}
static cdouble phialg(int kern, const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2)
{
    if (nc && nc->nc) {
        if (kern == 1) {
            cdouble m = nc->m1;
            cdouble f = cexp(csub(nc->lpref, cscale(0.5, cmul(s1, cmul(m, m)))));
            return cdiv(cmul(f, ccosh(cmul(cmul(m, s1), csqrt(cscale(2, X))))), csqrt(cscale(M_PI, X)));
        }
        double E; cdouble S = phialg_scaled(nc, X, sgn, s1, s2, &E);
        return cmul(S, cexp(CPX(E, 0)));
    }
    if (kern == 1) return cdiv(ONE, csqrt(cscale(M_PI, X)));
    cdouble arg = cmul(cscale(0.5, csub(s2, s1)), X);
    return cscale(1 / M_PI, dfl_k0x(sgn > 0 ? arg : cneg(arg)));
}
/* The noncentral density is a Gaussian in sqrt(2 X~) centered at the mean, e^{-s_d X~} cosh(b sqrt(2 X~)) with b = m s_d,
 * so its mass sits away from the edge by the factor e^{lam}, lam = m^2 s_d/2; the paths are built from the exponent
 * E = -t v^2 - s_d X + ncexp, ncexp = log cosh(b sqrt(2 X~)), and the nodes carry phialg e^{-ncexp}, which is bounded. */
static cdouble logcosh(cdouble z)
{
    cdouble zs = creal(z) >= 0 ? z : cneg(z);
    return csub(cadd(zs, clog(cadd(ONE, cexp(cscale(-2, zs))))), CPX(0.6931471805599453, 0));
}
/* tanh z without overflow: 1 - 2/(e^{2z} + 1) for Re z >= 0, odd otherwise */
static cdouble ctanh_safe(cdouble z)
{
    bool neg = creal(z) < 0; if (neg) z = cneg(z);
    cdouble t = csub(ONE, cdiv(CPX(2, 0), cadd(cexp(cscale(2, z)), ONE)));
    return neg ? cneg(t) : t;
}
static cdouble ncb(const nck_t *nc, int sgn, cdouble s1, cdouble s2)
{
    if (nc->kern == 1) return cmul(nc->m1, s1);
    return sgn > 0 ? cmul(nc->m1, s2) : cmul(nc->m2, s1);
}
static cdouble ncexp(const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2)
{
    if (!(nc && nc->nc)) return ZERO;
    cdouble Xt = sgn > 0 ? X : cneg(X);
    return logcosh(cmul(ncb(nc, sgn, s1, s2), csqrt(cscale(2, Xt))));
}
/* d ncexp / dv = b^2 (tanh z / z) (+-dX/dv), z = b sqrt(2 X~), finite at the edge */
static cdouble ncexp_dv(const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2, double dXdv)
{
    if (!(nc && nc->nc)) return ZERO;
    cdouble b = ncb(nc, sgn, s1, s2), Xt = sgn > 0 ? X : cneg(X), z = cmul(b, csqrt(cscale(2, Xt)));
    cdouble tz = cabs(z) < 1e-6 ? ONE : cdiv(ctanh_safe(z), z);
    return cmul(cmul(b, b), cscale(sgn > 0 ? dXdv : -dXdv, tz));
}
static cdouble phired(int kern, const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2)
{
    if (nc && nc->nc && kern == 2) {
        double E; cdouble S = phialg_scaled(nc, X, sgn, s1, s2, &E);
        return cmul(S, cexp(csub(CPX(E, 0), ncexp(nc, X, sgn, s1, s2))));
    }
    /* case DK: phialg e^{-ncexp} = pref e^{-s1 m^2/2} / sqrt(pi X~) exactly (the cosh of phialg is e^{ncexp}, which overflows for
     * Re z > 710, lam = s1 m^2/2 > ~350, e.g. c = 10, g0 = 5), with the constant e^{-lam} and pref (e^{+-lam}) in one exponential */
    if (nc && nc->nc) return cdiv(cexp(csub(nc->lpref, cscale(0.5, cmul(s1, cmul(nc->m1, nc->m1))))), csqrt(cscale(M_PI, X)));
    return phialg(kern, nc, X, sgn, s1, s2);
}
/* Psi_t(1/2) with a fixed start: P[J(1)^2 <= 1 + g0^2] (case DK), P[R <= 1/2] by the l-rule of Pkap (DU, DT), times the mass */
/* P[R <= 1/2] under the tilt with a fixed start, cases DU, DT: 2 int_0^inf phibar(l; m2, w2) Fbar(sqrt(1 + l^2); m1, w1) dl,
 * X1 ~ N(m1, w1 = 1/s2), X2 ~ N(m2, w2 = -1/s1), by the l-rule (the Gaussian in l peaks at |m2|) */
static cdouble ncP(cdouble s1, cdouble s2, cdouble m1, cdouble m2, int Nl)
{
    cdouble w1 = cdiv(ONE, s2), w2 = cneg(cdiv(ONE, s1)), iw2 = cdiv(ONE, w2), mw = cmul(m2, iw2);
    double ra = fmax(creal(iw2), 1e-300), bb = cabs(mw);
    double lm = (bb + sqrt(bb * bb + 2 * 41.4 * ra)) / ra;
    /* the integrand is smooth at l = 0: a peak of the Gaussian at l* separated from 0 by more than 8 widths gets its own part of
     * the l-rule (with the edge scale 1 a narrow peak at l* < 2 + 8 sigma fell to the single sinh rule, capped at NLMAX nodes:
     * P[R <= 1/2] off by 3e-7 at c = -60, g0 = 1.5, an error of F that does not depend on x; ncP in DFExact.jl) */
    double sep = fabs(creal(mw)) / ra - 8 / sqrt(ra), al = (sep > 0 && sep <= 2) ? sep / 2.000001 : 1.0;
    double ls[NLMAX], ws[NLMAX]; int n = lrule(al, lm, fabs(creal(mw)), ra, cimag(iw2), fabs(cimag(mw)), Nl, ls, ws);
    cdouble sq1 = csqrt(cscale(2, w1)), c2 = cdiv(ONE, cscale(2, csqrt(cscale(2 * M_PI, w2)))), P = ZERO;
    for (int k = 0; k < n; k++) {
        double l = ls[k], u = sqrt(1 + l * l);
        cdouble lm2 = csub(CPX(l, 0), m2), lp2 = cadd(CPX(l, 0), m2);
        cdouble f2 = cmul(cadd(cexp(cneg(cdiv(cmul(lm2, lm2), cscale(2, w2)))), cexp(cneg(cdiv(cmul(lp2, lp2), cscale(2, w2))))), c2);
        cdouble F1 = cscale(0.5, cadd(dfl_erf(cdiv(csub(CPX(u, 0), m1), sq1)), dfl_erf(cdiv(cadd(CPX(u, 0), m1), sq1))));
        P = cadd(P, cscale(ws[k], cmul(f2, F1)));
    }
    return cscale(2, P);
}
static cdouble psi_half_nc(const nck_t *nc, cdouble s1, cdouble s2, cdouble M0)
{
    if (nc->kern == 1) {
        /* pref P / sqrt(s1), P = (erf((r - m)/w) + erf((r + m)/w))/2 = P[|J(1)| <= r].  When the mean lies beyond r (Re z1 > 0
         * below, Re(m/w) >= 0 by the symmetry in m) the two erf are near -1 and 1 and their sum, P = (erfc(z1) - erfc(z2))/2, is
         * of the size e^{-lam} while pref reaches e^{lam}: with erfc(z) = e^{-z^2} erfcx(z), erfcx(z) = w(iz), each term carries
         * its exponential together with lpref (psi_half(K::NCK{1}) in DFExact.jl) */
        double r = sqrt(1 + nc->g0 * nc->g0); cdouble w = csqrt(cdiv(CPX(2, 0), s1)), m = nc->m1;
        if (creal(cdiv(m, w)) < 0) m = cneg(m);
        cdouble z1 = cdiv(csub(m, CPX(r, 0)), w), z2 = cdiv(cadd(m, CPX(r, 0)), w), den = cscale(2, csqrt(s1));
        cdouble e2 = cmul(cexp(csub(nc->lpref, cmul(z2, z2))), dfl_faddw(CPX(-cimag(z2), creal(z2))));
        if (creal(z1) > 0) return cdiv(csub(cmul(cexp(csub(nc->lpref, cmul(z1, z1))), dfl_faddw(CPX(-cimag(z1), creal(z1)))), e2), den);
        return cdiv(csub(csub(cscale(2, nc->pref), cmul(cexp(csub(nc->lpref, cmul(z1, z1))), dfl_faddw(CPX(cimag(z1), -creal(z1))))), e2), den);
    }
    return cmul(cmul(nc->pref, M0), ncP(s1, s2, nc->m1, nc->m2, DFL_NLNC));
}

/* ---- the pieces of the paths ----
 * A path of the inner integrals is a list of pieces [a, b], each with n Gauss-Legendre nodes (DFL_NODEFAC included) and the
 * substitution y = a + (b - a) u^p; the callers sum over the nodes of each piece (pnode) as they go, so that a path is
 * never stored node by node, and a path of more than DFL_NODEMAX nodes is refused before anything is evaluated on it (the
 * bound is recorded on the thread and the list is empty; Piece, pnode, NODEMAX of DFExact.jl).  The paths follow the phase
 * of the exponent with about 1.2 nodes per radian, and on a long segment between the support edges (a root b of X far
 * out, as with c0 < 0 near zero at x > 0, where b ~ x/|c0|) that phase grows like Im(t) b^2: about 6e11 nodes at
 * c0 = -1e-4, x = 1.  The largest path of the tests of DFExact.jl has 3.2e5 nodes. */
typedef cdouble (*efn_t)(double s, const void *ctx);
typedef struct { double a, b; int n, p; } piece_t;
typedef struct { piece_t *pc; int np, cap; } pieces_t;
static void pieces_add(pieces_t *P, double a, double b, int n, int p)
{
    if (P->np == P->cap) { P->cap = P->cap ? 2 * P->cap : 64; P->pc = realloc(P->pc, P->cap * sizeof(piece_t)); }
    piece_t q = {a, b, n, p}; P->pc[P->np++] = q;
}
/* node and weight k of the piece P, (u, gw) its rule */
static inline void pnode(const piece_t *P, const double *u, const double *gw, int k, double *y, double *w)
{
    double a = P->a, L = P->b - a;
    if (P->p == 1) { *y = a + L * u[k]; *w = L * gw[k]; }
    else if (P->p == 2) { *y = a + L * u[k] * u[k]; *w = 2 * L * u[k] * gw[k]; }
    else { double u2 = u[k] * u[k]; *y = a + L * u2 * u2; *w = 4 * L * u2 * u[k] * gw[k]; }
}
static const char PATH_WHAT[] = "a path of the inner integrals of the general kernel";
/* a path of at least n nodes: refused (the bound recorded on the thread, the list emptied); returns 1 */
static int path_refuse(pieces_t *P, double n) { P->np = 0; dfl_bound_set(DFL_BOUND_PATH, n, PATH_WHAT); return 1; }
#define NGROW 16
#define NLEV 5
#define NCON 16
/* the pieces for int_0^L: the first piece shortened until the exponent falls by at most 4, rises by at most 4 (the noncentral
 * kernels rise away from the edge) and its phase changes by at most 40 rad, then pieces growing by 4 with node counts from
 * the change of the exponent, until it is below -EXPO.  0, or 1 for a path of more than DFL_NODEMAX nodes (refused) */
static int pieces_nodes(efn_t Ef, const void *ctx, double L, double first, int p, int nmin, pieces_t *P)
{
    P->np = 0;
    if (dfl_bound_hit()) return 1;
    first = fmin(first, L);
    for (int it = 0; it < 60; it++) {
        cdouble E[9]; double mx = -INFINITY; for (int i = 0; i <= 8; i++) { E[i] = Ef(first * i / 8, ctx); mx = fmax(mx, creal(E[i])); }
        double dph = 0; for (int i = 0; i < 8; i++) dph += fabs(cimag(E[i + 1]) - cimag(E[i]));
        if ((creal(E[8]) > -4 && mx - creal(E[0]) < 4 && dph < 40) || first <= 1e-12 * L) break;
        first /= 2;
    }
    pieces_add(P, 0, first, nmin * DFL_NODEFAC, p);
    double tot = nmin * DFL_NODEFAC;
    double a = first;
    while (a < L) {
        double bn = fmin(L, 4 * a);
        cdouble E[9]; bool all = true; for (int i = 0; i <= 8; i++) { E[i] = Ef(a + (bn - a) * i / 8, ctx); if (creal(E[i]) >= -EXPO) all = false; }
        if (all) break;
        double dph = 0; for (int i = 0; i < 8; i++) dph += cabs(csub(E[i + 1], E[i]));
        if (tot + (1.2 * dph + 16) * DFL_NODEFAC > DFL_NODEMAX) return path_refuse(P, tot + (1.2 * dph + 16) * DFL_NODEFAC);
        int n = (int) fmax(NGROW, 16 * ceil(ceil(1.2 * dph + 16) / 16));
        int m = (n + 159) / 160; n = 16 * (int) ceil(ceil((double) n / m) / 16);
        tot += (double) m * n * DFL_NODEFAC;
        if (tot > DFL_NODEMAX) return path_refuse(P, tot);
        for (int k = 0; k < m; k++) pieces_add(P, a + (bn - a) * k / m, a + (bn - a) * (k + 1) / m, n * DFL_NODEFAC, 1);
        if (creal(E[8]) < -EXPO) break;
        a = bn;
    }
    return 0;
}
/* the pieces for int_0^1 graded geometrically toward both ends, node counts from the change of the exponent, sub-pieces on
 * which the exponent is below max(E1, E2) - EXPO skipped.  0, or 1 for a path of more than DFL_NODEMAX nodes (refused) */
static int phase_nodes(efn_t Ef, const void *ctx, cdouble E1, cdouble E2, pieces_t *P)
{
    P->np = 0;
    if (dfl_bound_hit()) return 1;
    /* ss = 0, 2^-NLEV, ..., 2^-1, 1 - 2^-2, ..., 1 - 2^-NLEV, 1 (ascending) */
    double ss[2 * NLEV + 1]; int ns = 0;
    ss[ns++] = 0; for (int k = NLEV; k >= 1; k--) ss[ns++] = pow(2.0, -k);
    for (int k = 2; k <= NLEV; k++) ss[ns++] = 1 - pow(2.0, -k); ss[ns++] = 1;
    double Emax = fmax(creal(E1), creal(E2)) - EXPO + 2, tot = 0;
    for (int k = 0; k < ns - 1; k++) {
        cdouble E[9]; double mx = -INFINITY; for (int i = 0; i <= 8; i++) { E[i] = Ef(ss[k] + (ss[k + 1] - ss[k]) * i / 8, ctx); mx = fmax(mx, creal(E[i])); }
        if (mx < Emax) continue;
        double dph = 0; for (int i = 0; i < 8; i++) dph += cabs(csub(E[i + 1], E[i]));
        if (tot + (1.2 * dph + 16) * DFL_NODEFAC > DFL_NODEMAX) return path_refuse(P, tot + (1.2 * dph + 16) * DFL_NODEFAC);
        int n = (int) fmax(NCON, 8 * ceil(ceil(1.2 * dph + 16) / 8));
        int mm = (n + 159) / 160; n = 8 * (int) ceil(ceil((double) n / mm) / 8);
        tot += (double) mm * n * DFL_NODEFAC;
        if (tot > DFL_NODEMAX) return path_refuse(P, tot);
        for (int i = 0; i < mm; i++) pieces_add(P, ss[k] + (ss[k + 1] - ss[k]) * i / mm, ss[k] + (ss[k + 1] - ss[k]) * (i + 1) / mm, n * DFL_NODEFAC, 1);
    }
    return 0;
}
/* the direction and length of the steepest-descent piece at an end */
static void descent(cdouble gam, cdouble alpha, int dir, double L, double dep, double sg, double deg, cdouble *u, double *l)
{
    cdouble uu = cabs(gam) > 0 ? cdiv(cneg(conj(gam)), CPX(cabs(gam), 0)) : CPX(0, -sg);
    double cmax = cos(deg * M_PI / 180);
    if (creal(uu) * dir < cmax) {
        double si = cimag(uu) > 0 ? 1 : cimag(uu) < 0 ? -1 : -sg;
        uu = CPX(dir * cmax, si * sin(deg * M_PI / 180));
    }
    double g1 = creal(cmul(gam, uu)), g2 = -creal(cmul(alpha, cmul(uu, uu)));
    double ll = L / 2;
    if (fabs(cimag(uu)) > 0) ll = fmin(ll, dep / fabs(cimag(uu)));
    if (g1 < 0) {
        if (g2 > 0) ll = fmin(ll, -g1 / (2 * g2));
        double disc = g1 * g1 - 4 * g2 * EXPO;
        if (g2 == 0) ll = fmin(ll, -EXPO / g1);
        else if (disc >= 0) { double sE = (-g1 - sqrt(disc)) / (2 * g2); if (sE > 0) ll = fmin(ll, sE); }
    }
    *l = fmax(ll, 1e-12 * L); *u = uu;
}

/* ---- the segment integral ---- */
typedef struct {
    int kern, kind; cdouble tj, sd, s1, s2; int sgn; const breaks_t *R; double q, c0;
    double a, b; cdouble ua, ub; double kap, m; /* partition */
    const nck_t *nc;                            /* the noncentral kernel of a fixed start, or NULL */
} sctx;
static cdouble W_of(const sctx *c, cdouble v) { return c->kind == 1 ? cscale(2, cmul(v, v)) : cadd(CPX(c->q, 0), cscale(2 * c->c0, v)); }
static cdouble Ez(const sctx *c, cdouble z, double o) { cdouble v = cadd(CPX(o, 0), z), X = xat(c->R, z, o);
    return cadd(csub(cneg(cmul(c->tj, cmul(v, v))), cmul(c->sd, X)), ncexp(c->nc, X, c->sgn, c->s1, c->s2)); }
static cdouble Fz(const sctx *c, cdouble z, double o) { cdouble X = xat(c->R, z, o);
    if (c->nc && c->nc->nc && c->kern == 2) {                             /* the factor as S e^E with S bounded: one exponential */
        double E; cdouble S = phialg_scaled(c->nc, X, c->sgn, c->s1, c->s2, &E);
        return cmul(cmul(W_of(c, cadd(CPX(o, 0), z)), S), cexp(cadd(csub(Ez(c, z, o), ncexp(c->nc, X, c->sgn, c->s1, c->s2)), CPX(E, 0)))); }
    return cmul(cmul(W_of(c, cadd(CPX(o, 0), z)), cexp(Ez(c, z, o))), phired(c->kern, c->nc, X, c->sgn, c->s1, c->s2)); }
typedef struct { const sctx *c; cdouble u; double o; cdouble E0; } ectx;
static cdouble E_piece(double s, const void *v) { const ectx *e = v; return csub(Ez(e->c, cscale(s, e->u), e->o), e->E0); }
typedef struct { const sctx *c; cdouble leg, dv; } lctx;
static cdouble E_leg(double s, const void *v) { const lctx *e = v; return Ez(e->c, csub(cadd(e->leg, cscale(s, e->dv)), CPX(e->c->a, 0)), e->c->a); }
static cdouble chi_seg(const sctx *c, cdouble v) { return cscale(0.5, cadd(ONE, dfl_erf(cscale(c->kap, csub(v, CPX(c->m, 0)))))); }

/* the length l of a descent piece of segint from v = o in the direction u, halved until the real part of the exponent (at 8
 * points along it) is at most Etop, the larger exponent of the two ends of the segment plus NOCLIMB; unchanged where it is
 * (a descent piece normally falls).  Where the exponent rises into the segment at an end, as that of the noncentral kernel
 * of case DK at v = 0 when the mass of the noncentral density lies further in, the steepest descent points out of the
 * segment and descent() clamps it to 80 degrees from the segment, where e^{-t v^2} grows like e^{Re(t) Im(v)^2} (at the
 * real point t = tau straight up): the piece climbed up to its depth L/8, with a root of X far out (c0 < 0 near zero,
 * x > 0) to e^{213} at c = 0, c0 = -0.01, g0 = 1, x = 1.5, and its terms cancelled against those of the connection back to
 * the real axis (F off by 6e74 there, by 1e-7 to 1e143 at x = 0.5 to 2, by 2.4e-3 at c = 2, x = 1.5; noclimb of DFExact.jl) */
#define NOCLIMB 4.0
static double noclimb(const sctx *c, cdouble u, double o, double l, double Etop, double L)
{
    while (l > 1e-12 * L) {
        double mx = -INFINITY;
        for (int i = 1; i <= 8; i++) { double e = creal(Ez(c, cscale(l * i / 8, u), o)); if (e != e) { mx = NAN; break; } mx = fmax(mx, e); }
        if (!(mx > Etop)) break;                                          /* (NaN: unchanged, as the maximum in DFExact.jl) */
        l /= 2;
    }
    return l;
}

/* The integral over the segment [a, b] split by the partition chi into the part near b (vr) and the part near a (vs: the
 * bump at a = 0, or vl: the edge at a), on three straight pieces: the descent pieces from a and from b and the connection
 * between their ends.  The two sums (with 1 - chi and with chi) run node by node in the order of the path, so that the path
 * is not stored; a path that exceeds the bounds of the engine leaves the bound on the thread and zero values. */
static void segint(sctx *c, int pa, int pb, double ella, double ellb, bool ws, bool wl, bool wr, double sg, int ka, int kb, double dep,
                   cdouble *vs, cdouble *vl, cdouble *vr)
{
    double a = c->a, b = c->b, L = b - a; cdouble alpha = cadd(c->tj, cscale(c->c0, c->sd));
    cdouble Eb = Ez(c, ZERO, b), Ea = Ez(c, ZERO, a);
    double dega = 80, fa = ella / 2; if (ka == 0) { dega = 45; fa = ella; }
    double degb = 80, fb = ellb / 2; if (kb == 0) { degb = 45; fb = ellb; }
    cdouble ua, ub; double la, lb;
    descent(cadd(csub(cscale(-2 * a, c->tj), cscale(dXv(c->R, a), c->sd)), ncexp_dv(c->nc, xat(c->R, ZERO, a), c->sgn, c->s1, c->s2, dXv(c->R, a))), alpha, 1, L, dep, sg, dega, &ua, &la);
    descent(cadd(csub(cscale(-2 * b, c->tj), cscale(dXv(c->R, b), c->sd)), ncexp_dv(c->nc, xat(c->R, ZERO, b), c->sgn, c->s1, c->s2, dXv(c->R, b))), alpha, -1, L, dep, sg, degb, &ub, &lb);
    /* a descent piece that climbs above the larger end of the segment is cut back: its terms would cancel against the
     * connection's */
    double Etop = (creal(Ea) != creal(Ea) || creal(Eb) != creal(Eb)) ? NAN : fmax(creal(Ea), creal(Eb)) + NOCLIMB;   /* (NaN: as max in DFExact.jl) */
    la = noclimb(c, ua, a, la, Etop, L);
    lb = noclimb(c, ub, b, lb, Etop, L);
    cdouble pa_ = cadd(CPX(a, 0), cscale(la, ua)), pb_ = cadd(CPX(b, 0), cscale(lb, ub));
    *vs = *vl = *vr = ZERO;
    pieces_t Pc = {NULL, 0, 0};
    long nv = 0;
    /* the integrand is evaluated only where the wanted parts of the partition (chi near b, 1 - chi near a) exceed 1e-15:
     * the edge part skips the nodes near a, the smooth part those near b; the sums with 1 - chi (acca) and chi (accb) */
    bool needa = ws || wl, needb = wr;
    cdouble acca = ZERO, accb = ZERO;
#define KEEP(V) (chi = chi_seg(c, (V)), (needa && cabs(csub(ONE, chi)) > 1e-15) || (needb && cabs(chi) > 1e-15))
#define ADDV(FEXPR) do { cdouble f_ = (FEXPR); if (needa) acca = cadd(acca, cmul(f_, csub(ONE, chi))); if (needb) accb = cadd(accb, cmul(f_, chi)); nv++; } while (0)
    /* the descent piece from a */
    ectx ea = {c, ua, a, Ea};
    if (pieces_nodes(E_piece, &ea, la, fmin(la / 8, fa), pa, 48 + 16 * (pa == 1), &Pc)) goto done;
    for (int ip = 0; ip < Pc.np; ip++) {
        const piece_t *P = &Pc.pc[ip]; const double *gw; const double *u = dfl_rule(P->n, &gw);
        for (int k = 0; k < P->n; k++) { double y, w; pnode(P, u, gw, k, &y, &w); cdouble chi, vc = cadd(CPX(a, 0), cscale(y, ua));
            if (KEEP(vc)) ADDV(cmul(Fz(c, cscale(y, ua), a), cscale(w, ua))); }
    }
    /* the descent piece from b, from b toward the segment: reversed */
    ectx eb = {c, ub, b, Eb};
    if (pieces_nodes(E_piece, &eb, lb, fmin(lb / 8, fb), pb, 48, &Pc)) goto done;
    for (int ip = 0; ip < Pc.np; ip++) {
        const piece_t *P = &Pc.pc[ip]; const double *gw; const double *u = dfl_rule(P->n, &gw);
        for (int k = 0; k < P->n; k++) { double y, w; pnode(P, u, gw, k, &y, &w); cdouble chi, vc = cadd(CPX(b, 0), cscale(y, ub));
            if (KEEP(vc)) ADDV(cneg(cmul(Fz(c, cscale(y, ub), b), cscale(w, ub)))); }
    }
    /* the connection: the straight line, unless the exponent rises along it above the ends; then the polyline through the
     * real axis */
    if (cabs(csub(pb_, pa_)) > 0) {
        double Emax = fmax(creal(Ea), creal(Eb)) + 2;
        cdouble legs[4] = {pa_, pb_}; int nl = 2;
        double mx = -INFINITY; for (int k = 0; k <= 32; k++) mx = fmax(mx, creal(Ez(c, csub(cadd(pa_, cscale(k / 32.0, csub(pb_, pa_))), CPX(a, 0)), a)));
        if (mx > Emax) { legs[0] = pa_; legs[1] = CPX(creal(pa_), 0); legs[2] = CPX(creal(pb_), 0); legs[3] = pb_; nl = 4; }
        if (getenv("DFL_DEBUG3")) printf("  legs %d (mx %.6g Emax %.6g Ea %.6g%+.6gi Eb %.6g%+.6gi)\n", nl, mx, Emax, creal(Ea), cimag(Ea), creal(Eb), cimag(Eb));
        for (int k = 0; k < nl - 1; k++) {
            cdouble dv = csub(legs[k + 1], legs[k]); if (cabs(dv) == 0) continue;
            lctx lc = {c, legs[k], dv};
            if (phase_nodes(E_leg, &lc, Ea, Eb, &Pc)) goto done;
            for (int ip = 0; ip < Pc.np; ip++) {
                const piece_t *P = &Pc.pc[ip]; const double *gw; const double *u = dfl_rule(P->n, &gw);
                for (int i = 0; i < P->n; i++) { double y, w; pnode(P, u, gw, i, &y, &w); cdouble chi, vc = cadd(legs[k], cscale(y, dv));
                    if (!KEEP(vc)) continue;
                    cdouble f = cabs(csub(vc, CPX(b, 0))) < cabs(csub(vc, CPX(a, 0))) ? Fz(c, csub(vc, CPX(b, 0)), b) : Fz(c, csub(vc, CPX(a, 0)), a);
                    ADDV(cmul(f, cscale(w, dv))); }
            }
        }
    }
    if (getenv("DFL_DEBUG3")) printf("segint a %.6g b %.6g sgn %d sd %.12g%+.12gi la %.12g lb %.12g ua %.12g%+.12gi ub %.12g%+.12gi pa_ %.12g%+.12gi pb_ %.12g%+.12gi nv %ld sums %.12g%+.12gi %.12g%+.12gi\n",
                                      a, b, c->sgn, creal(c->sd), cimag(c->sd), la, lb, creal(ua), cimag(ua), creal(ub), cimag(ub), creal(pa_), cimag(pa_), creal(pb_), cimag(pb_), nv,
                                      creal(acca), cimag(acca), creal(accb), cimag(accb));
#if defined(__GNUC__) || defined(__clang__)
    __atomic_add_fetch(&DFL_NCG_SEG, nv, __ATOMIC_RELAXED);
#endif
    if (needa) { if (ws) *vs = acca; else *vl = acca; }
    if (wr) *vr = accb;
done:
#undef ADDV
#undef KEEP
    free(Pc.pc);
}

/* ---- the transforms of the general kernel: Is and Ie[k] ---- */
typedef struct { const sctx *c; double a; } actx;
static cdouble A_ray(cdouble rho, const void *v) { const actx *x = v; return cmul(W_of(x->c, cadd(CPX(x->a, 0), rho)), phialg(x->c->kern, x->c->nc, xat(x->c->R, rho, x->a), x->c->sgn, x->c->s1, x->c->s2)); }
typedef struct { const actx *ax; cdouble e, be; double aa; } pctx;
static cdouble pseudo_E(double r, const void *v) { const pctx *p = v; return csub(CPX(-p->aa * r * r, 0), cscale(r, p->be)); }
typedef struct { cfn_t A; const void *Actx; cdouble alpha, beta, e; double aa; } rxctx;
static cdouble rx_f(double r, const void *v) { const rxctx *x = v; cdouble arg = csub(CPX(-x->aa * r * r, 0), cscale(r, cmul(x->beta, x->e)));
    return cmul(cmul(x->A(cscale(r, x->e), x->Actx), cexp(arg)), x->e); }
/* int_0^inf A(r) exp(-alpha r^2 - beta r) dr: along the ray of the Gaussian's decay, or by gaussexp's paths */
cdouble dfl_rayint(cdouble (*f)(double, const void *), const void *ctx, double R, double l, int n1, int n2, int p, double osc);
static cdouble rayexp(cfn_t A, const void *Actx, cdouble alpha, cdouble beta, double lA, int p)
{
    cdouble e = cexp(CPX(0, -carg(alpha) / 2)); double aa = cabs(alpha); cdouble be = cmul(beta, e);
    /* the straight ray when the exponential grows along it by at most e^2 and its phase stays below 150 radians (the
     * level-curve path of gaussexp, taken otherwise when the saddle z = beta / (2 sqrt(alpha)) has Re z >= 0, passes the
     * branch point of its square root at the distance |z| from the origin, where the Gaussian is e^{-|z|^2}: with the cap
     * of 60 radians of ray_ok it took over at |Im z| ~ 4, where that is 1e-7, and the inner transform lost seven digits
     * on a short stretch of the line, 1e-10 in the distribution function in case DK with c0 > 0 at x >= 0; at 150
     * radians the level curve begins at |Im z| ~ 10, where the branch point is below 1e-40) */
    double m = fmax(0, -creal(be));
    if (m * m / (4 * aa) <= 2 && fabs(cimag(be)) * sqrt(EXPO / aa) <= 150) {
        double Rr = sqrt(EXPO / aa);
        if (creal(be) > 0) Rr = fmin(Rr, EXPO / creal(be));
        rxctx x = {A, Actx, alpha, beta, e, aa};
        return dfl_rayint(rx_f, &x, Rr, lA, 48, 32, p, fabs(cimag(be)));
    }
    return dfl_gaussexp(A, Actx, alpha, 1.0, beta, lA, 0, p);
}
/* the ray from a for the noncentral kernel: the straight ray rho = r e in the direction of the Gaussian's decay, or, when
 * the exponential grows along it by more than e^2, the segment to the saddle x0 = -beta/(2 alpha) and the descent line from
 * it; the nodes from pieces_nodes with the full exponent (quadratic plus ncexp) */
typedef struct { const sctx *c; cdouble alpha, beta, e, x0, E0; double a; int mode; } nrctx;
static cdouble Efull_nc(const nrctx *x, cdouble rho) { const sctx *c = x->c;
    return cadd(csub(cneg(cmul(x->alpha, cmul(rho, rho))), cmul(x->beta, rho)), ncexp(c->nc, xat(c->R, rho, x->a), c->sgn, c->s1, c->s2)); }
static cdouble Ared_nc(const nrctx *x, cdouble rho) { const sctx *c = x->c;
    return cmul(W_of(c, cadd(CPX(x->a, 0), rho)), phired(c->kern, c->nc, xat(c->R, rho, x->a), c->sgn, c->s1, c->s2)); }
static cdouble Ef_nc(double r, const void *v) { const nrctx *x = v;
    if (x->mode == 0) return Efull_nc(x, cscale(r, x->e));
    if (x->mode == 1) return Efull_nc(x, cscale(r, x->x0));
    return csub(Efull_nc(x, cadd(x->x0, cscale(r, x->e))), x->E0); }
static double raylen(efn_t Ef, const void *ctx, double R0)
{
    double Rr = R0;
    for (int it = 0; it < 60; it++) { if (creal(Ef(Rr, ctx)) < -EXPO) break; Rr *= 1.5; }
    return Rr;
}
/* the full exponent's derivative along rho, and the scaled factor (S, E) at rho: Ared = S e^E */
static cdouble ncexp_dv_c(const nck_t *nc, cdouble X, int sgn, cdouble s1, cdouble s2, cdouble dXdv)   /* complex dX/dv */
{
    if (!(nc && nc->nc)) return ZERO;
    cdouble b = ncb(nc, sgn, s1, s2), Xt = sgn > 0 ? X : cneg(X), z = cmul(b, csqrt(cscale(2, Xt)));
    cdouble tz = cabs(z) < 1e-6 ? ONE : cdiv(ctanh_safe(z), z);
    return cmul(cmul(cmul(b, b), tz), sgn > 0 ? dXdv : cneg(dXdv));
}
static cdouble dEfull_nc(const nrctx *x, cdouble rho) { const sctx *c = x->c;
    cdouble dX = cadd(CPX(c->R->q, 0), cscale(2 * c->R->c0, cadd(CPX(x->a, 0), rho)));
    return cadd(csub(cscale(-2, cmul(x->alpha, rho)), x->beta), ncexp_dv_c(c->nc, xat(c->R, rho, x->a), c->sgn, c->s1, c->s2, dX)); }
static cdouble Ared_sc(const nrctx *x, cdouble rho, double *E) { const sctx *c = x->c;
    cdouble X = xat(c->R, rho, x->a), Wv = W_of(c, cadd(CPX(x->a, 0), rho));
    if (c->kern == 1) { *E = 0; return cmul(Wv, phired(c->kern, c->nc, X, c->sgn, c->s1, c->s2)); }   /* case DK: elementary, bounded */
    cdouble S = phialg_scaled(c->nc, X, c->sgn, c->s1, c->s2, E), ne = ncexp(c->nc, X, c->sgn, c->s1, c->s2);
    *E -= creal(ne);                                                   /* the real part; the phase stays in S */
    return cmul(cmul(Wv, S), cexp(CPX(0, -cimag(ne)))); }
/* a straight piece z0 + r d, r in [0, L]: the exponent relative to z0 */
typedef struct { const nrctx *x; cdouble z0, d, Ez0; } spctx;
static cdouble E_sp(double r, const void *v) { const spctx *p = v; return csub(Efull_nc(p->x, cadd(p->z0, cscale(r, p->d))), p->Ez0); }
/* the ray when the full exponent rises along the real axis by more than 2 above its value at the edge (the mass of the
 * noncentral density beyond the breakpoint): the straight piece from 0 to the saddle of the full exponent (Newton on E'
 * from the maximum of Re E on the real axis), then straight pieces in the direction of steepest descent at their start,
 * renewed where the descent along a piece ends, until E has fallen by EXPO.  Returns false when there is no such rise. */
static bool ray_saddle(const nrctx *x, double first, int p, double Rmax, cdouble *out)
{
    double E0 = creal(Efull_nc(x, ZERO)), Emax = -INFINITY, rmax = 0;
    for (int k = 0; k <= 80; k++) { double r = Rmax * pow(1.25, k - 80), Er = creal(Efull_nc(x, CPX(r, 0))); if (Er > Emax) { Emax = Er; rmax = r; } }
    if (!(Emax - E0 > 2)) return false;
    /* the maximum on the real axis refined (Newton on Re E' along the axis, within the scan's cell), so that the saddle is
     * judged against the true height of the bump */
    double rr = rmax;
    for (int it = 0; it < 20; it++) {
        double h = 1e-6 * rr, d1 = creal(dEfull_nc(x, CPX(rr, 0))), d2 = creal(csub(dEfull_nc(x, CPX(rr + h, 0)), dEfull_nc(x, CPX(rr - h, 0)))) / (2 * h);
        if (!(d2 < 0)) break;
        double rn = rr - d1 / d2; if (!(rn > rmax / 1.25 && rn < rmax * 1.25)) break;
        if (fabs(rn - rr) < 1e-10 * rr) { rr = rn; break; }
        rr = rn;
    }
    Emax = fmax(Emax, creal(Efull_nc(x, CPX(rr, 0)))); rmax = rr;
    cdouble rho = CPX(rmax, 0);
    for (int it = 0; it < 30; it++) {
        cdouble d1 = dEfull_nc(x, rho); double h = 1e-6 * cabs(rho);
        cdouble d2 = cscale(1 / (2 * h), csub(dEfull_nc(x, cadd(rho, CPX(h, 0))), dEfull_nc(x, csub(rho, CPX(h, 0)))));
        if (cabs(d2) == 0) break;
        cdouble step = cneg(cdiv(d1, d2));
        if (cabs(step) > cabs(rho)) step = cscale(cabs(rho) / cabs(step), step);
        rho = cadd(rho, step);
        if (cabs(step) < 1e-10 * cabs(rho)) break;
    }
    /* the saddle serves only if it is the bump of the real axis slightly displaced: one far above it (Re E beyond the
     * real-axis maximum) or far from it is not crossed by the real-axis path, and the rotated ray serves instead */
    if (!(isfinite(creal(rho)) && isfinite(cimag(rho)) && creal(rho) > 0 && creal(Efull_nc(x, rho)) > E0 + 1
          && creal(Efull_nc(x, rho)) <= Emax + 1 && fabs(cimag(rho)) <= 0.5 * fabs(creal(rho)))) {
        /* a small rise: the rotated ray serves (its cancellation is at most e^{rise}); a large one: the real-axis maximum
         * is the pivot, and the descent pieces start from it */
        if (!(Emax - E0 > 10)) return false;
        rho = CPX(rmax, 0);
    }
    cdouble val = ZERO; pieces_t Pc = {NULL, 0, 0};
    cdouble E_s = Efull_nc(x, rho);
    spctx sp = {x, ZERO, rho, ZERO};                                   /* lam in [0, 1] along rho */
    /* (a path beyond the bounds of the engine: the bound stays on the thread, the value is not used) */
    if (pieces_nodes(E_sp, &sp, 1.0, fmin(1.0 / 8, first / cabs(rho)), p, 48, &Pc)) { free(Pc.pc); *out = ZERO; return true; }
    for (int ip = 0; ip < Pc.np; ip++) {
        const piece_t *P = &Pc.pc[ip]; const double *gw; const double *u = dfl_rule(P->n, &gw);
        for (int k = 0; k < P->n; k++) { double y, w; pnode(P, u, gw, k, &y, &w); cdouble z = cscale(y, rho); double E; cdouble S = Ared_sc(x, z, &E);
            val = cadd(val, cscale(w, cmul(S, cexp(cadd(Efull_nc(x, z), CPX(E, 0)))))); }
    }
    val = cmul(val, rho);
    if (getenv("DFL_DEBUG4") && !(creal(val) == creal(val))) printf("  NaN already on the piece to the saddle: rho %g%+gi pieces %d\n", creal(rho), cimag(rho), Pc.np);
    cdouble z0 = rho; double h = 1e-6 * cabs(rho);
    cdouble d2 = cscale(1 / (2 * h), csub(dEfull_nc(x, cadd(rho, CPX(h, 0))), dEfull_nc(x, csub(rho, CPX(h, 0)))));
    cdouble d = cabs(d2) > 0 ? cexp(CPX(0, (M_PI - carg(d2)) / 2)) : ONE; if (creal(d) < 0) d = cneg(d);
    for (int piece = 0; piece < 16; piece++) {
        cdouble Ez0 = Efull_nc(x, z0);
        spctx sq = {x, z0, d, Ez0};
        /* the length: along the line while the exponent falls; stop where it has fallen by EXPO, or at the last point before
         * it rises (the direction is renewed there from the gradient), or at 10 Rmax */
        double L = 1e-3 * cabs(rho), Lend = NAN, Eprev = 0;
        for (int it = 0; it < 200; it++) {
            double Ec = creal(E_sp(L, &sq));
            if (Ec < -EXPO) { Lend = L; break; }
            if (!(Ec == Ec) || Ec > Eprev) { Lend = it == 0 ? L : L / 1.4; break; }
            if (L > 10 * Rmax) { Lend = L; break; }
            Eprev = Ec; L *= 1.4;
        }
        if (isnan(Lend)) Lend = L;
        if (pieces_nodes(E_sp, &sq, Lend, fmin(Lend / 8, first), 1, 48, &Pc)) { free(Pc.pc); *out = ZERO; return true; }
        for (int ip = 0; ip < Pc.np; ip++) {
            const piece_t *P = &Pc.pc[ip]; const double *gw; const double *u = dfl_rule(P->n, &gw);
            for (int k = 0; k < P->n; k++) { double y, w; pnode(P, u, gw, k, &y, &w); cdouble z = cadd(z0, cscale(y, d)); double E; cdouble S = Ared_sc(x, z, &E);
                val = cadd(val, cscale(w, cmul(cmul(S, cexp(cadd(cadd(E_sp(y, &sq), Ez0), CPX(E, 0)))), d))); }
        }
        if (getenv("DFL_DEBUG4")) printf("  piece %d: z0 %g%+gi d %g%+gi Lend %g pieces %d val %g%+gi Ez0 %g%+gi\n", piece, creal(z0), cimag(z0), creal(d), cimag(d), Lend, Pc.np, creal(val), cimag(val), creal(Ez0), cimag(Ez0));
        z0 = cadd(z0, cscale(Lend, d));
        if (creal(csub(Efull_nc(x, z0), E_s)) < -EXPO) break;
        cdouble g = dEfull_nc(x, z0); if (cabs(g) == 0) break;
        d = cscale(1 / cabs(g), cneg(conj(g)));
    }
    free(Pc.pc);
    if (getenv("DFL_DEBUG4") && !(creal(val) == creal(val))) {
        cdouble z = rho; double E; cdouble S = Ared_sc(x, z, &E);
        printf("ray_saddle NaN: alpha %g%+gi beta %g%+gi rho_s %g%+gi E_s %g%+gi rmax-scan Emax %g E0 %g S(rho_s) %g%+gi E %g d %g%+gi\n",
               creal(x->alpha), cimag(x->alpha), creal(x->beta), cimag(x->beta), creal(rho), cimag(rho), creal(E_s), cimag(E_s), Emax, E0, creal(S), cimag(S), E, creal(d), cimag(d));
    }
    *out = val;
    return true;
}
static cdouble rayexp_nc(const sctx *c, double a, cdouble alpha, cdouble beta, double first, int p)
{
    cdouble e = cexp(CPX(0, -carg(alpha) / 2)); double aa = cabs(alpha); cdouble be = cmul(beta, e);
    nrctx x = {c, alpha, beta, e, ZERO, ZERO, a, 0};
    pieces_t Pc = {NULL, 0, 0};
    cdouble val = ZERO;
    /* (a path beyond the bounds of the engine: the bound stays on the thread, the value is not used) */
#define RAYSUM(ACC, RHO) for (int ip = 0; ip < Pc.np; ip++) { \
        const piece_t *P = &Pc.pc[ip]; const double *gw; const double *u = dfl_rule(P->n, &gw); \
        for (int k = 0; k < P->n; k++) { double y, w; pnode(P, u, gw, k, &y, &w); cdouble rho = (RHO); double E; cdouble S = Ared_sc(&x, rho, &E); \
            ACC = cadd(ACC, cscale(w, cmul(S, cexp(cadd(Efull_nc(&x, rho), CPX(E, 0)))))); } }
    if (fmax(0, -creal(be)) * fmax(0, -creal(be)) / (4 * aa) <= 2) {
        double Rr = raylen(Ef_nc, &x, (-creal(be) + sqrt(creal(be) * creal(be) + 4 * aa * EXPO)) / (2 * aa));
        if (ray_saddle(&x, first, p, Rr, &val)) return val;
        if (pieces_nodes(Ef_nc, &x, Rr, fmin(Rr / 8, first), p, 48, &Pc)) { free(Pc.pc); return ZERO; }
        RAYSUM(val, cscale(y, e))
#if defined(__GNUC__) || defined(__clang__)
        { long nn = 0; for (int ip = 0; ip < Pc.np; ip++) nn += Pc.pc[ip].n; __atomic_add_fetch(&DFL_NCG_RAY, nn, __ATOMIC_RELAXED); }
#endif
        free(Pc.pc);
        return cmul(val, e);
    }
    x.x0 = cdiv(cneg(beta), cscale(2, alpha)); x.mode = 1;
    if (pieces_nodes(Ef_nc, &x, 1.0, fmin(1.0 / 8, first / cabs(x.x0)), p, 48, &Pc)) { free(Pc.pc); return ZERO; }
    RAYSUM(val, cscale(y, x.x0))
    val = cmul(val, x.x0);
    x.E0 = Efull_nc(&x, x.x0); x.mode = 2;
    double Rr = raylen(Ef_nc, &x, sqrt(EXPO / aa));
    if (pieces_nodes(Ef_nc, &x, Rr, fmin(Rr / 8, first), 1, 48, &Pc)) { free(Pc.pc); return ZERO; }
    cdouble v2 = ZERO;
    RAYSUM(v2, cadd(x.x0, cscale(y, e)))
#undef RAYSUM
    free(Pc.pc);
    return cadd(val, cmul(v2, e));
}
static void inner_gen(int kern, const nck_t *nc, cdouble tj, const breaks_t *R, cdouble s1, cdouble s2, cdouble M0, int kind, int part, cdouble *Is, cdouble *Ie)
{
    double q = R->q, c0 = R->c0; int ne = R->ne;
    double sg = cimag(tj) >= 0 ? 1.0 : -1.0;
    *Is = ZERO; for (int k = 0; k < 2; k++) Ie[k] = ZERO;
    bool wants = part == 0 || part == 1;
#define WANT(k) ((part == 0 || part == 2 || part == 10 + (k) + 1) && !R->negl[k])
    if (kind == 2 && wants) *Is = (nc && nc->nc) ? psi_half_nc(nc, s1, s2, M0) : dfl_psi_half(kern, s1, s2, M0);
    double lX = 1 / (cabs(kern == 1 ? s1 : cscale(0.5, csub(s2, s1))) + 2);
    double ed[4] = {0, ne > 0 ? R->b[0] : INFINITY, ne > 1 ? R->b[1] : INFINITY, INFINITY};
    for (int j = 0; j <= ne; j++) {
        double a = ed[j], bb = ed[j + 1]; int sgn = R->sgn[j];
        if (kern == 1 && sgn < 0) continue;
        cdouble sd = (kern == 1 || sgn < 0) ? s1 : s2;
        sctx c = {kern, kind, tj, sd, s1, s2, sgn, R, q, c0, a, bb, ZERO, ZERO, 0, 0, nc};
        if (!isfinite(bb)) {
            if (j == 0) { if (!wants) continue; } else { if (!WANT(j - 1)) continue; }
            cdouble alpha = cadd(tj, cscale(c0, sd)), beta = cadd(cscale(2 * a, alpha), cscale(q, sd));
            cdouble cst = cexp(csub(CPX(-creal(tj) * a * a, -cimag(tj) * a * a), cmul(sd, xat(R, ZERO, a))));
            actx ax = {&c, a};
            double lv = lX / fmax(fabs(dXv(R, a)), 1e-300);
            if (c0 != 0) lv = fmin(lv, sqrt(lX / fabs(c0)));
            lv = fmin(lv, R->ell0 / 16);
            int p = 1; cdouble val;
            if (nc && nc->nc) {
                double first = lv;
                if (j > 0 && R->kindb[j - 1] == 0) first = R->ell[j - 1];
                else if (j > 0) { lv = fmin(lv, R->ell[j - 1] / 16); first = lv; p = 4; }
                val = cmul(cst, rayexp_nc(&c, a, alpha, beta, first, p));
            } else if (j > 0 && R->kindb[j - 1] == 0) {
                cdouble e = cexp(CPX(0, -carg(alpha) / 2)); double aa = cabs(alpha); cdouble be = cmul(beta, e);
                double Rr = (-creal(be) + sqrt(creal(be) * creal(be) + 4 * aa * EXPO)) / (2 * aa);
                pctx pc = {&ax, e, be, aa};
                pieces_t Pc = {NULL, 0, 0};
                pieces_nodes(pseudo_E, &pc, Rr, fmin(Rr / 8, R->ell[j - 1]), 1, 48, &Pc);   /* (refused: no pieces, the bound on the thread) */
                val = ZERO;
                for (int ip = 0; ip < Pc.np; ip++) {
                    const piece_t *P = &Pc.pc[ip]; const double *gw; const double *u = dfl_rule(P->n, &gw);
                    for (int k = 0; k < P->n; k++) { double y, w; pnode(P, u, gw, k, &y, &w); val = cadd(val, cscale(w, cmul(A_ray(cscale(y, e), &ax), cexp(pseudo_E(y, &pc))))); }
                }
                val = cmul(cmul(val, cst), e);
                free(Pc.pc);
            } else {
                if (j > 0) { lv = fmin(lv, R->ell[j - 1] / 16); p = 4; }
                val = cmul(cst, rayexp(A_ray, &ax, alpha, beta, lv, p));
            }
            if (j == 0) *Is = cadd(*Is, val); else Ie[j - 1] = cadd(Ie[j - 1], val);
        } else {
            double L = bb - a, h = L / 2, kap = KAPPA / h, m = a + h, dep = L / 8;
            if (j == 0 && R->vsad > 0 && R->vsad < bb) {
                m = fmax(m, (R->vsad + bb) / 2); kap = fmax(kap, KAPPA / (bb - m));
                kap = fmin(kap, 4 * KAPPA / L); m = fmin(m, bb - KAPPA / kap);
                dep = fmin(dep, KAPPA / 4 / kap);
            }
            c.kap = kap; c.m = m;
            int pa = j > 0 ? 4 : 1;
            bool ws_ = j == 0 && wants, wl_ = j > 0 && WANT(j - 1), wr_ = WANT(j);
            if (!(ws_ || wl_ || wr_)) continue;
            double ella = j > 0 ? R->ell[j - 1] : R->ell0; int ka = j > 0 ? R->kindb[j - 1] : 1;
            cdouble vs_, vl_, vr_;
            segint(&c, pa, 1 + 3 * R->kindb[j], ella, R->ell[j], ws_, wl_, wr_, sg, ka, R->kindb[j], dep, &vs_, &vl_, &vr_);
            if (ws_) *Is = cadd(*Is, vs_);
            if (wl_) Ie[j - 1] = cadd(Ie[j - 1], vl_);
            if (wr_) Ie[j] = cadd(Ie[j], vr_);
        }
    }
#undef WANT
}

/* the saddle of the exponent on the first segment at t = tau + i */
static double saddle_loc(int model, double c, double q, double c0, double tau, bool stat, int kern, const breaks_t *R)
{
    if (R->ne == 0) return NAN;
    cdouble ts[2] = {CPX(tau, 0), CPX(tau, 1)}, P[2], s1[2], s2[2], M0[2];
    dfl_tparts(model, c, 2, ts, stat, P, s1, s2, M0);
    cdouble sd = (kern == 1 || R->sgn[0] < 0) ? s1[1] : s2[1];
    cdouble vs = cdiv(cscale(-q, sd), cscale(2, cadd(CPX(tau, 1), cscale(c0, sd))));
    return (creal(vs) > 0 && creal(vs) < R->b[0] && fabs(cimag(vs)) < 0.2 * (R->b[0] - creal(vs))) ? creal(vs) : NAN;
}
static void saddle_phase_gen(int kern, double q, double c0, int n, const cdouble *ts, const cdouble *s1, const cdouble *s2, const breaks_t *R, double *psi)
{
    for (int j = 0; j < n; j++) {
        cdouble sd = (kern == 1 || R->sgn[0] < 0) ? s1[j] : s2[j];
        cdouble alpha = cadd(ts[j], cscale(c0, sd)), beta = cscale(q, sd), z2 = cdiv(cmul(beta, beta), cscale(4, alpha));
        psi[j] = NAN;
        if (R->ne == 0) { cdouble z = cdiv(beta, cscale(2, csqrt(alpha))); if (creal(z) < 0 && creal(z2) > -30) psi[j] = cimag(z2); }
        else if (!isnan(R->vsad)) { cdouble vs = cdiv(cneg(beta), cscale(2, alpha)); if (creal(vs) > 0 && creal(vs) < R->b[0] && creal(z2) > -30) psi[j] = cimag(z2); }
    }
}
static double max_step(int n, const double *psi) { double m = 0; for (int j = 0; j + 1 < n; j++) if (!isnan(psi[j]) && !isnan(psi[j + 1])) m = fmax(m, fabs(psi[j + 1] - psi[j])); return m; }

/* ---- the outer integral for the general kernel ---- */
typedef struct { int kern, kind, part, k; const breaks_t *R; const cdouble *ts, *s1, *s2, *M0, *P; const double *wt; double *g; cdouble *G; double y;
                 int model; double c, g0; dfl_bound_t *fail; } gctx;
/* the transforms at the node j (ts[j]), with the noncentral kernel of a fixed start when g0 != 0 */
static void node_inner(const gctx *o, int j, int part, cdouble *Is, cdouble *Ie)
{
    if (o->g0 == 0) { inner_gen(o->kern, NULL, o->ts[j], o->R, o->s1[j], o->s2[j], o->M0[j], o->kind, part, Is, Ie); return; }
    nck_t K; nckernel(o->model, o->c, o->ts[j], o->g0, o->s1[j], o->s2[j], &K);
    inner_gen(o->kern, &K, o->ts[j], o->R, o->s1[j], o->s2[j], cmul(K.pref, o->M0[j]), o->kind, part, Is, Ie);
}
/* (a node whose inner integrals meet a bound of the engine records it in fail, and the other nodes are skipped) */
static void gsmooth_body(int j, void *v) { gctx *o = v; cdouble Is, Ie[2];
    if (dfl_bound_failed(o->fail)) return;
    dfl_bound_clear();
    node_inner(o, j, 1, &Is, Ie);
    if (dfl_bound_note(o->fail)) return;
    cdouble val = cmul(o->P[j], Is); if (o->kind == 2) val = cdiv(val, o->ts[j]); o->g[j] = creal(val) * o->wt[j]; }
static void gedge_body(int j, void *v) { gctx *o = v; cdouble Is, Ie[2], t = o->ts[j + 1];
    if (dfl_bound_failed(o->fail)) return;
    dfl_bound_clear();
    node_inner(o, j + 1, 10 + o->k + 1, &Is, Ie);
    if (dfl_bound_note(o->fail)) return;
    cdouble val = cmul(o->P[j + 1], Ie[o->k]); if (o->kind == 2) val = cdiv(val, t); o->G[j] = cmul(val, cexp(cscale(o->y, t))); }
static void guni_body(int j, void *v) { gctx *o = v; cdouble Is, Ie[2];
    if (dfl_bound_failed(o->fail)) return;
    dfl_bound_clear();
    node_inner(o, j, 0, &Is, Ie);
    if (dfl_bound_note(o->fail)) return;
    cdouble val = cmul(o->P[j], cadd(Is, cadd(Ie[0], Ie[1]))); if (o->kind == 2) val = cdiv(val, o->ts[j]); o->g[j] = creal(val); }

/* body(j) for j = 0..n-1 with the last node first: on the line of the outer integral the paths of the inner integrals grow
 * with Im t, so that a point whose paths exceed the bounds of the engine meets them at its last node, before the other
 * nodes are evaluated (tmap_last! of DFExact.jl); the values are those of dfl_parfor */
static void parfor_last(int n, void (*body)(int j, void *ctx), void *ctx, const dfl_bound_t *fail)
{
    if (n <= 0) return;
    body(n - 1, ctx);
    if (!fail->bound) dfl_parfor(n - 1, body, ctx);
}
/* the value in *out; 0, or the bound of the engine that a path of the inner integrals would exceed (recorded in *bd, *out
 * NaN) */
static int outer_gen(int kind, int model, double c, double q, double c0, double g0, breaks_t *R, double tau, bool stat, const dfl_single_opts *o, double *out, dfl_bound_t *bd)
{
    dfl_bound_t fail = {0, 0, NULL};
    *out = NAN; *bd = fail;
    int kern = dfl_kern(model, stat); double hu = o->hu * dfl_hufac(c), cde = tau, Wm = dfl_wmaxc_m(model, c); int N = o->N;
    double umax = asinh(Wm / cde); int nu = (int) floor(umax / hu) + 1;
    cdouble *ts = malloc(nu * sizeof(cdouble)), *P = malloc(nu * sizeof(cdouble)), *s1 = malloc(nu * sizeof(cdouble)), *s2 = malloc(nu * sizeof(cdouble)), *M0 = malloc(nu * sizeof(cdouble));
    double *wt = malloc(nu * sizeof(double)), *g = malloc(nu * sizeof(double)), *psi = malloc(nu * sizeof(double));
    for (int j = 0; j < nu; j++) { double u = j * hu; ts[j] = CPX(tau, cde * sinh(u)); wt[j] = cde * cosh(u); }
    dfl_tparts(model, c, nu, ts, stat, P, s1, s2, M0);
    saddle_phase_gen(kern, q, c0, nu, ts, s1, s2, R, psi);
    double val = NAN;
    if (max_step(nu, psi) > 0.25) {
        double w0 = fmin(0.5, tau / 2); int ne = (int) ceil(log2(Wm / w0)) + 2;
        double *edges0 = malloc(ne * sizeof(double)); edges0[0] = 0; for (int p = 1; p < ne; p++) edges0[p] = w0 * pow(2, p - 1); edges0[ne - 1] = Wm;
        int K = 8, nk = (ne - 1) * K + 1;
        cdouble *te0 = malloc(nk * sizeof(cdouble)), *P0 = malloc(nk * sizeof(cdouble)), *s1e = malloc(nk * sizeof(cdouble)), *s2e = malloc(nk * sizeof(cdouble)), *M0e = malloc(nk * sizeof(cdouble));
        for (int p = 0; p < ne - 1; p++) for (int k = 0; k < K; k++) te0[p * K + k] = CPX(tau, edges0[p] + (edges0[p + 1] - edges0[p]) * k / K);
        te0[nk - 1] = CPX(tau, Wm);
        dfl_tparts(model, c, nk, te0, stat, P0, s1e, s2e, M0e);
        double *psie = malloc(nk * sizeof(double)); saddle_phase_gen(kern, q, c0, nk, te0, s1e, s2e, R, psie);
        int cap_n = 0; int *mm = malloc(ne * sizeof(int)), *nn_ = malloc(ne * sizeof(int));
        for (int p = 0; p < ne - 1; p++) { double dp = 0; for (int k = 0; k < K; k++) { int i = p * K + k; if (!isnan(psie[i]) && !isnan(psie[i + 1])) dp += fabs(psie[i + 1] - psie[i]); }
            mm[p] = (int) fmax(1, ceil(dp / 50)); nn_[p] = (int) fmax(24, 16 * ceil(ceil(3 * dp / mm[p]) / 16)) * DFL_NODEFAC; cap_n += mm[p] * nn_[p]; }
        cdouble *tn = malloc((cap_n + 1) * sizeof(cdouble)); double *wn = malloc(cap_n * sizeof(double)); int nn = 0;
        for (int p = 0; p < ne - 1; p++) {
            double a = edges0[p], b = edges0[p + 1]; const double *gw; const double *u = dfl_rule(nn_[p], &gw);
            for (int k = 0; k < mm[p]; k++) { double aa = a + (b - a) * k / mm[p], bb = a + (b - a) * (k + 1) / mm[p];
                for (int i = 0; i < nn_[p]; i++) { tn[nn + 1] = CPX(tau, aa + (bb - aa) * u[i]); wn[nn] = (bb - aa) * gw[i]; nn++; } }
        }
        tn[0] = CPX(tau, 0);
        cdouble *Pn = malloc((nn + 1) * sizeof(cdouble)), *s1n = malloc((nn + 1) * sizeof(cdouble)), *s2n = malloc((nn + 1) * sizeof(cdouble)), *M0n = malloc((nn + 1) * sizeof(cdouble));
        dfl_tparts(model, c, nn + 1, tn, stat, Pn, s1n, s2n, M0n);
        double *gn = malloc(nn * sizeof(double));
        gctx on = {kern, kind, 1, 0, R, tn + 1, s1n + 1, s2n + 1, M0n + 1, Pn + 1, wn, gn, NULL, 0, model, c, g0, &fail};
        parfor_last(nn, gsmooth_body, &on, &fail);
        if (!fail.bound) { val = 0; for (int j = 0; j < nn; j++) val += gn[j]; val /= M_PI; }
        if (getenv("DFL_DEBUG") && !fail.bound) { int j0 = 0; for (int p = 0; p < ne - 1; p++) { double sp = 0, dp = 0; for (int k = 0; k < K; k++) { int i = p * K + k; if (!isnan(psie[i]) && !isnan(psie[i + 1])) dp += fabs(psie[i + 1] - psie[i]); }
            for (int k = 0; k < mm[p] * nn_[p]; k++) sp += gn[j0++]; printf("    panel %d [%.4g, %.4g] phase %.3g sub %d nodes %d psi0 %.3g sum %.15g\n", p, edges0[p], edges0[p + 1], dp, mm[p], nn_[p], psie[p * K], sp / M_PI); } }
        free(edges0); free(te0); free(P0); free(s1e); free(s2e); free(M0e); free(psie); free(mm); free(nn_); free(tn); free(wn); free(Pn); free(s1n); free(s2n); free(M0n); free(gn);
    } else {
        gctx oc = {kern, kind, 1, 0, R, ts, s1, s2, M0, P, wt, g, NULL, 0, model, c, g0, &fail};
        parfor_last(nu, gsmooth_body, &oc, &fail);
        if (!fail.bound) { g[0] *= 0.5; val = 0; for (int j = 0; j < nu; j++) val += g[j]; val *= hu / M_PI; }
    }
    free(ts); free(P); free(s1); free(s2); free(M0); free(wt); free(g); free(psi);
    if (fail.bound) { *bd = fail; return fail.bound; }
    if (getenv("DFL_DEBUG")) printf("  smooth part %.15g (nu %d)\n", val, nu);
    /* one Filon part per breakpoint */
    const double *gw; const double *u = dfl_rule(N, &gw);
    for (int k = 0; k < R->ne; k++) {
        if (R->negl[k]) continue;
        double y = R->y[k], w0 = fmin(fmin(0.5, 1 / y), tau / 2);
        int ne0 = (int) ceil(log2(Wm / w0)) + 2;
        double *edges0 = malloc(ne0 * sizeof(double)); edges0[0] = 0; for (int p = 1; p < ne0; p++) edges0[p] = w0 * pow(2, p - 1); edges0[ne0 - 1] = Wm;
        int cap_e = 1; for (int p = 0; p < ne0 - 1; p++) { double a = edges0[p], b = edges0[p + 1]; cap_e += dfl_filonm(a, b, y, c); }
        double *orig = malloc((cap_e + 1) * sizeof(double)); int no = 0; orig[0] = 0;
        for (int p = 0; p < ne0 - 1; p++) { double a = edges0[p], b = edges0[p + 1]; int m = dfl_filonm(a, b, y, c); for (int i = 1; i <= m; i++) orig[++no] = a + (b - a) * i / m; }
        int pf = o->pfac > 1 ? o->pfac : 1; int np = no * pf;
        double *edges = malloc((np + 1) * sizeof(double)); edges[0] = 0;
        for (int p = 0; p < no; p++) for (int r = 1; r <= pf; r++) edges[p * pf + r] = orig[p] + (orig[p + 1] - orig[p]) * r / pf;
        int nw = np * N;
        cdouble *te = malloc((nw + 1) * sizeof(cdouble)); te[0] = CPX(tau, 0);
        for (int p = 0; p < np; p++) for (int i = 0; i < N; i++) te[1 + p * N + i] = CPX(tau, edges[p] + (edges[p + 1] - edges[p]) * u[i]);
        cdouble *Pe = malloc((nw + 1) * sizeof(cdouble)), *s1e = malloc((nw + 1) * sizeof(cdouble)), *s2e = malloc((nw + 1) * sizeof(cdouble)), *M0e = malloc((nw + 1) * sizeof(cdouble));
        dfl_tparts(model, c, nw + 1, te, stat, Pe, s1e, s2e, M0e);
        cdouble *G = malloc(nw * sizeof(cdouble));
        gctx oe = {kern, kind, 2, k, R, te, s1e, s2e, M0e, Pe, NULL, NULL, G, y, model, c, g0, &fail};
        parfor_last(nw, gedge_body, &oe, &fail);
        if (fail.bound) { free(edges0); free(orig); free(edges); free(te); free(Pe); free(s1e); free(s2e); free(M0e); free(G); *bd = fail; return fail.bound; }
        cdouble acc = ZERO;
        for (int p = 0; p < np; p++) { cdouble fp = dfl_filon_panel(G + p * N, edges[p], edges[p + 1], y, N); acc = cadd(acc, fp);
            if (getenv("DFL_DEBUG2")) printf("    panel %d [%.6g, %.6g] %.15g%+.15gi  G0 %.15g%+.15gi\n", p, edges[p], edges[p + 1], creal(fp), cimag(fp), creal(G[p * N]), cimag(G[p * N])); }
        if (getenv("DFL_DEBUG")) printf("  edge part %d: y %.15g panels %d nodes %d value %.15g\n", k, y, np, nw, creal(cmul(cexp(CPX(-tau * y, 0)), acc)) / M_PI);
        val += creal(cmul(cexp(CPX(-tau * y, 0)), acc)) / M_PI;
        free(edges0); free(orig); free(edges); free(te); free(Pe); free(s1e); free(s2e); free(M0e); free(G);
    }
    *out = val;
    return 0;
}

/* the value of the density (kind 1) or the distribution function (kind 2) through the general kernel (c0 != 0, a fixed
 * start g0 != 0, or gen); in case DK with c0 > 0 the density is infinite at x = -sqrt(2 c0 (1 + g0^2)).  0 and the value in
 * *out, or the bound of the engine that a path of the inner integrals would exceed (*bd), with *out NaN */
int dfl_single_value_gen_b(int kind, double x, int model, double c, double c0, double g0, bool stat, const dfl_single_opts *o, double *out, dfl_bound_t *bd)
{
    dfl_bound_t none = {0, 0, NULL}; *bd = none;
    dfl_bound_clear();
    double q = (x == 0 && c0 == 0) ? 1e-12 : x;   /* with c0 != 0 the parabola has no root at v = 0 and x = 0 is taken exactly; the general kernel forced at c0 = 0 keeps the substitution */
    double tau = o->tau > 0 ? o->tau : dfl_default_tau(model, c, stat);
    /* case DK with a fixed start: the line at most half of t_* also for c <= 0 (e^{g0^2 Q(0,t)} has an essential singularity at t_*,
     * 1.23 at c = 0), and the outer step hu times 9/g0^2 for g0 > 3 (the phase of that factor grows with g0^2; default_tau
     * with g0 and value_gen in DFExact.jl) */
    dfl_single_opts og = *o;
    if (model == 1 && g0 != 0 && !stat) {
        if (!(o->tau > 0)) { double ts = dfl_tstar(1, c, false); if (ts / 2 < tau) tau = ts / 2; }
        if (g0 * g0 > 9) og.hu = o->hu * 9 / (g0 * g0);
    }
    o = &og;
    DFL_NODEFAC = o->nodefac > 0 ? o->nodefac : 1;
    int kern = dfl_kern(model, stat);
    if (stat) g0 = 0;
    double x0 = (g0 != 0 && model == 1) ? (1 + g0 * g0) / 2 : 0.5;
    breaks_t R; breakpts(q, c0, tau, x0, &R);
    if (R.dbl && kern == 1 && kind == 1) { *out = INFINITY; return 0; }
    R.vsad = saddle_loc(model, c, q, c0, tau, stat, kern, &R);
    if (o->uniform_hw > 0) {
        double hw = o->uniform_hw; int n = (int) floor(WMAX / hw) + 1;
        cdouble *t = malloc(n * sizeof(cdouble)), *P = malloc(n * sizeof(cdouble)), *s1 = malloc(n * sizeof(cdouble)), *s2 = malloc(n * sizeof(cdouble)), *M0 = malloc(n * sizeof(cdouble));
        double *g = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) t[j] = CPX(tau, j * hw);
        dfl_tparts(model, c, n, t, stat, P, s1, s2, M0);
        dfl_bound_t fail = {0, 0, NULL};
        gctx u = {kern, kind, 0, 0, &R, t, s1, s2, M0, P, NULL, g, NULL, 0, model, c, g0, &fail};
        dfl_parfor(n, guni_body, &u);
        g[0] *= 0.5; double v = 0; for (int j = 0; j < n; j++) v += g[j]; v *= hw / M_PI;
        free(t); free(P); free(s1); free(s2); free(M0); free(g);
        *bd = fail; *out = fail.bound ? NAN : v;
        return fail.bound;
    }
    return outer_gen(kind, model, c, q, c0, g0, &R, tau, stat, o, out, bd);
}
double dfl_single_value_gen(int kind, double x, int model, double c, double c0, double g0, bool stat, const dfl_single_opts *o)
{
    double v; dfl_bound_t bd;
    return dfl_single_value_gen_b(kind, x, model, c, c0, g0, stat, o, &v, &bd) ? NAN : v;
}

/* debugging aid: the breakpoints and the transforms at one node */
void dfl_debug_inner_gen(int model, double c, double c0, double x, bool stat, int kind, double tre, double tim, int part)
{
    double q = x == 0 ? 1e-12 : x, tau = dfl_default_tau(model, c, stat);
    int kern = dfl_kern(model, stat);
    breaks_t R; breakpts(q, c0, tau, 0.5, &R); R.vsad = saddle_loc(model, c, q, c0, tau, stat, kern, &R);
    printf("tau %.15g ne %d b %.15g %.15g y %.15g %.15g ell %.15g %.15g kindb %d %d ell0 %.15g sgn %d %d %d negl %d %d dbl %d vsad %.15g\n",
           tau, R.ne, R.b[0], R.b[1], R.y[0], R.y[1], R.ell[0], R.ell[1], R.kindb[0], R.kindb[1], R.ell0, R.sgn[0], R.sgn[1], R.sgn[2], R.negl[0], R.negl[1], R.dbl, R.vsad);
    cdouble ts[2] = {CPX(tau, 0), CPX(tre, tim)}, P[2], s1[2], s2[2], M0[2];
    dfl_tparts(model, c, 2, ts, stat, P, s1, s2, M0);
    cdouble Is, Ie[2];
    dfl_bound_clear();
    inner_gen(kern, NULL, ts[1], &R, s1[1], s2[1], M0[1], kind, part, &Is, Ie);
    if (dfl_bound_hit()) printf("(a path of the inner integrals exceeds the bounds of the engine)\n");
    printf("part %d t %.15g%+.15gi s1 %.15g%+.15gi s2 %.15g%+.15gi M0 %.15g%+.15gi\nIs %.15g%+.15gi Ie1 %.15g%+.15gi Ie2 %.15g%+.15gi\n",
           part, creal(ts[1]), cimag(ts[1]), creal(s1[1]), cimag(s1[1]), creal(s2[1]), cimag(s2[1]), creal(M0[1]), cimag(M0[1]),
           creal(Is), cimag(Is), creal(Ie[0]), cimag(Ie[0]), creal(Ie[1]), cimag(Ie[1]));
}

/* =====================================================================================================================
 * The contour integral representation (dfl_contour): a fixed start g0 != 0 in cases DU, DT (models 2, 3) as a double
 * integral with an explicit integrand; a port of ctr_h / ctr_rend / ctr_moments / ctr_kernel / ctr_pieces / ctr_logM /
 * ctr_kbound / ctr_xscale / ctr_bound / ctr_golden / ctr_place / ctr_seg / ctr_upath / ctr_inner_n / ctr_layout / ctr_phi
 * / ctr_path / ctr_eval / ctr_level! / ctr_sum / ctr_value / tr_contour of DFExact.jl (November 2026), where the derivation
 * is described:
 *   f(x) = (1/2 pi i) int_C I_f(t) dt,  F(x) = (1/2 pi i) int_C I_F(t) dt/t,
 * C the hyperbola t(u) = tau + a (cosh u - 1) + i b sinh u (trapezoidal rule in u >= 0), and at each node two U-shaped
 * paths in s, G+ around the root s1 (opening right) and G- around s2 (opening left), of Mt(s,t) times the v-integrals in
 * closed form (incomplete Gaussian integrals: ctr_h, ctr_moments) over the pieces of {v > 0} where X(v) = c0 v^2 + x v
 * + 1/2 is positive (G+) or negative (G-) (at each node the pieces with Re(t) a^2 <= 50, ctr_active), and for the
 * distribution function the kernels (1 - e^{-s/2})/s on G+ and -1/s on G-.  Two structurally different evaluations:
 * A (the hyperbola at the angle ctr_phi(c), the placement weight 0.5, 64 nodes per panel and per ray) and B (the angle
 * larger by pi/24, the weight 1, 48 nodes); the error estimate is |A - B| plus 10 eps times the sums of the largest terms
 * of both; while it exceeds 1e-10 both get the next level (A 96, 128, 192; B 64, 96, 128) at all their nodes; the value
 * is A at the last level, status 2 where the estimate still exceeds 1e-10 or the value is not finite.
 * ===================================================================================================================== */
#include "dfl_lock.h"

#define CTR_NL 4
static const int CTR_LEVELS[CTR_NL] = {64, 96, 128, 192};      /* nodes per panel and per ray of evaluation A: the levels of the refinement */
static const int CTR_LEVELS_B[CTR_NL] = {48, 64, 96, 128};     /* ... of evaluation B (one level lower, so that the difference also changes the rule) */
#define CTR_DPHI (M_PI / 24)      /* evaluation B: the asymptotic angle of its hyperbola exceeds that of A by 7.5 degrees */
#define CTR_PEN_B 1.0             /* ... and its placement weighs the grading cost by 1 (A: CTR_PEN) */
#define CTR_NGRID 24              /* the grid of the placement scan */
#define CTR_MARG 1e-3             /* the admissible abscissas keep this fraction of |Re s1 - Re s2| from the roots */
#define CTR_PEN 0.5               /* the weight of the grading cost in the placement objective (evaluation A) */
#define CTR_RMAX 1e5              /* the outer contour ends at |t - tau| = CTR_RMAX */
#define CTR_SUB 8                 /* points per outer step for the continuation of the branches */
#define CTR_ERRTOL 1e-10
#define CTR_CHUNK 32              /* the outer nodes in chunks of 32, until a chunk whose terms all stay below 1e-18 */

/* max, min and clamp of Julia (NaN propagates) */
static inline double ctr_max(double a, double b) { return (a != a || b != b) ? NAN : (a > b ? a : b); }
static inline double ctr_min(double a, double b) { return (a != a || b != b) ? NAN : (a < b ? a : b); }
static inline double ctr_clamp(double x, double lo, double hi) { return x > hi ? hi : x < lo ? lo : x; }
static inline cdouble ctr_inv(cdouble z) { return cdiv(ONE, z); }
static inline cdouble ctr_erfcx(cdouble z) { return dfl_faddw(CPX(-cimag(z), creal(z))); }   /* erfcx(z) = w(i z) */

/* h_k(z) = int_0^inf u^k exp(-u^2 - 2 z u) du, k = 0, 1, 2, Re z >= 0: (sqrt(pi)/2) erfcx(z) and the recurrence forward
 * for |z| < 6, the asymptotic series to its smallest term otherwise */
static void ctr_h(cdouble z, cdouble *h)
{
    if (cabs(z) < 6) {
        cdouble z2 = cscale(2, z);
        cdouble h0 = cscale(sqrt(M_PI) / 2, ctr_erfcx(z));
        cdouble h1 = cscale(0.5, csub(ONE, cmul(z2, h0)));
        h[0] = h0; h[1] = h1; h[2] = cscale(0.5, csub(h0, cmul(z2, h1)));
        return;
    }
    cdouble w = ctr_inv(cscale(2, z)), w2 = cmul(w, w);
    cdouble a0 = w, a1 = w2, a2 = cmul(cscale(2, w2), w);
    cdouble s0 = a0, s1 = a1, s2 = a2;
    for (int j = 0; ; ) {
        cdouble r0 = cscale((double) (-(2 * j + 1) * (2 * j + 2)) / (j + 1), w2);
        cdouble r1 = cscale((double) (-(2 * j + 2) * (2 * j + 3)) / (j + 1), w2);
        cdouble r2 = cscale((double) (-(2 * j + 3) * (2 * j + 4)) / (j + 1), w2);
        if (!(cabs(r2) < 1)) break;        /* (abs(r2) >= 1 in DFExact.jl, which loops forever at z = NaN) */
        a0 = cmul(a0, r0); a1 = cmul(a1, r1); a2 = cmul(a2, r2);
        s0 = cadd(s0, a0); s1 = cadd(s1, a1); s2 = cadd(s2, a2);
        j++;
        if (cabs(a0) < 1e-17 * cabs(s0) && cabs(a2) < 1e-17 * cabs(s2)) break;
    }
    h[0] = s0; h[1] = s1; h[2] = s2;
}
/* the remainders R_m(a) = int_a^inf v^m e^{-t v^2 - s X(v)} dv less the full-line Gaussian when it belongs to a (the
 * returned flag 1: Re z_a < 0), m = 0, 1, 2; sa = sqrt(t + c0 s), beta = x s, Ea = e^{-t a^2 - s X(a)} */
static int ctr_rend(double a, cdouble sa, cdouble beta, cdouble Ea, cdouble *R)
{
    cdouble z = cadd(cscale(a, sa), cdiv(beta, cscale(2, sa))), g[3]; int n;
    if (creal(z) >= 0) { n = 0; ctr_h(z, g); }
    else { cdouble hh[3]; n = 1; ctr_h(cneg(z), hh); g[0] = cneg(hh[0]); g[1] = hh[1]; g[2] = cneg(hh[2]); }
    cdouble i1 = ctr_inv(sa), i2 = cmul(i1, i1), i3 = cmul(i2, i1);
    R[0] = cmul(cmul(Ea, i1), g[0]);
    R[1] = cmul(Ea, cadd(cmul(cscale(a, i1), g[0]), cmul(i2, g[1])));
    R[2] = cmul(Ea, cadd(cadd(cmul(cscale(a * a, i1), g[0]), cmul(cscale(2 * a, i2), g[1])), cmul(i3, g[2])));
    return n;
}
/* the pieces (a, b) of {v > 0} (b may be Inf), at most three */
typedef struct { int n; double a[3], b[3]; } ctr_pcs;
/* int over the pieces of v^m e^{-t v^2 - s X(v)} dv, m = 0, 1, 2: the endpoint remainders and the full-line Gaussian as
 * often as the flags at the lower ends exceed those at the upper ends */
static void ctr_moments(cdouble t, cdouble s, double c0, double q, const ctr_pcs *pc, cdouble *M)
{
    cdouble alpha = cadd(t, cscale(c0, s)), beta = cscale(q, s), sa = csqrt(alpha), mt = cneg(t);
    cdouble M0 = ZERO, M1 = ZERO, M2 = ZERO, R[3]; int nf = 0;
    for (int k = 0; k < pc->n; k++) {
        double a = pc->a[k], b = pc->b[k];
        int na = ctr_rend(a, sa, beta, cexp(csub(cscale(a, cscale(a, mt)), cscale(c0 * a * a + q * a + 0.5, s))), R);
        M0 = cadd(M0, R[0]); M1 = cadd(M1, R[1]); M2 = cadd(M2, R[2]); nf += na;
        if (isfinite(b)) {
            int nb = ctr_rend(b, sa, beta, cexp(csub(cscale(b, cscale(b, mt)), cscale(c0 * b * b + q * b + 0.5, s))), R);
            M0 = csub(M0, R[0]); M1 = csub(M1, R[1]); M2 = csub(M2, R[2]); nf -= nb;
        }
    }
    if (nf != 0) {
        cdouble G0 = cmul(cscale(sqrt(M_PI), ctr_inv(sa)), cexp(csub(cdiv(cmul(beta, beta), cscale(4, alpha)), cscale(0.5, s))));
        cdouble b2 = cdiv(beta, cscale(2, alpha));
        M0 = cadd(M0, cscale(nf, G0));
        M1 = csub(M1, cmul(cscale(nf, b2), G0));
        M2 = cadd(M2, cmul(cscale(nf, cadd(cmul(b2, b2), ctr_inv(cscale(2, alpha)))), G0));
    }
    M[0] = M0; M[1] = M1; M[2] = M2;
}
/* the v-kernel of the density (kind 1, 2 v^2) or the distribution function (kind 2, X'(v) = q + 2 c0 v) */
static cdouble ctr_kernel(int kind, cdouble t, cdouble s, double c0, double q, const ctr_pcs *pc)
{
    if (pc->n == 0) return ZERO;
    cdouble M[3]; ctr_moments(t, s, c0, q, pc, M);
    return kind == 1 ? cscale(2, M[2]) : cadd(cscale(q, M[0]), cscale(2 * c0, M[1]));
}
/* the pieces of {v > 0} with X(v) > 0 (plus) and < 0 (minus), split at the positive roots of X (a double root is no split) */
static void ctr_pieces(double q, double c0, ctr_pcs *plus, ctr_pcs *minus)
{
    double r[2]; int nr = 0;
    if (c0 == 0) { if (q < 0) r[nr++] = -0.5 / q; }
    else {
        double dsc = q * q - 2 * c0;
        if (dsc > 0) {
            double x1 = (-q - sqrt(dsc)) / (2 * c0), x2 = (-q + sqrt(dsc)) / (2 * c0);
            if (x1 > 0) r[nr++] = x1;
            if (x2 > 0) r[nr++] = x2;
            if (nr == 2 && r[1] < r[0]) { double tmp = r[0]; r[0] = r[1]; r[1] = tmp; }
        }
    }
    double bk[4]; int nb = 0; bk[nb++] = 0; for (int k = 0; k < nr; k++) bk[nb++] = r[k]; bk[nb++] = INFINITY;
    plus->n = minus->n = 0;
    for (int j = 0; j < nb - 1; j++) { ctr_pcs *p = j % 2 == 0 ? plus : minus; p->a[p->n] = bk[j]; p->b[p->n] = bk[j + 1]; p->n++; }
}
static bool ctr_semi(const ctr_pcs *pc) { for (int k = 0; k < pc->n; k++) if (!isfinite(pc->b[k])) return true; return false; }
/* the pieces that are not negligible at a point t of the outer contour: a piece (a, b) with Re(t) a^2 > 50 (its Gaussian
 * factor below e^{-50} from its start; far out at x -> 0- with c0 = 0 the root of X runs off to 1/(2|x|)) is dropped */
static void ctr_active(const ctr_pcs *in, double tr, ctr_pcs *out)
{
    out->n = 0;
    for (int k = 0; k < in->n; k++) if (tr * in->a[k] * in->a[k] <= 50) { out->a[out->n] = in->a[k]; out->b[out->n] = in->b[k]; out->n++; }
}

/* the data of a point t of the outer contour: the roots (s1 the one with Re > 0 at tau, s2 the other), P(t) and
 * KQ = c^{2d} e^c g(z) P^2 */
typedef struct { cdouble t, s1, s2, P, KQ; } ctr_t;
/* log Mt(s,t) - log P: -(log(s1 - s) + log(s - s2))/2 + g0^2 Q(s,t) (principal logarithms) */
static inline cdouble ctr_logM(const ctr_t *d, cdouble s, double g0, double c)
{
    cdouble a = csub(d->s1, s), b = csub(s, d->s2);
    cdouble num = cadd(cadd(d->t, cscale(0.5, cmul(s, s))), cscale(c, s));
    return cadd(cscale(-0.5, cadd(clog(a), clog(b))), cdiv(cmul(cscale(g0 * g0, d->KQ), num), cmul(a, b)));
}
/* max over v in the pieces of Re(-t v^2) - sig X(v): +Inf when a semi-infinite piece has a growing Gaussian */
static double ctr_kbound(double tr, double sig, double c0, double q, const ctr_pcs *pc)
{
    double m = -INFINITY;
    for (int k = 0; k < pc->n; k++) {
        double a = pc->a[k], b = pc->b[k], A = tr + sig * c0;
#define CTR_HV(v) (-A * (v) * (v) - sig * q * (v) - sig / 2)
        if (A > 0) { double v = ctr_clamp(-sig * q / (2 * A), a, b); m = ctr_max(m, CTR_HV(v)); }
        else { if (!isfinite(b)) return INFINITY; m = ctr_max(ctr_max(m, CTR_HV(a)), CTR_HV(b)); }
#undef CTR_HV
    }
    return pc->n == 0 ? 0.0 : m;
}
/* the slowest decay scale along the rays of the v-kernel of the pieces (1/min(max|X|, |X(v0)|) per piece; semi-infinite
 * pieces cut where e^{-Re(t) v^2} = e^{-18}); 0 without pieces */
static double ctr_xscale(double tr, double c0, double q, const ctr_pcs *pc)
{
    if (pc->n == 0) return 0.0;
    double sc = 0.0;
    for (int j = 0; j < pc->n; j++) {
        double a = pc->a[j], b = pc->b[j], bb = isfinite(b) ? b : a + 6 / sqrt(tr), m = 0.0;
        for (int k = 0; k <= 32; k++) { double v = a + (bb - a) * k / 32; m = ctr_max(m, fabs(c0 * v * v + q * v + 0.5)); }
        double sp = 0.0;
        if (c0 != 0) {
            double v0 = -q / (2 * c0); if (a < v0 && v0 < b) m = ctr_min(m, fabs(c0 * v0 * v0 + q * v0 + 0.5));
            /* at an end that is a root v_j of X the kernel decays like s^(-1/2) (the Gaussian width of c0 (v - v_j)^2) up to
             * |s| ~ |c0|/X'(v_j)^2 before the slope takes over: near a tangency of X (two close roots) on the pieces beside
             * it as well as between them, where the vertex is not inside the piece */
            double e[2] = {a, b};
            for (int i = 0; i < 2; i++) {
                if (!(e[i] > 0 && isfinite(e[i]))) continue;
                double sl = 2 * c0 * e[i] + q; sp = ctr_max(sp, fabs(c0) / ctr_max(sl * sl, 1e-12));
            }
        }
        sc = ctr_max(sc, ctr_max(1 / ctr_max(m, 1e-12), sp));
    }
    return sc;
}
/* the placement problem of one U-shape: around sk (side +1: opening right, -1: left), so the other root, the pieces of
 * its v-kernel, the focus points, and kbx: 1 the exponent bound of the kernel taken at least max(0, -sig/2) (the kernel
 * (1 - e^{-s/2})/s of the distribution function on G+), 2 at least 0 (-1/s on G-), 0 the v-kernel's alone */
typedef struct { const ctr_t *d; double g0, c, tr, c0, q; const ctr_pcs *pc; cdouble sk, so; int side, nfoc; const cdouble *foc; int kbx; double pen; } ctr_pctx;
static double ctr_kb(const ctr_pctx *p, double x)
{
    double k = ctr_kbound(p->tr, x, p->c0, p->q, p->pc);
    return p->kbx == 1 ? ctr_max(k, ctr_max(0.0, -x / 2)) : p->kbx == 2 ? ctr_max(k, 0.0) : k;
}
/* the largest of log|Mt| + kb(Re s) over the 8 probe points of the U-shape with its segment at sig */
static double ctr_bound(const ctr_pctx *p, double sig)
{
    double dd = fabs(creal(p->sk) - sig), y0 = cimag(p->sk), yo = ctr_clamp(cimag(p->so), y0 - dd, y0 + dd);
    double rs = creal(p->sk), rx = rs + p->side * dd;
    cdouble pr[8] = {CPX(sig, y0), CPX(sig, y0 + dd), CPX(sig, y0 - dd), CPX(rs, y0 + dd), CPX(rs, y0 - dd), CPX(rx, y0 + dd), CPX(rx, y0 - dd), CPX(sig, yo)};
    double m = -INFINITY;
    for (int k = 0; k < 8; k++) m = ctr_max(m, creal(ctr_logM(p->d, pr[k], p->g0, p->c)) + ctr_kb(p, creal(pr[k])));
    return m;
}
/* the objective B of the placement: ctr_bound plus the grading cost toward the focus points */
static double ctr_obj(const ctr_pctx *p, double sig)
{
    double v = ctr_bound(p, sig), dd = fabs(creal(p->sk) - sig);
    for (int k = 0; k < p->nfoc; k++) {
        double dist = hypot(sig - creal(p->foc[k]), ctr_max(0.0, fabs(cimag(p->foc[k]) - cimag(p->sk)) - dd));
        v += p->pen * log(1 + dd / ctr_max(dist, 1e-300));
    }
    return v;
}
/* golden-section minimum of the objective on [a, b], n steps */
static double ctr_golden(const ctr_pctx *p, double a, double b, int n)
{
    double g = (sqrt(5.0) - 1) / 2;
    double x1 = b - g * (b - a), x2 = a + g * (b - a), f1 = ctr_obj(p, x1), f2 = ctr_obj(p, x2);
    for (int i = 0; i < n; i++) {
        if (f1 < f2) { b = x2; x2 = x1; f2 = f1; x1 = b - g * (b - a); f1 = ctr_obj(p, x1); }
        else { a = x1; x1 = x2; f1 = f2; x2 = a + g * (b - a); f2 = ctr_obj(p, x2); }
    }
    return f1 < f2 ? x1 : x2;
}
/* the abscissa of the U-shape in the admissible interval [lo, hi]: the objective scanned on a geometric grid of CTR_NGRID
 * distances d = |Re sk - sig| and refined by 12 golden-section steps between the neighbours of the best */
static double ctr_place(const ctr_pctx *p, double lo, double hi)
{
    double rs = creal(p->sk), dmin, dmax;
    if (p->side == 1) { dmin = rs - hi; dmax = rs - lo; } else { dmin = lo - rs; dmax = hi - rs; }
    dmin = ctr_max(dmin, 1e-3 * dmax);
    /* an empty admissible interval (the branch point -t/c0 or the condition Re s < 0 of G- beyond the root; not met in the
     * supported region, met for c > 2): NaN, so that the value is not finite and flagged (status 2) */
    if (!(dmax > 0 && dmin < dmax)) return NAN;
    double best = INFINITY, sgs[CTR_NGRID]; int jb = 0;
    for (int k = 0; k < CTR_NGRID; k++) {
        double dk = dmin * pow(dmax / dmin, (double) k / (CTR_NGRID - 1));
        sgs[k] = p->side == 1 ? rs - dk : rs + dk;
        double v = ctr_obj(p, sgs[k]); if (v < best) { best = v; jb = k; }
    }
    double a = sgs[jb > 0 ? jb - 1 : 0], b = sgs[jb + 1 < CTR_NGRID ? jb + 1 : CTR_NGRID - 1];
    return ctr_golden(p, ctr_min(a, b), ctr_max(a, b), 12);
}
/* the integrand on G+ (minus = 0) or G- (minus = 1): Mt/P times the v-kernel, plus (1 - e^{-s/2})/s on G+ and -1/s on G-
 * for the distribution function */
typedef struct { const ctr_t *d; int kind, minus; double q, c0, c, g0; const ctr_pcs *pc; } ctr_fctx;
static cdouble ctr_f(const ctr_fctx *x, cdouble s)
{
    cdouble e = cexp(ctr_logM(x->d, s, x->g0, x->c)), k = ctr_kernel(x->kind, x->d->t, s, x->c0, x->q, x->pc);
    if (x->kind == 2) {
        if (x->minus) k = csub(k, ctr_inv(s));
        else k = cadd(k, cabs(s) < 1e-8 ? CPX(0.5 - creal(s) / 8, -(cimag(s) / 8)) : cdiv(csub(ONE, cexp(cscale(-0.5, s))), s));
    }
    return cmul(e, k);
}
/* int of f(sig + i y) i dy between st and en by Gauss-Legendre panels (n nodes) graded from st: the first length h0
 * (h0 = Inf: one panel), growing by the factor 3; the sum into *acc, the largest |term| into *mx */
static void ctr_seg(const ctr_fctx *f, double sig, double st, double en, double h0, const double *z, const double *w, int n, cdouble *acc, double *mx)
{
    double len = fabs(en - st), dir = en >= st ? 1.0 : -1.0, a = 0.0, h = isfinite(h0) ? ctr_min(h0, len) : len, m = 0.0;
    cdouble sum = ZERO;
    while (a < len * (1 - 1e-14)) {
        double b = ctr_min(a + h, len); if ((len - b) < 0.5 * h) b = len;
        for (int k = 0; k < n; k++) {
            double y = (b - a) * w[k];
            cdouble fv = ctr_f(f, CPX(sig, st + dir * (a + (b - a) * z[k]))), val = CPX(-(cimag(fv) * y), creal(fv) * y);   /* f i y */
            sum = cadd(sum, val); m = ctr_max(m, cabs(val));
        }
        a = b; h *= 3;
    }
    *acc = sum; *mx = m;
}
/* (1/2 pi i) int over the U-shape around sk with its segment at sig of f(s) ds: the segment split and graded toward the
 * focus points, the rays r = rho u/(1 - u), rho = max(d, rscale), with graded panels up to rho first when rho > 2d;
 * returns the value and max |term|/(2 pi) */
static void ctr_upath(const ctr_fctx *f, cdouble sk, double sig, int side, int n, const double *z, const double *w, double rscale,
                      const cdouble *foc, int nf, cdouble *value, double *mxo)
{
    double dd = fabs(creal(sk) - sig), y0 = cimag(sk), ylo = y0 - dd, yhi = y0 + dd, rho = ctr_max(dd, rscale);
    cdouble acc = ZERO, v; double mx = 0.0, m;
    /* the breakpoints: the ends (no grading) and the focus levels with their first panel lengths */
    double lev[3], hs[3]; int ord[3];
    for (int k = 0; k < nf; k++) { lev[k] = ctr_clamp(cimag(foc[k]), ylo, yhi); hs[k] = ctr_max(fabs(sig - creal(foc[k])), 1e-6 * dd); }
    /* the focus points in the order of their scales: one whose level lies within its own scale of a level already kept is
     * dropped (stable sorts, as sortperm) */
    for (int k = 0; k < nf; k++) ord[k] = k;
    for (int i = 1; i < nf; i++) { int o = ord[i], j = i - 1; while (j >= 0 && hs[ord[j]] > hs[o]) { ord[j + 1] = ord[j]; j--; } ord[j + 1] = o; }
    double kl[3], kh[3]; int nk = 0;
    for (int i = 0; i < nf; i++) {
        int k = ord[i]; bool near_ = false;
        for (int j = 0; j < nk; j++) if (fabs(lev[k] - kl[j]) <= hs[k]) { near_ = true; break; }
        if (near_) continue;
        kl[nk] = lev[k]; kh[nk] = hs[k]; nk++;
    }
    for (int k = 0; k < nk; k++) ord[k] = k;
    for (int i = 1; i < nk; i++) { int o = ord[i], j = i - 1; while (j >= 0 && kl[ord[j]] > kl[o]) { ord[j + 1] = ord[j]; j--; } ord[j + 1] = o; }
    double py[5], ph[5], tol = 1e-12 * ctr_max(dd, 1.0); int np = 0;
    py[np] = ylo; ph[np] = INFINITY; np++;
    for (int i = 0; i < nk; i++) {
        int k = ord[i];
        if (fabs(kl[k] - py[np - 1]) <= tol) ph[np - 1] = ctr_min(ph[np - 1], kh[k]);
        else { py[np] = kl[k]; ph[np] = kh[k]; np++; }
    }
    if (!(fabs(yhi - py[np - 1]) <= tol)) { py[np] = yhi; ph[np] = INFINITY; np++; }
    for (int k = 0; k < np - 1; k++) {
        double A = py[k], hA = ph[k], B = py[k + 1], hB = ph[k + 1];
        if (B - A <= 0) continue;
        if (!isfinite(hA) && !isfinite(hB)) { ctr_seg(f, sig, A, B, INFINITY, z, w, n, &v, &m); acc = cadd(acc, v); mx = ctr_max(mx, m); }
        else if (!isfinite(hA)) { ctr_seg(f, sig, B, A, hB, z, w, n, &v, &m); acc = cadd(acc, v); mx = ctr_max(mx, m); }
        else if (!isfinite(hB)) { ctr_seg(f, sig, A, B, hA, z, w, n, &v, &m); acc = cadd(acc, v); mx = ctr_max(mx, m); }
        else {
            double M = (A + B) / 2;
            ctr_seg(f, sig, A, M, hA, z, w, n, &v, &m); acc = cadd(acc, v); mx = ctr_max(mx, m);
            ctr_seg(f, sig, B, M, hB, z, w, n, &v, &m); acc = cadd(acc, v); mx = ctr_max(mx, m);
        }
    }
    for (int r = 0; r < 2; r++) {
        double yc = r == 0 ? yhi : ylo; int dir = r == 0 ? 1 : -1, sgn = side == 1 ? dir : -dir;
        double r0 = 0.0;
        if (rho > 2 * dd) {
            /* two scales: graded panels from the corner (first length d, growing by 3) up to rho, the map beyond */
            double a = 0.0, h = dd;
            while (a < rho * (1 - 1e-14)) {
                double b = ctr_min(a + h, rho); if ((rho - b) < 0.5 * h) b = rho;
                for (int k = 0; k < n; k++) {
                    cdouble val = cscale(sgn * (b - a) * w[k], ctr_f(f, CPX(sig + side * (a + (b - a) * z[k]), yc)));
                    acc = cadd(acc, val); mx = ctr_max(mx, cabs(val));
                }
                a = b; h *= 3;
            }
            r0 = rho;
        }
        for (int k = 0; k < n; k++) {
            double u = z[k], rr = r0 + rho * u / (1 - u);
            cdouble val = cscale(sgn * rho / ((1 - u) * (1 - u)) * w[k], ctr_f(f, CPX(sig + side * rr, yc)));
            acc = cadd(acc, val); mx = ctr_max(mx, cabs(val));
        }
    }
    *value = CPX(cimag(acc) / (2 * M_PI), -creal(acc) / (2 * M_PI));   /* acc/(2 pi i) */
    *mxo = mx / (2 * M_PI);
}
/* the placement at one point t: the abscissas, the focus points and the ray scales of G+ and G- */
typedef struct { double sp, sm, rsp, rsm; int nfp, nfm; cdouble focp[3], focm[3]; ctr_pcs pl, mi; } ctr_lay;   /* (pl, mi: the active pieces) */
/* the inner function I_f(t) (kind 1) or I_F(t) (kind 2) at one point t, n nodes per panel and per ray; *mx the largest
 * |term| */
static void ctr_inner_n(int kind, const ctr_t *d, double q, double c0, double c, double g0, const ctr_pcs *plus, const ctr_pcs *minus,
                        int n, const double *z, const double *w, const ctr_lay *L, cdouble *Iv, double *mx)
{
    ctr_fctx fp = {d, kind, 0, q, c0, c, g0, plus}, fm = {d, kind, 1, q, c0, c, g0, minus};
    cdouble vp, vm = ZERO; double mp, mm = 0.0;
    /* no admissible abscissa (ctr_place: NaN): the value is NaN, hence flagged */
    if (isnan(L->sp) || ((minus->n > 0 || kind == 2) && isnan(L->sm))) { *Iv = CPX(NAN, NAN); *mx = 0.0; return; }
    ctr_upath(&fp, d->s1, L->sp, 1, n, z, w, L->rsp, L->focp, L->nfp, &vp, &mp);
    if (minus->n > 0 || kind == 2) ctr_upath(&fm, d->s2, L->sm, -1, n, z, w, L->rsm, L->focm, L->nfm, &vm, &mm);
    *Iv = cmul(d->P, cadd(vp, vm)); *mx = cabs(d->P) * ctr_max(mp, mm);
}
static void ctr_layout(int kind, const ctr_t *d, double q, double c0, double c, double g0, const ctr_pcs *plus, const ctr_pcs *minus, double pen, ctr_lay *L)
{
    cdouble s1 = d->s1, s2 = d->s2; double tr = creal(d->t);
    double lo = creal(s2), hi = creal(s1), Lw = hi - lo;
    cdouble sb = c0 == 0 ? ZERO : CPX(-creal(d->t) / c0, -cimag(d->t) / c0);
    bool semip = c0 > 0 && ctr_semi(plus);
    L->nfp = 0; L->focp[L->nfp++] = ZERO; if (semip) L->focp[L->nfp++] = sb; L->focp[L->nfp++] = s2;
    double lop = lo + CTR_MARG * Lw; if (semip) lop = ctr_max(lop, creal(sb) + CTR_MARG * Lw);
    /* the exponent bound of the kernel on G+ (and on G- below): for the distribution function also that of the kernels of
     * Psi_t(1/2), so that a negligible v-kernel does not leave the placement blind to Mt */
    ctr_pctx pp = {d, g0, c, tr, c0, q, plus, s1, s2, 1, L->nfp, L->focp, kind == 2 ? 1 : 0, pen};
    L->sp = ctr_place(&pp, lop, hi - CTR_MARG * Lw);
    bool semim = c0 < 0 && ctr_semi(minus);
    L->nfm = 0; L->focm[L->nfm++] = ZERO; if (semim) L->focm[L->nfm++] = sb; L->focm[L->nfm++] = s1;
    L->sm = NAN;
    if (minus->n > 0 || kind == 2) {
        double him = hi - CTR_MARG * Lw; if (semim) him = ctr_min(him, creal(sb) - CTR_MARG * Lw);
        if (kind == 2) him = ctr_min(him, -CTR_MARG * Lw);                      /* Re s < 0 on G- (the kernel -1/s) */
        ctr_pctx pm = {d, g0, c, tr, c0, q, minus, s2, s1, -1, L->nfm, L->focm, kind == 2 ? 2 : 0, pen};
        L->sm = ctr_place(&pm, lo + CTR_MARG * Lw, him);
    }
    /* (the kernel (1 - e^{-s/2})/s of the distribution function on G+ decays on the scale 2 = 1/X(0)) */
    L->rsp = kind == 2 ? ctr_max(ctr_xscale(tr, c0, q, plus), 2.0) : ctr_xscale(tr, c0, q, plus);
    L->rsm = ctr_xscale(tr, c0, q, minus);
}

/* the asymptotic angle of the outer hyperbola: 45 degrees for c >= -10, rising linearly to 70 degrees at c = -40, 70 below */
static double ctr_phi(double c) { return (45 + 25 * ctr_clamp((-c - 10) / 30, 0.0, 1.0)) * M_PI / 180; }
/* the outer contour of (model, c) (variant 1: evaluation A, the angle ctr_phi(c); variant 2: evaluation B, the angle
 * ctr_phi(c) + CTR_DPHI): the nodes u_k = k h, the data along it (branches continued on CTR_SUB points per step) and
 * t'(u_k); cached per (model, c, variant) under a lock (CTR_PMAX entries; beyond, each call builds its own) */
typedef struct { int model, variant; double c, h; int K1; ctr_t *T; cdouble *dt; } ctr_path_t;
#define CTR_PMAX 256
static ctr_path_t *CTRPATH[CTR_PMAX]; static int nctrpath = 0; static dfl_lock_t ctrlock = DFL_LOCK_INIT;
static ctr_path_t *ctr_path_new(int model, double c, int variant)
{
    double tau = dfl_default_tau(model, c, false), ts = dfl_tstar(model, c, false);
    double phi = ctr_phi(c) + (variant == 2 ? CTR_DPHI : 0.0), y = ctr_min(0.5, 0.9 * (M_PI / 2 - phi));
    double b = 0.9 * ctr_min(tau, ts - tau) / sin(y), a = b / tan(phi);
    double h = 2 * M_PI * y / (37 + 37 * tan(phi) * y);
    int K = (int) ceil(asinh(CTR_RMAX / hypot(a, b)) / h), nd = CTR_SUB * K + 1;
    cdouble *tt = malloc(nd * sizeof(cdouble)), *P = malloc(nd * sizeof(cdouble)), *s1e = malloc(nd * sizeof(cdouble)), *s2e = malloc(nd * sizeof(cdouble)), *M0 = malloc(nd * sizeof(cdouble));
    for (int j = 0; j < nd; j++) { double u = j * h / CTR_SUB; tt[j] = CPX(tau + a * (cosh(u) - 1), b * sinh(u)); }
    dfl_tparts(model, c, nd, tt, false, P, s1e, s2e, M0);            /* (engine labels: s1e the root with Re < 0 at tau) */
    ctr_path_t *C = malloc(sizeof *C);
    C->model = model; C->variant = variant; C->c = c; C->h = h; C->K1 = K + 1;
    C->T = malloc((K + 1) * sizeof(ctr_t)); C->dt = malloc((K + 1) * sizeof(cdouble));
    double ek = (model == 2 ? c * c : pow(c, 4)) * exp(c);
    for (int k = 0; k <= K; k++) {
        int j = CTR_SUB * k; cdouble t = tt[j];
        cdouble KQ = cmul(cscale(ek, dfl_gstat(model, CPX(c * c - 2 * creal(t), -2 * cimag(t)))), cmul(P[j], P[j]));
        ctr_t d = {t, s2e[j], s1e[j], P[j], KQ}; C->T[k] = d;
        C->dt[k] = CPX(a * sinh(k * h), b * cosh(k * h));
    }
    free(tt); free(P); free(s1e); free(s2e); free(M0);
    return C;
}
static void ctr_path_free(ctr_path_t *C) { if (C) { free(C->T); free(C->dt); free(C); } }
static ctr_path_t *ctr_path(int model, double c, int variant, bool *owned)
{
    *owned = false;
#define CTR_HIT(k) (CTRPATH[k]->model == model && CTRPATH[k]->c == c && CTRPATH[k]->variant == variant)
    dfl_lock(&ctrlock);
    for (int k = 0; k < nctrpath; k++) if (CTR_HIT(k)) { ctr_path_t *r = CTRPATH[k]; dfl_unlock(&ctrlock); return r; }
    dfl_unlock(&ctrlock);
    ctr_path_t *C = ctr_path_new(model, c, variant);
    dfl_lock(&ctrlock);
    for (int k = 0; k < nctrpath; k++) if (CTR_HIT(k)) { ctr_path_t *r = CTRPATH[k]; dfl_unlock(&ctrlock); ctr_path_free(C); return r; }
#undef CTR_HIT
    if (nctrpath < CTR_PMAX) CTRPATH[nctrpath++] = C; else *owned = true;
    dfl_unlock(&ctrlock);
    return C;
}
/* one evaluation (variant 1: A, 2: B; ctr_eval of DFExact.jl): the path, the pieces, the per-node layouts and values G
 * (outer node k, level l at G[k CTR_NL + l]), the largest terms gm at the first level, the nodes done, the levels and
 * their rules; the outer nodes in chunks of CTR_CHUNK on the workers */
typedef struct {
    int kind, variant, lev, k1, done; bool owned; double x, c, c0, g0, pen; ctr_path_t *C; ctr_pcs plus, minus;
    double *G, *gm; ctr_lay *lay; const int *levels; const double *z[CTR_NL], *w[CTR_NL];
} ctr_eval_t;
static cdouble ctr_wt(const ctr_eval_t *o, int k) { return o->kind == 1 ? o->C->dt[k] : cdiv(o->C->dt[k], o->C->T[k].t); }
static void ctr_body_first(int j, void *v)
{
    ctr_eval_t *o = v; int k = o->k1 + j; const ctr_t *d = &o->C->T[k]; ctr_lay *L = &o->lay[k];
    ctr_active(&o->plus, creal(d->t), &L->pl); ctr_active(&o->minus, creal(d->t), &L->mi);
    ctr_layout(o->kind, d, o->x, o->c0, o->c, o->g0, &L->pl, &L->mi, o->pen, L);
    cdouble wt = ctr_wt(o, k), Iv; double mx;
    ctr_inner_n(o->kind, d, o->x, o->c0, o->c, o->g0, &L->pl, &L->mi, o->levels[0], o->z[0], o->w[0], L, &Iv, &mx);
    o->G[k * CTR_NL] = cimag(cmul(Iv, wt)); o->gm[k] = mx * cabs(wt);
}
static void ctr_body_level(int k, void *v)
{
    ctr_eval_t *o = v; const ctr_t *d = &o->C->T[k]; int l = o->lev;
    cdouble Iv; double mx;
    ctr_inner_n(o->kind, d, o->x, o->c0, o->c, o->g0, &o->lay[k].pl, &o->lay[k].mi, o->levels[l], o->z[l], o->w[l], &o->lay[k], &Iv, &mx);
    o->G[k * CTR_NL + l] = cimag(cmul(Iv, ctr_wt(o, k)));
}
/* the evaluation at its first level, until a chunk whose terms all stay below 1e-18 */
static void ctr_eval(ctr_eval_t *o, int kind, double x, int model, double c, double c0, double g0, int variant)
{
    memset(o, 0, sizeof *o);
    o->kind = kind; o->variant = variant; o->x = x; o->c = c; o->c0 = c0; o->g0 = g0;
    o->C = ctr_path(model, c, variant, &o->owned);
    ctr_pieces(x, c0, &o->plus, &o->minus);
    o->pen = variant == 2 ? CTR_PEN_B : CTR_PEN; o->levels = variant == 2 ? CTR_LEVELS_B : CTR_LEVELS;
    int K1 = o->C->K1;
    o->G = calloc((size_t) K1 * CTR_NL, sizeof(double)); o->gm = calloc(K1, sizeof(double)); o->lay = malloc(K1 * sizeof(ctr_lay));
    for (int l = 0; l < CTR_NL; l++) o->z[l] = dfl_rule(o->levels[l], &o->w[l]);
    int done = 0;
    while (done < K1) {
        int k1 = done, k2 = done + CTR_CHUNK < K1 ? done + CTR_CHUNK : K1;
        o->k1 = k1; dfl_parfor(k2 - k1, ctr_body_first, o);
        done = k2;
        double mg = 0, mm = 0;
        for (int k = k1; k < k2; k++) { mg = ctr_max(mg, fabs(o->G[k * CTR_NL])); mm = ctr_max(mm, o->gm[k]); }
        if (mg < 1e-18 && mm < 1e-18) break;
    }
    o->done = done;
}
/* the next level l (0-based) of an evaluation, at all its nodes */
static void ctr_level(ctr_eval_t *o, int l) { o->lev = l; dfl_parfor(o->done, ctr_body_level, o); }
static double ctr_sum(const ctr_eval_t *o, int l)
{
    double w = o->C->h / M_PI, s = 0;
    for (int k = 0; k < o->done; k++) s += (k == 0 ? w / 2 : w) * o->G[k * CTR_NL + l];
    return s;
}
static double ctr_gmsum(const ctr_eval_t *o)
{
    double w = o->C->h / M_PI, s = 0;
    for (int k = 0; k < o->done; k++) s += (k == 0 ? w / 2 : w) * o->gm[k];
    return s;
}
static void ctr_eval_free(ctr_eval_t *o) { free(o->G); free(o->gm); free(o->lay); if (o->owned) ctr_path_free(o->C); }
/* the value: evaluations A and B, the error estimate |A - B| plus the cancellation, both refined while it exceeds
 * CTR_ERRTOL, the value A at the last level (ctr_value of DFExact.jl); level 1-based (1..CTR_NL), nodes the outer nodes
 * of A and B together */
static double ctr_value(int kind, double x, int model, double c, double c0, double g0, int verified, double *err, int *status, int *nodes, int *level)
{
    ctr_eval_t A, B;
    ctr_eval(&A, kind, x, model, c, c0, g0, 1);
    if (!verified) {                                  /* one evaluation: A at the first level, no estimate */
        double v = ctr_sum(&A, 0);
        if (err) *err = NAN;
        if (status) *status = isfinite(v) ? 0 : 2;
        if (nodes) *nodes = A.done;
        if (level) *level = 1;
        ctr_eval_free(&A);
        return v;
    }
    ctr_eval(&B, kind, x, model, c, c0, g0, 2);
    double canc = 10 * 2.220446049250313e-16 * (ctr_gmsum(&A) + ctr_gmsum(&B));
    int l = 0; double e = fabs(ctr_sum(&A, 0) - ctr_sum(&B, 0)) + canc;
    while (e > CTR_ERRTOL && l < CTR_NL - 1) {
        l++; ctr_level(&A, l); ctr_level(&B, l);
        e = fabs(ctr_sum(&A, l) - ctr_sum(&B, l)) + canc;
    }
    double v = ctr_sum(&A, l);
    if (err) *err = e;
    if (status) *status = (e > CTR_ERRTOL || !isfinite(v)) ? 2 : 0;
    if (nodes) *nodes = A.done + B.done;
    if (level) *level = l + 1;
    ctr_eval_free(&A); ctr_eval_free(&B);
    return v;
}

int dfl_contour_lev(int kind, int n, const double *x, int model, double c, double c0, double g0, int verified, double *out, double *err, int *status, int *nodes, int *level, char *errmsg, size_t errlen)
{
    if (kind != 1 && kind != 2) { snprintf(errmsg, errlen, "kind must be 1 (density) or 2 (distribution function)"); return 1; }
    /* the checks of dfl_check_args but the region of a fixed start (dfl_fixed_ok), which the user-level functions check
     * (dfl_dist, dfl_icdf): like tr_contour of DFExact.jl and dfl_double, the engine function computes outside it too */
    if (!isfinite(g0)) { snprintf(errmsg, errlen, "g0 must be a finite real scalar"); return 1; }
    if (dfl_check_args(model, c, c0, 0.0, false, errmsg, errlen)) return 1;
    if (!((model == 2 || model == 3) && g0 != 0)) { snprintf(errmsg, errlen, "%s", DFL_CONTOUR_MSG); return 1; }
    if (!isfinite(dfl_tstar(model, c, false))) { snprintf(errmsg, errlen, "no zero of D(0, t) was found (t_*), c = %g", c); return 1; }
    for (int k = 0; k < n; k++) {
        if (!isfinite(x[k])) {
            out[k] = isnan(x[k]) ? NAN : (x[k] > 0 && kind == 2) ? 1.0 : 0.0;
            if (err) err[k] = 0; if (status) status[k] = 0; if (nodes) nodes[k] = 0; if (level) level[k] = 0;
            continue;
        }
        out[k] = ctr_value(kind, x[k], model, c, c0, g0, verified, err ? &err[k] : NULL, status ? &status[k] : NULL, nodes ? &nodes[k] : NULL, level ? &level[k] : NULL);
    }
    return 0;
}
int dfl_contour(int kind, int n, const double *x, int model, double c, double c0, double g0, int verified, double *out, double *err, int *status, int *nodes, char *errmsg, size_t errlen)
{
    return dfl_contour_lev(kind, n, x, model, c, c0, g0, verified, out, err, status, nodes, NULL, errmsg, errlen);
}
