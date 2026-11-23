/* dfl_det_inner.c: the inversion on paths around the roots (a port of DFDet.jl): at each node t_k of the outer contour two
 * U-shaped paths in s, G+ around the group of positive roots and G- around the negative group (ushapes, ushape!,
 * graded_panels, ulevels, place and nsplit for the fixed start, whose roots are essential singularities), the integrals
 * over v in closed form (incomplete Gaussian integrals: ctr_h, ctr_rend, ctr_moments) over the pieces where the parabola
 * X(v) is positive or negative, the terms of the t-ratio and of the events bounded by parabolas in sqrt(V) (tnode_inner!,
 * tnode_specs!, XSpec), the outer trapezoidal rule or Gauss-Legendre panels on the vertical line with the estimate of the
 * rounding error (contour_sum), Filon's rule for the parts at the root of the parabola on the line (contour_sum_filon,
 * filon_int), and the law of V alone (vsum). */
#include "dfl_det_int.h"

typedef struct { int n; double a[4], b[4]; } pcs_t;
static inline void pcs_add(pcs_t *p, double a, double b) { p->a[p->n] = a; p->b[p->n] = b; p->n++; }

/* ---------- the incomplete Gaussian integrals ---------- */
static inline cdouble erfcx_(cdouble z) { return dfl_faddw(cI(z)); }     /* erfcx(z) = w(i z) */
/* h_k(z) = int_0^inf u^k exp(-u^2 - 2 z u) du, k = 0, 1, 2 (ctr_h) */
static void ctr_h(cdouble z, cdouble *h)
{
    if (cabs2(z) < 36) {
        cdouble h0 = cscale(sqrt(M_PI) / 2, erfcx_(z)), z2 = cscale(2, z);
        cdouble h1 = cscale(0.5, csub(DONE, cmul(z2, h0)));
        h[0] = h0; h[1] = h1; h[2] = cscale(0.5, csub(h0, cmul(z2, h1)));
        return;
    }
    cdouble w = cinv(cscale(2, z)), w2 = cmul(w, w);
    cdouble a0 = w, a1 = w2, a2 = cmul(cscale(2, w2), w), s0 = a0, s1 = a1, s2 = a2;
    for (int j = 0; ; ) {
        double r0 = (double) (-(2 * j + 1) * (2 * j + 2)) / (j + 1), r1 = (double) (-(2 * j + 2) * (2 * j + 3)) / (j + 1), r2 = (double) (-(2 * j + 3) * (2 * j + 4)) / (j + 1);
        cdouble R2 = cscale(r2, w2);
        if (!(cabs2(R2) < 1)) break;
        a0 = cmul(a0, cscale(r0, w2)); a1 = cmul(a1, cscale(r1, w2)); a2 = cmul(a2, R2);
        s0 = cadd(s0, a0); s1 = cadd(s1, a1); s2 = cadd(s2, a2);
        j++;
        if (cabs2(a0) < 1e-34 * cabs2(s0) && cabs2(a2) < 1e-34 * cabs2(s2)) break;
    }
    h[0] = s0; h[1] = s1; h[2] = s2;
}
/* h_k(z), k = 0..3 (ctr_h3) */
static void ctr_h3(cdouble z, cdouble *h)
{
    if (cabs2(z) < 36) {
        cdouble h0 = cscale(sqrt(M_PI) / 2, erfcx_(z)), z2 = cscale(2, z);
        cdouble h1 = cscale(0.5, csub(DONE, cmul(z2, h0))), h2 = cscale(0.5, csub(h0, cmul(z2, h1)));
        h[0] = h0; h[1] = h1; h[2] = h2; h[3] = csub(h1, cmul(z, h2));
        return;
    }
    cdouble w = cinv(cscale(2, z)), w2 = cmul(w, w);
    cdouble a0 = w, a1 = w2, a2 = cmul(cscale(2, w2), w), a3 = cscale(6, cmul(w2, w2)), s0 = a0, s1 = a1, s2 = a2, s3 = a3;
    for (int j = 0; ; ) {
        double r0 = (double) (-(2 * j + 1) * (2 * j + 2)) / (j + 1), r1 = (double) (-(2 * j + 2) * (2 * j + 3)) / (j + 1), r2 = (double) (-(2 * j + 3) * (2 * j + 4)) / (j + 1),
               r3 = (double) (-(2 * j + 4) * (2 * j + 5)) / (j + 1);
        cdouble R3 = cscale(r3, w2);
        if (!(cabs2(R3) < 1)) break;
        a0 = cmul(a0, cscale(r0, w2)); a1 = cmul(a1, cscale(r1, w2)); a2 = cmul(a2, cscale(r2, w2)); a3 = cmul(a3, R3);
        s0 = cadd(s0, a0); s1 = cadd(s1, a1); s2 = cadd(s2, a2); s3 = cadd(s3, a3);
        j++;
        if (cabs2(a0) < 1e-34 * cabs2(s0) && cabs2(a3) < 1e-34 * cabs2(s3)) break;
    }
    h[0] = s0; h[1] = s1; h[2] = s2; h[3] = s3;
}
static int ctr_rend(double a, cdouble sa, cdouble beta, cdouble Ea, cdouble *R)
{
    cdouble z = cadd(cscale(a, sa), cdiv(beta, cscale(2, sa))), g[3]; int n;
    if (creal(z) >= 0) { n = 0; ctr_h(z, g); }
    else { cdouble hh[3]; n = 1; ctr_h(cneg(z), hh); g[0] = cneg(hh[0]); g[1] = hh[1]; g[2] = cneg(hh[2]); }
    cdouble i1 = cinv(sa), i2 = cmul(i1, i1), i3 = cmul(i2, i1);
    R[0] = cmul(cmul(Ea, i1), g[0]);
    R[1] = cmul(Ea, cadd(cmul(cscale(a, i1), g[0]), cmul(i2, g[1])));
    R[2] = cmul(Ea, cadd(cadd(cmul(cscale(a * a, i1), g[0]), cmul(cscale(2 * a, i2), g[1])), cmul(i3, g[2])));
    return n;
}
static int ctr_rend3(double a, cdouble sa, cdouble beta, cdouble Ea, cdouble *R)
{
    cdouble z = cadd(cscale(a, sa), cdiv(beta, cscale(2, sa))), g[4]; int n;
    if (creal(z) >= 0) { n = 0; ctr_h3(z, g); }
    else { cdouble hh[4]; n = 1; ctr_h3(cneg(z), hh); g[0] = cneg(hh[0]); g[1] = hh[1]; g[2] = cneg(hh[2]); g[3] = hh[3]; }
    cdouble i1 = cinv(sa), i2 = cmul(i1, i1), i3 = cmul(i2, i1), i4 = cmul(i3, i1);
    R[0] = cmul(cmul(Ea, i1), g[0]);
    R[1] = cmul(Ea, cadd(cmul(cscale(a, i1), g[0]), cmul(i2, g[1])));
    R[2] = cmul(Ea, cadd(cadd(cmul(cscale(a * a, i1), g[0]), cmul(cscale(2 * a, i2), g[1])), cmul(i3, g[2])));
    R[3] = cmul(Ea, cadd(cadd(cmul(cscale(a * a * a, i1), g[0]), cmul(cscale(3 * a * a, i2), g[1])), cadd(cmul(cscale(3 * a, i3), g[2]), cmul(i4, g[3]))));
    return n;
}
static inline cdouble expX(cdouble t, cdouble s, double c0, double q, double A, double v)
{   /* e^{-t v^2 - s X(v)}, X(v) = c0 v^2 + q v + A */
    return cexp(csub(cscale(-v * v, t), cscale(c0 * v * v + q * v + A, s)));
}
/* int over the pieces of v^m e^{-t v^2 - s X(v)} dv, m = 0, 1, 2 (ctr_moments) */
static void ctr_moments(cdouble t, cdouble s, double c0, double q, double A, const pcs_t *pc, cdouble *M)
{
    cdouble alpha = c0 == 0 ? t : cadd(t, cscale(c0, s)), beta = cscale(q, s), sa = csq(alpha), R[3];
    cdouble M0 = DZERO, M1 = DZERO, M2 = DZERO; int nf = 0;
    for (int k = 0; k < pc->n; k++) {
        double a = pc->a[k], b = pc->b[k];
        nf += ctr_rend(a, sa, beta, expX(t, s, c0, q, A, a), R);
        M0 = cadd(M0, R[0]); M1 = cadd(M1, R[1]); M2 = cadd(M2, R[2]);
        if (isfinite(b)) {
            nf -= ctr_rend(b, sa, beta, expX(t, s, c0, q, A, b), R);
            M0 = csub(M0, R[0]); M1 = csub(M1, R[1]); M2 = csub(M2, R[2]);
        }
    }
    if (nf != 0) {
        cdouble G0 = cmul(cdiv(CPX(sqrt(M_PI), 0), sa), cexp(csub(cdiv(cmul(beta, beta), cscale(4, alpha)), cscale(A, s))));
        cdouble b2 = cdiv(beta, cscale(2, alpha));
        M0 = cadd(M0, cscale(nf, G0));
        M1 = csub(M1, cmul(cscale(nf, b2), G0));
        M2 = cadd(M2, cmul(cscale(nf, cadd(cmul(b2, b2), cinv(cscale(2, alpha)))), G0));
    }
    M[0] = M0; M[1] = M1; M[2] = M2;
}
static void ctr_moments3(cdouble t, cdouble s, double c0, double q, double A, const pcs_t *pc, cdouble *M)
{
    cdouble alpha = c0 == 0 ? t : cadd(t, cscale(c0, s)), beta = cscale(q, s), sa = csq(alpha), R[4];
    cdouble M0 = DZERO, M1 = DZERO, M2 = DZERO, M3 = DZERO; int nf = 0;
    for (int k = 0; k < pc->n; k++) {
        double a = pc->a[k], b = pc->b[k];
        nf += ctr_rend3(a, sa, beta, expX(t, s, c0, q, A, a), R);
        M0 = cadd(M0, R[0]); M1 = cadd(M1, R[1]); M2 = cadd(M2, R[2]); M3 = cadd(M3, R[3]);
        if (isfinite(b)) {
            nf -= ctr_rend3(b, sa, beta, expX(t, s, c0, q, A, b), R);
            M0 = csub(M0, R[0]); M1 = csub(M1, R[1]); M2 = csub(M2, R[2]); M3 = csub(M3, R[3]);
        }
    }
    if (nf != 0) {
        cdouble G0 = cmul(cdiv(CPX(sqrt(M_PI), 0), sa), cexp(csub(cdiv(cmul(beta, beta), cscale(4, alpha)), cscale(A, s))));
        cdouble b2 = cdiv(beta, cscale(2, alpha)), i2a = cinv(cscale(2, alpha));
        M0 = cadd(M0, cscale(nf, G0));
        M1 = csub(M1, cmul(cscale(nf, b2), G0));
        M2 = cadd(M2, cmul(cscale(nf, cadd(cmul(b2, b2), i2a)), G0));
        M3 = csub(M3, cmul(cscale(nf, cadd(cmul(cmul(b2, b2), b2), cmul(cscale(3, b2), i2a))), G0));
    }
    M[0] = M0; M[1] = M1; M[2] = M2; M[3] = M3;
}
/* ctr_moments with the factors of the node given: sa = sqrt(t + c0 s), i1 = 1/sa, eA = e^{-A s} (ctr_moments_pre) */
static void ctr_moments_pre(cdouble t, cdouble s, double c0, double q, double A, const pcs_t *pc, cdouble sa, cdouble i1, cdouble eA, cdouble *M)
{
    cdouble beta = cscale(q, s), hb = cscale(0.5, cmul(beta, i1)), i2 = cmul(i1, i1), i3 = cmul(i2, i1), g[3], hh[3];
    cdouble M0 = DZERO, M1 = DZERO, M2 = DZERO; int nf = 0;
    for (int k = 0; k < pc->n; k++) {
        double a = pc->a[k], b = pc->b[k];
        cdouble Ea = a == 0 ? eA : expX(t, s, c0, q, A, a);
        cdouble z = cadd(cscale(a, sa), hb);
        if (creal(z) >= 0) ctr_h(z, g); else { ctr_h(cneg(z), hh); g[0] = cneg(hh[0]); g[1] = hh[1]; g[2] = cneg(hh[2]); nf++; }
        M0 = cadd(M0, cmul(cmul(Ea, i1), g[0]));
        M1 = cadd(M1, cmul(Ea, cadd(cmul(cscale(a, i1), g[0]), cmul(i2, g[1]))));
        M2 = cadd(M2, cmul(Ea, cadd(cadd(cmul(cscale(a * a, i1), g[0]), cmul(cscale(2 * a, i2), g[1])), cmul(i3, g[2]))));
        if (isfinite(b)) {
            cdouble Eb = expX(t, s, c0, q, A, b);
            z = cadd(cscale(b, sa), hb);
            if (creal(z) >= 0) ctr_h(z, g); else { ctr_h(cneg(z), hh); g[0] = cneg(hh[0]); g[1] = hh[1]; g[2] = cneg(hh[2]); nf--; }
            M0 = csub(M0, cmul(cmul(Eb, i1), g[0]));
            M1 = csub(M1, cmul(Eb, cadd(cmul(cscale(b, i1), g[0]), cmul(i2, g[1]))));
            M2 = csub(M2, cmul(Eb, cadd(cadd(cmul(cscale(b * b, i1), g[0]), cmul(cscale(2 * b, i2), g[1])), cmul(i3, g[2]))));
        }
    }
    if (nf != 0) {
        cdouble alpha = cmul(sa, sa);
        cdouble G0 = cmul(cscale(sqrt(M_PI), i1), cexp(csub(cdiv(cmul(beta, beta), cscale(4, alpha)), cscale(A, s))));
        cdouble b2 = cdiv(beta, cscale(2, alpha));
        M0 = cadd(M0, cscale(nf, G0));
        M1 = csub(M1, cmul(cscale(nf, b2), G0));
        M2 = cadd(M2, cmul(cscale(nf, cadd(cmul(b2, b2), cinv(cscale(2, alpha)))), G0));
    }
    M[0] = M0; M[1] = M1; M[2] = M2;
}
/* the terms of ctr_rend at the end v with the factor E in place of e^{-t v^2 - s X(v)} (rend_part) */
static void rend_part(double v, cdouble sa, cdouble i1, cdouble beta, cdouble E, cdouble *e)
{
    cdouble z = cadd(cscale(v, sa), cscale(0.5, cmul(beta, i1))), g[3], hh[3];
    if (creal(z) >= 0) ctr_h(z, g); else { ctr_h(cneg(z), hh); g[0] = cneg(hh[0]); g[1] = hh[1]; g[2] = cneg(hh[2]); }
    cdouble i2 = cmul(i1, i1), i3 = cmul(i2, i1);
    e[0] = cmul(cmul(E, i1), g[0]);
    e[1] = cmul(E, cadd(cmul(cscale(v, i1), g[0]), cmul(i2, g[1])));
    e[2] = cmul(E, cadd(cadd(cmul(cscale(v * v, i1), g[0]), cmul(cscale(2 * v, i2), g[1])), cmul(i3, g[2])));
}

/* ---------- the pieces of v ---------- */
/* {v > 0} split at the positive roots of X(v) = c0 v^2 + x v + A (xpieces) */
static void xpieces(double x, double c0, double A, pcs_t *plus, pcs_t *minus)
{
    plus->n = minus->n = 0;
    if (c0 == 0) {
        if (!(x < 0)) { pcs_add(plus, 0, INFINITY); return; }
        double v = -A / x; pcs_add(plus, 0, v); pcs_add(minus, v, INFINITY); return;
    }
    double dsc = x * x - 4 * c0 * A, r[2]; int nr = 0;
    if (dsc > 0) {
        double y1 = (-x - sqrt(dsc)) / (2 * c0), y2 = (-x + sqrt(dsc)) / (2 * c0);
        if (y1 > 0) r[nr++] = y1;
        if (y2 > 0) r[nr++] = y2;
        if (nr == 2 && r[1] < r[0]) { double tt = r[0]; r[0] = r[1]; r[1] = tt; }
    }
    double bk[4]; int nb = 0; bk[nb++] = 0; for (int k = 0; k < nr; k++) bk[nb++] = r[k]; bk[nb++] = INFINITY;
    for (int j = 0; j < nb - 1; j++) pcs_add(j % 2 == 0 ? plus : minus, bk[j], bk[j + 1]);
}
/* the pieces of (0, vmax) with X > 0 and X < 0 for any sign of A (xpieces_gen) */
static void xpieces_gen(double q, double c0, double A, double vmax, pcs_t *plus, pcs_t *minus)
{
    double r[2]; int nr = 0;
    if (c0 == 0) { if (q != 0) { double v = -A / q; if (v > 0) r[nr++] = v; } }
    else {
        double dsc = q * q - 4 * c0 * A;
        if (dsc > 0) {
            double qq = -(q + (q >= 0 ? 1.0 : -1.0) * sqrt(dsc)) / 2, ys[2] = {qq / c0, A / qq};
            for (int i = 0; i < 2; i++) if (ys[i] > 0 && isfinite(ys[i])) r[nr++] = ys[i];
            if (nr == 2 && r[1] < r[0]) { double tt = r[0]; r[0] = r[1]; r[1] = tt; }
        }
    }
    double bk[4]; int nb = 0; bk[nb++] = 0; for (int k = 0; k < nr; k++) if (r[k] < vmax) bk[nb++] = r[k]; bk[nb++] = vmax;
    plus->n = minus->n = 0;
    for (int j = 0; j < nb - 1; j++) {
        double a = bk[j], b = bk[j + 1];
        if (!(b > a)) continue;
        double m = isfinite(b) ? (a + b) / 2 : a + 1.0, X = c0 * m * m + q * m + A;
        if (X > 0) pcs_add(plus, a, b); else if (X < 0) pcs_add(minus, a, b);
    }
}
/* the slowest decay scale of the v-kernel (xscale) */
static double xscale(double tr, double c0, double q, double A, const pcs_t *pc)
{
    if (pc->n == 0) return 0.0;
    double sc = 0;
    for (int i = 0; i < pc->n; i++) {
        double a = pc->a[i], b = pc->b[i], bb = isfinite(b) ? b : a + 6 / sqrt(tr), m = 0;
        for (int k = 0; k <= 32; k++) { double v = a + (bb - a) * k / 32; m = dmax(m, fabs(c0 * v * v + q * v + A)); }
        double sp = 0;
        if (c0 != 0) {
            double v0 = -q / (2 * c0);
            if (a < v0 && v0 < b) m = dmin(m, fabs(c0 * v0 * v0 + q * v0 + A));
            double es[2] = {a, b};
            for (int l = 0; l < 2; l++) { double e = es[l]; if (!(e > 0 && isfinite(e))) continue; sp = dmax(sp, fabs(c0) / dmax((2 * c0 * e + q) * (2 * c0 * e + q), 1e-12)); }
        }
        sc = dmax(sc, dmax(1 / dmax(m, 1e-12), sp));
    }
    return sc;
}
static void active(const pcs_t *in, double tr, pcs_t *out) { out->n = 0; for (int i = 0; i < in->n; i++) if (tr * in->a[i] * in->a[i] <= 50) pcs_add(out, in->a[i], in->b[i]); }

/* ---------- the U-shaped paths ---------- */
typedef struct { const cdouble *rts; int r, rp; cdouble lP; const det_fixt *fx; double g2; cdouble *Bf; } logmt_t;
static inline cdouble logMt(const logmt_t *L, cdouble s)
{
    cdouble v = det_logMtilde(L->rts, L->r, L->rp, L->lP, s);
    if (L->fx) v = cadd(v, cscale(L->g2, cadd(L->fx->cf, det_fixE1(L->fx, s, L->Bf))));
    return v;
}
static double relogMt(const logmt_t *L, cdouble s)
{
    double acc = creal(L->lP);
    for (int j = 0; j < L->rp; j++) acc -= log(cabs2(csub(L->rts[j], s))) / 4;
    for (int j = L->rp; j < L->r; j++) acc -= log(cabs2(csub(s, L->rts[j]))) / 4;
    if (L->fx) acc += L->g2 * creal(cadd(L->fx->cf, det_fixE1(L->fx, s, L->Bf)));
    return acc;
}
typedef struct { double a, b; } pan_t;
typedef struct { pan_t *v; int n, cap; } panv;
static void pan_push(panv *o, double a, double b) { if (o->n == o->cap) { o->cap = o->cap ? 2 * o->cap : 32; o->v = realloc(o->v, o->cap * sizeof(pan_t)); } o->v[o->n].a = a; o->v[o->n].b = b; o->n++; }
static void seg_panels(panv *out, double st, double en, double h0)
{
    double len = fabs(en - st), dir = en >= st ? 1.0 : -1.0;
    if (len == 0) return;
    double a = 0, h = isfinite(h0) ? dmin(h0, len) : len;
    while (a < len * (1 - 1e-14)) {
        double b = dmin(a + h, len);
        if ((len - b) < 0.5 * h) b = len;
        double u1 = st + dir * a, u2 = st + dir * b;
        pan_push(out, dmin(u1, u2), dmax(u1, u2));
        a = b; h *= 2;
    }
}
typedef struct { double pos, h; } foc_t;
/* panels of [A, B] graded toward the focus points (graded_panels).  Points within 1e-12 (B - A) of each other or of an
 * end are merged with it, a tolerance relative to the length of the interval: far above the unit root the segments of
 * the U-shapes are tiny (the positive root lies within about 2c e^{-2c} of the branch point -t/c0 of the coefficient
 * statistic, 1e-16 at c = 20), and with the tolerance 1e-12 max(B - A, 1) used before a segment shorter than 1e-12 got
 * no panel at all (from c = 16 on the coefficient statistic was off by up to 0.2) */
static void graded_panels(double A, double B, const foc_t *foc, int nf, panv *out)
{
    out->n = 0;
    int *ord = malloc((nf + 1) * sizeof(int));
    for (int i = 0; i < nf; i++) ord[i] = i;
    for (int i = 1; i < nf; i++) { int k = ord[i], j = i - 1; while (j >= 0 && foc[ord[j]].h > foc[k].h) { ord[j + 1] = ord[j]; j--; } ord[j + 1] = k; }
    double *kl = malloc((nf + 1) * sizeof(double)), *kh = malloc((nf + 1) * sizeof(double)); int nk = 0;
    for (int ii = 0; ii < nf; ii++) {
        int k = ord[ii]; double pos = dclamp(foc[k].pos, A, B), h = foc[k].h; int skip = 0;
        for (int l = 0; l < nk; l++) if (fabs(pos - kl[l]) <= h) { skip = 1; break; }
        if (skip) continue;
        kl[nk] = pos; kh[nk] = h; nk++;
    }
    for (int i = 0; i < nk; i++) ord[i] = i;
    for (int i = 1; i < nk; i++) { int k = ord[i], j = i - 1; while (j >= 0 && kl[ord[j]] > kl[k]) { ord[j + 1] = ord[j]; j--; } ord[j + 1] = k; }
    double tol = 1e-12 * (B - A);
    double *pp = malloc((nk + 2) * sizeof(double)), *ph = malloc((nk + 2) * sizeof(double)); int np = 0;
    pp[np] = A; ph[np] = INFINITY; np++;
    for (int ii = 0; ii < nk; ii++) {
        int k = ord[ii];
        if (fabs(kl[k] - pp[np - 1]) <= tol) ph[np - 1] = dmin(ph[np - 1], kh[k]);
        else { pp[np] = kl[k]; ph[np] = kh[k]; np++; }
    }
    if (fabs(B - pp[np - 1]) <= tol) pp[np - 1] = B; else { pp[np] = B; ph[np] = INFINITY; np++; }
    for (int k = 0; k < np - 1; k++) {
        double a = pp[k], ha = ph[k], b = pp[k + 1], hb = ph[k + 1];
        if (b - a <= 0) continue;
        if (!isfinite(ha) && !isfinite(hb)) pan_push(out, a, b);
        else if (!isfinite(ha)) seg_panels(out, b, a, hb);
        else if (!isfinite(hb)) seg_panels(out, a, b, ha);
        else { double m = (a + b) / 2; seg_panels(out, a, m, ha); seg_panels(out, b, m, hb); }
    }
    free(ord); free(kl); free(kh); free(pp); free(ph);
}
static void ulevels(const cdouble *grp, int ng, double sig, int side, int wide, double *ylo, double *yhi)
{
    if (wide) {
        double lo = INFINITY, hi = -INFINITY;
        for (int i = 0; i < ng; i++) { double d = fabs(creal(grp[i]) - sig); lo = dmin(lo, cimag(grp[i]) - d); hi = dmax(hi, cimag(grp[i]) + d); }
        *ylo = lo; *yhi = hi; return;
    }
    double d = INFINITY, mi = INFINITY, ma = -INFINITY;
    for (int i = 0; i < ng; i++) { d = dmin(d, side * (creal(grp[i]) - sig)); mi = dmin(mi, cimag(grp[i])); ma = dmax(ma, cimag(grp[i])); }
    *ylo = mi - d; *yhi = ma + d;
}
#define NSPLIT_MAX 64
/* the number of parts of a straight panel of a U-shape of the fixed start (nsplit) */
static int nsplit(const logmt_t *lm, cdouble za, cdouble zb, int n, double A, int side)
{
    if (!lm) return 1;
    double len = cabs(csub(zb, za));
    if (!(len > 0)) return 1;
    cdouble zm = cscale(0.5, cadd(za, zb)), La = logMt(lm, za), Lm = logMt(lm, zm), Lb = logMt(lm, zb);
    if (!(cisfinite(La) && cisfinite(Lm) && cisfinite(Lb))) return 1;
#define KB(z) (side == 1 ? dmax(0.0, -A * creal(z)) : 0.0)
    double big = dmax(dmax(creal(La) + KB(za), creal(Lm) + KB(zm)), creal(Lb) + KB(zb)) + log(len);
#undef KB
    if (big < -40) return 1;
    double om = dmax(cabs(csub(Lm, La)), cabs(csub(Lb, Lm))) / (len / 2) + A;
    double m = ceil(dmin(om * len / n, 1e6));
    return (int) dclamp(m, 1, NSPLIT_MAX);
}
/* ushape!: the nodes S and weights W of (1/2 pi i) int over the U-shape around the group grp */
static int ushape(cvec *S, cvec *W, const cdouble *grp, int ng, const cdouble *oth, int no, double sig, int side, int n, int ntail,
                  const cdouble *extra, int nex, double rs, double hmax, const logmt_t *lm, double akern, det_status *st)
{
    const double *w, *z = det_gl(n, &w), *wt, *zt = det_gl(ntail, &wt);
    double d = INFINITY; for (int i = 0; i < ng; i++) d = dmin(d, side * (creal(grp[i]) - sig));
    if (!(d > 0)) { det_fail(st, "ushape: segment not separated from its roots"); return 1; }
    double ylo, yhi; ulevels(grp, ng, sig, side, lm != NULL, &ylo, &yhi);
    int nf = 0; foc_t *foc = malloc((ng + no + nex + 2) * sizeof(foc_t));
    for (int i = 0; i < ng; i++) { foc[nf].pos = cimag(grp[i]); foc[nf].h = dmax(fabs(creal(grp[i]) - sig), 1e-6 * d); nf++; }
    for (int i = 0; i < no; i++) { foc[nf].pos = cimag(oth[i]); foc[nf].h = dmax(fabs(creal(oth[i]) - sig), 1e-6 * d); nf++; }
    foc[nf].pos = 0; foc[nf].h = dmax(fabs(sig), 1e-6 * d); nf++;
    for (int i = 0; i < nex; i++) { foc[nf].pos = cimag(extra[i]); foc[nf].h = dmax(fabs(creal(extra[i]) - sig), 1e-6 * d); nf++; }
    panv pv = {0};
    graded_panels(ylo, yhi, foc, nf, &pv);
    int start = S->n;
    for (int ip = 0; ip < pv.n; ip++) {
        double a0 = pv.v[ip].a, b0 = pv.v[ip].b;
        int m = isfinite(hmax) ? imax(1, (int) ceil((b0 - a0) / hmax)) : 1;
        for (int j = 0; j < m; j++) {
            double a1 = a0 + (b0 - a0) * j / m, b1 = a0 + (b0 - a0) * (j + 1) / m;
            int ms = nsplit(lm, CPX(sig, a1), CPX(sig, b1), n, akern, side);
            for (int i = 0; i < ms; i++) {
                double a = a1 + (b1 - a1) * i / ms, b = a1 + (b1 - a1) * (i + 1) / ms;
                for (int q = 0; q < n; q++) { cvec_push(S, CPX(sig, a + (b - a) * z[q])); cvec_push(W, CPX(0, (b - a) * w[q])); }
            }
        }
    }
    foc_t *rf = malloc((ng + 1) * sizeof(foc_t));
    for (int pass = 0; pass < 2; pass++) {
        double yc = pass == 0 ? yhi : ylo; int dir = pass == 0 ? 1 : -1, sgn = side == 1 ? dir : -dir;
        double mx = -INFINITY;
        for (int i = 0; i < ng; i++) { rf[i].pos = side * (creal(grp[i]) - sig); rf[i].h = dmax(fabs(cimag(grp[i]) - yc), 1e-6 * d); mx = dmax(mx, rf[i].pos); }
        double R = dmax(mx + dmax(2 * d, 1.0), rs);
        graded_panels(0.0, R, rf, ng, &pv);
        for (int ip = 0; ip < pv.n; ip++) {
            double a1 = pv.v[ip].a, b1 = pv.v[ip].b;
            int ms = nsplit(lm, CPX(sig + side * a1, yc), CPX(sig + side * b1, yc), n, akern, side);
            for (int i = 0; i < ms; i++) {
                double a = a1 + (b1 - a1) * i / ms, b = a1 + (b1 - a1) * (i + 1) / ms;
                for (int q = 0; q < n; q++) { cvec_push(S, CPX(sig + side * (a + (b - a) * z[q]), yc)); cvec_push(W, CPX(sgn * (b - a) * w[q], 0)); }
            }
        }
        double Rt = dmax(R, 2.0);
        for (int q = 0; q < ntail; q++) {
            double u = zt[q], yv = u / (1 - u), r = R + Rt * yv * yv, dr = 2 * Rt * yv / ((1 - u) * (1 - u));
            cvec_push(S, CPX(sig + side * r, yc)); cvec_push(W, CPX(sgn * dr * wt[q], 0));
        }
    }
    cdouble i2p = cinv(CPX(0, 2 * M_PI));
    for (int i = start; i < W->n; i++) W->v[i] = cmul(W->v[i], i2p);
    free(foc); free(rf); free(pv.v);
    return 0;
}
/* place: the abscissa of the U-shape of the fixed start in (lo, hi) */
typedef struct { const logmt_t *L; const cdouble *grp; int ng; const cdouble *oth; int no; int side; double A, xq, tr, c0p, foc[3]; int nfoc; } place_ctx;
static double place_obj(const place_ctx *P, double sig)
{
    int side = P->side;
    double d = INFINITY, span = -INFINITY;
    for (int i = 0; i < P->ng; i++) { d = dmin(d, side * (creal(P->grp[i]) - sig)); span = dmax(span, side * (creal(P->grp[i]) - sig)); }
    span += d;
    double ylo, yhi; ulevels(P->grp, P->ng, sig, side, 1, &ylo, &yhi);
    double m = -INFINITY;
#define PR(s) do { cdouble s_ = (s); double v_ = relogMt(P->L, s_); if (side == 1) { v_ += dmax(0.0, -P->A * creal(s_)); if (P->xq > 0 && creal(s_) < 0) { double dd_ = P->tr + P->c0p * creal(s_); v_ += dd_ > 0 ? creal(s_) * creal(s_) * P->xq * P->xq / (4 * dd_) : INFINITY; } } m = dmax(m, isfinite(v_) ? v_ : INFINITY); } while (0)
    PR(CPX(sig, ylo)); PR(CPX(sig, yhi)); PR(CPX(sig, dclamp(0.0, ylo, yhi)));
    for (int i = 0; i < P->ng; i++) { PR(CPX(sig, cimag(P->grp[i]))); PR(CPX(creal(P->grp[i]), yhi)); PR(CPX(creal(P->grp[i]), ylo)); }
    for (int i = 0; i < P->no; i++) PR(CPX(sig, dclamp(cimag(P->oth[i]), ylo, yhi)));
    double ys[2] = {ylo, yhi};
    for (int l = 0; l < 2; l++) { PR(CPX(sig + side * span, ys[l])); PR(CPX(sig + side * 1e3 * dmax(span, 1.0), ys[l])); }
#undef PR
    for (int i = 0; i < P->nfoc; i++) m += 0.5 * log(1 + d / dmax(fabs(sig - P->foc[i]), 1e-300));
    return m;
}
static double place(const logmt_t *L, const cdouble *grp, int ng, const cdouble *oth, int no, int side, double lo, double hi, double A, double sig0,
                    double xq, double tr, double c0p)
{
    double edge = side == 1 ? INFINITY : -INFINITY;
    for (int i = 0; i < ng; i++) edge = side == 1 ? dmin(edge, creal(grp[i])) : dmax(edge, creal(grp[i]));
    place_ctx P = {L, grp, ng, oth, no, side, A, xq, tr, c0p, {0, 0, 0}, 1};
    if (no > 0) {
        double e = side == 1 ? -INFINITY : INFINITY;
        for (int i = 0; i < no; i++) e = side == 1 ? dmax(e, creal(oth[i])) : dmin(e, creal(oth[i]));
        P.foc[1] = e; P.nfoc = 2;
    }
    /* (above the unit root with c0 > 0 the branch point -Re(t)/c0 of the v-kernels, the left end of the admissible interval
     * of G+, is a focus of the grading cost too: the kernel of the density of the coefficient statistic has a double pole
     * there, (t + c0 s)^{-2}, which the objective did not see, and with a fixed start at c from 1 to 5 the density was off
     * by up to 0.12 (case DU, c = 5, g0 = 0.5, at the 90% point) without a warning; as place of DFDet.jl) */
    if (side == 1 && c0p > 0) P.foc[P.nfoc++] = -tr / c0p;
    double D = side == 1 ? edge - lo : hi - edge, m = 1e-3 * D, ds[24];
    const int nh = 12; int ng2 = 0;
    double g1[12];
    for (int k = 1; k <= nh; k++) g1[k - 1] = m * pow(D / 2 / m, (double) (k - 1) / (nh - 1));
    for (int k = 0; k < nh; k++) ds[ng2++] = g1[k];
    for (int k = 0; k < nh; k++) ds[ng2++] = D - g1[k];
    /* sort and unique */
    for (int i = 1; i < ng2; i++) { double v = ds[i]; int j = i - 1; while (j >= 0 && ds[j] > v) { ds[j + 1] = ds[j]; j--; } ds[j + 1] = v; }
    int nu = 0; for (int i = 0; i < ng2; i++) if (nu == 0 || ds[i] != ds[nu - 1]) ds[nu++] = ds[i];
    int kbest = 0; double vb = INFINITY;
    for (int i = 0; i < nu; i++) { double v = place_obj(&P, edge - side * ds[i]); if (i == 0 || v < vb) { vb = v; kbest = i; } }
    double a = ds[imax(kbest - 1, 0)], b = ds[imin(kbest + 1, nu - 1)], g = (sqrt(5.0) - 1) / 2;
    double x1 = b - g * (b - a), x2 = a + g * (b - a), f1 = place_obj(&P, edge - side * x1), f2 = place_obj(&P, edge - side * x2);
    for (int it = 0; it < 12; it++) {
        if (f1 < f2) { b = x2; x2 = x1; f2 = f1; x1 = b - g * (b - a); f1 = place_obj(&P, edge - side * x1); }
        else { a = x1; x1 = x2; f1 = f2; x2 = a + g * (b - a); f2 = place_obj(&P, edge - side * x2); }
    }
    double sb = edge - side * (f1 < f2 ? x1 : x2);
    return dmin(f1, f2) < place_obj(&P, sig0) - 2 ? sb : sig0;
}

/* the gap (lo, hi) between the root groups of a node and the default abscissas sp, sm of its U-shapes (uabscissas) */
static void uabscissas(const cdouble *gp, int rp, const cdouble *gm, int rm, double alpha, double *hi_, double *lo_, double *sp, double *sm)
{
    double hi = INFINITY, lo = -INFINITY;
    for (int i = 0; i < rp; i++) hi = dmin(hi, creal(gp[i]));
    if (rm > 0) for (int i = 0; i < rm; i++) lo = dmax(lo, creal(gm[i]));
    if (hi > 0 && lo < 0) { *sp = alpha * hi; *sm = alpha * lo; }
    else { double lo2 = dmax(lo, hi - 2 * dmax(fabs(hi), 1.0)); *sp = hi - alpha * (hi - lo2); *sm = lo2 + alpha * (dmin(0.0, hi) - lo2); }
    *hi_ = hi; *lo_ = lo;
}
/* sigma+ moved right of the branch point -Re(t)/c0 of every c0 > 0 (ubranch; NaN where a branch point is not left of hi) */
static double ubranch(double sp, double hi, double tr, const double *c0s, int nc0, double alpha)
{
    for (int i = 0; i < nc0; i++) {
        if (!(c0s[i] > 0)) continue;
        double lb = -tr / c0s[i];
        if (!(lb < hi)) return NAN;
        if (sp <= lb) sp = lb + alpha * (hi - lb);
    }
    return sp;
}
/* the caps of the panels of the segments of G+ and G- (ucaps): 20/A, below c = DET_FAR_CMIN = -60 20/min(A, 40/|sigma|) */
static void ucaps(double c, double sp, double sm, double A, double *hp, double *hm)
{
    double hseg = 20 / A;
    *hp = c < DET_FAR_CMIN && sp > 0 ? 20 / dmin(A, 40 / sp) : hseg; *hm = c < DET_FAR_CMIN && sm < 0 ? 20 / dmin(A, 40 / fabs(sm)) : hseg;
}
/* the nodes of the segments of the narrow U-shapes of the zero and the stationary start, a lower bound of their count
 * (within 1% of it beyond 1e5), and the longest segment (unarrow) */
static double unarrow(const cdouble *gp, int rp, const cdouble *gm, int rm, double sp, double sm, double hp, double hm, int n, double *hseg)
{
    double nest = 0, ylo, yhi; *hseg = 0;
    if (rp > 0) { ulevels(gp, rp, sp, 1, 0, &ylo, &yhi); nest += (yhi - ylo) / hp * n; *hseg = dmax(*hseg, yhi - ylo); }
    if (rm > 0) { ulevels(gm, rm, sm, -1, 0, &ylo, &yhi); nest += (yhi - ylo) / hm * n; *hseg = dmax(*hseg, yhi - ylo); }
    return nest;
}
/* the count of the wide U-shapes of the fixed start at the abscissas sp, sm (the heights to hmx) */
static double ufixed_count(const cdouble *rts, int r, double sp, double sm, double hp, double hm, int n, double *hmx)
{
    *hmx = 0;
    for (int i = 0; i < r; i++) { cdouble z = rts[i]; *hmx = dmax(*hmx, fabs(cimag(z)) + fabs(creal(z) - (creal(z) > 0 ? sp : sm))); }
    return 2 * *hmx / dmin(hp, hm) * n;
}
/* the abscissas of the U-shapes of the fixed start (ufixed_place): the defaults replaced by those of place */
static void ufixed_place(const det_path *CP, const logmt_t *L, const double *c0s, int nc0, const cdouble *gp, int rp, const cdouble *gm, int rm,
                         double hi, double lo, double tr, double alpha, double A, double xq, double *sp, double *sm)
{
    double xg = CP->c > 0 ? xq : 0.0, lbp = -INFINITY, c0p = 0;
    if (CP->c > 0) for (int i = 0; i < nc0; i++) if (c0s[i] > 0) { lbp = dmax(lbp, -tr / c0s[i]); c0p = dmax(c0p, c0s[i]); }
    double sp0 = isfinite(lbp) ? dmax(*sp, lbp + alpha * (hi - lbp)) : *sp;
    if (rm > 0) {
        *sp = place(L, gp, rp, gm, rm, 1, dmax(lo, lbp), hi, A, sp0, xg, tr, c0p);
        *sm = place(L, gm, rm, gp, rp, -1, lo, dmin(hi, 0.0), A, *sm, 0.0, 1.0, 0.0);
    } else *sp = place(L, gp, rp, gm, rm, 1, dmax(hi - 2 * dmax(fabs(hi), 1.0), lbp), hi, A, sp0, xg, tr, c0p);
}
#define USHAPE_MSG "the U-shapes of the %s start at t = %g%+gi would need some %.3g nodes (segments to %.3g long): a root runs off (more than %.0f nodes)"
#define USHAPE_MSG_FIXED "the U-shapes of the fixed start at t = %g%+gi would need some %.0f nodes (heights to %.3g): the roots run off (more than %.0f nodes)"
static void logmt_of(const det_path *CP, int k, logmt_t *L, cdouble *Bf);
/* ushape_guard: the guard on the nodes of the U-shapes before any U-shape is built or any inner integral evaluated (the
 * outer path carries the roots at its nodes): at every node t_k the count that the guard of ushapes compares with
 * DET_USHAPE_MAXNODES, the segments of the narrow U-shapes with the zero and the stationary start, the wide heights at
 * the abscissas of place with a fixed start (placed only where an upper bound of the count exceeds the cap: every abscissa
 * lies in the interval of place, every cap of the segments is at least 20/A); the first node beyond the cap at which the
 * transform has not fallen below the tolerance tol of the outer sum, |M_V(t_k) t'_k| max(1, 1/|t_k|) >= tol, refuses the
 * call with the error of ushapes (1 and the message in err).  The nodes where the transform is below tol are left to the
 * guard of ushapes, as the outer sum stops before them unless its terms are not small there (d(r) = 1 + r on the
 * vertical line at c = 0 is beyond the cap from |t| = 3e4 on, where |M_V| is 1e-36, and its sum stops at |t| = 1.6e4).
 * The design (1 + r, 1{r <= 0.4}) on the vertical line is beyond the cap from |t| = 6300 on, where |M_V| is 1e-15: it
 * took seconds and some 0.8 GB before ushapes refused, and is refused here before any inner integral (ushape_guard of
 * DFDet.jl) */
static int ushape_guard(const det_path *CP, const double *c0s, int nc0, double A, int n, double alpha, double tol, double xq, char *err, size_t errlen)
{
    const det_model *Mo = CP->Mo;
    int r = CP->r, rp = Mo->rp, rm = Mo->rm, fixed = Mo->start == 1;
    double lt = log(tol);
    cdouble *Bf = NULL;
    for (int k = 0; k < CP->K; k++) {
        const cdouble *rts = CP->R + (size_t) k * r, *gp = rts, *gm = rts + rp;
        double hi, lo, sp, sm, hp, hm, nest = 0, hseg = 0, tr = creal(CP->t[k]);
        uabscissas(gp, rp, gm, rm, alpha, &hi, &lo, &sp, &sm);
        if (!(lo < dmin(hi, 0.0))) continue;                 /* (ushapes refuses such a node with its own error) */
        if (fixed) {
            /* (an upper bound of the count: sigma+ >= L, the lower end of the interval of place, sigma- <= min(hi, 0)) */
            double L = rm > 0 ? lo : hi - 2 * dmax(fabs(hi), 1.0), hmx = 0;
            for (int i = 0; i < rp; i++) hmx = dmax(hmx, fabs(cimag(gp[i])) + creal(gp[i]) - L);
            for (int i = 0; i < rm; i++) hmx = dmax(hmx, fabs(cimag(gm[i])) + dmin(hi, 0.0) - creal(gm[i]));
            if (2 * hmx * A / 20 * n <= DET_USHAPE_MAXNODES) continue;
        } else {
            sp = ubranch(sp, hi, tr, c0s, nc0, alpha);
            if (isnan(sp)) continue;
            ucaps(CP->c, sp, sm, A, &hp, &hm);
            nest = unarrow(gp, rp, gm, rm, sp, sm, hp, hm, n, &hseg);
            if (!(nest > DET_USHAPE_MAXNODES)) continue;
        }
        /* log |M_V(t_k)| = Re log Mt(0) (with a fixed start and g0^2 Re cf(t_k): E1(0) = 0) */
        const det_fixt *fx = CP->fx ? CP->fx[k] : NULL;
        double lmv = creal(det_logMtilde(rts, r, rp, CP->lP[k], DZERO));
        if (fx) lmv += Mo->g0 * Mo->g0 * creal(fx->cf);
        if (!(lmv + log(cabs(CP->dt[k]) * dmax(1.0, 1 / cabs(CP->t[k]))) >= lt)) continue;
        if (fixed) {
            if (!fx) continue;
            /* (the count of ushapes, with its placed abscissas) */
            if (!Bf) Bf = malloc((size_t) imax(r, 1) * (r + 2) * sizeof(cdouble));
            logmt_t Lm; logmt_of(CP, k, &Lm, Bf);
            ufixed_place(CP, &Lm, c0s, nc0, gp, rp, gm, rm, hi, lo, tr, alpha, A, xq, &sp, &sm);
            sp = ubranch(sp, hi, tr, c0s, nc0, alpha);
            if (isnan(sp)) continue;
            ucaps(CP->c, sp, sm, A, &hp, &hm);
            double hmx; nest = ufixed_count(rts, r, sp, sm, hp, hm, n, &hmx);
            if (!(nest > DET_USHAPE_MAXNODES)) continue;
            snprintf(err, errlen, USHAPE_MSG_FIXED, creal(CP->t[k]), cimag(CP->t[k]), nest, hmx, (double) DET_USHAPE_MAXNODES);
            free(Bf); return 1;
        }
        snprintf(err, errlen, USHAPE_MSG, Mo->start == 2 ? "stationary" : "zero", creal(CP->t[k]), cimag(CP->t[k]), nest, hseg, (double) DET_USHAPE_MAXNODES);
        free(Bf); return 1;
    }
    free(Bf);
    return 0;
}

/* the U-shapes of node k: nodes Sp, Sm, Mt times the weights (Mp, Mm), the factors 1 + |log Mt| (Ep, Em) (ushapes) */
typedef struct { cvec Sp, Mp, Sm, Mm; dvec Ep, Em; } ushapes_t;
static void ushapes_free(ushapes_t *U) { cvec_free(&U->Sp); cvec_free(&U->Mp); cvec_free(&U->Sm); cvec_free(&U->Mm); dvec_free(&U->Ep); dvec_free(&U->Em); }
static int ushapes(const det_path *CP, int k, const logmt_t *L, const double *c0s, int nc0, double rsp, double rsm, double alpha, double A, int n, int ntail,
                   double xq, ushapes_t *U, det_status *st)
{
    const det_model *Mo = CP->Mo;
    memset(U, 0, sizeof *U);
    cdouble t = CP->t[k]; double tr = creal(t);
    int rp = Mo->rp, rm = Mo->rm;
    const cdouble *gp = L->rts, *gm = L->rts + rp;
    double hi, lo, sp, sm;
    uabscissas(gp, rp, gm, rm, alpha, &hi, &lo, &sp, &sm);
    if (!(lo < dmin(hi, 0.0))) { det_fail(st, "the root groups are not separated by a line Re s = sigma < 0 at t = %g%+gi (lo = %g, hi = %g)", creal(t), cimag(t), lo, hi); return 1; }
    cdouble exp_p[8], exp_m[8]; int nep = 0, nem = 0;
    int fixed = L->fx != NULL;
    if (fixed) ufixed_place(CP, L, c0s, nc0, gp, rp, gm, rm, hi, lo, tr, alpha, A, xq, &sp, &sm);
    for (int i = 0; i < nc0; i++) {
        double c0 = c0s[i];
        if (c0 == 0) continue;
        cdouble sb = cdivr(cneg(t), c0);
        if (c0 > 0) {
            if (!(-tr / c0 < hi)) { det_fail(st, "no abscissa for G+ right of the branch point -t/c0 at t = %g%+gi", creal(t), cimag(t)); return 1; }
            exp_p[nep++] = sb;
        } else exp_m[nem++] = sb;
    }
    sp = ubranch(sp, hi, tr, c0s, nc0, alpha);
    double hp, hm;
    ucaps(CP->c, sp, sm, A, &hp, &hm);
    /* (the guard on the size of the U-shapes, which ushape_guard checks before the outer sum at the nodes where the
     * transform is not negligible; here at the other nodes, which the sum reaches where its terms are not small) */
    if (fixed) {
        double hmx, nest = ufixed_count(L->rts, rp + rm, sp, sm, hp, hm, n, &hmx);
        if (nest > DET_USHAPE_MAXNODES) { det_fail(st, USHAPE_MSG_FIXED, creal(t), cimag(t), nest, hmx, (double) DET_USHAPE_MAXNODES); return 1; }
    } else {
        double hseg, nest = unarrow(gp, rp, gm, rm, sp, sm, hp, hm, n, &hseg);
        if (nest > DET_USHAPE_MAXNODES) { det_fail(st, USHAPE_MSG, Mo->start == 2 ? "stationary" : "zero", creal(t), cimag(t), nest, hseg, (double) DET_USHAPE_MAXNODES); return 1; }
    }
    const logmt_t *lmf = fixed ? L : NULL;
    cvec Wp = {0};
    if (ushape(&U->Sp, &Wp, gp, rp, gm, rm, sp, 1, n, ntail, exp_p, nep, rsp, hp, lmf, A, st)) { cvec_free(&Wp); return 1; }
    for (int j = 0; j < U->Sp.n; j++) {
        cdouble l = logMt(L, U->Sp.v[j]);
        cvec_push(&U->Mp, cmul(cexp(l), Wp.v[j])); dvec_push(&U->Ep, 1 + cabs(l));
    }
    cvec_free(&Wp);
    if (rm > 0) {
        cvec Wm = {0};
        if (ushape(&U->Sm, &Wm, gm, rm, gp, rp, sm, -1, n, ntail, exp_m, nem, rsm, hm, lmf, A, st)) { cvec_free(&Wm); return 1; }
        for (int j = 0; j < U->Sm.n; j++) {
            cdouble l = logMt(L, U->Sm.v[j]);
            cvec_push(&U->Mm, cmul(cexp(l), Wm.v[j])); dvec_push(&U->Em, 1 + cabs(l));
        }
        cvec_free(&Wm);
    }
    return 0;
}

/* ---------- the terms of one node ---------- */
typedef struct {
    const det_path *CP; int nx; const double *xs; double c0; int vpow, n, ntail; double alpha;
    const double *vr;                              /* Filon's rule: the root of each x (0: none), or NULL */
    double *GF, *Gf, *AF, *Af; cdouble *JF, *Jf;   /* K x nx, row-major in the node */
    int ne; const int *nt, *off; const det_xspec *specs;   /* the events (tnode_specs) */
    int k0;
    det_status st;
} node_ctx;
static void logmt_of(const det_path *CP, int k, logmt_t *L, cdouble *Bf)
{
    const det_model *Mo = CP->Mo;
    L->rts = CP->R + (size_t) k * CP->r; L->r = CP->r; L->rp = Mo->rp; L->lP = CP->lP[k];
    L->fx = CP->fx ? CP->fx[k] : NULL; L->g2 = Mo->g0 * Mo->g0; L->Bf = Bf;
}
static inline cdouble cps(cdouble s, double A)
{   /* (1 - e^{-A s})/s */
    return cabs2(s) < 1e-16 ? csub(CPX(A, 0), cscale(A * A / 2, s)) : cdiv(cneg(cexpm1(cscale(-A, s))), s);
}
/* tnode_inner!: the terms Im{I_F t'/t} and Im{I_f t'} of the t-ratio at node k for every x */
static void tnode_inner(node_ctx *o, int k)
{
    const det_path *CP = o->CP; const det_model *Mo = CP->Mo;
    double A = Mo->a, c0 = o->c0;
    cdouble t = CP->t[k], dt = CP->dt[k]; double tr = creal(t);
    int nx = o->nx;
    cdouble *Bf = malloc((size_t) imax(CP->r, 1) * (CP->r + 2) * sizeof(cdouble));
    logmt_t L; logmt_of(CP, k, &L, Bf);
    double rsp = 0, rsm = 0;
    pcs_t plus, minus, pa, ma;
    if (c0 != 0) {
        for (int ix = 0; ix < nx; ix++) {
            xpieces(o->xs[ix], c0, A, &plus, &minus); active(&plus, tr, &pa); active(&minus, tr, &ma);
            rsp = dmax(rsp, xscale(tr, c0, o->xs[ix], A, &pa)); rsm = dmax(rsm, xscale(tr, c0, o->xs[ix], A, &ma));
        }
        rsp = dmin(rsp, 1e4); rsm = dmin(rsm, 1e4);
    }
    double xq = 0; for (int ix = 0; ix < nx; ix++) xq = dmax(xq, o->xs[ix]);
    ushapes_t U;
    if (ushapes(CP, k, &L, &c0, 1, rsp, rsm, o->alpha, A, o->n, o->ntail, dmax(0.0, xq), &U, &o->st)) { ushapes_free(&U); free(Bf); return; }
    int np_ = U.Sp.n, nm_ = U.Sm.n;
    cdouble *sqp = malloc((np_ + nm_ + 1) * 4 * sizeof(cdouble)), *iqp = sqp + np_, *eAp = iqp + np_, *psp = eAp + np_;
    cdouble *sqm = psp + np_, *iqm = sqm + nm_, *eAm = iqm + nm_;
    for (int j = 0; j < np_; j++) { cdouble s = U.Sp.v[j]; sqp[j] = csq(c0 == 0 ? t : cadd(t, cscale(c0, s))); iqp[j] = cinv(sqp[j]); eAp[j] = cexp(cscale(-A, s)); psp[j] = cps(s, A); }
    for (int j = 0; j < nm_; j++) { cdouble s = U.Sm.v[j]; sqm[j] = csq(c0 == 0 ? t : cadd(t, cscale(c0, s))); iqm[j] = cinv(sqm[j]); eAm[j] = cexp(cscale(-A, s)); }
    for (int ix = 0; ix < nx; ix++) {
        double x = o->xs[ix];
        xpieces(x, c0, A, &plus, &minus); active(&plus, tr, &pa); active(&minus, tr, &ma);
        cdouble IF = DZERO, If = DZERO, JF = DZERO, Jf = DZERO; double aF = 0, af = 0;
        double v1 = o->vr ? o->vr[ix] : 0.0;
        int sgp = 0, sgm = 0;
        if (v1 > 0) {
            for (int i = 0; i < pa.n; i++) { if (pa.b[i] == v1) sgp--; if (pa.a[i] == v1) sgp++; }
            for (int i = 0; i < ma.n; i++) { if (ma.b[i] == v1) sgm--; if (ma.a[i] == v1) sgm++; }
        }
        cdouble M[4], e[4];
        for (int side = 0; side < 2; side++) {
            const pcs_t *pc = side == 0 ? &pa : &ma;
            int ns = side == 0 ? np_ : nm_, sg = side == 0 ? sgp : sgm;
            const cdouble *Ss = side == 0 ? U.Sp.v : U.Sm.v, *Ms = side == 0 ? U.Mp.v : U.Mm.v, *sq = side == 0 ? sqp : sqm, *iq = side == 0 ? iqp : iqm, *eA = side == 0 ? eAp : eAm;
            const double *Es = side == 0 ? U.Ep.v : U.Em.v;
            for (int j = 0; j < ns; j++) {
                cdouble s = Ss[j], m = Ms[j], tF, tf;
                cdouble kc = side == 0 ? psp[j] : cneg(cinv(s));
                if (o->vpow == 2) {
                    if (pc->n == 0) M[0] = M[1] = M[2] = DZERO; else ctr_moments_pre(t, s, c0, x, A, pc, sq[j], iq[j], eA[j], M);
                    tF = cmul(m, cadd(cadd(cscale(x, M[0]), cscale(2 * c0, M[1])), kc)); tf = cmul(m, cscale(2, M[2]));
                } else {
                    if (pc->n == 0) M[0] = M[1] = M[2] = M[3] = DZERO; else ctr_moments3(t, s, c0, x, A, pc, M);
                    tF = cmul(m, cadd(cadd(cscale(x, M[0]), cscale(2 * c0, M[1])), kc)); tf = cmul(m, cscale(2, M[3]));
                }
                if (sg != 0) {
                    cdouble E = cscale(sg, cexp(cscale(-(c0 * v1 * v1 + x * v1 + A), s)));
                    if (o->vpow == 2) { rend_part(v1, sq[j], iq[j], cscale(x, s), E, e); Jf = cadd(Jf, cmul(cscale(2, m), e[2])); }
                    else { ctr_rend3(v1, sq[j], cscale(x, s), E, e); Jf = cadd(Jf, cmul(cscale(2, m), e[3])); }
                    JF = cadd(JF, cmul(m, cadd(cscale(x, e[0]), cscale(2 * c0, e[1]))));
                }
                IF = cadd(IF, tF); If = cadd(If, tf); aF += Es[j] * cfabs(tF); af += Es[j] * cfabs(tf);
            }
        }
        size_t row = (size_t) (k - o->k0) * nx + ix;
        if (v1 > 0) {
            cdouble eb = cexp(cscale(-v1 * v1, t));
            IF = csub(IF, cmul(eb, JF)); If = csub(If, cmul(eb, Jf));
            o->JF[row] = cmul(JF, cdiv(dt, t)); o->Jf[row] = cmul(Jf, dt);
        }
        o->GF[row] = cimag(cmul(IF, cdiv(dt, t))); o->Gf[row] = cimag(cmul(If, dt));
        o->AF[row] = aF * cabs(cdiv(dt, t)); o->Af[row] = af * cabs(dt);
    }
    free(sqp); ushapes_free(&U); free(Bf);
}
/* psi_pm: the constant terms of the kernels on G+ and G- (the split of P(R <= A under the tilt) at R = 0) */
static inline void psi_pm(cdouble s, double A, cdouble *pp, cdouble *pm)
{
    if (A > 0) { *pp = cps(s, A); *pm = cneg(cinv(s)); return; }
    if (A < 0) { *pp = DZERO; *pm = cneg(cdiv(cexp(cscale(-A, s)), s)); return; }
    *pp = DZERO; *pm = cneg(cinv(s));
}
/* tnode_specs!: the terms of node k for the events made of parabola terms */
static void tnode_specs(node_ctx *o, int k)
{
    const det_path *CP = o->CP; const det_model *Mo = CP->Mo;
    cdouble t = CP->t[k], dt = CP->dt[k]; double tr = creal(t);
    cdouble *Bf = malloc((size_t) imax(CP->r, 1) * (CP->r + 2) * sizeof(cdouble));
    logmt_t L; logmt_of(CP, k, &L, Bf);
    int ntot = o->off[o->ne];
    pcs_t *PL = malloc((ntot + 1) * sizeof(pcs_t)), *MI = malloc((ntot + 1) * sizeof(pcs_t));
    double c0s[64]; int nc0 = 0;
    double rsp = 0, rsm = 0, rAp = 0, rAm = 0, ak = Mo->a, xq = 0; int d3 = 0;
    for (int i = 0; i < ntot; i++) {
        const det_xspec *sp = &o->specs[i]; pcs_t p0, m0;
        xpieces_gen(sp->q, sp->c0, sp->A, sp->vmax, &p0, &m0); active(&p0, tr, &PL[i]); active(&m0, tr, &MI[i]);
        ak = dmax(ak, fabs(sp->A)); if (sp->dc0 != 0) d3 = 1;
        if (sp->A != 0) { if (sp->A > 0) rAp = dmax(rAp, 1 / sp->A); else rAm = dmax(rAm, -1 / sp->A); }
        xq = dmax(xq, sp->q);
        if (sp->c0 == 0) continue;
        int in = 0; for (int l = 0; l < nc0; l++) if (c0s[l] == sp->c0) in = 1;
        if (!in && nc0 < 64) c0s[nc0++] = sp->c0;
        rsp = dmax(rsp, xscale(tr, sp->c0, sp->q, sp->A, &PL[i])); rsm = dmax(rsm, xscale(tr, sp->c0, sp->q, sp->A, &MI[i]));
    }
    rsp = dmax(dmin(rsp, 1e4), dmin(rAp, 1e8)); rsm = dmax(dmin(rsm, 1e4), dmin(rAm, 1e8));
    ushapes_t U;
    if (ushapes(CP, k, &L, c0s, nc0, rsp, rsm, o->alpha, ak, o->n, o->ntail, dmax(0.0, xq), &U, &o->st)) { ushapes_free(&U); free(Bf); free(PL); free(MI); return; }
    for (int ie = 0; ie < o->ne; ie++) {
        cdouble IF = DZERO, If = DZERO, JF = DZERO, Jf = DZERO; double aF = 0, af = 0;
        double v1 = o->vr ? o->vr[ie] : 0.0;
        for (int side = 0; side < 2; side++) {
            const cvec *Ss = side == 0 ? &U.Sp : &U.Sm, *Ms = side == 0 ? &U.Mp : &U.Mm; const dvec *Es = side == 0 ? &U.Ep : &U.Em;
            for (int j = 0; j < Ss->n; j++) {
                cdouble s = Ss->v[j], m = Ms->v[j], tF = DZERO, tf = DZERO, M[4], e[3];
                for (int it = o->off[ie]; it < o->off[ie + 1]; it++) {
                    const det_xspec *sp = &o->specs[it]; const pcs_t *pc = side == 0 ? &PL[it] : &MI[it];
                    double c0 = sp->c0, q = sp->q;
                    if (v1 > 0) {
                        int sg = 0;
                        for (int i = 0; i < pc->n; i++) { if (pc->b[i] == v1) sg--; if (pc->a[i] == v1) sg++; }
                        if (sg != 0) {
                            cdouble sa = csq(c0 == 0 ? t : cadd(t, cscale(c0, s)));
                            rend_part(v1, sa, cinv(sa), cscale(q, s), cscale(sg, cexp(cscale(-(c0 * v1 * v1 + q * v1 + sp->A), s))), e);
                            JF = cadd(JF, cmul(cscale(sp->w, m), cadd(cscale(q, e[0]), cscale(2 * c0, e[1]))));
                            Jf = cadd(Jf, cmul(cscale(sp->w * 2, m), cadd(cscale(sp->dq, e[2]), cscale(sp->dA, e[1]))));
                        }
                    }
                    if (pc->n == 0) M[0] = M[1] = M[2] = M[3] = DZERO;
                    else if (d3) ctr_moments3(t, s, c0, q, sp->A, pc, M);
                    else { ctr_moments(t, s, c0, q, sp->A, pc, M); M[3] = DZERO; }
                    cdouble pp, pm; psi_pm(s, sp->A, &pp, &pm);
                    cdouble kF = cadd(cadd(cscale(q, M[0]), cscale(2 * c0, M[1])), side == 0 ? pp : pm);
                    if (sp->bnd && isfinite(sp->vmax)) {
                        double vm = sp->vmax; cdouble bp, bm; psi_pm(s, c0 * vm * vm + q * vm + sp->A, &bp, &bm);
                        kF = csub(kF, cmul(cexp(cscale(-vm * vm, t)), side == 0 ? bp : bm));
                    }
                    tF = cadd(tF, cscale(sp->w, kF));
                    tf = cadd(tf, cscale(sp->w * 2, cadd(cadd(cscale(sp->dc0, M[3]), cscale(sp->dq, M[2])), cscale(sp->dA, M[1]))));
                }
                tF = cmul(tF, m); tf = cmul(tf, m);
                IF = cadd(IF, tF); If = cadd(If, tf); aF += Es->v[j] * cfabs(tF); af += Es->v[j] * cfabs(tf);
            }
        }
        size_t row = (size_t) (k - o->k0) * o->ne + ie;
        if (v1 > 0) {
            cdouble eb = cexp(cscale(-v1 * v1, t));
            IF = csub(IF, cmul(eb, JF)); If = csub(If, cmul(eb, Jf));
            o->JF[row] = cmul(JF, cdiv(dt, t)); o->Jf[row] = cmul(Jf, dt);
        }
        o->GF[row] = cimag(cmul(IF, cdiv(dt, t))); o->Gf[row] = cimag(cmul(If, dt));
        o->AF[row] = aF * cabs(cdiv(dt, t)); o->Af[row] = af * cabs(dt);
    }
    ushapes_free(&U); free(Bf); free(PL); free(MI);
}
static void node_body(int j, void *v)
{
    node_ctx *o = v;
    if (det_failed(&o->st)) return;
    if (o->specs) tnode_specs(o, o->k0 + j); else tnode_inner(o, o->k0 + j);
}

/* ---------- the outer sums ---------- */
#define WARN_TOL 1e-4
static double nanmax(double a, double b) { return (a != a || b != b) ? NAN : (a > b ? a : b); }
/* Filon's rule on a panel: int_0^1 p(x) e^{-i om x} dx, p through the values g at the nodes of F (filon_int) */
static cdouble filon_int(const det_filon_table *F, const cdouble *g, double om)
{
    int nq = F->nq;
    if (om <= 2000) {
        int m = imax(1, (int) ceil(om / 20));
        const double *wr, *zr = det_gl(64, &wr);
        cdouble acc = DZERO;
        for (int i = 0; i < m; i++) for (int r = 0; r < 64; r++) {
            double x = (i + zr[r]) / m;
            cdouble num = DZERO; double den = 0; int hit = -1;
            for (int q = 0; q < nq; q++) {
                double d = x - F->z[q];
                if (d == 0) { hit = q; break; }
                double c = F->lam[q] / d; num = cadd(num, cscale(c, g[q])); den += c;
            }
            acc = cadd(acc, cmul(cscale(wr[r] / m, hit >= 0 ? g[hit] : cdivr(num, den)), cis_(-om * x)));
        }
        return acc;
    }
    cdouble io = cinv(CPX(0, om)), e = cis_(-om), acc = DZERO, fac = io;
    for (int k = 0; k < nq; k++) {
        cdouble d0 = DZERO, d1 = DZERO;
        for (int q = 0; q < nq; q++) { d0 = cadd(d0, cscale(F->D0[k + q * nq], g[q])); d1 = cadd(d1, cscale(F->D1[k + q * nq], g[q])); }
        acc = cadd(acc, cmul(fac, csub(d0, cmul(d1, e))));
        fac = cmul(fac, io);
    }
    return acc;
}
static const det_filon_table *filon_table(int nq) { for (int i = 0; i < 4; i++) if (DET_FILON_TABLES[i].nq == nq) return &DET_FILON_TABLES[i]; return NULL; }
#define FILON_OMIN 30.0
/* contour_sum and contour_sum_filon (beta != NULL: the parts at the roots, beta = vr^2) */
static int contour_sum(const det_path *CP, int nx, node_ctx *o, double tol, int warn, const double *beta, double *F, double *f, double *eF, double *ef,
                       char *err, size_t errlen)
{
    int K1 = CP->K, CH = 32, done = 0;
    size_t sz = (size_t) imax(K1, 1) * nx;
    o->GF = calloc(sz, sizeof(double)); o->Gf = calloc(sz, sizeof(double)); o->AF = calloc(sz, sizeof(double)); o->Af = calloc(sz, sizeof(double));
    o->JF = beta ? calloc(sz, sizeof(cdouble)) : NULL; o->Jf = beta ? calloc(sz, sizeof(cdouble)) : NULL;
    o->k0 = 0;
    double last = INFINITY;
    double *eb = beta ? malloc(nx * sizeof(double)) : NULL;
    if (beta) for (int ix = 0; ix < nx; ix++) eb[ix] = exp(-CP->tau * beta[ix]);
    /* (the arrays hold all nodes: node k at row k; the parallel loop of each chunk writes its rows) */
    while (done < K1) {
        int k1 = done, k2 = imin(done + CH, K1);
        node_ctx oc = *o; oc.k0 = k1;
        oc.GF = o->GF + (size_t) k1 * nx; oc.Gf = o->Gf + (size_t) k1 * nx; oc.AF = o->AF + (size_t) k1 * nx; oc.Af = o->Af + (size_t) k1 * nx;
        if (beta) { oc.JF = o->JF + (size_t) k1 * nx; oc.Jf = o->Jf + (size_t) k1 * nx; }
        dfl_parfor(k2 - k1, node_body, &oc);
        if (det_failed(&oc.st)) { snprintf(err, errlen, "%s", oc.st.msg); free(o->GF); free(o->Gf); free(o->AF); free(o->Af); free(o->JF); free(o->Jf); free(eb); return 1; }
        done = k2;
        double mx = 0;
        for (int k = k1; k < k2; k++) for (int ix = 0; ix < nx; ix++) {
            mx = nanmax(mx, nanmax(fabs(o->GF[(size_t) k * nx + ix]), fabs(o->Gf[(size_t) k * nx + ix])));
            if (beta) mx = nanmax(mx, nanmax(eb[ix] * cabs(o->JF[(size_t) k * nx + ix]), eb[ix] * cabs(o->Jf[(size_t) k * nx + ix])));
        }
        double wm = 0; for (int k = k1; k < k2; k++) wm = dmax(wm, CP->wt[k]);
        last = mx * wm;
        if (mx < tol) break;
    }
    double emax = 0; int nonfin = 0;
    for (int ix = 0; ix < nx; ix++) {
        double sF = 0, sf = 0, aF = 0, af = 0;
        for (int k = 0; k < done; k++) {
            double w = CP->wt[k]; size_t i = (size_t) k * nx + ix;
            sF += w * o->GF[i]; sf += w * o->Gf[i]; aF += w * o->AF[i]; af += w * o->Af[i];
        }
        F[ix] = sF; f[ix] = sf; if (eF) eF[ix] = 2.220446049250313e-16 * aF; if (ef) ef[ix] = 2.220446049250313e-16 * af;
        /* (eF, ef may be NULL: the estimate then only for the warning) */
        double e1 = 2.220446049250313e-16 * aF, e2 = 2.220446049250313e-16 * af;
        if (beta && beta[ix] > 0) {
            const det_filon_table *Fd = filon_table(CP->nq);
            int nq = CP->nq;
            for (int p = 0; p < CP->npan; p++) {
                int j1 = p * nq;
                if (j1 >= done) break;
                double wa = CP->cuts[p], Lp = CP->cuts[p + 1] - CP->cuts[p], b = beta[ix];
                if (b * Lp < FILON_OMIN || !Fd) {
                    for (int kk = j1; kk < imin(j1 + nq, done); kk++) {
                        cdouble e = cexp(cscale(-b, CP->t[kk]));
                        F[ix] += CP->wt[kk] * cimag(cmul(e, o->JF[(size_t) kk * nx + ix])); f[ix] += CP->wt[kk] * cimag(cmul(e, o->Jf[(size_t) kk * nx + ix]));
                    }
                    continue;
                }
                cdouble gF[64], gf[64];
                for (int q = 0; q < nq; q++) { int kk = j1 + q; gF[q] = kk < K1 ? o->JF[(size_t) kk * nx + ix] : DZERO; gf[q] = kk < K1 ? o->Jf[(size_t) kk * nx + ix] : DZERO; }
                cdouble ph = cscale(Lp * eb[ix] / M_PI, cis_(-b * wa));
                F[ix] += cimag(cmul(ph, filon_int(Fd, gF, b * Lp))); f[ix] += cimag(cmul(ph, filon_int(Fd, gf, b * Lp)));
            }
        }
        emax = dmax(emax, dmax(e1, e2)); if (!isfinite(F[ix]) || !isfinite(f[ix])) nonfin = 1;
    }
    if (warn && (emax > WARN_TOL || nonfin))
        det_warn("contour_eval: the terms of the sums are large: the rounding error may reach %.2g; the result may be inaccurate (start %d, gamma0 %g, c = %g)", emax, CP->Mo->start, CP->Mo->g0, CP->c);
    if (warn && last > 1e-10) det_warn("contour_eval: the outer contour ends with terms of size %.2g (it is cut at |t - tau| = %g); the result may be inaccurate (start %d, gamma0 %g, c = %g)", last, det_default_rmax(CP->c), CP->Mo->start, CP->Mo->g0, CP->c);
    free(o->GF); free(o->Gf); free(o->AF); free(o->Af); free(o->JF); free(o->Jf); free(eb);
    return 0;
}
/* (Filon's data of the line: nq of the panels, from the path) */

/* the warning of the vertical line for a parabola with a positive root b, b^2 = b2, where Filon's rule does not take the
 * part at the root (line_edge_warn) */
static void line_edge_warn(const det_path *CP, double b2, int c0neg)
{
    if (!(b2 > 0)) return;
    const det_model *Mo = CP->Mo;
    if (c0neg && CP->tau * b2 < 40) {
        det_warn("contour_eval: on the vertical line Re t = %.3g (c = %g%s: no hyperbola keeps the positive roots right of the origin) with c0 < 0 the parabola has a root b with b^2 = %.3g, whose part of the outer integrand the line does not resolve: the result may be off by 1e-5 to 1e-3", CP->tau, CP->c, isnan(Mo->P->cbar) ? "" : ", GLS detrending", b2);
        return;
    }
    int nq = CP->nq, np = CP->npan;
    double est = 0, worst = 0;
    for (int k = 0; k < np; k++) {
        int j1 = k * nq, j2 = k * nq + nq - 1;
        double w1 = cimag(CP->t[j1]), w2 = cimag(CP->t[j2]);
        double L = k == 0 ? 2 * w2 : (w2 - w1) * 1.02;
        if (!(b2 * L > 30)) continue;
        double lmv = dmax(creal(det_logMtilde(CP->R + (size_t) j1 * CP->r, CP->r, Mo->rp, CP->lP[j1], DZERO)), creal(det_logMtilde(CP->R + (size_t) j2 * CP->r, CP->r, Mo->rp, CP->lP[j2], DZERO)));
        est += exp(lmv - CP->tau * b2) * L / (M_PI * dmax(cabs(CP->t[j1]), CP->tau));
        worst = dmax(worst, b2 * L / 30);
    }
    if (est > 1e-6) det_warn("contour_eval: on the vertical line Re t = %.3g (c = %g%s: no hyperbola keeps the positive roots right of the origin) the parabola has a root b with b^2 = %.3g, whose part of the outer integrand (of size %.2g) oscillates up to %.2g times faster than its panels resolve: the result may be off by more than 1e-5", CP->tau, CP->c, isnan(Mo->P->cbar) ? "" : ", GLS detrending", b2, est, worst);
}

void det_eval_defaults(const det_path *CP, det_eval_opts *o)
{
    int fixed = CP->Mo->start == 1;
    o->n = fixed ? 48 : 24; o->ntail = fixed ? 64 : 40; o->alpha = 0.5; o->tol = 1e-17; o->warn = 1; o->filon = 1;
}
int det_contour_eval(const det_path *CP, int nx, const double *xs, double c0, int vpow, const det_eval_opts *o0,
                     double *F, double *f, double *eF, double *ef, char *err, size_t errlen)
{
    det_eval_opts o; if (o0) o = *o0; else det_eval_defaults(CP, &o);
    if (nx == 0) return 0;
    {   /* the guard on the nodes of the U-shapes, before the outer sum */
        double xq = 0; for (int ix = 0; ix < nx; ix++) xq = dmax(xq, xs[ix]);
        if (ushape_guard(CP, &c0, 1, CP->Mo->a, o.n, o.alpha, o.tol, dmax(0.0, xq), err, errlen)) return 1;
    }
    double *vr = NULL;
    /* Filon's rule on the line for the t-ratio with one root of the parabola per x (filon_roots_x) */
    if (o.filon && isnan(CP->h) && c0 != 0 && CP->line && filon_table(CP->nq)) {
        vr = calloc(nx, sizeof(double)); int found = 0, two = 0;
        for (int i = 0; i < nx && !two; i++) {
            if (!isfinite(xs[i])) continue;
            pcs_t pl, mi; xpieces(xs[i], c0, CP->Mo->a, &pl, &mi);
            double rr[4]; int nr = 0;
            for (int l = 0; l < pl.n; l++) if (isfinite(pl.b[l]) && pl.b[l] > 0) { int dup = 0; for (int u = 0; u < nr; u++) if (rr[u] == pl.b[l]) dup = 1; if (!dup) rr[nr++] = pl.b[l]; }
            for (int l = 0; l < mi.n; l++) if (isfinite(mi.b[l]) && mi.b[l] > 0) { int dup = 0; for (int u = 0; u < nr; u++) if (rr[u] == mi.b[l]) dup = 1; if (!dup) rr[nr++] = mi.b[l]; }
            if (nr == 0) continue;
            if (nr > 1) { two = 1; break; }
            vr[i] = rr[0]; found = 1;
        }
        if (two || !found) { free(vr); vr = NULL; }
    }
    node_ctx nc; memset(&nc, 0, sizeof nc);
    nc.CP = CP; nc.nx = nx; nc.xs = xs; nc.c0 = c0; nc.vpow = vpow; nc.n = o.n; nc.ntail = o.ntail; nc.alpha = o.alpha; nc.vr = vr;
    int rc;
    if (vr) {
        double *beta = malloc(nx * sizeof(double)); for (int i = 0; i < nx; i++) beta[i] = vr[i] * vr[i];
        rc = contour_sum(CP, nx, &nc, o.tol, o.warn, beta, F, f, eF, ef, err, errlen);
        free(beta); free(vr);
        return rc;
    }
    if (o.warn && isnan(CP->h)) {
        double b2 = 0;
        for (int i = 0; i < nx; i++) {
            pcs_t pl, mi; xpieces(xs[i], c0, CP->Mo->a, &pl, &mi);
            for (int l = 0; l < pl.n; l++) if (isfinite(pl.b[l])) b2 = dmax(b2, pl.b[l] * pl.b[l]);
            for (int l = 0; l < mi.n; l++) if (isfinite(mi.b[l])) b2 = dmax(b2, mi.b[l] * mi.b[l]);
        }
        line_edge_warn(CP, b2, c0 < 0);
    }
    return contour_sum(CP, nx, &nc, o.tol, o.warn, NULL, F, f, eF, ef, err, errlen);
}
int det_contour_eval_specs(const det_path *CP, int ne, const int *nt, const det_xspec *specs, const det_eval_opts *o0,
                           double *F, double *f, double *eF, double *ef, char *err, size_t errlen)
{
    det_eval_opts o; if (o0) o = *o0; else det_eval_defaults(CP, &o);
    if (ne == 0) return 0;
    int *off = malloc((ne + 1) * sizeof(int)); off[0] = 0; for (int i = 0; i < ne; i++) off[i + 1] = off[i] + nt[i];
    {   /* the guard on the nodes of the U-shapes, with the c0 and the constant of the kernels of tnode_specs */
        double c0s[64], ak = CP->Mo->a, xq = 0; int nc0 = 0;
        for (int i = 0; i < off[ne]; i++) {
            const det_xspec *sp = &specs[i];
            ak = dmax(ak, fabs(sp->A)); xq = dmax(xq, sp->q);
            if (sp->c0 == 0) continue;
            int in = 0; for (int l = 0; l < nc0; l++) if (c0s[l] == sp->c0) in = 1;
            if (!in && nc0 < 64) c0s[nc0++] = sp->c0;
        }
        if (ushape_guard(CP, c0s, nc0, ak, o.n, o.alpha, o.tol, dmax(0.0, xq), err, errlen)) { free(off); return 1; }
    }
    double *vr = NULL;
    if (o.filon && isnan(CP->h) && CP->line && filon_table(CP->nq)) {
        /* filon_roots: one term with q = 0, c0 != 0, no derivative in c0, vmax = Inf, one root */
        vr = calloc(ne, sizeof(double)); int found = 0, bad = 0;
        for (int i = 0; i < ne && !bad; i++) {
            double rr[8]; int nr = 0, nu = 0;
            for (int it = off[i]; it < off[i + 1]; it++) {
                const det_xspec *sp = &specs[it]; pcs_t pl, mi; xpieces_gen(sp->q, sp->c0, sp->A, sp->vmax, &pl, &mi);
                for (int l = 0; l < pl.n; l++) if (isfinite(pl.b[l]) && pl.b[l] > 0 && nr < 8) rr[nr++] = pl.b[l];
                for (int l = 0; l < mi.n; l++) if (isfinite(mi.b[l]) && mi.b[l] > 0 && nr < 8) rr[nr++] = mi.b[l];
            }
            if (nr == 0) continue;
            for (int u = 0; u < nr; u++) { int dup = 0; for (int v = 0; v < u; v++) if (rr[v] == rr[u]) dup = 1; if (!dup) nu++; }
            const det_xspec *s0 = &specs[off[i]];
            if (!(nt[i] == 1 && s0->q == 0 && s0->c0 != 0 && s0->dc0 == 0 && !isfinite(s0->vmax) && nu == 1)) { bad = 1; break; }
            vr[i] = rr[0]; found = 1;
        }
        if (bad || !found) { free(vr); vr = NULL; }
    }
    node_ctx nc; memset(&nc, 0, sizeof nc);
    nc.CP = CP; nc.nx = ne; nc.n = o.n; nc.ntail = o.ntail; nc.alpha = o.alpha; nc.vr = vr; nc.ne = ne; nc.nt = nt; nc.off = off; nc.specs = specs;
    int rc;
    if (vr) {
        double *beta = malloc(ne * sizeof(double)); for (int i = 0; i < ne; i++) beta[i] = vr[i] * vr[i];
        rc = contour_sum(CP, ne, &nc, o.tol, o.warn, beta, F, f, eF, ef, err, errlen);
        free(beta); free(vr); free(off);
        return rc;
    }
    if (o.warn && isnan(CP->h)) {
        double b2 = 0; int neg = 0;
        for (int it = 0; it < off[ne]; it++) {
            const det_xspec *sp = &specs[it]; pcs_t pl, mi; xpieces_gen(sp->q, sp->c0, sp->A, sp->vmax, &pl, &mi);
            for (int side = 0; side < 2; side++) { const pcs_t *pc = side ? &mi : &pl; for (int l = 0; l < pc->n; l++) { double b = pc->b[l]; if (!(isfinite(b) && b < sp->vmax)) continue; b2 = dmax(b2, b * b); if (sp->c0 < 0) neg = 1; } }
        }
        line_edge_warn(CP, b2, neg);
    }
    rc = contour_sum(CP, ne, &nc, o.tol, o.warn, NULL, F, f, eF, ef, err, errlen);
    free(off);
    return rc;
}

/* the law of V alone: (1/pi) sum' Im{M_V(t_k) e^{-t_k y} t'(u_k)/t_k} (cdf, P(V > y)) or without 1/t_k (vsum) */
double det_vsum(const det_path *CP, double y, int cdf)
{
    const det_model *Mo = CP->Mo;
    int r = CP->r;
    cdouble lg[64]; double acc = 0;
    for (int k = 0; k < CP->K; k++) {
        cdouble lM = CP->lP[k];
        for (int j = 0; j < r; j++) {
            cdouble z = j < Mo->rp ? CP->R[(size_t) k * r + j] : cneg(CP->R[(size_t) k * r + j]);
            cdouble l = clog(z);
            if (k > 0) l = cadd(l, CPX(0, 2 * M_PI * jround((cimag(lg[j]) - cimag(l)) / (2 * M_PI))));
            lg[j] = l; lM = csub(lM, cscale(0.5, l));
        }
        if (CP->fx && CP->fx[k]) lM = cadd(lM, cscale(Mo->g0 * Mo->g0, CP->fx[k]->cf));
        cdouble t = CP->t[k];
        acc += CP->wt[k] * cimag(cmul(cexp(csub(lM, cscale(y, t))), cdf ? cdiv(CP->dt[k], t) : CP->dt[k]));
    }
    return acc;
}
