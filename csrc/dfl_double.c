/* dfl_double.c: the double integral (the conditional integral representation): the inner inversion in the argument of V along a path through a
 * saddle point, the outer integral over V = b with an adaptive rule in log b.  Zero start: the closed forms
 * (case DK: E[e^{sV}; R <= a] by Girsanov; DU, DT: M_V g_s(r)) through the saddle point of the whole integrand on the
 * better of two paths (three in case DK).  With an initial observation: the Talbot parabola through the saddle point
 * of log M_V(s) - s b, the start terms, the noncentral probability for a fixed start in DU, DT.  A port of
 * double_zero / double_start of DFExact.jl (Raymond Kan's TR_LIMDIST). */
#include "dflimit.h"
#include "dfl_lock.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const cdouble ZERO = {0}, ONE = {1};
cdouble dfl_girs_detu(int model, double c, cdouble u, cdouble v, dfl_girs_t *G);
static inline cdouble zof(double c, cdouble t) { return CPX(c * c - 2 * creal(t), -2 * cimag(t)); }

/* ---- the outer rules ---- */
typedef struct { int M; double *u, *gw, *z1, *w1, *z2, *w2, *LM; } orules_t;
static orules_t *ORULES[257];
static dfl_lock_t orlock = DFL_LOCK_INIT;
static const orules_t *outer_rules(int M)
{
    dfl_lock(&orlock);
    if (ORULES[M] == NULL) {
        orules_t *o = calloc(1, sizeof *o); o->M = M;
        const double *gw; const double *u = dfl_rule(M, &gw);
        o->u = malloc(M * sizeof(double)); o->gw = malloc(M * sizeof(double)); o->z1 = malloc(M * sizeof(double)); o->w1 = malloc(M * sizeof(double)); o->z2 = malloc(M * sizeof(double)); o->w2 = malloc(M * sizeof(double));
        for (int k = 0; k < M; k++) {
            o->u[k] = u[k]; o->gw[k] = gw[k];
            o->z1[k] = (1 - cos(M_PI * u[k])) / 2; o->w1[k] = gw[k] * M_PI / 2 * sin(M_PI * u[k]);
            o->z2[k] = (1 - cos(M_PI * o->z1[k])) / 2; o->w2[k] = gw[k] * (M_PI / 2) * (M_PI / 2) * sin(M_PI * o->z1[k]) * sin(M_PI * u[k]);
        }
        double *Pn = malloc(M * M * sizeof(double));
        for (int k = 0; k < M; k++) { double x = 2 * u[k] - 1; Pn[k] = 1; if (M > 1) Pn[M + k] = x;
            for (int n = 1; n < M - 1; n++) Pn[(n + 1) * M + k] = ((2 * n + 1) * x * Pn[n * M + k] - n * Pn[(n - 1) * M + k]) / (n + 1); }
        o->LM = malloc(M * M * sizeof(double));                      /* LM[n][k] = (2n+1)/2 Pn(x_k) 2 gw_k */
        for (int n = 0; n < M; n++) for (int k = 0; k < M; k++) o->LM[n * M + k] = (2 * n + 1) / 2.0 * Pn[n * M + k] * 2 * gw[k];
        free(Pn);
        ORULES[M] = o;
    }
    dfl_unlock(&orlock);
    return ORULES[M];
}

/* ---- the adaptive outer rule in log b ---- */
typedef void (*evalH_t)(int n, const double *B, const int *IX, double *H, double *E, void *ctx);
typedef struct { double *F, *err; int *nun, *nref; int npairs; } adapt_out;
static void adaptb(int np0, const double *PA0, const double *PL0, const bool *PEL0, const bool *PER0, const int *PX0, int nx,
                   const double *zs, const double *ws, const double *ze, const double *we, const orules_t *OR, evalH_t evalH, void *ctx,
                   double otol, int maxlev, adapt_out *out)
{
    int M = OR->M, mh = M / 2;
    for (int k = 0; k < nx; k++) { out->F[k] = 0; out->err[k] = 0; out->nref[k] = 0; out->nun[k] = 0; }
    out->npairs = 0;
    int np = np0; double *PA = malloc(np * sizeof(double)), *PL = malloc(np * sizeof(double)); bool *PEL = malloc(np), *PER = malloc(np); int *PX = malloc(np * sizeof(int));
    memcpy(PA, PA0, np * sizeof(double)); memcpy(PL, PL0, np * sizeof(double)); memcpy(PEL, PEL0, np); memcpy(PER, PER0, np); memcpy(PX, PX0, np * sizeof(int));
    double *sc = calloc(nx, sizeof(double));
    /* accepted pairs for the unresolved count */
    int cap_a = 1024, na = 0; double *Ha = malloc(cap_a * sizeof(double)), *Ea = malloc(cap_a * sizeof(double)), *Wa = malloc(cap_a * sizeof(double)); int *Xa = malloc(cap_a * sizeof(int));
    for (int lev = 0; lev <= maxlev && np > 0; lev++) {
        int n = M * np;
        double *B = malloc(n * sizeof(double)), *W = malloc(n * sizeof(double)), *H = malloc(n * sizeof(double)), *E = malloc(n * sizeof(double)); int *IX = malloc(n * sizeof(int));
        for (int p = 0; p < np; p++) {
            const double *z = (PEL[p] || PER[p]) ? ze : zs, *w = (PEL[p] || PER[p]) ? we : ws;
            for (int k = 0; k < M; k++) { int i = p * M + k; B[i] = exp(PA[p] + PL[p] * z[k]); W[i] = PL[p] * w[k] * B[i]; IX[i] = PX[p]; }
        }
        evalH(n, B, IX, H, E, ctx);
        out->npairs += n;
        double *est = malloc(np * sizeof(double)); bool *rf = malloc(np);
        int nrf = 0;
        for (int p = 0; p < np; p++) {
            double Iv = 0, nse = 0, sumG = 0; double C[256];
            for (int nn = 0; nn < M; nn++) { double cn = 0; for (int k = 0; k < M; k++) { int i = p * M + k; double G = W[i] == 0 ? 0 : W[i] * H[i]; cn += OR->LM[nn * M + k] * G / OR->gw[k]; } C[nn] = fabs(cn); }
            for (int k = 0; k < M; k++) { int i = p * M + k; double G = W[i] == 0 ? 0 : W[i] * H[i]; Iv += G; sumG += fabs(G); nse += W[i] == 0 ? 0 : fabs(W[i]) * E[i]; }
            double e = 0;
            if (M >= 4) { double chi = fmax(C[M - 1], C[M - 2]), cmid = fmax(C[mh], C[mh - 1]); e = chi < cmid ? chi * chi * chi / (cmid * cmid) : chi; }
            if (!isfinite(e)) e = 0;
            est[p] = e;
            double floor_ = nse + 1e3 * 2.220446049250313e-16 * sumG;
            rf[p] = e > fmax(otol, 10 * floor_) && lev < maxlev;
            if (!rf[p]) {
                out->F[PX[p]] += Iv; out->err[PX[p]] += e + nse;
                for (int k = 0; k < M; k++) { int i = p * M + k; if (na == cap_a) { cap_a *= 2; Ha = realloc(Ha, cap_a * sizeof(double)); Ea = realloc(Ea, cap_a * sizeof(double)); Wa = realloc(Wa, cap_a * sizeof(double)); Xa = realloc(Xa, cap_a * sizeof(int)); }
                    Ha[na] = H[i]; Ea[na] = E[i]; Wa[na] = W[i]; Xa[na] = IX[i]; na++; }
            } else { nrf++; out->nref[PX[p]]++; }
        }
        double *PA2 = malloc(2 * nrf * sizeof(double)), *PL2 = malloc(2 * nrf * sizeof(double)); bool *PEL2 = malloc(2 * nrf + 1), *PER2 = malloc(2 * nrf + 1); int *PX2 = malloc(2 * nrf * sizeof(int));
        int q = 0;
        for (int p = 0; p < np; p++) if (rf[p]) { PA2[q] = PA[p]; PL2[q] = PL[p] / 2; PX2[q] = PX[p]; PEL2[q] = PEL[p]; PER2[q] = false; q++; }
        for (int p = 0; p < np; p++) if (rf[p]) { PA2[q] = PA[p] + PL[p] / 2; PL2[q] = PL[p] / 2; PX2[q] = PX[p]; PEL2[q] = false; PER2[q] = PER[p]; q++; }
        free(PA); free(PL); free(PEL); free(PER); free(PX); free(B); free(W); free(H); free(E); free(IX); free(est); free(rf);
        PA = PA2; PL = PL2; PEL = PEL2; PER = PER2; PX = PX2; np = q;
    }
    for (int k = 0; k < na; k++) sc[Xa[k]] = fmax(sc[Xa[k]], fabs(Ha[k]));
    for (int k = 0; k < na; k++) if (Ea[k] > 1e-9 * sc[Xa[k]] && fabs(Wa[k]) * Ea[k] > 1e-14) out->nun[Xa[k]]++;
    free(PA); free(PL); free(PEL); free(PER); free(PX); free(sc); free(Ha); free(Ea); free(Wa); free(Xa);
}

/* the trapezoidal sum along a path with the nested-rule estimate */
static void fin(int n, const cdouble *G, double *Iv, double *E)
{
    double s = 0, s2 = 0, sa = 0; bool bad = false;
    for (int k = 0; k < n; k++) { if (!isfinite(creal(G[k])) || !isfinite(cimag(G[k]))) bad = true; s += creal(G[k]); if (k % 2 == 0) s2 += 2 * creal(G[k]); sa += cabs(G[k]); }
    if (bad) { *Iv = 0; *E = INFINITY; return; }
    *Iv = s; *E = fabs(s - s2) + 1e-14 * sa + 1e3 * cabs(G[n - 1]);
}

/* ---- the saddle point of a real convex function ---- */
typedef double (*rfn_t)(double s, void *ctx);
static void wscan1(rfn_t Lf, void *ctx, double s, double sc, double smx, double *a1, double *L0)
{
    double l0 = Lf(s, ctx), best = INFINITY, dd = sc, Db = NAN;
    for (int k = -6; k <= 12; k++) {
        double d = fmin(sc * pow(10, k), fmax((smx - s) / 2, 2.220446049250313e-16 * fabs(s)));
        double Dl = (Lf(s - d, ctx) + Lf(s + d, ctx)) / 2 - l0;
        double m = fabs(log(fmax(Dl, 2.2250738585072014e-308) / 0.5));
        if (m < best) { best = m; dd = d; Db = Dl; }
    }
    double g2 = 2 * Db / (dd * dd);
    if (!(g2 > 0) || !isfinite(g2)) g2 = 1 / (sc * sc);
    *a1 = 1 / sqrt(g2); *L0 = l0;
}
static double dfd(rfn_t Lf, void *ctx, double s, double h) { double Lp = Lf(s + h, ctx), Lm = Lf(s - h, ctx); return isfinite(Lp) ? (Lp - Lm) / (2 * h) : INFINITY; }
static void rsad1(rfn_t Lf, void *ctx, double b, double smx, double *s_, double *a1_, double *g2_, double *L0_)
{
    double sc = 1 / b;
    double s0 = fmin(-sc, smx - sc / 2), t0 = -(sc / 2) * (sc / 2) / 2 - fabs(smx); if (b < 0.05 && t0 < s0) s0 = t0;
    double w0, l0; wscan1(Lf, ctx, s0, sc, smx, &w0, &l0);
#define HD(s, w) fmax(1e-4 * (w), 1e-12 * fabs(s))
    double sL = s0, gL = dfd(Lf, ctx, sL, HD(sL, w0)), sR = smx - 1e-12 * fmax(1, fabs(smx));
    for (int k = 1; k <= 200; k++) { if (gL < 0) break; sR = fmin(sR, sL); sL -= fmax(sc, w0) * pow(4, k); gL = dfd(Lf, ctx, sL, HD(sL, w0)); }
    double gR = dfd(Lf, ctx, sR, HD(sR, w0)); if (!isfinite(gR) || gR <= 0) gR = INFINITY;
    int side = 0;
    for (int it = 0; it < 300; it++) {
        double w = sR - sL; if (!(w > 1e-9 * w0)) break;
        double sn = isfinite(gR) ? sL - gL * w / (gR - gL) : (sL + sR) / 2;
        if (!(sn > sL && sn < sR)) sn = (sL + sR) / 2;
        double gn = dfd(Lf, ctx, sn, HD(sn, w0));
        if (gn < 0) { sL = sn; gL = gn; if (side == -1) gR /= 2; side = -1; }
        else { sR = sn; gR = gn; if (side == 1) gL /= 2; side = 1; }
        if (!isfinite(gR)) gR = INFINITY;
    }
#undef HD
    double s = (sL + sR) / 2, a1, L0; wscan1(Lf, ctx, s, sc, smx, &a1, &L0);
    *s_ = s; *a1_ = a1; *g2_ = 1 / (a1 * a1); *L0_ = L0;
}
static void rsad_warm(rfn_t Lf, void *ctx, double b, double smx, double s0, double a10, double *s_, double *a1_, double *g2_, double *L0_)
{
    double s = s0, a1 = a10;
    if (!(isfinite(s) && isfinite(a1) && a1 > 0)) { rsad1(Lf, ctx, b, smx, s_, a1_, g2_, L0_); return; }
    s = fmin(s, smx - a1 / 4);
    for (int it = 0; it < 30; it++) {
        double h = fmax(1e-4 * a1, 1e-12 * fabs(s)); h = fmin(h, (smx - s) / 2);
        double L0 = Lf(s, ctx), Lp = Lf(s + h, ctx), Lm = Lf(s - h, ctx);
        if (!(isfinite(L0) && isfinite(Lp) && isfinite(Lm))) { rsad1(Lf, ctx, b, smx, s_, a1_, g2_, L0_); return; }
        double g1 = (Lp - Lm) / (2 * h), g2 = (Lp - 2 * L0 + Lm) / (h * h);
        if (!(g2 > 0)) { rsad1(Lf, ctx, b, smx, s_, a1_, g2_, L0_); return; }
        a1 = 1 / sqrt(g2);
        double ds = -g1 / g2; if (ds > 4 * a1) ds = 4 * a1; if (ds < -4 * a1) ds = -4 * a1;
        double sn = s + ds; if (!(sn < smx - 1e-12 * fmax(1, fabs(smx)))) sn = (s + smx) / 2;
        if (fabs(ds) < 1e-3 * a1) { *s_ = sn; *a1_ = a1; *g2_ = g2; *L0_ = Lf(sn, ctx); return; }
        s = sn;
    }
    rsad1(Lf, ctx, b, smx, s_, a1_, g2_, L0_);
}

/* ---- case DK, zero start ---- */
static double dketa(double s, double c)
{
    cdouble th = csqrt(CPX(c * c - 2 * s, 0));
    cdouble del = c > 0 ? cdiv(CPX(-2 * s, 0), cadd(th, CPX(c, 0))) : csub(th, CPX(c, 0));
    if (cabs(th) < 1e-4) return creal(cadd(del, csub(cadd(ONE, cscale(1.0 / 3, cmul(th, th))), th)));
    cdouble e2 = cexp(cscale(-2, th));
    return creal(cadd(del, cdiv(cscale(2, cmul(th, e2)), csub(ONE, e2))));
}
static double dksmax(double c)
{
    if (c == 1) return 0.5;
    double lo = fmin(0, c * c / 2) - 1, hi = (c * c + M_PI * M_PI) / 2;
    while (dketa(lo, c) <= 0) lo = 2 * lo - 1;
    for (int k = 0; k < 2000; k++) { double m = (lo + hi) / 2; if (m == lo || m == hi) break; if (dketa(m, c) > 0) lo = m; else hi = m; }
    return lo;
}
/* log of the integrand times e^{-s b} along the path S (sqrt(eta) continuous along it) */
static void dklam(int n, const cdouble *S, double b, double c, double a, double v, int kind, cdouble *L)
{
    cdouble rprev = ZERO;
    for (int k = 0; k < n; k++) {
        cdouble s = S[k], q = CPX(c * c - 2 * creal(s), -2 * cimag(s)), th = csqrt(q);
        cdouble del = c > 0 ? cdiv(cscale(-2, s), cadd(th, CPX(c, 0))) : csub(th, CPX(c, 0));
        cdouble ls, ep;
        if (cabs(th) < 1) {
            cdouble r = ONE, tm = ONE, ch = ONE;
            for (int j = 1; j <= 14; j++) { tm = cscale(1.0 / ((2 * j) * (2 * j + 1)), cmul(tm, q)); r = cadd(r, tm); }
            tm = ONE; for (int j = 1; j <= 14; j++) { tm = cscale(1.0 / ((2 * j - 1) * (2 * j)), cmul(tm, q)); ch = cadd(ch, tm); }
            ls = cscale(-0.5, clog(r)); ep = csub(cdiv(ch, r), th);
        } else {
            cdouble e2 = cexp(cscale(-2, th));
            ep = cdiv(cscale(2, cmul(th, e2)), csub(ONE, e2));
            ls = cscale(0.5, csub(csub(clog(th), th), clog(cscale(0.5, csub(ONE, e2)))));
        }
        cdouble eta = cadd(del, ep);
        if (kind == 3) { L[k] = cadd(cadd(CPX(-c / 2, 0), ls), csub(csub(CPX(-log(M_PI * a) / 2, 0), cscale(v, eta)), csub(cscale(c * b, ep), cscale(b / 2, cmul(del, del))))); continue; }
        if (kind > 0) {
            cdouble r = csqrt(eta);
            if (k > 0 && creal(cmul(r, conj(rprev))) < 0) r = cneg(r);
            rprev = r;
            if (kind == 1) L[k] = cadd(cadd(CPX(-c / 2, 0), ls), csub(cscale(b / 2, cmul(del, cadd(th, CPX(c, 0)))), clog(r)));
            else L[k] = cadd(cadd(cadd(CPX(-c / 2, 0), ls), csub(csub(cneg(clog(r)), cscale(v, eta)), csub(cscale(c * b, ep), cscale(b / 2, cmul(del, del))))),
                             clog(dfl_faddw(cmul(CPX(0, 1), cscale(sqrt(a), r)))));
            continue;
        }
        cdouble t = cscale(a, eta), zt = csqrt(t);
        if (creal(t) > 2) {
            L[k] = cadd(cadd(CPX(-c / 2, 0), ls), cadd(cscale(b / 2, cmul(del, cadd(th, CPX(c, 0)))), csub(cadd(clog(csub(ONE, cmul(cexp(cneg(t)), dfl_faddw(cmul(CPX(0, 1), zt))))), CPX(log(a) / 2, 0)), clog(zt))));
        } else {
            cdouble lg;
            if (cabs(zt) < 0.2) { cdouble ser = ZERO, term = ONE; for (int j = 0; j <= 20; j++) { ser = cadd(ser, cscale(1.0 / (2 * j + 1), term)); term = cscale(-1.0 / (j + 1), cmul(term, t)); } lg = cadd(t, clog(cscale(2 / sqrt(M_PI), ser))); }
            else lg = clog(cdiv(csub(cexp(t), dfl_faddw(cmul(CPX(0, 1), zt))), zt));
            L[k] = cadd(cadd(cadd(CPX(-c / 2 + log(a) / 2, 0), ls), csub(csub(cneg(cscale(v, eta)), cscale(c * b, ep)), cneg(cscale(b / 2, cmul(del, del))))), lg);
        }
    }
}
typedef void (*lk_t)(int n, const cdouble *S, cdouble *L, void *ctx);
static void pathW(lk_t lk, void *ctx, double b, double sq, double mu, int NN, bool divS, double *Iv, double *E)
{
    double kap = 2, tl = 45, tmax = fmax(10, (tl + 5) / (2 * kap * b * mu)), ht = asinh(tmax) / NN;
    cdouble *S = malloc((NN + 1) * sizeof(cdouble)), *dSW = malloc((NN + 1) * sizeof(cdouble)), *G = malloc((NN + 1) * sizeof(cdouble));
    for (int j = 0; j <= NN; j++) {
        double tau = j * ht, TH = sinh(tau), Wt = cosh(tau) * ht; if (j == 0) Wt /= 2;
        double rq = sqrt(1 + (TH / (2 * kap)) * (TH / (2 * kap)));
        S[j] = CPX(sq + mu * (TH * TH / rq), -2 * mu * TH);
        dSW[j] = cscale(Wt / M_PI, cmul(CPX(0, 1), CPX(mu * (2 * TH / rq - TH * TH * TH / (4 * kap * kap * rq * rq * rq)), -2 * mu)));
    }
    lk(NN + 1, S, G, ctx);
    for (int j = 0; j <= NN; j++) { G[j] = cmul(cexp(G[j]), dSW[j]); if (divS) G[j] = cdiv(G[j], S[j]); }
    fin(NN + 1, G, Iv, E);
    free(S); free(dSW); free(G);
}
static void pathT(lk_t lk, void *ctx, double b, double a, double sq, double a1q, int NN, double c, bool divS, bool polegap, double *Iv, double *E, double *sA_)
{
    double tl = 45, qh = c * c - 2 * sq, th0 = qh > 0 ? sqrt(qh) : 0;
    double tA = fmax(fmax(th0, asinh(fabs(a) / sqrt(6 * b)) / 2), 1.0);
    if (polegap) { double sA = (c * c - tA * tA) / 2; if (fabs(sA) < a1q) tA = sqrt(fmax(c * c + 2 * a1q, 1.0)); }
    double sA = (c * c - tA * tA) / 2, hq = 1e-4 * a1q;
    cdouble S3[3] = {CPX(sA - hq, 0), CPX(sA, 0), CPX(sA + hq, 0)}, L3[3]; lk(3, S3, L3, ctx);
    double Ls = (creal(L3[2]) - creal(L3[0])) / (2 * hq), Lss = (creal(L3[2]) - 2 * creal(L3[1]) + creal(L3[0])) / (hq * hq);
    double sg = 1 / sqrt(fmax(Lss * tA * tA - Ls, b)); if (!isfinite(sg)) sg = 1 / sqrt(b);
    double ym = fmax(12 * sg, sqrt(2 * (tl + 10) / b)), um = asinh(ym / sg);
    cdouble *S = malloc((NN + 1) * sizeof(cdouble)), *TW = malloc((NN + 1) * sizeof(cdouble)), *G = malloc((NN + 1) * sizeof(cdouble));
    for (int j = 0; j <= NN; j++) {
        double U = j * um / NN, Y = sg * sinh(U), Wt = sg * cosh(U) * um / NN; if (j == 0) Wt /= 2;
        cdouble T = CPX(tA, Y); S[j] = cscale(0.5, csub(CPX(c * c, 0), cmul(T, T))); TW[j] = cscale(Wt / M_PI, T);
    }
    lk(NN + 1, S, G, ctx);
    for (int j = 0; j <= NN; j++) { G[j] = cmul(cexp(G[j]), TW[j]); if (divS) G[j] = cdiv(G[j], S[j]); }
    fin(NN + 1, G, Iv, E);
    *sA_ = sA;
    free(S); free(TW); free(G);
}
typedef struct { double b, c, a, v; int kind; } dkctx;
static void dk_lk(int n, const cdouble *S, cdouble *L, void *v) { dkctx *d = v; dklam(n, S, d->b, d->c, d->a, d->v, d->kind, L); }
static double dk_real(double s, void *v) { cdouble S = CPX(s, 0), L; dk_lk(1, &S, &L, v); return creal(L); }
typedef struct { double s, a1, g2, L0; } sad_t;
typedef struct { sad_t main, B, C; bool valid; } dksad_t;
static void dksaddles(double b, double a, double c, bool pdfm, const dksad_t *prev, dksad_t *out)
{
    double v = a - c * b; int k0 = pdfm ? 3 : 0;
    double s1 = (c * c + M_PI * M_PI) / 2, smx = dksmax(c);
    dkctx d0 = {b, c, a, v, k0};
    if (prev) rsad_warm(dk_real, &d0, b, s1, prev->main.s, prev->main.a1, &out->main.s, &out->main.a1, &out->main.g2, &out->main.L0);
    else rsad1(dk_real, &d0, b, s1, &out->main.s, &out->main.a1, &out->main.g2, &out->main.L0);
    out->valid = true;
    if (pdfm) return;
    dkctx d1 = {b, c, a, v, 1}, d2 = {b, c, a, v, 2};
    if (prev) { rsad_warm(dk_real, &d1, b, smx, prev->B.s, prev->B.a1, &out->B.s, &out->B.a1, &out->B.g2, &out->B.L0); rsad_warm(dk_real, &d2, b, smx, prev->C.s, prev->C.a1, &out->C.s, &out->C.a1, &out->C.g2, &out->C.L0); }
    else { rsad1(dk_real, &d1, b, smx, &out->B.s, &out->B.a1, &out->B.g2, &out->B.L0); rsad1(dk_real, &d2, b, smx, &out->C.s, &out->C.a1, &out->C.g2, &out->C.L0); }
}
/* ---- the inner inversion by several paths, accepted when two of them agree ----
 * evalpath(pth, NN, ctx, &I, &E) gives the trapezoidal value and the nested-rule estimate of path pth with NN nodes.  Each
 * path is first refined on its own (the node count doubled up to three times until its estimate is below tol max(1, |I|));
 * a value is accepted as soon as two paths agree to within that tolerance, the one with the smaller estimate being kept
 * and the discrepancy entering the error.  The nested rule alone can be fooled: an oscillating tail that is not
 * resolved by the rule and its half aliases into the same wrong value on both (case DK, c = 5, c0 = -5, v = 0.086,
 * a = 35: the parabola with 96 and 192 nodes agrees with itself to 1e-15 and is off by 1e-7, which the vertical line
 * exposes).  When no two paths agree, the value with the smallest estimate is kept if every other path is consistent
 * with it, the discrepancy within ten times the sum of the two estimates (a path that converges slowly, or not at all,
 * is no evidence against a resolved one); otherwise the inconsistent paths and the best one are recomputed with the node
 * count doubled, up to three more times, until two agree or all are consistent, and if they still disagree the best
 * value is returned with the smallest inconsistent discrepancy as its error, which the outer rule then reports. */
typedef void (*pathfn_t)(int pth, int NN, void *ctx, double *Iv, double *E);
static bool paths_agree(const double *Iv, int p, int q, double tol)
{
    return isfinite(Iv[p]) && isfinite(Iv[q]) && fabs(Iv[p] - Iv[q]) <= 2 * tol * fmax(1, fmax(fabs(Iv[p]), fabs(Iv[q])));
}
static void paths_accept(const double *Iv, const double *E, int p, int q, double *H, double *Eout)
{
    int b = E[p] <= E[q] ? p : q; *H = Iv[b]; *Eout = fmax(E[b], fabs(Iv[p] - Iv[q]));
}
static bool paths_inconsistent(const double *Iv, const double *E, int b, int q) { return q != b && isfinite(Iv[q]) && fabs(Iv[b] - Iv[q]) > 10 * (E[b] + E[q]); }
static int paths_best(const double *E, int np) { int b = 0; for (int p = 1; p < np; p++) if (E[p] < E[b]) b = p; return b; }
/* (verified = false: one evaluation, the second path, the vertical line, once with NN nodes: the oscillating tails that alias
 * on the parabola do not fool it; ONEPATH of julia/DFExact.jl) */
static void agreepaths(pathfn_t evalpath, void *ctx, int np, int NN, double tol, double *H, double *Eout, bool verified)
{
    if (!verified) { evalpath(np >= 2 ? 2 : 1, NN, ctx, H, Eout); return; }
    double Iv[3], E[3]; int N[3];
    for (int p = 0; p < np; p++) { Iv[p] = NAN; E[p] = INFINITY; N[p] = NN; }
    for (int p = 0; p < np; p++) {
        int NNq = NN;
        for (int rf = 0; rf <= 3; rf++) {
            double Ip, Ep; evalpath(p + 1, NNq, ctx, &Ip, &Ep);
            if (Ep < E[p]) { Iv[p] = Ip; E[p] = Ep; N[p] = NNq; }
            if (Ep <= tol * fmax(1, fabs(Ip))) break;
            NNq *= 2;
        }
        for (int q = 0; q < p; q++) if (paths_agree(Iv, p, q, tol)) { paths_accept(Iv, E, p, q, H, Eout); return; }
    }
    int b = paths_best(E, np); bool bad[3], anybad = false;
    if (!isfinite(E[b])) { *H = 0; *Eout = INFINITY; return; }
    for (int q = 0; q < np; q++) { bad[q] = paths_inconsistent(Iv, E, b, q); anybad |= bad[q]; }
    if (!anybad) { *H = Iv[b]; *Eout = E[b]; return; }
    for (int round = 0; round < 3; round++) {
        for (int p = 0; p < np; p++) if (p == b || bad[p]) { N[p] *= 2; evalpath(p + 1, N[p], ctx, &Iv[p], &E[p]); }   /* the finer rule replaces the value */
        for (int p = 0; p < np; p++) for (int q = 0; q < p; q++) if (paths_agree(Iv, p, q, tol)) { paths_accept(Iv, E, p, q, H, Eout); return; }
        b = paths_best(E, np); anybad = false;
        if (!isfinite(E[b])) { *H = 0; *Eout = INFINITY; return; }
        for (int q = 0; q < np; q++) { bad[q] = paths_inconsistent(Iv, E, b, q); anybad |= bad[q]; }
        if (!anybad) { *H = Iv[b]; *Eout = E[b]; return; }
    }
    double dmin = INFINITY;
    for (int q = 0; q < np; q++) if (bad[q]) dmin = fmin(dmin, fabs(Iv[b] - Iv[q]));
    *H = Iv[b]; *Eout = fmax(E[b], dmin);
}
typedef struct { double b, c, a, v; int k0; const dksad_t *sad; } dkpath_ctx;
static void dkpath(int pth, int NNq, void *v, double *Iv, double *Eq)
{
    dkpath_ctx *p = v; double dummy;
    dkctx dk = {p->b, p->c, p->a, p->v, p->k0};
    if (pth == 1) pathW(dk_lk, &dk, p->b, p->sad->main.s, p->sad->main.a1 / 2, NNq, false, Iv, Eq);
    else if (pth == 2) pathT(dk_lk, &dk, p->b, p->a, p->sad->main.s, p->sad->main.a1, NNq, p->c, false, false, Iv, Eq, &dummy);
    else { double IB, EB, IC, EC; dkctx d1 = {p->b, p->c, p->a, p->v, 1}, d2 = {p->b, p->c, p->a, p->v, 2};
        pathW(dk_lk, &d1, p->b, p->sad->B.s, p->sad->B.a1 / 2, NNq, false, &IB, &EB); pathT(dk_lk, &d2, p->b, p->a, p->sad->C.s, p->sad->C.a1, NNq, p->c, false, false, &IC, &EC, &dummy);
        *Iv = IB - IC; *Eq = EB + EC; }
}
static void dkpair(double b, double a, double c, bool pdfm, int NN, const dksad_t *sad, double *H, double *E, bool verified)
{
    *H = 0; *E = 0;
    if (!(a > 0)) return;
    double v = a - c * b, tol = 1e-13; int k0 = pdfm ? 3 : 0;
    dksad_t sd; if (sad == NULL) { dksaddles(b, a, c, pdfm, NULL, &sd); sad = &sd; }
    double g2 = sad->main.g2, L0 = sad->main.L0;
    double Lh = L0 - log(2 * M_PI * fmax(g2, 1e-6 * b * b)) / 2, lw = pdfm ? 1.5 * log(b) : log(b);
    if (Lh + lw < -35) return;
    dkpath_ctx pc = {b, c, a, v, k0, sad};
    agreepaths(dkpath, &pc, pdfm ? 2 : 3, NN, tol, H, E, verified);
}

/* ---- cases DU, DT, zero start ---- */
typedef struct { double b, c, r; int model; } iictx;
static void iilog(int n, const cdouble *S, cdouble *L, void *v)
{
    iictx *d = v; double *lcd = malloc(n * sizeof(double)), *la0 = malloc(n * sizeof(double));
    for (int k = 0; k < n; k++) {
        dfl_girs_t G; dfl_girsanov(d->model, d->c, S[k], &G);
        cdouble lcdk = cadd(G.lch, clog(G.dS)), la0k = clog(cneg(G.A0));
        lcd[k] = cimag(lcdk); la0[k] = cimag(la0k);
        cdouble Sq = csqrt(csub(cmul(G.A1, G.A1), cscale(4, cmul(G.A0, G.A2))));
        cdouble z = cdiv(cscale(fabs(d->r), Sq), cscale(-2, G.A0));
        cdouble ex = d->r < 0 ? cdiv(cscale(fabs(d->r), cadd(G.A1, Sq)), cscale(2, G.A0)) : cdiv(cscale(-2 * d->r, G.A2), cadd(G.A1, Sq));
        L[k] = cadd(cadd(CPX(-d->c / 2 - creal(lcdk) / 2 - log(M_PI) - creal(la0k) / 2, 0), ex), cadd(clog(dfl_k0scaled(z)), cscale(d->b / 2, cmul(G.del, cadd(G.th, CPX(d->c, 0))))));
    }
    for (int k = 1; k < n; k++) { lcd[k] -= 2 * M_PI * round((lcd[k] - lcd[k - 1]) / (2 * M_PI)); la0[k] -= 2 * M_PI * round((la0[k] - la0[k - 1]) / (2 * M_PI)); }
    for (int k = 0; k < n; k++) L[k] = csub(L[k], CPX(0, lcd[k] / 2 + la0[k] / 2));
    free(lcd); free(la0);
}
static double ii_real(double s, void *v) { cdouble S = CPX(s, 0), L; iilog(1, &S, &L, v); return creal(L); }
static struct { int model; double c, sab; } IIABS[256]; static int niiabs = 0; static dfl_lock_t iilock = DFL_LOCK_INIT;
static bool ii_okp(int model, double c, double s) { dfl_girs_t G; dfl_girsanov(model, c, CPX(s, 0), &G); cdouble v = cexp(cadd(G.lch, clog(G.dS)));
    return creal(v) > 0 && fabs(cimag(v)) < 1e-9 * cabs(v) && creal(csub(cmul(G.A1, G.A1), cscale(4, cmul(G.A0, G.A2)))) > 0 && creal(cneg(G.A0)) > 0; }
static double iiabs(double c, int model)
{
    dfl_lock(&iilock);
    for (int k = 0; k < niiabs; k++) if (IIABS[k].model == model && IIABS[k].c == c) { double r = IIABS[k].sab; dfl_unlock(&iilock); return r; }
    dfl_unlock(&iilock);
    double sp = (c * c + M_PI * M_PI / 4) / 2, lo = 0, hi = sp * (1 - 1e-12), sab;
    if (ii_okp(model, c, hi)) sab = hi;
    else { for (int k = 0; k < 200; k++) { double m = (lo + hi) / 2; if (m == lo || m == hi) break; if (ii_okp(model, c, m)) lo = m; else hi = m; } sab = lo; }
    dfl_lock(&iilock); if (niiabs < 256) { IIABS[niiabs].model = model; IIABS[niiabs].c = c; IIABS[niiabs].sab = sab; niiabs++; } dfl_unlock(&iilock);
    return sab;
}
/* r = a(b) kept away from the logarithmic singularity at r = 0: |r| at least 1e-200, and for c < -20 at least 2^-54, the rounding
 * error of a(b) = 1/2 + x sqrt(b) + c0 b; the node next to the zero of a(b) has a true |a| of 1e-16 on the short pieces graded toward
 * the step, and 1e-200 in place of a computed 0 costs 2e-9 in F at c = -480 (rfloor in DFExact.jl; unchanged on [-20, inf)).
 * The floor 2^-54 also for c >= -20 changes F by at most 1.6e-15 and f by at most 1.4e-12 relative over 352 values of cases
 * DU, DT, c in [-20, 4.7], c0 = 0 and c0 != 0 (measured in November 2026, not adopted).  gden uses the same floor. */
static double rfloor(double r, double c) { double m = c < -20 ? 5.5511151231257827e-17 /* 2^-54 */ : 1e-200; return fabs(r) < m ? (r < 0 ? -m : m) : r; }
static void iisaddle(double b, double r, double c, int model, const sad_t *prev, sad_t *out)
{
    r = rfloor(r, c);
    iictx d = {b, c, r, model}; double sab = iiabs(c, model);
    if (prev) rsad_warm(ii_real, &d, b, sab, prev->s, prev->a1, &out->s, &out->a1, &out->g2, &out->L0);
    else rsad1(ii_real, &d, b, sab, &out->s, &out->a1, &out->g2, &out->L0);
    if (out->s > sab - 1 / b) { out->s = sab - 1 / b; out->a1 = 1 / b; out->g2 = b * b; out->L0 = ii_real(out->s, &d); }
}
typedef struct { iictx d; double b, r, c, s, a1, sab; bool pdfm; } iipath_ctx;
static void iipath(int pth, int NNq, void *v, double *Iv, double *Eq)
{
    iipath_ctx *p = v; iictx d = p->d;
    if (pth == 1) {
        double sq = p->s;
        if (!p->pdfm) { if (fabs(sq) < p->a1) sq = sq >= 0 ? p->a1 : -p->a1; sq = fmin(sq, p->sab - 0.5 * p->a1); }
        pathW(iilog, &d, p->b, sq, p->a1 / 2, NNq, !p->pdfm, Iv, Eq);
        if (!p->pdfm && sq < 0) *Iv += creal(cexp(CPX(ii_real(0, &d), 0)));
    } else {
        double sA; pathT(iilog, &d, p->b, p->r, p->s, p->a1, NNq, p->c, !p->pdfm, !p->pdfm, Iv, Eq, &sA);
        if (!p->pdfm && sA < 0) *Iv += creal(cexp(CPX(ii_real(0, &d), 0)));
    }
}
static void iipair(double b, double r, double c, int model, bool pdfm, int NN, const sad_t *sad, double *H, double *E, bool verified)
{
    double tol = 1e-13; r = rfloor(r, c);
    iictx d = {b, c, r, model}; double sab = iiabs(c, model);
    sad_t sd; if (sad == NULL) { iisaddle(b, r, c, model, NULL, &sd); sad = &sd; }
    double s = sad->s, a1 = sad->a1, g2 = sad->g2, L0 = sad->L0;
    double Lh = L0 - log(2 * M_PI * fmax(g2, 1e-6 * b * b)) / 2;
    double lw = pdfm ? 1.5 * log(b) : log(b) + log(1 + fabs(c) + fabs(r) / b) - log(fmax(fabs(s), a1));
    *H = 0; *E = 0;
    if (Lh + lw < -35) { if (!pdfm && s < 0) *H = creal(cexp(CPX(ii_real(0, &d), 0))); return; }
    iipath_ctx pc = {d, b, r, c, s, a1, sab, pdfm};
    agreepaths(iipath, &pc, 2, NN, tol, H, E, verified);
}
/* P[k1 Z1^2 + k2 Z2^2 <= a] for real weights: composite Gauss-Legendre over [l0, l0 + 40 sd] */
static double Pk0(double k1, double k2, double a)
{
    double s1 = 2 * k1, s2 = -2 * k2, l0 = sqrt(fmax(0, -2 * a)), sd = sqrt(s2), P = 0;
    const double *gw; const double *u = dfl_rule(32, &gw); int npan = 40;
    for (int p = 0; p < npan; p++) {
        double A = l0 + 40 * sd * p / npan, B = l0 + 40 * sd * (p + 1) / npan;
        for (int k = 0; k < 32; k++) { double l = A + (B - A) * u[k]; P += (B - A) * gw[k] * 2 * exp(-l * l / (2 * s2)) / sqrt(2 * M_PI * s2) * erf(sqrt(fmax(2 * a + l * l, 0) / (2 * s1))); }
    }
    return P;
}

/* ---- the step of the outer integrand in the far upper tail ---- */
typedef struct { int model; double c, b; } cmctx;
static double cm_L(double s, void *v) { cmctx *x = v; if (x->model == 1) { cdouble S = CPX(s, 0), L; dklam(1, &S, x->b, x->c, 1, 1, 1, &L); return creal(L); }
    dfl_girs_t G; dfl_girsanov(x->model, x->c, CPX(s, 0), &G); return creal(cadd(csub(CPX(-x->c / 2, 0), cadd(cscale(0.5, G.lch), cscale(0.5, clog(G.D)))), cscale(x->b / 2, cmul(G.del, cadd(G.th, CPX(x->c, 0)))))); }
static void cm_m(int model, double c, double s, double *m, double *v) { if (model == 1) { double k1 = 1 / (2 * dketa(s, c)); *m = k1; *v = 2 * k1 * k1; return; }
    dfl_girs_t G; dfl_girsanov(model, c, CPX(s, 0), &G); double A2 = creal(G.A2), A1 = creal(G.A1), A0 = creal(G.A0); *m = A1 / (2 * A2); *v = ((A1 / A2) * (A1 / A2) - 2 * A0 / A2) / 2; }
static void condmom(int model, double c, double smx, double lb, double *m, double *sd)
{
    double b = exp(lb); cmctx x = {model, c, b}; double s, a1, g2, L0;
    rsad1(cm_L, &x, b, smx, &s, &a1, &g2, &L0);
    double h = fmin(1e-3 * a1, (smx - s) / 4), mm, vt, mp, vp, mn, vn;
    cm_m(model, c, s, &mm, &vt); cm_m(model, c, s + h, &mp, &vp); cm_m(model, c, s - h, &mn, &vn);
    double dm = (mp - mn) / (2 * h), kss = (cm_L(s - h, &x) - 2 * cm_L(s, &x) + cm_L(s + h, &x)) / (h * h);
    double vc = fmax(vt - dm * dm / kss, 1e-8 * vt);
    *m = mm; *sd = sqrt(vc);
}
typedef struct { int model; double c, lo, hi, smx, lb[61], m[61], sd[61]; } stepgrid_t;
static stepgrid_t SG[64]; static int nsg = 0; static dfl_lock_t sglock = DFL_LOCK_INIT;
typedef struct { stepgrid_t *G; } sgjob;
static void sg_body(int j, void *v) { sgjob *x = v; condmom(x->G->model, x->G->c, x->G->smx, x->G->lb[j], &x->G->m[j], &x->G->sd[j]); }
static const stepgrid_t *stepgrid(int model, double c, double lo, double hi)
{
    dfl_lock(&sglock);
    for (int k = 0; k < nsg; k++) if (SG[k].model == model && SG[k].c == c && SG[k].lo == lo && SG[k].hi == hi) { dfl_unlock(&sglock); return &SG[k]; }
    stepgrid_t *G = &SG[nsg < 64 ? nsg : 63]; G->model = model; G->c = c; G->lo = lo; G->hi = hi;
    if (model == 1) G->smx = dksmax(c); else { double t = dfl_tstar(model, c, false); G->smx = isfinite(t) ? t : (c * c + M_PI * M_PI / 4) / 2; }
    for (int j = 0; j < 61; j++) G->lb[j] = lo + (hi - lo) * j / 60;
    sgjob jb = {G}; dfl_parfor(61, sg_body, &jb);
    if (nsg < 64) nsg++;
    dfl_unlock(&sglock);
    return G;
}
static int steproots(const stepgrid_t *G, double x, double c0, double wmax, double *ls, double *ws)
{
    int nr = 0; double ph[61]; bool ok[61];
    for (int j = 0; j < 61; j++) { ph[j] = 0.5 + x * exp(G->lb[j] / 2) + c0 * exp(G->lb[j]) - G->m[j]; ok[j] = isfinite(ph[j]) && isfinite(G->sd[j]); }
    for (int j = 0; j < 60; j++) {
        if (!(ok[j] && ok[j + 1] && (ph[j] < 0) != (ph[j + 1] < 0))) continue;
        double la = G->lb[j], lc = G->lb[j + 1], fa = ph[j], fc = ph[j + 1], sa = G->sd[j], sdc = G->sd[j + 1]; int side = 0;
        for (int it = 0; it < 6; it++) {
            double l = la - fa * (lc - la) / (fc - fa), mn, sn; condmom(G->model, G->c, G->smx, l, &mn, &sn);
            double fn = 0.5 + x * exp(l / 2) + c0 * exp(l) - mn; if (!isfinite(fn)) break;
            if ((fn < 0) == (fa < 0)) { la = l; fa = fn; sa = sn; if (side == -1) fc /= 2; side = -1; }
            else { lc = l; fc = fn; sdc = sn; if (side == 1) fa /= 2; side = 1; }
        }
        double t = fa / (fa - fc), l = la + t * (lc - la), sd = sa + t * (sdc - sa), m1, s1, m2, s2;
        condmom(G->model, G->c, G->smx, l - 1e-3, &m1, &s1); condmom(G->model, G->c, G->smx, l + 1e-3, &m2, &s2);
        double sl = ((0.5 + x * exp((l + 1e-3) / 2) + c0 * exp(l + 1e-3) - m2) - (0.5 + x * exp((l - 1e-3) / 2) + c0 * exp(l - 1e-3) - m1)) / 2e-3;
        double wd = sd / fabs(sl);
        if (wd < wmax && nr < 8) { ls[nr] = l; ws[nr] = fmax(wd, 1e-6); nr++; }
    }
    return nr;
}
/* the pieces in log b of the points x */
typedef struct { int cap, n; double *PA, *PL; bool *PEL, *PER; int *PX; } pieces_t;
static void push_piece(pieces_t *P, double a, double L, bool el, bool er, int x)
{
    if (P->n == P->cap) { P->cap = P->cap ? 2 * P->cap : 256; P->PA = realloc(P->PA, P->cap * sizeof(double)); P->PL = realloc(P->PL, P->cap * sizeof(double)); P->PEL = realloc(P->PEL, P->cap); P->PER = realloc(P->PER, P->cap); P->PX = realloc(P->PX, P->cap * sizeof(int)); }
    P->PA[P->n] = a; P->PL[P->n] = L; P->PEL[P->n] = el; P->PER[P->n] = er; P->PX[P->n] = x; P->n++;
}
static int cmp_double(const void *a, const void *b) { double x = *(const double *) a, y = *(const double *) b; return x < y ? -1 : x > y; }
static void pieces_logb(int nx, const double *x, int model, double c, double c0, double lo, double hi, double plen, double stepw, bool dk, pieces_t *P)
{
    const stepgrid_t *Gr = stepgrid(model, c, lo, hi);
    for (int k = 0; k < nx; k++) {
        double xk = x[k], rts[2]; int nrt = 0;
        if (c0 == 0) { rts[0] = -1 / (2 * xk); nrt = 1; }
        else { double d = xk * xk - 2 * c0; if (d >= 0) { rts[0] = (-xk - sqrt(d)) / (2 * c0); rts[1] = (-xk + sqrt(d)) / (2 * c0); nrt = 2; } }
        double lr[2]; int nl = 0;
        for (int i = 0; i < nrt; i++) if (rts[i] > 0) { double l = 2 * log(rts[i]); if (l > lo && l < hi) lr[nl++] = l; }
        if (nl == 2 && lr[0] > lr[1]) { double t = lr[0]; lr[0] = lr[1]; lr[1] = t; }
        double sp = NAN, sg = INFINITY;
        if (c0 > 0 && xk < 0) {
            double r0 = -xk / (2 * c0), amin = 0.5 - xk * xk / (4 * c0), l0 = 2 * log(r0);
            if (nl == 2 && lr[1] - lr[0] < 1e-10) { lr[0] = (lr[0] + lr[1]) / 2; nl = 1; }
            else if (nl == 2) sg = lr[1] - lr[0];
            else if (l0 > lo && l0 < hi) { if (amin <= 1e-13) { lr[0] = l0; nl = 1; } else { sg = 2 * sqrt(amin / c0) / r0; sp = l0; } }
        }
        double pts[512]; int np = 0;
        pts[np++] = lo; pts[np++] = hi; for (int i = 0; i < nl; i++) pts[np++] = lr[i];
        if (sg < plen) {
            double gr[3]; int ng = 0; for (int i = 0; i < nl; i++) gr[ng++] = lr[i]; if (!isnan(sp)) gr[ng++] = sp;
            for (int g = 0; g < ng; g++) for (int e = 0; e <= (int) ceil(log2(plen / sg)); e++) { double d = sg * pow(2, e); if (np < 500) { pts[np++] = gr[g] - d; pts[np++] = gr[g] + d; } }
        }
        double ls[8], ws[8]; int ns = steproots(Gr, xk, c0, stepw, ls, ws);
        for (int g = 0; g < ns; g++) { if (np < 500) pts[np++] = ls[g]; for (int e = 0; e <= (int) ceil(log2(plen / (2 * ws[g]))); e++) { double d = 2 * ws[g] * pow(2, e); if (np < 500) { pts[np++] = ls[g] - d; pts[np++] = ls[g] + d; } } }
        /* keep those in [lo, hi], drop slivers at a zero, sort, unique */
        double q[512]; int nq = 0;
        for (int i = 0; i < np; i++) {
            if (!(pts[i] >= lo && pts[i] <= hi)) continue;
            bool sliver = false; for (int g = 0; g < nl; g++) if (fabs(pts[i] - lr[g]) < 1e-9 * fmin(fmax(sg, 1e-3), plen) && pts[i] != lr[g]) sliver = true;
            if (!sliver) q[nq++] = pts[i];
        }
        qsort(q, nq, sizeof(double), cmp_double);
        int nu = 0; for (int i = 0; i < nq; i++) if (i == 0 || q[i] != q[nu - 1]) q[nu++] = q[i];
        if (dk && nl == 2) { int m = 0; for (int i = 0; i < nu; i++) if (q[i] <= lr[0] || q[i] >= lr[1]) q[m++] = q[i]; nu = m; }
        for (int p = 0; p < nu - 1; p++) {
            if (dk && nl == 2 && q[p] == lr[0]) continue;
            int npp = (int) fmax(1, ceil((q[p + 1] - q[p]) / plen));
            for (int i = 0; i < npp; i++) {
                double a = q[p] + (q[p + 1] - q[p]) * i / npp, b = q[p] + (q[p + 1] - q[p]) * (i + 1) / npp;
                bool el = false, er = false; for (int g = 0; g < nl; g++) { if (i == 0 && q[p] == lr[g]) el = true; if (i == npp - 1 && q[p + 1] == lr[g]) er = true; }
                push_piece(P, a, b - a, el, er, k);
            }
        }
    }
}
static double EVdk(double c) { return fabs(c) < 1e-8 ? 0.5 : (exp(2 * c) - 1 - 2 * c) / (4 * c * c); }

/* ---- the zero-start route ---- */
typedef struct { int model, M, Nv; double c, c0; bool pdfm; const double *x; void *sads;
                 double (*bnd)(double b, const void *ctx); const void *bctx;   /* a general boundary a(b) (case DK; dfl_region) */
                 bool two; void *sads2; bool verified; } zctx;                              /* the region |R| <= a(b) (cases DU, DT; dfl_adfmax) */
typedef struct { int n, M; const double *B; const int *IX; zctx *z; void *sads; } blockjob;
static void zsweep_body(int p, void *v)
{
    blockjob *j = v; zctx *z = j->z; int M = j->M;
    if (z->model == 1) { dksad_t *S = j->sads; const dksad_t *prev = NULL;
        for (int i = p * M; i < (p + 1) * M; i++) { double b = j->B[i], a = z->bnd ? z->bnd(b, z->bctx) : 0.5 + z->x[j->IX[i]] * sqrt(b) + z->c0 * b; S[i].valid = false; if (a > 0) { dksaddles(b, a, z->c, z->pdfm, prev, &S[i]); prev = &S[i]; } } }
    else { sad_t *S = j->sads; const sad_t *prev = NULL;
        for (int i = p * M; i < (p + 1) * M; i++) { double b = j->B[i], a = 0.5 + z->x[j->IX[i]] * sqrt(b) + z->c0 * b; iisaddle(b, a, z->c, z->model, prev, &S[i]); prev = &S[i]; }
        if (z->two) { sad_t *S2 = z->sads2; const sad_t *pm = NULL;
            for (int i = p * M; i < (p + 1) * M; i++) { double b = j->B[i], a = 0.5 + z->x[j->IX[i]] * sqrt(b) + z->c0 * b; iisaddle(b, -a, z->c, z->model, pm, &S2[i]); pm = &S2[i]; } } }
}
typedef struct { const double *B; const int *IX; double *H, *E; zctx *z; void *sads; } pairjob;
static void zpair_body(int i, void *v)
{
    pairjob *j = v; zctx *z = j->z; double b = j->B[i], xx = z->x[j->IX[i]], a = z->bnd ? z->bnd(b, z->bctx) : 0.5 + xx * sqrt(b) + z->c0 * b, h, e;
    if (z->model == 1) { dksad_t *S = j->sads; dkpair(b, a, z->c, z->pdfm, z->Nv, S[i].valid ? &S[i] : NULL, &h, &e, z->verified); if (z->pdfm) { h *= sqrt(b); e *= sqrt(b); } }
    else { sad_t *S = j->sads; iipair(b, a, z->c, z->model, z->pdfm, z->Nv, &S[i], &h, &e, z->verified);
        if (z->two) { sad_t *S2 = z->sads2; double h2, e2; iipair(b, -a, z->c, z->model, z->pdfm, z->Nv, &S2[i], &h2, &e2, z->verified); h += h2; e += e2; }
        double fac = z->pdfm ? sqrt(b) : xx / (2 * sqrt(b)) + z->c0; h *= fac; e *= fabs(fac); }
    j->H[i] = h; j->E[i] = e;
}
static void zero_evalH(int n, const double *B, const int *IX, double *H, double *E, void *v)
{
    zctx *z = v; int M = z->M, nblk = n / M;
    void *sads = z->model == 1 ? (void *) malloc(n * sizeof(dksad_t)) : (void *) malloc(n * sizeof(sad_t));
    if (z->two) z->sads2 = malloc(n * sizeof(sad_t));
    blockjob bj = {n, M, B, IX, z, sads}; dfl_parfor(nblk, zsweep_body, &bj);
    pairjob pj = {B, IX, H, E, z, sads}; dfl_parfor(n, zpair_body, &pj);
    free(sads); if (z->two) { free(z->sads2); z->sads2 = NULL; }
}
static int double_zero(int kind, int nx, const double *x, int model, double c, double c0, const dfl_double_opts *o, double *F, dfl_double_info *info)
{
    bool pdfm = kind == 1; int M = o->M > 0 ? o->M : (model == 1 ? 32 : 64);
    /* the lower end in log b: E V e^{-blo} (at most bmin) in case DK; bmin in cases DU, DT for c >= -20, but E V ~ 1/(2|c|) falls below
     * bmin = 3e-3 at c = -167: for c < -20 the end is E V e^{-3} (at most bmin), below which nothing is lost (P(V <= u E V) <=
     * exp(-|c| (1 - u)^2 / (4u)) up to a power of 1/u), and no deeper, since at E V e^{-8} the logarithm of the integrand keeps seven
     * digits only at the saddle point and the search stops on noise at some nodes (dzlo in DFExact.jl) */
    double EB = EVdk(c), lo = model == 1 ? fmin(log(o->bmin), log(EB) - o->blo) : (c < -20 ? fmin(log(o->bmin), log(EB) - 3.0) : log(o->bmin)), hi = log(EB) + o->bhi;
    const orules_t *OR = outer_rules(M);
    pieces_t P = {0}; pieces_logb(nx, x, model, c, c0, lo, hi, o->plen, o->stepw, model == 1, &P);
    zctx z = {model, M, o->Nv, c, c0, pdfm, x, NULL, NULL, NULL, false, NULL, o->verified != 0};
    double *err = malloc(nx * sizeof(double)); int *nun = malloc(nx * sizeof(int)), *nref = malloc(nx * sizeof(int));
    adapt_out ao = {F, err, nun, nref, 0};
    /* (verified = false: one evaluation: one path per inner inversion; the adaptive outer rule stays, without its bisection
     * the values far from the unit root were off by up to 1e-2) */
    adaptb(P.n, P.PA, P.PL, P.PEL, P.PER, P.PX, nx, OR->u, OR->gw, model == 1 ? OR->z1 : OR->z2, model == 1 ? OR->w1 : OR->w2, OR, zero_evalH, &z, o->otol, o->maxlev, &ao);
    if (model != 1 && !pdfm) {
        dfl_girs_t G0; dfl_girsanov(model, c, ZERO, &G0); cdouble S0 = csqrt(csub(cmul(G0.A1, G0.A1), cscale(4, cmul(G0.A0, G0.A2))));
        double k10 = creal(cdiv(cadd(G0.A1, S0), cscale(4, G0.A2))), k20 = creal(cdiv(G0.A0, cadd(G0.A1, S0))), blo_ = exp(lo);
        for (int k = 0; k < nx; k++) F[k] += Pk0(k10, k20, 0.5 + x[k] * sqrt(blo_) + c0 * blo_);
    }
    /* case DK: where the zeros of a(b) coincide (c0 > 0, x = -sqrt(2 c0)) the density is infinite (a logarithmic singularity); in
     * cases DU, DT it is finite there.  A value that is not finite counts as flagged, except that infinite density of case DK. */
    for (int k = 0; k < nx; k++) {
        bool dkinf = model == 1 && pdfm && c0 > 0 && fabs(x[k] + sqrt(2 * c0)) <= 8 * 2.220446049250313e-16 * fmax(1, sqrt(2 * c0));
        if (dkinf) F[k] = INFINITY;
        info[k].err = err[k]; info[k].nbad = 0; info[k].nunres = nun[k]; info[k].nrefine = nref[k]; info[k].status = 2 * (err[k] > o->errtol || (!isfinite(F[k]) && !dkinf));
    }
    free(P.PA); free(P.PL); free(P.PEL); free(P.PER); free(P.PX); free(err); free(nun); free(nref);
    return 0;
}

/* ---- with an initial observation ---- */
typedef struct {
    int model, kind, M, Nv, Nl; double c, c0, g0, s02, smax0, smax, mumin, tdec, brng; bool stat, st, nz, nzR, useg, dkfixed, pdfm;
    cdouble k10, k20, mq0; double xx; int nbad, nunres;
    bool two;                                  /* the region |R| <= a(b): also r = -a(b) on the same nodes (dfl_adfmax) */
    bool verified;                             /* false: one evaluation (no recomputation of unresolved columns) */
} sctx_t;
/* log M_V(s) at real s with the initial-value terms; ok: s in the domain */
static double LB(const sctx_t *S, double s, bool *ok)
{
    *ok = false;
    if (!(s < S->smax0)) return -INFINITY;
    double L, v; dfl_girs_t G;
    if (S->model == 1) { dfl_girsanov(1, S->c, CPX(s, 0), &G); v = creal(G.eta); L = -S->c / 2 - creal(G.lsh) / 2 - log(fabs(v)) / 2; }
    else { cdouble mu, Dt = dfl_Dscaled(S->model, S->c, ZERO, CPX(s, 0), &mu); v = creal(mu) > 1 ? creal(Dt) : creal(cmul(Dt, cexp(mu))); L = -(creal(clog(Dt)) + creal(mu)) / 2; }
    bool okk = isfinite(v) && v > 0 && isfinite(L);
    if (S->nzR) {
        double Q0 = creal(dfl_qstart_d(S->model, S->c, ZERO, CPX(s, 0))); okk = okk && isfinite(Q0);
        if (S->stat) { double d1 = 1 - 2 * S->s02 * Q0; okk = okk && d1 > 0; L -= log(fabs(d1)) / 2; } else L += S->g0 * S->g0 * Q0;
    }
    if (S->nz) {
        if (S->model != 1) dfl_girsanov(1, S->c, CPX(s, 0), &G);
        double aq = creal(cdiv(CPX(s, 0), G.eta)); okk = okk && isfinite(aq);
        if (S->st) { double d1 = 1 - 2 * aq * S->s02; okk = okk && d1 > 0; L -= log(fabs(d1)) / 2; } else L += S->g0 * S->g0 * aq;
    }
    *ok = okk; return L;
}
/* the continuous logarithm along a path, refined where the phase jumps by more than pi/4 (f evaluates the function) */
typedef cdouble (*pfn_t)(cdouble s, const void *ctx);
static void pathlog(pfn_t f, const void *ctx, int n, const cdouble *S, const cdouble *Z, bool round0, cdouble *out)
{
    int cap = 4 * n + 64, m = n; double *tt = malloc(cap * sizeof(double)); cdouble *lv = malloc(cap * sizeof(cdouble));
    for (int k = 0; k < n; k++) { tt[k] = k; lv[k] = clog(Z[k]); }
    for (int it = 0; it < 40; it++) {
        int nbad = 0; double *tm = malloc(m * sizeof(double));
        for (int k = 0; k + 1 < m; k++) { double d = carg(cexp(CPX(0, cimag(lv[k + 1]) - cimag(lv[k])))); if (fabs(d) > M_PI / 4) tm[nbad++] = (tt[k] + tt[k + 1]) / 2; }
        if (nbad == 0) { free(tm); break; }
        if (m + nbad > cap) { cap = 2 * (m + nbad); tt = realloc(tt, cap * sizeof(double)); lv = realloc(lv, cap * sizeof(cdouble)); }
        for (int q = 0; q < nbad; q++) {
            double xq = tm[q]; int fl = (int) floor(xq); if (fl > n - 2) fl = n - 2; double fr = xq - fl;
            cdouble sm = cadd(cscale(1 - fr, S[fl]), cscale(fr, S[fl + 1]));
            tt[m] = xq; lv[m] = clog(f(sm, ctx)); m++;
        }
        free(tm);
        /* insertion sort by tt (small arrays) */
        for (int i = 1; i < m; i++) { double t = tt[i]; cdouble l = lv[i]; int j = i - 1; while (j >= 0 && tt[j] > t) { tt[j + 1] = tt[j]; lv[j + 1] = lv[j]; j--; } tt[j + 1] = t; lv[j + 1] = l; }
    }
    double *ph = malloc(m * sizeof(double)); ph[0] = cimag(lv[0]);
    for (int k = 1; k < m; k++) ph[k] = ph[k - 1] + carg(cexp(CPX(0, cimag(lv[k]) - cimag(lv[k - 1]))));
    if (round0) { double sh = 2 * M_PI * round(ph[0] / (2 * M_PI)); for (int k = 0; k < m; k++) ph[k] -= sh; }
    int j = 0;
    for (int k = 0; k < n; k++) { while (tt[j] != k) j++; out[k] = CPX(creal(lv[j]), ph[j]); }
    free(tt); free(lv); free(ph);
}
typedef struct { int model; double c; } gctx1;
static cdouble f_eta(cdouble s, const void *v) { const gctx1 *g = v; dfl_girs_t G; dfl_girsanov(1, g->c, s, &G); return G.eta; }
static cdouble f_Dt(cdouble s, const void *v) { const gctx1 *g = v; cdouble mu; return dfl_Dscaled(g->model, g->c, ZERO, s, &mu); }
void dfl_qpieces(int model, double c, cdouble S, double gg, cdouble s1, cdouble s2, bool girs, cdouble *Q0, cdouble *m1s, cdouble *m2s)
{
    cdouble (*qf)(int, double, cdouble, cdouble) = girs ? dfl_qstart : dfl_qstart_d;
    cdouble la = cdiv(CPX(0.25, 0), s1), lb = cdiv(CPX(-0.25, 0), s2);
    *Q0 = cscale(gg * gg, qf(model, c, ZERO, S));
    cdouble Qa = csub(cscale(gg * gg, qf(model, c, la, S)), *Q0), Qb = csub(cscale(gg * gg, qf(model, c, lb, S)), *Q0);
    cdouble A11 = cdiv(cscale(0.5, la), csub(ONE, cmul(la, s1))), A12 = cdiv(cscale(-0.5, la), cadd(ONE, cmul(la, s2)));
    cdouble A21 = cdiv(cscale(0.5, lb), csub(ONE, cmul(lb, s1))), A22 = cdiv(cscale(-0.5, lb), cadd(ONE, cmul(lb, s2)));
    cdouble dt = csub(cmul(A11, A22), cmul(A12, A21));
    *m1s = cdiv(csub(cmul(Qa, A22), cmul(A12, Qb)), dt); *m2s = cdiv(csub(cmul(A11, Qb), cmul(A21, Qa)), dt);
}
/* log M_V and the weights (and means) at the nodes of a path */
static void nodepar(const sctx_t *S, int n, const cdouble *Sn, cdouble *L, cdouble *k1, cdouble *k2, cdouble *mq, cdouble *m2s)
{
    gctx1 g = {S->model, S->c};
    cdouble *Z = malloc(n * sizeof(cdouble)), *lz = malloc(n * sizeof(cdouble));
    dfl_girs_t *Gs = S->model == 1 ? malloc(n * sizeof(dfl_girs_t)) : NULL;
    if (S->model == 1) {
        for (int k = 0; k < n; k++) { dfl_girsanov(1, S->c, Sn[k], &Gs[k]); Z[k] = Gs[k].eta; }
        pathlog(f_eta, &g, n, Sn, Z, true, lz);
        for (int k = 0; k < n; k++) { L[k] = csub(CPX(-S->c / 2, 0), cadd(cscale(0.5, Gs[k].lsh), cscale(0.5, lz[k]))); k1[k] = cdiv(ONE, cscale(2, Gs[k].eta)); k2[k] = ZERO; }
    } else {
        cdouble *mu = malloc(n * sizeof(cdouble));
        for (int k = 0; k < n; k++) Z[k] = dfl_Dscaled(S->model, S->c, ZERO, Sn[k], &mu[k]);
        pathlog(f_Dt, &g, n, Sn, Z, false, lz);
        double sh = 2 * M_PI * round(cimag(cadd(lz[0], mu[0])) / (2 * M_PI));
        for (int k = 0; k < n; k++) {
            cdouble lt = csub(cadd(lz[k], mu[k]), CPX(0, sh)); L[k] = cscale(-0.5, lt);
            cdouble m2, dp = dfl_Dscaled(S->model, S->c, CPX(-1, 0), Sn[k], &m2), dm = dfl_Dscaled(S->model, S->c, CPX(1, 0), Sn[k], &m2), a0 = Z[k];
            cdouble a1 = cscale(0.5, csub(dp, dm)), a2 = csub(cscale(0.5, cadd(dp, dm)), a0);
            cdouble p1 = cdiv(a1, cscale(2, a0)), p2 = cdiv(a2, cscale(4, a0)), dsc = csqrt(csub(cmul(p1, p1), cscale(4, p2)));
            k1[k] = cscale(0.5, cadd(p1, dsc)); k2[k] = cscale(0.5, csub(p1, dsc));
            if (creal(k2[k]) > creal(k1[k])) { cdouble t = k1[k]; k1[k] = k2[k]; k2[k] = t; }
        }
        free(mu);
    }
    for (int k = 0; k < n; k++) { mq[k] = ZERO; m2s[k] = ZERO; }
    if (S->nzR) {
        if (S->stat) {
            double *ph = malloc(n * sizeof(double)); cdouble *d1 = malloc(n * sizeof(cdouble)), *mu1 = malloc(n * sizeof(cdouble)), *mu2 = malloc(n * sizeof(cdouble));
            for (int k = 0; k < n; k++) { cdouble Q0; dfl_qpieces(S->model, S->c, Sn[k], 1.0, cscale(2, k1[k]), cscale(-2, k2[k]), false, &Q0, &mu1[k], &mu2[k]); d1[k] = csub(ONE, cscale(2 * S->s02, Q0)); ph[k] = cimag(clog(d1[k])); }
            for (int k = 1; k < n; k++) ph[k] -= 2 * M_PI * round((ph[k] - ph[k - 1]) / (2 * M_PI));
            for (int k = 0; k < n; k++) {
                L[k] = csub(L[k], cscale(0.5, CPX(log(cabs(d1[k])), ph[k])));
                cdouble st2 = cdiv(CPX(S->s02, 0), d1[k]);
                cdouble p1 = cadd(cadd(k1[k], k2[k]), cscale(0.5, cmul(st2, csub(mu1[k], mu2[k]))));
                cdouble p2 = cadd(cmul(k1[k], k2[k]), cscale(0.5, cmul(st2, csub(cmul(mu1[k], k2[k]), cmul(mu2[k], k1[k])))));
                cdouble dsc = csqrt(csub(cmul(p1, p1), cscale(4, p2)));
                k1[k] = cscale(0.5, cadd(p1, dsc)); k2[k] = cscale(0.5, csub(p1, dsc));
                if (creal(k2[k]) > creal(k1[k])) { cdouble t = k1[k]; k1[k] = k2[k]; k2[k] = t; }
            }
            free(ph); free(d1); free(mu1); free(mu2);
        } else {
            for (int k = 0; k < n; k++) { cdouble Q0; dfl_qpieces(S->model, S->c, Sn[k], S->g0, cscale(2, k1[k]), cscale(-2, k2[k]), false, &Q0, &mq[k], &m2s[k]); L[k] = cadd(L[k], Q0); }
        }
    }
    if (S->nz) {
        cdouble *aq = malloc(n * sizeof(cdouble)), *mqk = malloc(n * sizeof(cdouble));
        for (int k = 0; k < n; k++) { aq[k] = cdiv(Sn[k], Gs[k].eta); mqk[k] = cdiv(cexp(cneg(Gs[k].lsh)), Gs[k].eta); }
        if (S->st) {
            double *ph = malloc(n * sizeof(double)); cdouble *d1 = malloc(n * sizeof(cdouble));
            for (int k = 0; k < n; k++) { d1[k] = csub(ONE, cscale(2 * S->s02, aq[k])); ph[k] = cimag(clog(d1[k])); }
            for (int k = 1; k < n; k++) ph[k] -= 2 * M_PI * round((ph[k] - ph[k - 1]) / (2 * M_PI));
            for (int k = 0; k < n; k++) {
                L[k] = csub(L[k], cscale(0.5, CPX(log(cabs(d1[k])), ph[k])));
                cdouble t2 = cdiv(CPX(S->s02, 0), d1[k]), v = cscale(2, k1[k]);
                cdouble p1 = cscale(0.5, csub(cadd(cmul(cmul(mqk[k], mqk[k]), t2), v), t2)), p2 = cscale(-0.25, cmul(t2, v));
                cdouble dsc = csqrt(csub(cmul(p1, p1), cscale(4, p2)));
                k1[k] = cscale(0.5, cadd(p1, dsc)); k2[k] = cscale(0.5, csub(p1, dsc));
                if (creal(k2[k]) > creal(k1[k])) { cdouble t = k1[k]; k1[k] = k2[k]; k2[k] = t; }
            }
            free(ph); free(d1);
        } else { for (int k = 0; k < n; k++) { L[k] = cadd(L[k], cscale(S->g0 * S->g0, aq[k])); mq[k] = mqk[k]; } }
        free(aq); free(mqk);
    }
    free(Z); free(lz); free(Gs);
}
static cdouble gden(const sctx_t *S, double aa, cdouble k1, cdouble k2, cdouble mq)
{
    if (S->dkfixed) {
        if (S->g0 != 0) { cdouble v = cscale(2, k1), m = cscale(S->g0, mq); double r2 = 2 * aa + S->g0 * S->g0; if (!(r2 > 0)) return ZERO; double r = sqrt(r2);
            cdouble rm = csub(CPX(r, 0), m), rp = cadd(CPX(r, 0), m);
            return cdiv(cadd(cexp(cneg(cdiv(cmul(rm, rm), cscale(2, v)))), cexp(cneg(cdiv(cmul(rp, rp), cscale(2, v))))), cscale(r, csqrt(cscale(2 * M_PI, v)))); }
        if (!(aa > 0)) return ZERO;
        return cdiv(cexp(cneg(cdiv(CPX(aa, 0), cscale(2, k1)))), csqrt(cscale(2 * M_PI * aa, k1)));
    }
    /* |aa| at least rfloor(aa, c) from the logarithmic singularity at aa = 0: next to a zero of a(b), and at the zero itself when
     * two zeros coincide (c0 > 0, x = -sqrt(2 c0)), the computed a(b) can be exactly 0, where K_0 is infinite (gden in DFExact.jl) */
    aa = rfloor(aa, S->c);
    cdouble s2 = cdiv(ONE, cscale(2, k1)), s1 = cdiv(ONE, cscale(2, k2)), al = cscale(0.5, cadd(s1, s2)), be = cscale(0.5, csub(s2, s1));
    cdouble z = cscale(fabs(aa), be);
    return cdiv(cmul(cexp(csub(cneg(cscale(aa, al)), z)), dfl_k0scaled(z)), cscale(2 * M_PI, cmul(csqrt(k1), csqrt(cneg(k2)))));
}
cdouble dfl_Pkap(cdouble k1, cdouble k2, double a, bool pdfm, int Nl, bool nc, cdouble m1s, cdouble m2s)
{
    const double *wl; const double *zl = dfl_rule(Nl, &wl);
    cdouble s1 = cscale(2, k1), s2 = cscale(-2, k2), m1 = csqrt(m1s), m2 = csqrt(m2s);
    double lm = sqrt(2 * 41.4 / fmax(creal(cdiv(ONE, s2)), 1e-12)) + (nc ? cabs(m2) : 0);
    cdouble P = ZERO, sq2s1 = csqrt(cscale(2, s1)), sq2ps1 = csqrt(cscale(2 * M_PI, s1)), sq2ps2 = csqrt(cscale(2 * M_PI, s2));
#define F1(u) (nc ? cscale(0.5, cadd(dfl_erf(cdiv(csub((u), m1), sq2s1)), dfl_erf(cdiv(cadd((u), m1), sq2s1)))) : dfl_erf(cdiv((u), sq2s1)))
#define f1(u) (nc ? cdiv(cadd(cexp(cneg(cdiv(cmul(csub((u), m1), csub((u), m1)), cscale(2, s1)))), cexp(cneg(cdiv(cmul(cadd((u), m1), cadd((u), m1)), cscale(2, s1))))), sq2ps1) \
                  : cscale(2 / sqrt(M_PI), cdiv(cexp(cneg(cdiv(cmul((u), (u)), cscale(2, s1)))), sq2s1)))
#define f2(l) (nc ? cdiv(cadd(cexp(cneg(cdiv(cmul(csub((l), m2), csub((l), m2)), cscale(2, s2)))), cexp(cneg(cdiv(cmul(cadd((l), m2), cadd((l), m2)), cscale(2, s2))))), cscale(2, sq2ps2)) \
                  : cdiv(cexp(cneg(cdiv(cmul((l), (l)), cscale(2, s2)))), sq2ps2))
    if (a >= 0) {
        double al = fmax(sqrt(2 * a), 1e-8 * lm), tm = asinh(lm / al);
        for (int k = 0; k < Nl; k++) {
            double T = zl[k] * tm, Lg = al * sinh(T), dL = al * cosh(T) * wl[k] * tm, u = sqrt(2 * a + Lg * Lg);
            cdouble Lc = CPX(Lg, 0), uc = CPX(u, 0);
            cdouble term = pdfm ? cdiv(cmul(f2(Lc), f1(uc)), uc) : cmul(f2(Lc), F1(uc));
            P = cadd(P, cscale(dL, term));
        }
    } else {
        double l0 = sqrt(-2 * a), um = sqrt(fmax(lm * lm - l0 * l0, 0) + 1e-30);
        for (int k = 0; k < Nl; k++) {
            double U = zl[k] * um, WU = wl[k] * um, Lg = sqrt(U * U + l0 * l0);
            cdouble Lc = CPX(Lg, 0), Uc = CPX(U, 0);
            cdouble term = pdfm ? cdiv(cmul(f2(Lc), f1(Uc)), Lc) : cscale(U / Lg, cmul(f2(Lc), F1(Uc)));
            P = cadd(P, cscale(WU, term));
        }
    }
#undef F1
#undef f1
#undef f2
    return cscale(2, P);
}
static cdouble Pk(const sctx_t *S, cdouble k1, cdouble k2, double a, cdouble mq, cdouble m2)
{
    if (S->nzR && !S->stat) return dfl_Pkap(k1, k2, a, S->pdfm, S->Nl, true, mq, m2);
    if (S->dkfixed) {
        cdouble m = cscale(S->g0, mq); double r = sqrt(2 * a + S->g0 * S->g0); cdouble sg = csqrt(cscale(4, k1));
        cdouble rm = cdiv(csub(CPX(r, 0), m), sg), rp = cdiv(cadd(CPX(r, 0), m), sg);
        if (S->pdfm) return cscale(1.0 / (2 * r), cdiv(cadd(cexp(cneg(cmul(rm, rm))), cexp(cneg(cmul(rp, rp)))), cscale(sqrt(M_PI) / 2, sg)));
        return cscale(0.5, cadd(dfl_erf(rm), dfl_erf(rp)));
    }
    return dfl_Pkap(k1, k2, a, S->pdfm, S->Nl, false, ZERO, ZERO);
}
/* the path integral of the reflected boundary r = -a on the nodes of the path of a (the density form; dfl_adfmax) */
static void hb_reflect(const sctx_t *S, int NN, double bs, double as, const cdouble *Sn, const cdouble *W, const cdouble *L, const cdouble *k1,
                       const cdouble *k2, const cdouble *mq, double *Iout, double *Ihout)
{
    double Ij = 0, Ihj = 0;
    for (int k = 0; k <= NN; k++) {
        cdouble ex = cexp(csub(L[k], cscale(bs, Sn[k]))), g = gden(S, -as, k1[k], k2[k], mq[k]);
        cdouble G = S->pdfm ? cscale(sqrt(bs), cmul(ex, g)) : cdiv(cmul(ex, g), Sn[k]);
        Ij += creal(cmul(G, W[k])); if (k % 2 == 0) Ihj += creal(cmul(G, cscale(2, W[k])));
    }
    *Iout = Ij; *Ihout = Ihj;
}
/* H(b) for the b's of one piece */
typedef struct { sctx_t *S; const double *bv; int nb; double *H; } hbjob;
static void Hb(sctx_t *S, int nb, const double *bv, double *H)
{
    double *a = malloc(nb * sizeof(double)); int *act = malloc(nb * sizeof(int)); int na = 0;
    for (int j = 0; j < nb; j++) { a[j] = 0.5 + S->xx * sqrt(bv[j]) + S->c0 * bv[j]; H[j] = 0; if (!S->dkfixed || a[j] + S->g0 * S->g0 / 2 > 0) act[na++] = j; }
    if (na == 0) { free(a); free(act); return; }
    double *s = malloc(na * sizeof(double)), *a1 = malloc(na * sizeof(double)), *sbd = malloc(na * sizeof(double));
    for (int jj = 0; jj < na; jj++) {
        double bs = bv[act[jj]];
        double sj = jj > 0 ? s[jj - 1] : 0.0, slo = -INFINITY, shi = S->smax, sb = S->smax, g2 = 1.0; bool ok0, okt;
        for (int it = 0; it < 200; it++) {
            double L0 = LB(S, sj, &ok0); bool bad = !ok0 || sj >= S->smax;
            if (bad) { shi = sj; sb = fmin(sb, sj); }
            double e = fmin(1e-4 * fmax(1, fabs(sj)), 0.1 * (sb - sj)); bool okp = false; double Lp = L0, Lm = L0;
            for (int k = 0; k < 30 && !okp && !bad; k++) { double Lpt = LB(S, sj + e, &okt); bool okm; double Lmt = LB(S, sj - e, &okm); okt = okt && sj + e < S->smax; Lp = Lpt; Lm = Lmt; okp = okt; if (!okt) { e /= 10; sb = fmin(sb, sj + 10 * e); } }
            double g1 = (Lp - Lm) / (2 * e) - bs, g2a = (Lp - 2 * L0 + Lm) / (e * e);
            bool gd = !bad && okp;
            if (gd && g1 < 0) slo = fmax(slo, sj);
            if (gd && g1 >= 0) shi = fmin(shi, sj);
            if (gd && g2a > 0) g2 = g2a;
            double sn = (gd && g2a > 0) ? sj - g1 / g2a : NAN, A0 = slo, B0 = shi;
            if (!(sn > A0 && sn < B0)) { if (isfinite(A0) && isfinite(B0)) sn = (A0 + B0) / 2; else if (!isfinite(A0)) sn = B0 - fmax(1, 2 * fabs(B0)); else sn = A0 + fmax(1, fabs(A0)); }
            bool conv = gd && (fabs(g1) < 1e-9 * bs || (B0 - A0) < 1e-12 * (1 + fabs(sj)));
            if (conv) break;
            sj = sn;
        }
        if (S->nzR && !S->stat && isfinite(S->smax0)) sj = fmin(sj, S->smax0 - 0.02 * fmax(1, fabs(S->smax0)));
        for (int pass = 0; pass < 2; pass++) {
            double a1j = 1 / sqrt(g2), e = fmin(0.05 * a1j, 0.3 * (sb - sj)); bool okp, okm;
            double L0 = LB(S, sj, &ok0), Lp = LB(S, sj + e, &okp), Lm = LB(S, sj - e, &okm); okp = okp && sj + e < S->smax;
            double g2n = (Lp - 2 * L0 + Lm) / (e * e);
            if (okp && okm && g2n > 0 && isfinite(g2n)) g2 = g2n;
        }
        double a1j = 1 / sqrt(g2);
        if (S->useg && !S->pdfm && fabs(sj) < a1j) sj = (sj >= 0 && a1j < 0.5 * sb) ? a1j : -a1j;
        s[jj] = sj; a1[jj] = a1j; sbd[jj] = sb;
    }
    double *Iv = malloc(na * sizeof(double)), *Ih = malloc(na * sizeof(double));
    double *Iv2 = S->two ? malloc(na * sizeof(double)) : NULL, *Ih2 = S->two ? malloc(na * sizeof(double)) : NULL;
    for (int jj = 0; jj < na; jj++) {
        int NN = S->Nv; double bs = bv[act[jj]], as = a[act[jj]];
        double mu = fmax(a1[jj] / 2, S->mumin), sth = a1[jj] / (2 * mu), thm = sqrt(S->tdec / (bs * mu));
        for (int rf = 0; rf <= 2; rf++) {
            double ht = asinh(thm / sth) / NN;
            cdouble *Sn = malloc((NN + 1) * sizeof(cdouble)), *W = malloc((NN + 1) * sizeof(cdouble)), *L = malloc((NN + 1) * sizeof(cdouble)), *k1 = malloc((NN + 1) * sizeof(cdouble)), *k2 = malloc((NN + 1) * sizeof(cdouble)), *mq = malloc((NN + 1) * sizeof(cdouble)), *m2 = malloc((NN + 1) * sizeof(cdouble)), *G = malloc((NN + 1) * sizeof(cdouble));
            for (int k = 0; k <= NN; k++) { double tau = k * ht, TH = sth * sinh(tau), w = sth * cosh(tau) * ht; if (k == 0) w /= 2;
                Sn[k] = CPX(s[jj] + mu * TH * TH, -2 * mu * TH); W[k] = cscale(2 * mu / M_PI * w, CPX(1, TH)); }
            nodepar(S, NN + 1, Sn, L, k1, k2, mq, m2);
            for (int k = 0; k <= NN; k++) { if (creal(k1[k]) <= 0) S->nbad++; if ((S->model != 1 || S->st) && creal(k2[k]) >= 0) S->nbad++; }
            for (int k = 0; k <= NN; k++) {
                cdouble ex = cexp(csub(L[k], cscale(bs, Sn[k])));
                if (S->useg) { cdouble g = gden(S, as, k1[k], k2[k], mq[k]); G[k] = S->pdfm ? cscale(sqrt(bs), cmul(ex, g)) : cdiv(cmul(ex, g), Sn[k]); }
                else { cdouble P = Pk(S, k1[k], k2[k], as, mq[k], m2[k]); if (S->pdfm) P = cscale(sqrt(bs), P); G[k] = cmul(ex, P); }
            }
            double Ij = 0, Ihj = 0; for (int k = 0; k <= NN; k++) { Ij += creal(cmul(G[k], W[k])); if (k % 2 == 0) Ihj += creal(cmul(G[k], cscale(2, W[k]))); }
            Iv[jj] = Ij; Ih[jj] = Ihj;
            if (S->two) hb_reflect(S, NN, bs, as, Sn, W, L, k1, k2, mq, &Iv2[jj], &Ih2[jj]);
            free(Sn); free(W); free(L); free(k1); free(k2); free(mq); free(m2); free(G);
            /* the refinement decision needs the scale of the piece: done after the first pass over all columns */
            if (rf == 0) break;
        }
    }
    double sc = 2.2250738585072014e-308; for (int jj = 0; jj < na; jj++) sc = fmax(sc, fabs(Iv[jj]));
    if (S->two) for (int jj = 0; jj < na; jj++) sc = fmax(sc, fabs(Iv2[jj]));
    double tabs = 1e-11 / S->brng;
#define HBBAD(jj) ((fabs(Iv[jj] - Ih[jj]) > 1e-9 * sc && fabs(Iv[jj] - Ih[jj]) > tabs) || (S->two && fabs(Iv2[jj] - Ih2[jj]) > 1e-9 * sc && fabs(Iv2[jj] - Ih2[jj]) > tabs))
    for (int rf = 1; rf <= (S->verified ? 2 : 0); rf++) {
        bool any = false;
        for (int jj = 0; jj < na; jj++) {
            if (!HBBAD(jj)) continue;
            any = true; int NN = S->Nv << rf; double bs = bv[act[jj]], as = a[act[jj]];
            double mu = fmax(a1[jj] / 2, S->mumin), sth = a1[jj] / (2 * mu), thm = sqrt(S->tdec / (bs * mu)), ht = asinh(thm / sth) / NN;
            cdouble *Sn = malloc((NN + 1) * sizeof(cdouble)), *W = malloc((NN + 1) * sizeof(cdouble)), *L = malloc((NN + 1) * sizeof(cdouble)), *k1 = malloc((NN + 1) * sizeof(cdouble)), *k2 = malloc((NN + 1) * sizeof(cdouble)), *mq = malloc((NN + 1) * sizeof(cdouble)), *m2 = malloc((NN + 1) * sizeof(cdouble)), *G = malloc((NN + 1) * sizeof(cdouble));
            for (int k = 0; k <= NN; k++) { double tau = k * ht, TH = sth * sinh(tau), w = sth * cosh(tau) * ht; if (k == 0) w /= 2; Sn[k] = CPX(s[jj] + mu * TH * TH, -2 * mu * TH); W[k] = cscale(2 * mu / M_PI * w, CPX(1, TH)); }
            nodepar(S, NN + 1, Sn, L, k1, k2, mq, m2);
            for (int k = 0; k <= NN; k++) { cdouble ex = cexp(csub(L[k], cscale(bs, Sn[k])));
                if (S->useg) { cdouble g = gden(S, as, k1[k], k2[k], mq[k]); G[k] = S->pdfm ? cscale(sqrt(bs), cmul(ex, g)) : cdiv(cmul(ex, g), Sn[k]); }
                else { cdouble P = Pk(S, k1[k], k2[k], as, mq[k], m2[k]); if (S->pdfm) P = cscale(sqrt(bs), P); G[k] = cmul(ex, P); } }
            double Ij = 0, Ihj = 0; for (int k = 0; k <= NN; k++) { Ij += creal(cmul(G[k], W[k])); if (k % 2 == 0) Ihj += creal(cmul(G[k], cscale(2, W[k]))); }
            Iv[jj] = Ij; Ih[jj] = Ihj;
            if (S->two) hb_reflect(S, NN, bs, as, Sn, W, L, k1, k2, mq, &Iv2[jj], &Ih2[jj]);
            free(Sn); free(W); free(L); free(k1); free(k2); free(mq); free(m2); free(G);
        }
        if (!any) break;
    }
    for (int jj = 0; jj < na; jj++) if (HBBAD(jj)) S->nunres++;
#undef HBBAD
    if (S->useg && !S->pdfm) for (int jj = 0; jj < na; jj++) { if (s[jj] < 0) Iv[jj] += creal(gden(S, a[act[jj]], S->k10, S->k20, S->mq0)); Iv[jj] *= S->xx / (2 * sqrt(bv[act[jj]])) + S->c0; }
    if (S->two && S->useg && !S->pdfm) for (int jj = 0; jj < na; jj++) { if (s[jj] < 0) Iv2[jj] += creal(gden(S, -a[act[jj]], S->k10, S->k20, S->mq0)); Iv2[jj] *= S->xx / (2 * sqrt(bv[act[jj]])) + S->c0; }
    for (int jj = 0; jj < na; jj++) H[act[jj]] = S->two ? Iv[jj] + Iv2[jj] : Iv[jj];
    free(a); free(act); free(s); free(a1); free(sbd); free(Iv); free(Ih); free(Iv2); free(Ih2);
}
typedef struct { sctx_t *S; const double *B; double *H; int M; } startjob;
static void start_block(int p, void *v) { startjob *j = v; sctx_t Sc = *j->S; Sc.nbad = 0; Sc.nunres = 0; Hb(&Sc, j->M, j->B + p * j->M, j->H + p * j->M);
    /* (the counters of the copies are summed by the caller through the shared struct: tolerated race on diagnostics) */
    j->S->nbad += Sc.nbad; j->S->nunres += Sc.nunres; }
static void start_evalH(int n, const double *B, const int *IX, double *H, double *E, void *v)
{
    startjob *j = v; int nblk = n / j->M; j->B = B; j->H = H; (void) IX;
    dfl_parfor(nblk, start_block, j);
    for (int i = 0; i < n; i++) E[i] = 0;
}
static int double_start_x(int kind, int nx, const double *x, int model, double c, double c0, double g0, bool stat, const dfl_double_opts *o, double *F, dfl_double_info *info, bool two);
static int double_start(int kind, int nx, const double *x, int model, double c, double c0, double g0, bool stat, const dfl_double_opts *o, double *F, dfl_double_info *info)
{
    return double_start_x(kind, nx, x, model, c, c0, g0, stat, o, F, info, false);
}
/* two: the region |R| <= a(b) of the ADFmax statistic (c0 = 0, a stationary start), ending at the zero v0 = 1/(4x^2) of a(b)
 * when x < 0 (stat_two of julia/DFExact.jl) */
static int double_start_x(int kind, int nx, const double *x, int model, double c, double c0, double g0, bool stat, const dfl_double_opts *o, double *F, dfl_double_info *info, bool two)
{
    sctx_t S = {0}; S.two = two; S.verified = two || o->verified;           /* (the ADFmax statistic as before) */
    S.model = model; S.kind = kind; S.M = o->M > 0 ? o->M : 64; S.Nv = o->Nv; S.Nl = o->Nl; S.c = c; S.c0 = c0; S.g0 = stat ? 0 : g0; S.stat = stat; S.pdfm = kind == 1;
    S.s02 = stat ? -1 / (2 * c) : 0; S.st = stat && model == 1; S.nz = S.st || (S.g0 != 0 && model == 1); S.nzR = model != 1 && (stat || S.g0 != 0);
    S.mumin = o->mumin; S.tdec = o->tdec;
    S.smax0 = dfl_tstar(model, c, false);
    bool ok;
    double hh = 1e-5, EB = (LB(&S, hh, &ok) - LB(&S, -hh, &ok)) / (2 * hh);
    S.smax = S.smax0;
    LB(&S, S.smax * (1 - 1e-9), &ok);
    if (!ok) { double lo = 0, hi = S.smax; for (int it = 0; it < 80; it++) { double md = (lo + hi) / 2; LB(&S, md, &ok); if (ok) lo = md; else hi = md; } S.smax = lo; }
    S.useg = !(S.nzR && !stat) && !(model == 1 && !S.st);
    S.dkfixed = model == 1 && !S.st;
    if (S.useg && !S.pdfm) { cdouble s0 = ZERO, L, k1, k2, mq, m2; nodepar(&S, 1, &s0, &L, &k1, &k2, &mq, &m2); S.k10 = k1; S.k20 = k2; S.mq0 = mq; }
    const orules_t *OR = outer_rules(S.M);
    /* the lower end in log b: E V e^{-blo}, with a stationary start at most bmin (as in case DK of the zero start): in case SK,
     * E V ~ 1/(2|c|) as c -> 0-, while V keeps the mass of the start's chi-square law near 0, and E V e^{-8} left out 1e-4 of
     * F at c = -0.005 and 1e-3 at c = -0.001 without a flag (cases SU, ST, whose E V stays below 1, are unchanged;
     * double_start of DFExact.jl) */
    double lo = stat ? fmin(log(o->bmin), log(EB) - o->blo) : log(EB) - o->blo, hi = log(EB) + o->bhi, blo_ = exp(lo); S.brng = exp(hi) - blo_;
    double q0 = S.dkfixed ? (1 + S.g0 * S.g0) / 2 : 0.5;
    for (int i = 0; i < nx; i++) {
        double xx = x[i]; S.xx = xx; S.nbad = 0; S.nunres = 0;
        double sr[2]; int nsr = 0;
        if (c0 == 0) { sr[0] = -q0 / xx; nsr = 1; } else { double d = xx * xx - 4 * c0 * q0; if (d >= 0) { sr[0] = (-xx - sqrt(d)) / (2 * c0); sr[1] = (-xx + sqrt(d)) / (2 * c0); nsr = 2; } }
        double bk[2]; int nbk = 0; for (int k = 0; k < nsr; k++) if (sr[k] > 0) { double l = log(sr[k] * sr[k]); if (lo < l && l < hi) bk[nbk++] = l; }
        if (nbk == 2 && bk[0] > bk[1]) { double t = bk[0]; bk[0] = bk[1]; bk[1] = t; }
        if (nbk == 2 && bk[0] == bk[1]) nbk = 1;                           /* a double root once */
        double br[4]; int nbr = 0; br[nbr++] = lo; for (int k = 0; k < nbk; k++) br[nbr++] = bk[k]; br[nbr++] = hi;
        pieces_t P = {0};
        double top = INFINITY;                                              /* the two-sided region ends at v0 (x < 0) */
        if (two && xx < 0) { double sv = -q0 / xx; top = log(sv * sv); if (!(top > lo)) { F[i] = 0; info[i].err = 0; info[i].nbad = 0; info[i].nunres = 0; info[i].nrefine = 0; info[i].status = 0; continue; } }
        for (int jj = 1; jj < nbr; jj++) { int np0 = (int) fmax(1, ceil((br[jj] - br[jj - 1]) / o->plen));
            if (br[jj - 1] >= top) break;
            for (int q = 0; q < np0; q++) { double a = br[jj - 1] + (br[jj] - br[jj - 1]) * q / np0, b = br[jj - 1] + (br[jj] - br[jj - 1]) * (q + 1) / np0;
                bool el = false, er = false; for (int k = 0; k < nbk; k++) { if (S.useg && fabs(a - bk[k]) < 1e-12) el = true; if (S.useg && fabs(b - bk[k]) < 1e-12) er = true; }
                push_piece(&P, a, b - a, el, er, 0); } }
        startjob job = {&S, NULL, NULL, S.M};
        double Fi, erri; int nuni, nrefi; adapt_out ao = {&Fi, &erri, &nuni, &nrefi, 0};
        adaptb(P.n, P.PA, P.PL, P.PEL, P.PER, P.PX, 1, OR->z1, OR->w1, OR->z2, OR->w2, OR, start_evalH, &job, o->otol, o->maxlev, &ao);
        F[i] = Fi;
        if (S.useg && !S.pdfm) { double alo = 0.5 + xx * sqrt(blo_) + c0 * blo_; F[i] += creal(Pk(&S, S.k10, S.k20, alo, S.mq0, ZERO));
            if (two) F[i] -= creal(Pk(&S, S.k10, S.k20, -alo, S.mq0, ZERO)); }
        info[i].err = erri; info[i].nbad = S.nbad; info[i].nunres = S.nunres; info[i].nrefine = nrefi; info[i].status = (S.nbad > 0 ? 1 : 0) + 2 * (S.nunres > 0 || erri > o->errtol || !isfinite(F[i]));   /* (a value that is not finite counts as flagged) */
        free(P.PA); free(P.PL); free(P.PEL); free(P.PER); free(P.PX);
    }
    return 0;
}

/* ---- the applications (julia/DFExact.jl): the ADFmax statistic and a general boundary ---- */
/* the steps of the outer integrand of the boundary sg (1/2 + x sqrt(b)) (steps_bnd of DFExact.jl; STEPROOTS with c0 = 0 and
 * the reflected boundary) */
static int steproots_sg(const stepgrid_t *G, double sg, double x, double wmax, double *ls, double *ws)
{
    int nr = 0; double ph[61]; bool ok[61];
    for (int j = 0; j < 61; j++) { ph[j] = sg * (0.5 + x * exp(G->lb[j] / 2)) - G->m[j]; ok[j] = isfinite(ph[j]) && isfinite(G->sd[j]); }
    for (int j = 0; j < 60; j++) {
        if (!(ok[j] && ok[j + 1] && (ph[j] < 0) != (ph[j + 1] < 0))) continue;
        double la = G->lb[j], lc = G->lb[j + 1], fa = ph[j], fc = ph[j + 1], sa = G->sd[j], sdc = G->sd[j + 1]; int side = 0;
        for (int it = 0; it < 6; it++) {
            double l = la - fa * (lc - la) / (fc - fa), mn, sn; condmom(G->model, G->c, G->smx, l, &mn, &sn);
            double fn = sg * (0.5 + x * exp(l / 2)) - mn; if (!isfinite(fn)) break;
            if ((fn < 0) == (fa < 0)) { la = l; fa = fn; sa = sn; if (side == -1) fc /= 2; side = -1; }
            else { lc = l; fc = fn; sdc = sn; if (side == 1) fa /= 2; side = 1; }
        }
        double t = fa / (fa - fc), l = la + t * (lc - la), sd = sa + t * (sdc - sa), m1, s1, m2, s2;
        condmom(G->model, G->c, G->smx, l - 1e-3, &m1, &s1); condmom(G->model, G->c, G->smx, l + 1e-3, &m2, &s2);
        double sl = ((sg * (0.5 + x * exp((l + 1e-3) / 2)) - m2) - (sg * (0.5 + x * exp((l - 1e-3) / 2)) - m1)) / 2e-3;
        double wd = sd / fabs(sl);
        if (wd < wmax && nr < 8) { ls[nr] = l; ws[nr] = fmax(wd, 1e-6); nr++; }
    }
    return nr;
}
/* the pieces in log b over [lo, hi] (pieces_two of adfmax.jl): the zero lv0 of a(b) an edge, pieces of at most plen graded
 * toward the steps, with trunc the pieces beyond lv0 dropped */
static void pieces_two(double lo, double hi, double lv0, bool trunc, double plen, int ns, const double *ls, const double *ws, pieces_t *P)
{
    bool edge = lo < lv0 && lv0 < hi; double pts[600]; int np = 0;
    pts[np++] = lo; pts[np++] = hi; if (edge) pts[np++] = lv0;
    for (int g = 0; g < ns; g++) { if (np < 590) pts[np++] = ls[g]; int ne = (int) ceil(log2(plen / (2 * ws[g])));
        for (int e = 0; e <= ne; e++) { double d = 2 * ws[g] * pow(2, e); if (np < 590) { pts[np++] = ls[g] - d; pts[np++] = ls[g] + d; } } }
    double top = (trunc && edge) ? lv0 : hi, q[600]; int nq = 0;
    for (int i = 0; i < np; i++) { if (!(pts[i] >= lo && pts[i] <= top)) continue; if (edge && fabs(pts[i] - lv0) < 1e-9 && pts[i] != lv0) continue; q[nq++] = pts[i]; }
    qsort(q, nq, sizeof(double), cmp_double);
    int nu = 0; for (int i = 0; i < nq; i++) if (i == 0 || q[i] != q[nu - 1]) q[nu++] = q[i];
    for (int p = 0; p < nu - 1; p++) {
        int npp = (int) fmax(1, ceil((q[p + 1] - q[p]) / plen));
        for (int i = 0; i < npp; i++) {
            double a = q[p] + (q[p + 1] - q[p]) * i / npp, b = q[p] + (q[p + 1] - q[p]) * (i + 1) / npp;
            push_piece(P, a, b - a, i == 0 && edge && q[p] == lv0, i == npp - 1 && edge && q[p + 1] == lv0, 0);
        }
    }
}
/* Pk0 for a < 0 with u = sqrt(2a + l^2) (the integrand of PK0 has a square-root edge at l0 = sqrt(-2a), where its panels lose
 * 1e-6; quadgk in DFExact.jl): 2 int_0^U phi(l; s2) erf(u/sqrt(2 s1)) u/l du, l = sqrt(u^2 + l0^2), U = sqrt((l0 + 40 sd)^2 - l0^2) */
static double Pk0_neg(double k1, double k2, double a)
{
    double s1 = 2 * k1, s2 = -2 * k2, l0 = sqrt(-2 * a), sd = sqrt(s2), L = l0 + 40 * sd, U = sqrt(L * L - l0 * l0), P = 0;
    const double *gw; const double *u = dfl_rule(32, &gw); int npan = 40;
    for (int p = 0; p < npan; p++) {
        double A = U * p / npan, B = U * (p + 1) / npan;
        for (int k = 0; k < 32; k++) { double uu = A + (B - A) * u[k], l = sqrt(uu * uu + l0 * l0); P += (B - A) * gw[k] * 2 * exp(-l * l / (2 * s2)) / sqrt(2 * M_PI * s2) * erf(uu / sqrt(2 * s1)) * uu / l; }
    }
    return P;
}
/* ADFmax, cases DU, DT with a zero start (zero_two of adfmax.jl): P(|R| <= a(b_lo)) + int a'(b) [G(a, b) + G(-a, b)] db, the
 * density int sqrt(b) [h(a, b) + h(-a, b)] db, over [b_min, E V e^6] up to v0 = 1/(4x^2) (x < 0) */
static int adfmax_zero(int kind, int nx, const double *x, int model, double c, const dfl_double_opts *o, double *F, dfl_double_info *info)
{
    bool pdfm = kind == 1; int M = o->M > 0 ? o->M : 64;
    double EB = EVdk(c), lo = log(o->bmin), hi = log(EB) + o->bhi;
    const orules_t *OR = outer_rules(M);
    const stepgrid_t *Gr = stepgrid(model, c, lo, hi);
    dfl_girs_t G0; dfl_girsanov(model, c, ZERO, &G0); cdouble S0 = csqrt(csub(cmul(G0.A1, G0.A1), cscale(4, cmul(G0.A0, G0.A2))));
    double k10 = creal(cdiv(cadd(G0.A1, S0), cscale(4, G0.A2))), k20 = creal(cdiv(G0.A0, cadd(G0.A1, S0)));
    for (int i = 0; i < nx; i++) {
        double xk = x[i], lv0 = xk < 0 ? 2 * log(-1 / (2 * xk)) : INFINITY;
        info[i].err = 0; info[i].nbad = 0; info[i].nunres = 0; info[i].nrefine = 0; info[i].status = 0;
        if (lv0 <= lo) { F[i] = 0; continue; }
        double ls[16], ws[16]; int ns = steproots_sg(Gr, 1, xk, o->stepw, ls, ws); ns += steproots_sg(Gr, -1, xk, o->stepw, ls + ns, ws + ns);
        pieces_t P = {0}; pieces_two(lo, hi, lv0, true, o->plen, ns, ls, ws, &P);
        zctx z = {model, M, o->Nv, c, 0, pdfm, &xk, NULL, NULL, NULL, true, NULL, true};
        double err; int nun, nref; adapt_out ao = {&F[i], &err, &nun, &nref, 0};
        adaptb(P.n, P.PA, P.PL, P.PEL, P.PER, P.PX, 1, OR->u, OR->gw, OR->z2, OR->w2, OR, zero_evalH, &z, o->otol, o->maxlev, &ao);
        if (!pdfm) { double alo = 0.5 + xk * sqrt(exp(lo)); F[i] += (alo >= 0 ? Pk0(k10, k20, alo) : Pk0_neg(k10, k20, alo)) - (alo <= 0 ? Pk0(k10, k20, -alo) : Pk0_neg(k10, k20, -alo)); }
        info[i].err = err; info[i].nunres = nun; info[i].nrefine = nref; info[i].status = 2 * (err > o->errtol || !isfinite(F[i]));
        free(P.PA); free(P.PL); free(P.PEL); free(P.PER); free(P.PX);
    }
    return 0;
}
int dfl_adfmax(int kind, int n, const double *x, int model, double c, bool stat, const dfl_double_opts *o, double *out, dfl_double_info *info, char *err, size_t errlen)
{
    if (kind != 1 && kind != 2) { snprintf(err, errlen, "kind must be 1 (density) or 2 (distribution function)"); return 1; }
    if (model != 2 && model != 3) { snprintf(err, errlen, "the ADFmax statistic: model must be 2 (constant) or 3 (constant and trend)"); return 1; }
    if (dfl_check_args(model, c, 0, 0, stat, err, errlen)) return 1;
    dfl_double_opts d; if (o == NULL) { dfl_double_defaults(&d); o = &d; }
    dfl_double_opts o96 = *o; if (o96.Nv <= 0) o96.Nv = 96; if (o96.Nl <= 0) o96.Nl = 96; o = &o96;   /* (no rule of one evaluation here) */
    int nf = 0; double *xf = malloc((n + 1) * sizeof(double)); int *id = malloc((n + 1) * sizeof(int));
    for (int k = 0; k < n; k++) {
        info[k].err = 0; info[k].status = 0; info[k].nbad = 0; info[k].nunres = 0; info[k].nrefine = 0;
        if (isfinite(x[k])) { xf[nf] = x[k]; id[nf++] = k; } else out[k] = isnan(x[k]) ? NAN : (x[k] > 0 && kind == 2) ? 1.0 : 0.0;
    }
    if (nf > 0) {
        double *Ff = malloc(nf * sizeof(double)); dfl_double_info *inf = malloc(nf * sizeof(dfl_double_info));
        if (!stat) adfmax_zero(kind, nf, xf, model, c, o, Ff, inf);
        else {
            if (!isfinite(dfl_tstar(model, c, false))) { free(Ff); free(inf); free(xf); free(id); snprintf(err, errlen, "no zero of D(0, t) was found (t_*), c = %g", c); return 1; }
            double_start_x(kind, nf, xf, model, c, 0, 0, true, o, Ff, inf, true);
        }
        for (int j = 0; j < nf; j++) { out[id[j]] = Ff[j]; info[id[j]] = inf[j]; }
        free(Ff); free(inf);
    }
    if (kind == 1) for (int k = 0; k < n; k++) if (out[k] < 0) out[k] = 0;
    free(xf); free(id);
    return 0;
}

/* a region {U <= g(V)}, case DK, zero start (Pregion of wap.jl): the boundary families of dfl_region */
typedef struct { int type; double p1, p2; } bnd_t;
static double lerfcx(double x) { return x < 0 ? x * x + log(erfc(x)) : log(creal(dfl_faddw(CPX(0, x)))); }   /* log erfcx */
static double wap_z(double v, double k, double w)
{
    double lr = log(k) + log(1 + w * w * v) / 2;
#define WF(z) (lerfcx((z) / sqrt(2.0)) - lr)
    if (WF(60.0) > 0) return 60.0;
    double lo = -60, hi = 60, flo = WF(lo);
    for (int it = 0; it < 200; it++) { double m = (lo + hi) / 2, fm = WF(m); if (fm * flo < 0) hi = m; else { lo = m; flo = fm; } if (hi - lo < 1e-11) break; }
#undef WF
    return (lo + hi) / 2;
}
static double bnd_g(double v, const bnd_t *B)
{
    switch (B->type) {
        case 1: return B->p2 + B->p1 * v;                                               /* line: b + a v */
        case 2: return B->p1 * sqrt(v) + B->p2 * v;                                     /* parabola: x sqrt(v) + c0 v */
        default: return wap_z(v, B->p1, B->p2) * sqrt(v + 1 / (B->p2 * B->p2));         /* the weighted-average-power test (k, omega) */
    }
}
static double bnd_a(double b, const void *ctx) { return 0.5 + bnd_g(b, ctx); }
/* P(V < v), case DK, zero start, by Gil-Pelaez (P_V of wap.jl) */
static double gp_V_grid(double c, double v, int N, double umax, double *tail)
{
    double vmax = asinh(umax), h = vmax / N, sum = 0, ph = 0, ph0 = 0;
    *tail = 0;
    for (int k = 0; k <= N; k++) {
        double u = k == 0 ? 1e-7 : sinh(k * h), wk = cosh(k * h) * h; if (k == 0) wk /= 2;
        cdouble T = CPX(0, u), mu = csqrt(csub(CPX(c * c, 0), cscale(2, T))), E = cexp(cscale(-2, mu));
        cdouble q = cabs(mu) < 1e-6 ? csub(CPX(2, 0), cscale(2, mu)) : cdiv(csub(ONE, E), mu);
        cdouble lm = csub(mu, CPX(log(2.0), 0)), br = csub(cadd(ONE, E), cscale(c, q));
        double a = carg(br); if (k == 0) ph = a; else ph += a - ph0 - 2 * M_PI * round((a - ph0) / (2 * M_PI)); ph0 = a;
        cdouble phi = cexp(csub(CPX(-c / 2, 0), cscale(0.5, cadd(lm, CPX(log(cabs(br)), ph)))));
        sum += cimag(cmul(phi, cexp(CPX(0, -u * v)))) / u * wk;
        if (k >= N - 100) *tail = fmax(*tail, cabs(phi));
    }
    return 0.5 - sum / M_PI;
}
/* P(V < v), on the grid of dfl_vgrid where the default one does not serve (c <= -50, c > 2), clamped to [0, 1]; NaN where
 * the grid would exceed its bound */
static double gp_V(double c, double v)
{
    double tail, P = gp_V_grid(c, v, 60000, 4000, &tail), umax; int N; char err[256];
    int r = dfl_vgrid(c, tail, &N, &umax, err, sizeof err);
    if (r < 0) return NAN;
    if (r > 0) P = gp_V_grid(c, v, N, umax, &tail);
    return fmin(1.0, fmax(0.0, P));
}
int dfl_region(int type, const double *par, double v0, int nbrk, const double *brks, double c, const dfl_double_opts *o, double *out, dfl_double_info *info, char *err, size_t errlen)
{
    if (type < 1 || type > 3) { snprintf(err, errlen, "the boundary type must be 1 (line), 2 (parabola) or 3 (weighted-average-power test)"); return 1; }
    if (dfl_check_args(1, c, 0, 0, false, err, errlen)) return 1;
    if (type == 3 && !(par[0] > 0 && par[1] > 0)) { snprintf(err, errlen, "the weighted-average-power boundary needs k > 0 and omega > 0"); return 1; }
    dfl_double_opts d; if (o == NULL) { dfl_double_defaults(&d); o = &d; }
    dfl_double_opts o96 = *o; if (o96.Nv <= 0) o96.Nv = 96; if (o96.Nl <= 0) o96.Nl = 96; o = &o96;   /* (no rule of one evaluation here) */
    bnd_t B = {type, par[0], par[1]};
    int M = o->M > 0 ? o->M : 32;
    double EB = EVdk(c), lo = fmin(log(o->bmin), log(EB) - o->blo), hi = log(EB) + o->bhi;
    if (v0 > 0) lo = fmax(lo, log(v0));
    /* the zeros of a(b) in log b: sign changes on 2000 points, then bisection to 1e-13 */
    int ng = 2000; double *va = malloc((ng + 1) * sizeof(double)), zs[64]; int nz = 0;
    for (int k = 0; k <= ng; k++) va[k] = bnd_a(exp(lo + (hi - lo) * k / ng), &B);
    for (int k = 0; k < ng && nz < 60; k++) {
        double l1 = lo + (hi - lo) * k / ng, l2 = lo + (hi - lo) * (k + 1) / ng;
        if (va[k] == 0) zs[nz++] = l1;
        else if (va[k] * va[k + 1] < 0) { double f1 = va[k];
            for (int it = 0; it < 200; it++) { double m = (l1 + l2) / 2, fm = bnd_a(exp(m), &B); if (fm * f1 < 0) l2 = m; else { l1 = m; f1 = fm; } if (l2 - l1 < 1e-13) break; }
            zs[nz++] = (l1 + l2) / 2; }
    }
    free(va);
    double *pts = malloc((nz + nbrk + 2) * sizeof(double)); int np = 0;
    pts[np++] = lo; pts[np++] = hi; for (int k = 0; k < nz; k++) pts[np++] = zs[k];
    for (int k = 0; k < nbrk; k++) { double l = log(brks[k]); if (lo < l && l < hi) pts[np++] = l; }
    qsort(pts, np, sizeof(double), cmp_double);
    int nu = 0; for (int i = 0; i < np; i++) if (i == 0 || pts[i] != pts[nu - 1]) pts[nu++] = pts[i];
    pieces_t P = {0};
    for (int p = 0; p < nu - 1; p++) {
        int npp = (int) fmax(1, ceil((pts[p + 1] - pts[p]) / o->plen));
        bool zl = false, zr = false; for (int k = 0; k < nz; k++) { if (pts[p] == zs[k]) zl = true; if (pts[p + 1] == zs[k]) zr = true; }
        for (int i = 0; i < npp; i++) {
            double a = pts[p] + (pts[p + 1] - pts[p]) * i / npp, b = pts[p] + (pts[p + 1] - pts[p]) * (i + 1) / npp;
            push_piece(&P, a, b - a, i == 0 && zl, i == npp - 1 && zr, 0);
        }
    }
    free(pts);
    const orules_t *OR = outer_rules(M);
    double x0 = 0; zctx z = {1, M, o->Nv, c, 0, false, &x0, NULL, bnd_a, &B, false, NULL, true};
    double erri; int nun, nref; adapt_out ao = {out, &erri, &nun, &nref, 0};
    adaptb(P.n, P.PA, P.PL, P.PEL, P.PER, P.PX, 1, OR->u, OR->gw, OR->z1, OR->w1, OR, zero_evalH, &z, o->otol, o->maxlev, &ao);
    if (v0 > 0) *out += gp_V(c, v0);
    info->err = erri; info->nbad = 0; info->nunres = nun; info->nrefine = nref; info->status = 2 * (erri > o->errtol || !isfinite(*out));
    free(P.PA); free(P.PL); free(P.PEL); free(P.PER); free(P.PX);
    return 0;
}

void dfl_double_defaults(dfl_double_opts *o)
{
    o->M = 0; o->Nv = 0; o->Nl = 0;                  /* (0: the rule of one evaluation, dfl_cond_rule; 96 with verified) */
    o->maxlev = 12; o->plen = 2.0; o->otol = 1e-12; o->stepw = 0.1; o->errtol = 1e-10; o->blo = 8.0; o->bhi = 6.0; o->bmin = 3e-3; o->mumin = 4.5; o->tdec = 41.5;
    o->verified = 0;
}
/* The rule of one evaluation of the conditional integral (verified = 0; cond_rule of DFExact.jl): the nodes Nv of the path of
 * each inner inversion and Nl of the integral over l, by regions measured in November 2026 against the rule with 192 nodes and
 * the outer tolerance 1e-14 on a dense grid of x (the README, "The rules of one evaluation").  Before, 96 nodes everywhere.
 * (Warm calls with 48 nodes: 0.60 times the time with a stationary start, 0.81 to 0.87 with a zero start; medians.)  48
 * nodes hold with a stationary start for c <= -2 and with a zero start in cases DU, DT for c >= -5 and in case DK for
 * c <= -13.5; 64 with a stationary start for c > -2 (48 lost 7e-8 of the peak of the density in case SK at c = -0.5); 96
 * elsewhere (with a zero start in cases DU, DT 48 nodes lost 1.8e-8 of the peak at c = -13.5 and up to 1e-3 far from the
 * unit root), with a fixed start (measured at one configuration only: 48 nodes lost 7e-7 in case DU at c = 2, c0 = -2) and
 * above c = 2.  The options Nv, Nl (> 0) override it. */
#define COND_RULE_CMAX 2.0
int dfl_cond_rule(int model, double c, double c0, double g0, bool stat)
{
    (void) c0;
    int n = stat ? (c <= -2 ? 48 : 64) : g0 != 0 ? 96 : model == 1 ? (c <= -13.5 ? 48 : 96) : (c >= -5 ? 48 : 96);
    if (c > COND_RULE_CMAX) n = 96;                  /* (beyond the range measured) */
    return n;
}
/* the options with Nv, Nl resolved (0: the rule of one evaluation, or 96 with verified; rule 0: 96) */
static dfl_double_opts nodes_of(const dfl_double_opts *o, int rule, int model, double c, double c0, double g0, bool stat)
{
    dfl_double_opts r = *o;
    int n = (rule && !o->verified) ? dfl_cond_rule(model, c, c0, g0, stat) : 96;
    if (r.Nv <= 0) r.Nv = n;
    if (r.Nl <= 0) r.Nl = n;
    return r;
}
int dfl_double(int kind, int n, const double *x, int model, double c, double c0, double g0, bool stat, const dfl_double_opts *o0, double *out, dfl_double_info *info, char *err, size_t errlen)
{
    dfl_double_opts d0; if (o0 == NULL) { dfl_double_defaults(&d0); o0 = &d0; }
    dfl_double_opts oo = nodes_of(o0, 1, model, c, c0, g0, stat); const dfl_double_opts *o = &oo;
    if (model == 4) { snprintf(err, errlen, "the GLS-detrended case (model 4) has the single integral only"); return 1; }
    if (stat && g0 != 0) { snprintf(err, errlen, "a fixed start g0 applies to the zero-start cases only"); return 1; }
    if (stat && !(c <= DFL_CSTAT)) { snprintf(err, errlen, "%s", DFL_STAT_MSG); return 1; }
    if (!stat && g0 == 0) return double_zero(kind, n, x, model, c, c0, o, out, info);
    /* the domain end of the zero-start M_V: an error, not NaN, where no zero of D(0, t) is found */
    if (!isfinite(dfl_tstar(model, c, false))) { snprintf(err, errlen, "no zero of D(0, t) was found (t_*), c = %g", c); return 1; }
    return double_start(kind, n, x, model, c, c0, g0, stat, o, out, info);
}
