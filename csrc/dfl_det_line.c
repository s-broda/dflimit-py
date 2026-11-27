/* dfl_det_line.c: the line integral with the explicit kernel for the designs of the general engine whose numerator has
 * rank one or two (a port of src/DFDet_line.jl of DFLimit.jl).  Where the determinant D(., t) has degree one (a positive
 * root) or two (one root of each sign) the inversion in s is explicit, and the distribution function and the density are
 * one integral over t along the line Re t = tau of the classical engine's inner integrals (dfl_single.c, dfl_gen.c).  A
 * provider stands for the classical model code (DFL_MODEL_PROV, the provider set on the calling thread): it supplies P(t),
 * s_1(t), s_2(t) and M_0(t) = (-s_1 s_2)^{-1/2} from the general engine's transform (det_mgfparts: log D_V(t) and the
 * reciprocal roots), the kernel (the rank), the line and its end, and the classical integrator does the rest unchanged:
 * its outer rules, its adaptive inner rules, the general kernel for c0 != 0 and its bounds on the paths.
 *
 * Coverage (what the papers state): no deterministic terms; the one-step statistic for continuous designs that do not
 * vanish at the origin and whose span is closed under differentiation (polynomial trends of any order, an intercept with
 * sines and cosines); the statistic that detrends first for continuous designs that do not vanish at the origin; a zero
 * or a stationary start (c <= -1e-4); least squares detrending.  Not covered: GLS detrending, a fixed start, designs with a
 * jump or a numerator of another rank, the statistic that detrends first with d(0) = 0, c below DFL_CMIN.
 *
 * Guards: the roots labelled by the sign of their real part (one root in each group), the argument of log P(t) at each
 * point from the pivots of the precision matrix (line_point, after Bao and Kan; until November 2026 continued from the real
 * point along the line), the end of the line extended until |P| is small (line_end), the cap of the first piece of the edge
 * segment and of the ray X < 0 by the root scale (dfl_k0first), and the bounds of the classical integrator's paths (DFL_BOUND_*: method auto
 * takes the inversion on paths at such a point, method line returns an error).  The classical integrator's fixed node layouts
 * (dfl_batch.c) take the inner integrals where they serve, as for the classical cases (the line moved off tau = 1 where the
 * positive root is
 * small at tau = 1, prov_get). */
#include "dfl_det_int.h"
#include <stdint.h>

/* ---------------------------------------------------------------------------------------------------------------- */
/* the provider */
typedef struct { cdouble t; cdouble v[3]; int used; } memo_ent;
struct dfl_prov {
    det_prep *P; det_model Mo; double c; int r, stat, start; double tstar, tau, wend;
    unsigned long long key;
    memo_ent *tab; size_t cap, n;     /* the memo of the transform per point of the line (grown to PROV_MEMO_CAP, then emptied) */
    dfl_lock_t lock;
    int refs;
};
static _Thread_local const dfl_prov *CUR;   /* the provider of DFL_MODEL_PROV on this thread */
static _Thread_local int PROVFAIL;          /* 1: the transform failed */
#define PROV_MEMO_CAP0 (1 << 11)     /* the memo starts with this many slots and doubles while half full, up to */
#define PROV_MEMO_CAP (1 << 16)      /* (65536 slots, 4.7 MB; then it is emptied: the values do not depend on it) */

static size_t hash_t(cdouble t, size_t cap)
{
    double a = creal(t), b = cimag(t); uint64_t x, y; memcpy(&x, &a, 8); memcpy(&y, &b, 8);
    uint64_t h = x * 0x9E3779B97F4A7C15ULL ^ (y + 0x632BE59BD9B4E019ULL + (x << 6) + (x >> 2));
    h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 29;
    return (size_t) h & (cap - 1);
}
static memo_ent *memo_find(const dfl_prov *pv, cdouble t)
{
    size_t i = hash_t(t, pv->cap);
    for (;;) {
        memo_ent *e = &pv->tab[i];
        if (!e->used || (creal(e->t) == creal(t) && cimag(e->t) == cimag(t))) return e;
        i = (i + 1) & (pv->cap - 1);
    }
}
/* (log D_V(t), mu_+(t), mu_-(t)) at one point: mu_+ the reciprocal of the positive root, mu_- of the negative one (0 for rank
 * one); the groups by the sign of Re mu, which on the line is that of Re s (the roots stay on their side of the gap).  The
 * argument of log D_V at the point from the pivots of the precision matrix (det_detlog_piv, Sigma as det_mgfparts left it:
 * one small elimination), after Bao and Kan; with the principal arguments of mu_+ and -mu_-, each in the right half-plane on
 * the line, log P(t)^{-2} has its continuous imaginary part at the point.  wk: 2 N^2 + N0^2 complex entries, piv N ints */
static void line_point(const dfl_prov *pv, det_work *W, cdouble t, cdouble *v, cdouble *wk, int *piv)
{
    cdouble th, ld, mu[2];
    if (det_mgfparts(W, pv->c, t, &th, &ld, mu, NULL)) PROVFAIL = 1;
    size_t N = W->N;
    cdouble lp = det_detlog_piv(W, pv->c, DZERO, t, th, wk, wk + N * N, wk + 2 * N * N, piv);
    v[0] = CPX(creal(ld), cimag(th) + cimag(lp));
    if (pv->r == 1) { v[1] = mu[0]; v[2] = DZERO; return; }
    if (creal(mu[0]) >= creal(mu[1])) { v[1] = mu[0]; v[2] = mu[1]; } else { v[1] = mu[1]; v[2] = mu[0]; }
}
typedef struct { const dfl_prov *pv; const cdouble *t; cdouble *out; const int *miss; volatile int fail; } raw_ctx;
static void raw_body(int j, void *v)
{
    raw_ctx *r = v; char buf[DET_WORK_STACK]; det_work W;
    if (det_work_init(&W, &r->pv->Mo, buf, sizeof buf)) { r->fail = 1; return; }
    size_t N = W.N, N0 = W.N0;
    cdouble wkb[3 * 64], *wk = 2 * N * N + N0 * N0 <= 3 * 64 ? wkb : malloc((2 * N * N + N0 * N0) * sizeof(cdouble));
    int pvb[16], *piv = N <= 16 ? pvb : malloc(N * sizeof(int));
    if (!wk || !piv) { r->fail = 1; det_work_free(&W); return; }
    int k = r->miss[j];
    PROVFAIL = 0;
    line_point(r->pv, &W, r->t[k], r->out + 3 * k, wk, piv);
    if (PROVFAIL) r->fail = 1;
    if (wk != wkb) free(wk);
    if (piv != pvb) free(piv);
    det_work_free(&W);
}
/* the transform at the points t (3 values each into out): from the memo, the missing points on the workers */
static void line_raw(dfl_prov *pv, int n, const cdouble *t, cdouble *out)
{
    int *miss = malloc((n + 1) * sizeof(int)), nm = 0;
    dfl_lock(&pv->lock);
    for (int k = 0; k < n; k++) {
        const memo_ent *e = memo_find(pv, t[k]);
        if (e->used) memcpy(out + 3 * k, e->v, 3 * sizeof(cdouble)); else miss[nm++] = k;
    }
    dfl_unlock(&pv->lock);
    if (nm > 0) {
        raw_ctx r = {pv, t, out, miss, 0};
        if (nm < 8 || dfl_nworkers() <= 1) for (int j = 0; j < nm; j++) raw_body(j, &r);
        else dfl_parfor(nm, raw_body, &r);
        if (r.fail) PROVFAIL = 1;
        dfl_lock(&pv->lock);
        while (pv->n + nm > pv->cap / 2 && pv->cap < PROV_MEMO_CAP) {       /* (grow: rehash into twice the slots) */
            memo_ent *old = pv->tab; size_t oc = pv->cap;
            memo_ent *nt = calloc(2 * oc, sizeof(memo_ent));
            if (!nt) break;
            pv->tab = nt; pv->cap = 2 * oc;
            for (size_t i = 0; i < oc; i++) if (old[i].used) *memo_find(pv, old[i].t) = old[i];
            free(old);
        }
        if (pv->n + nm > pv->cap / 2) { memset(pv->tab, 0, pv->cap * sizeof(memo_ent)); pv->n = 0; }
        /* (at most half the slots filled, so that memo_find always meets an empty slot: a call with more points keeps the first
         * ones.  Until November 27, 2026 all were entered; a call with more points than slots filled the table and memo_find
         * then searched forever: case DU at c = 10, x = 7860, engine general, 67680 points on the line, 65536 slots) */
        int nput = (size_t) nm > pv->cap / 2 ? (int) (pv->cap / 2) : nm;
        for (int j = 0; j < nput; j++) {
            int k = miss[j]; memo_ent *e = memo_find(pv, t[k]);
            if (!e->used) { e->used = 1; e->t = t[k]; pv->n++; }
            memcpy(e->v, out + 3 * k, 3 * sizeof(cdouble));
        }
        dfl_unlock(&pv->lock);
    }
    free(miss);
}
/* log P(t)^{-2} = log D_V + log mu_+ + log(-mu_-), with the argument of log D_V fixed at the point (line_point) */
static inline cdouble lineLr(int r, const cdouble *v) { return r == 1 ? cadd(v[0], clog(v[1])) : cadd(cadd(v[0], clog(v[1])), clog(cneg(v[2]))); }

/* P(t), s_1(t), s_2(t), M_0(t) on the points t of the line (every grid of the classical integrator starts at the real
 * point; otherwise it is put in front): s_1 the negative root and s_2 the positive one (rank two), s_1 the positive root
 * (rank one, s_2 and M_0 unused); M_0 = (-s_1)^{-1/2} s_2^{-1/2} with the principal branches; P from log P^{-2} with its
 * argument fixed at each point (line_point; until November 2026 continued from the real point along the line, within
 * 1.7e-14 of these values on the engine's grids, at the same cost) */
static void prov_tparts(dfl_prov *pv, int nt, const cdouble *t, cdouble *P, cdouble *s1, cdouble *s2, cdouble *M0)
{
    if (nt <= 0) return;
    if (cimag(t[0]) != 0) {
        cdouble *tt = malloc((nt + 1) * sizeof(cdouble)), *Q = malloc(4 * (size_t) (nt + 1) * sizeof(cdouble));
        tt[0] = CPX(creal(t[0]), 0); memcpy(tt + 1, t, nt * sizeof(cdouble));
        int m = nt + 1;
        prov_tparts(pv, m, tt, Q, Q + m, Q + 2 * m, Q + 3 * m);
        memcpy(P, Q + 1, nt * sizeof(cdouble)); memcpy(s1, Q + m + 1, nt * sizeof(cdouble));
        memcpy(s2, Q + 2 * m + 1, nt * sizeof(cdouble)); memcpy(M0, Q + 3 * m + 1, nt * sizeof(cdouble));
        free(tt); free(Q); return;
    }
    cdouble *raw = malloc(3 * (size_t) nt * sizeof(cdouble));
    line_raw(pv, nt, t, raw);
    for (int j = 0; j < nt; j++) {
        const cdouble *v = raw + 3 * j;
        cdouble Lr = lineLr(pv->r, v), L;
        if (j == 0) L = CPX(creal(Lr), 0);       /* (D_V mu_+ (-mu_-) is positive at the real point) */
        else L = Lr;                             /* (the argument fixed at the point, line_point) */
        P[j] = cexp(cscale(-0.5, L));
        if (pv->r == 1) { s1[j] = cinv(v[1]); s2[j] = DZERO; M0[j] = DZERO; }
        else { s1[j] = cinv(v[2]); s2[j] = cinv(v[1]); M0[j] = cmul(csq(v[1]), csq(cneg(v[2]))); }
    }
    free(raw);
}
/* the hooks of the classical integrator for DFL_MODEL_PROV (dflimit.h) */
void dfl_prov_tparts(int nt, const cdouble *t, cdouble *P, cdouble *s1, cdouble *s2, cdouble *M0) { prov_tparts((dfl_prov *) CUR, nt, t, P, s1, s2, M0); }
int dfl_prov_kern(void) { return CUR->r; }
double dfl_prov_wend(void) { return CUR->wend; }
double dfl_prov_tau(void) { return CUR->tau; }
/* the end of the outer integrals: dfl_wmaxc(c), set by the decay of the kernel's exponential factor alone, extended in steps of
 * 25% until |P(tau + iW)| <= 2.5e-8 (the largest value the classical engine leaves at its end in its cases), at most 64
 * dfl_wmaxc(c): the transform carries a power of t that grows with the number of trend functions (at c = -20 |P| at the
 * classical end is 9.5e-9 in case DU, 7e-7 for a trend of order four) */
static double line_end(dfl_prov *pv, double tau)
{
    double W0 = dfl_wmaxc(pv->c), W = W0;
    for (;;) {
        cdouble t = CPX(tau, W), v[3]; line_raw(pv, 1, &t, v);
        if (!(exp(-creal(lineLr(pv->r, v)) / 2) > 2.5e-8 && W < 64 * W0)) break;
        W *= 1.25;
    }
    return W;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* which designs, starts and options the line integral covers */
/* whether d jumps at a break by more than 1e-9 times its size (the Chebyshev re-expansions of fdesign meet at their
 * computational breaks to about 1e-12) */
static int line_jumps(const dfl_design *D)
{
    int K = D->K, d1 = D->deg + 1;
    double v0[64], v5[64], v1[64], sz = 1;
    if (K > 64) return 1;
    dfl_design_value(D, 0.0, v0); dfl_design_value(D, 0.5, v5); dfl_design_value(D, 1.0, v1);
    for (int i = 0; i < K; i++) sz = dmax(sz, dmax(fabs(v0[i]), dmax(fabs(v5[i]), fabs(v1[i]))));
    for (int j = 0; j + 1 < D->np; j++) {
        double lam = D->br[j + 1], oa = D->loc ? D->br[j] : 0, ob = D->loc ? D->br[j + 1] : 0;
        for (int i = 0; i < K; i++) {
            double a = evalpoly(lam - oa, D->C + (size_t) j * d1 * K + (size_t) i * d1, d1);
            double b = evalpoly(lam - ob, D->C + (size_t) (j + 1) * d1 * K + (size_t) i * d1, d1);
            if (fabs(b - a) > 1e-9 * sz) return 1;
        }
    }
    return 0;
}
/* whether the span of the design is closed under differentiation: the derivatives d' lie in span{d} (the residual
 * E - C' G^{-1} C of the Gram matrix of (d, d') at most 1e-9 of its size; closed_under_derivative) */
static int closed_under_derivative(const dfl_design *D)
{
    int K = D->K, d1 = D->deg + 1, np = D->np, n = 2 * K;
    if (K == 0) return 1;
    double *F = calloc((size_t) np * d1 * n, sizeof(double)), *G = malloc((size_t) n * n * sizeof(double));
    double *Gk = malloc((size_t) K * K * sizeof(double)), *X = malloc((size_t) K * K * sizeof(double));
    for (int j = 0; j < np; j++) for (int i = 0; i < K; i++) {
        const double *Cj = D->C + (size_t) j * d1 * K + (size_t) i * d1;
        double *f = F + (size_t) j * d1 * n + (size_t) i * d1, *fd = F + (size_t) j * d1 * n + (size_t) (K + i) * d1;
        for (int k = 0; k < d1; k++) { f[k] = Cj[k]; fd[k] = k + 1 < d1 ? (k + 1) * Cj[k + 1] : 0; }
    }
    det_gram(np, D->deg, n, F, D->br, D->loc, G);
    for (int a = 0; a < K; a++) for (int b = 0; b < K; b++) { Gk[a + b * K] = G[a + b * n]; X[a + b * K] = G[a + (K + b) * n]; }
    int ok = 0;
    if (!det_rsolve(K, Gk, K, X)) {
        double emax = 1, rmax = 0;
        for (int a = 0; a < K; a++) for (int b = 0; b < K; b++) {
            double e = G[(K + a) + (K + b) * n], s = 0;
            for (int k = 0; k < K; k++) s += G[k + (K + a) * n] * X[k + b * K];
            emax = dmax(emax, fabs(e)); rmax = dmax(rmax, fabs(e - s));
        }
        ok = rmax <= 1e-9 * emax;
    }
    free(F); free(G); free(Gk); free(X);
    return ok;
}
#define LINE_RANK_MSG "the line integral needs a numerator of rank one (a positive root) or two (one root of each sign)"
/* 0 where the line integral covers the design with its start and options, otherwise 1 and the reason in why */
static int line_refusal(const dfl_design *D, double c, const dfl_det_opts *o, char *why, size_t len)
{
    if (!isnan(o->gls)) { snprintf(why, len, "GLS detrending is not covered: the papers state the line integral for least squares detrending"); return 1; }
    if (o->start == DFL_START_FIXED && o->gamma0 != 0) { snprintf(why, len, "a fixed start is not covered: its kernel is the noncentral density of the numerator, a further integral with an intercept or a trend (the classical engine takes its means from closed forms case by case)"); return 1; }
    if (!(c >= DFL_CMIN)) { snprintf(why, len, "c below %g, the range in which the classical line integral is checked", DFL_CMIN); return 1; }
    if (o->start == DFL_START_STAT && !(c <= DFL_CSTAT)) { snprintf(why, len, "a stationary start needs c <= %g for the classical line integral's quadrature", DFL_CSTAT); return 1; }
    if (D->K == 0) return 0;
    int jumps = line_jumps(D);
    double v0[64], v5[64], m0 = 0, m5 = 1;
    if (D->K > 64) { snprintf(why, len, "more than 64 regressors"); return 1; }
    dfl_design_value(D, 0.0, v0); dfl_design_value(D, 0.5, v5);
    for (int i = 0; i < D->K; i++) { m0 = dmax(m0, fabs(v0[i])); m5 = dmax(m5, fabs(v5[i])); }
    int vanish = m0 <= 1e-12 * m5;
    if (D->first) {
        if (jumps || vanish) {
            snprintf(why, len, "the statistic that detrends first is covered for continuous trends that do not vanish at the origin only (the design has %s)",
                     jumps ? "a jump" : "d(0) = 0, a numerator of rank one for which the papers do not state the line integral");
            return 1;
        }
        return 0;
    }
    if (closed_under_derivative(D) && !jumps && !vanish) return 0;
    char e2[64]; det_prep *P = det_prep_cached(D, NAN, e2, sizeof e2);
    int r = P ? P->r : -1, rp = P ? P->rp : -1, rm = P ? P->rm : -1;
    det_prep_release(P);
    snprintf(why, len, "the one-step statistic is covered for continuous designs that do not vanish at the origin and whose span is closed under differentiation (polynomial and trigonometric trends with an intercept) only: %s (the numerator has rank %d, %d positive and %d negative roots)",
             jumps ? "the design has a jump" : vanish ? "d(0) = 0" : "the span of the design is not closed under differentiation", r, rp, rm);
    return 1;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the cache of the providers, per (design, start, c) (at most DET_CACHE_MAX, the oldest dropped) */
#define DET_LINE_CACHE_MAX 64
typedef struct { unsigned long long key; int start; double c; dfl_prov *pv; } prov_entry;
static prov_entry PROVC[DET_LINE_CACHE_MAX]; static int NPROVC = 0;
static dfl_lock_t PLOCK = DFL_LOCK_INIT;
static void prov_free(dfl_prov *pv) { det_model_free(&pv->Mo); det_prep_release(pv->P); free(pv->tab); free(pv); }
static void prov_unref_locked(dfl_prov *pv) { if (--pv->refs == 0) prov_free(pv); }
void det_line_release(dfl_prov *pv) { if (!pv) return; dfl_lock(&PLOCK); prov_unref_locked(pv); dfl_unlock(&PLOCK); }
void det_line_empty_cache(void)
{
    dfl_lock(&PLOCK);
    for (int i = 0; i < NPROVC; i++) prov_unref_locked(PROVC[i].pv);
    NPROVC = 0;
    dfl_unlock(&PLOCK);
}
int det_line_cache_size(void) { return NPROVC; }
/* the provider of the design with its start at c (a new reference), or NULL with the reason in why (the rank of the
 * numerator with the start) or an error in err */
static dfl_prov *prov_get(const dfl_design *D, double c, int start, char *why, size_t wlen, char *err, size_t errlen, int *isrank)
{
    *isrank = 0;
    unsigned long long key = det_design_key(D, NAN);
    dfl_lock(&PLOCK);
    for (int i = 0; i < NPROVC; i++) if (PROVC[i].key == key && PROVC[i].start == start && PROVC[i].c == c) { dfl_prov *pv = PROVC[i].pv; pv->refs++; dfl_unlock(&PLOCK); return pv; }
    dfl_unlock(&PLOCK);
    dfl_prov *pv = calloc(1, sizeof *pv);
    if (!pv) { snprintf(err, errlen, "out of memory"); return NULL; }
    pv->P = det_prep_cached(D, NAN, err, errlen);
    if (!pv->P) { free(pv); return NULL; }
    if (det_model_make(pv->P, start, 0.0, &pv->Mo, err, errlen)) { det_prep_release(pv->P); free(pv); return NULL; }
    const det_model *Mo = &pv->Mo;
    if (!((Mo->r == 1 && Mo->rp == 1) || (Mo->r == 2 && Mo->rp == 1 && Mo->rm == 1))) {
        snprintf(why, wlen, "%s; with this start it has rank %d (%d positive, %d negative)", LINE_RANK_MSG, Mo->r, Mo->rp, Mo->rm);
        det_model_free(&pv->Mo); det_prep_release(pv->P); free(pv); *isrank = 1; return NULL;
    }
    int mult;
    if (det_tstar(Mo, c, &pv->tstar, &mult, err, errlen)) { det_model_free(&pv->Mo); det_prep_release(pv->P); free(pv); return NULL; }
    pv->c = c; pv->r = Mo->r; pv->start = start; pv->stat = start == DFL_START_STAT; pv->key = key;
    pv->cap = PROV_MEMO_CAP0; pv->tab = calloc(pv->cap, sizeof(memo_ent));
    if (!pv->tab) { det_model_free(&pv->Mo); det_prep_release(pv->P); free(pv); snprintf(err, errlen, "out of memory"); return NULL; }
    { dfl_lock_t l0 = DFL_LOCK_INIT; pv->lock = l0; }
    { char buf[DET_WORK_STACK]; det_work W; det_work_init(&W, Mo, buf, sizeof buf); det_work_free(&W); }   /* (the one-time setup, serially) */
    /* the line: Re t = 1 for c <= 0 with a zero start, min(1, t_* / 2) otherwise (as the classical engine).  The fixed layouts of
     * the K_0 kernel's distribution function (dfl_batch.c), which the classical integrator takes at tau = 1 exactly, end the
     * ray of the constant part over X > 1/2 where e^{-s_2 X} has fallen for s_2(tau) >= 2.5 (as in the classical cases); where
     * the positive root lies below 2.5 at tau = 1 the line moves to the next double below 1, so that the adaptive rules,
     * which follow the decay of the kernel, take the distribution function (line_tau of DFDet_line.jl) */
    PROVFAIL = 0;
    pv->tau = (c <= 0 && !pv->stat) ? 1.0 : dmin(1.0, pv->tstar / 2);
    if (pv->tau == 1.0 && pv->r == 2) {
        cdouble t1 = CPX(1.0, 0), v[3]; line_raw(pv, 1, &t1, v);
        if (creal(cinv(v[1])) < 2.5) pv->tau = nextafter(1.0, 0.0);
    }
    pv->wend = line_end(pv, pv->tau);
    if (PROVFAIL) { snprintf(err, errlen, "the line integral: the transform failed at the end of the line (c = %g)", c); prov_free(pv); return NULL; }
    pv->refs = 1;
    dfl_lock(&PLOCK);
    for (int i = 0; i < NPROVC; i++) if (PROVC[i].key == key && PROVC[i].start == start && PROVC[i].c == c) { dfl_prov *q = PROVC[i].pv; q->refs++; dfl_unlock(&PLOCK); prov_free(pv); return q; }
    if (NPROVC == DET_LINE_CACHE_MAX) { prov_unref_locked(PROVC[0].pv); memmove(PROVC, PROVC + 1, (NPROVC - 1) * sizeof(prov_entry)); NPROVC--; }
    PROVC[NPROVC].key = key; PROVC[NPROVC].start = start; PROVC[NPROVC].c = c; PROVC[NPROVC].pv = pv; NPROVC++;
    pv->refs++;      /* (the cache's) */
    dfl_unlock(&PLOCK);
    return pv;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the option method of dfl_det_dist and dfl_det_quantile */
/* the rule of method auto where the line integral covers the arguments: the line integral at c0 = 0 for c >= -20
 * (DET_LINE_AUTO_CMIN), except just above the unit root with deterministic terms, 0 < c <= 0.5 (DET_LINE_PATHS_CMAX), where
 * the paths are the faster; the inversion on paths otherwise.  Measured on the C engines with the classical integrator's fixed
 * node layouts (c/test/line_timing.c, November 2026: warm calls, F and f at mean + (-2, -1, 0, 1, 2) sd and the 5% quantile
 * at new points, 1 and 4 threads; polynomial trends of order 0 to 4, the intercept with the first sine and cosine, the
 * quadratic trend with a stationary start): at c0 = 0 for -20 <= c <= 0 the line integral was 2.2 to 3.5 times faster than
 * the paths per value and 1.8 to 2.7 times per quantile; at c = -30 and -50 1.1 to 2.3 times per value but 0.77 to 1.8 per
 * quantile (mostly slower: the classical integrator's line grows longer beyond c = -20), so the paths there; with c0 != 0
 * the general kernel of the classical integrator (breakpoints of X(v), adaptive pieces) makes the line integral the slower
 * (as in the Julia engine, 2 to 5 times).  Since the thinner rules of the paths (re-measured November 26, 2026, interleaved
 * warm calls, 4 threads): above the unit root the paths are 1.5 to 2.3 times faster per value and per quantile for every
 * design with deterministic terms at c = 0.25 and 0.5; up to c = 2 (the intercept 1.5) for some designs by 1.3 to 2.2 times,
 * but below 1.5 for others (the Julia rule takes the paths up to c = 1.5 with one regressor and c = 2 with more);
 * without deterministic terms the line integral. */
/* whether method auto takes the line integral, where it covers the arguments, with c0 > 0 above the unit root on the vertical
 * line (the outer contour of the paths where no hyperbola keeps the positive roots right of the origin): there the parabola
 * X(v) of a point x < 0 has two positive roots, whose parts of the outer integrand the line's rule does not resolve, and the
 * paths' one evaluation was off by up to 1e-3 in the lower tail, below zero or NaN (November 26, 2026; line_vline of the
 * Julia engine); the line integral agrees with the paths at x > 0 and with the classical engine for the linear trend */
int det_line_vline(const dfl_design *D, double c, const dfl_det_opts *o)
{
    if (!(o->c0 > 0 && c > 0 && o->start != DFL_START_FIXED)) return 0;
    char e[256];
    det_path *CP = det_path_cached(D, o->gls, o->start, o->gamma0, c, NAN, 1.0, e, sizeof e);
    if (!CP) return 0;
    int v = isnan(CP->h);
    det_path_release(CP);
    return v;
}
int det_line_auto(const dfl_design *D, double c, double c0)
{
    if (!(c0 == 0 && c >= DET_LINE_AUTO_CMIN)) return 0;
    return !(c > 0 && D->K >= 1 && c <= DET_LINE_PATHS_CMAX);
}
/* the provider when method (auto or line) takes the line integral, NULL when the inversion on paths computes; 1 (an error in
 * err) where method line is refused or the provider fails */
int det_line_select(const dfl_design *D, double c, const dfl_det_opts *o, dfl_prov **out, char *err, size_t errlen)
{
    *out = NULL;
    int method = o->method;
    if (method == DFL_DET_PATHS) return 0;
    if (method != DFL_DET_AUTO && method != DFL_DET_LINE) { snprintf(err, errlen, "method must be 0 (auto), 1 (the line integral with the explicit kernel) or 2 (the inversion on paths around the roots)"); return 1; }
    char why[512] = "";
    int refused = 0;
    /* (the options of the paths: n, ntail, phi, hfac other than 1; the rule of one evaluation is not one, as in Julia) */
    int hf = !isnan(o->hfac) && o->hfac != 1;
    if (o->n > 0 || o->ntail > 0 || !isnan(o->phi) || hf) {
        const char *nm[4] = {"n", "ntail", "phi", "hfac"}; int on[4] = {o->n > 0, o->ntail > 0, !isnan(o->phi), hf}, k = snprintf(why, sizeof why, "the options"), first = 1;
        for (int i = 0; i < 4; i++) if (on[i] && k < (int) sizeof why) { k += snprintf(why + k, sizeof why - k, "%s %s", first ? "" : ",", nm[i]); first = 0; }
        if (k < (int) sizeof why) snprintf(why + k, sizeof why - k, " belong to the inversion on paths");
        refused = 1;
    }
    if (!refused && method == DFL_DET_AUTO && !det_line_auto(D, c, o->c0) && !det_line_vline(D, c, o)) return 0;     /* (before the provider is built) */
    if (!refused) refused = line_refusal(D, c, o, why, sizeof why);
    if (!refused) {
        int start = o->start == DFL_START_STAT ? DFL_START_STAT : DFL_START_ZERO, isrank;
        dfl_prov *pv = prov_get(D, c, start, why, sizeof why, err, errlen, &isrank);
        if (pv) { *out = pv; return 0; }
        if (!isrank) return 1;
        refused = 1;
    }
    if (method == DFL_DET_LINE) { snprintf(err, errlen, "method line: %s", why); return 1; }
    return 0;
}
/* one value of the line integral at x: 0 and the value, or the bound of the classical integrator's paths (DFL_BOUND_*), or
 * -1 where the provider failed (the transform) */
int det_line_value(dfl_prov *pv, int kind, double x, double c0, double *out)
{
    dfl_single_opts so; dfl_single_defaults(&so);
    so.tau = pv->tau;
    dfl_bound_t bd;
    const dfl_prov *prev = CUR; CUR = pv; PROVFAIL = 0;
    int rc = c0 != 0 ? dfl_single_value_gen_b(kind, x, DFL_MODEL_PROV, pv->c, c0, 0.0, pv->stat, &so, out, &bd)
                     : dfl_single_value_c0_b(kind, x, DFL_MODEL_PROV, pv->c, pv->stat, &so, out, &bd);
    CUR = prev;
    if (PROVFAIL) return -1;
    return rc ? bd.bound : 0;
}
/* (for the tests) the transform at n points of the provider's line in one call of line_raw: 0 (1 where the transform failed),
 * and the number of points its memo then holds in *nmemo (at most half its slots) */
int det_line_raw_test(dfl_prov *pv, int n, size_t *nmemo)
{
    cdouble *t = malloc((size_t) n * sizeof(cdouble)), *out = malloc(3 * (size_t) n * sizeof(cdouble));
    if (!t || !out) { free(t); free(out); return 1; }
    for (int k = 0; k < n; k++) t[k] = CPX(pv->tau, pv->wend * (k + 1) / n);
    PROVFAIL = 0; line_raw(pv, n, t, out);
    int rc = PROVFAIL;
    dfl_lock(&pv->lock); *nmemo = pv->n; dfl_unlock(&pv->lock);
    free(t); free(out);
    return rc;
}
/* the message of a point the line integral cannot take under method line */
void det_line_point_error(char *err, size_t errlen, const char *fn, const dfl_prov *pv, double x, double c0, int rc)
{
    if (rc < 0) snprintf(err, errlen, "%s: method line: c = %g, c0 = %g, x = %g: the transform of the provider failed on the line; method auto takes the inversion on paths there", fn, pv->c, c0, x);
    else snprintf(err, errlen, "%s: method line: c = %g, c0 = %g, x = %g: the line integral needs more nodes than the engine allows here (%s); method auto takes the inversion on paths at such points",
                  fn, pv->c, c0, x, rc == DFL_BOUND_RULE ? "a Gauss-Legendre rule beyond its bound, at large x above the unit root" : "a path of the inner integrals beyond its bound, c0 < 0 near zero with x > 0");
}
