/* dfl_det_mgf.c: the joint moment generating function of the general engine (a port of DFDet.jl): the covariance
 * Sigma(theta) of the Gaussian vector ell under the Ornstein-Uhlenbeck law with parameter -theta (sigma!, by the transfer
 * functions in the functions e_k(z) = int_0^1 x^k e^{-zx} dx and Gauss-Legendre panels graded toward the right end of each
 * piece), the matrix M0 = I - 2 (t B + gamma E11) Sigma and its determinant, the reciprocal roots in s as the eigenvalues
 * of the small matrix H = 2 Dv L' Sigma M0^{-1} L (mgfparts), the joint moment generating function with a fixed start
 * (fixparts, fixE1: Woodbury's identity, bounded at infinity in s) and with a stationary start (the extended covariance),
 * the logarithm of the transform at one point (logmgf_point), and t_* (the Nystrom estimate, bisection on D_V, the scan
 * from the branch point far from the unit root). */
#include "dfl_det_int.h"

double det_e0r(double z) { return fabs(z) < 1e-3 ? 1 - z / 2 + z * z / 6 - z * z * z / 24 : (1 - exp(-z)) / z; }
static inline cdouble e0c(cdouble z)
{
    if (cfabs(z) < 1e-3) { cdouble z2 = cmul(z, z); return cadd(csub(DONE, cscale(0.5, z)), csub(cdivr(z2, 6), cdivr(cmul(z2, z), 24))); }
    return cdiv(csub(DONE, cexp(cneg(z))), z);
}
/* 1 - (c + theta) e_0(2 theta) = 1 - 2 gamma Var J(1), without the cancellation of theta - c for c > 0 */
static cdouble m011(double c, cdouble t, cdouble th)
{
    if (cabs(th) < 0.5) return csub(DONE, cmul(caddr(th, c), e0c(cscale(2, th))));
    cdouble del = c > 0 ? cdiv(cscale(-2, t), caddr(th, c)) : caddr(th, -c);
    return cdiv(cadd(del, cmul(caddr(th, c), cexp(cscale(-2, th)))), cscale(2, th));
}

/* e_k(z) = int_0^1 x^k e^{-zx} dx, k = 0..D (ek!: the power series for |z| < 2; upward recurrence for k <= |z|, the
 * downward one from the series in 1 - x beyond) */
static double INVN[1025];
static volatile int INVN_DONE = 0;
static dfl_lock_t INVN_LOCK = DFL_LOCK_INIT;
static void invn_init(void)
{
    if (INVN_DONE) return;
    dfl_lock(&INVN_LOCK);
    if (!INVN_DONE) { for (int m = 1; m <= 1024; m++) INVN[m] = 1.0 / m; INVN_DONE = 1; }
    dfl_unlock(&INVN_LOCK);
}
static void ek(cdouble *e, cdouble z, int D)
{
    double a2 = cabs2(z);
    if (a2 < 4) {
        for (int k = 0; k <= D; k++) e[k] = CPX(INVN[k + 1], 0);
        cdouble term = DONE, mz = cneg(z);
        for (int n = 1; n <= 300; n++) {
            term = cscale(INVN[n], cmul(term, mz));
            if (n + D + 1 <= 1024) for (int k = 0; k <= D; k++) e[k] = cadd(e[k], cscale(INVN[n + k + 1], term));
            else for (int k = 0; k <= D; k++) e[k] = cadd(e[k], cdivr(term, n + k + 1));
            if (cabs2(term) < 1e-36) break;
        }
        return;
    }
    cdouble em = cexp(cneg(z)), iz = cinv(z);
    int K1 = a2 < (double) D * D ? (int) floor(sqrt(a2)) : D;
    e[0] = cmul(csub(DONE, em), iz);
    for (int k = 1; k <= K1; k++) e[k] = cmul(csub(cscale(k, e[k - 1]), em), iz);
    if (K1 < D) {
        cdouble term = CPX(1.0 / (D + 1), 0), acc = term;
        for (int m = 1; m <= 5000; m++) {
            term = cdivr(cmul(term, z), D + m + 1);
            acc = cadd(acc, term);
            if (cabs2(term) < 1e-36 * cabs2(acc)) break;
        }
        e[D] = cmul(em, acc);
        for (int k = D; k >= K1 + 2; k--) e[k - 1] = cdivr(cadd(cmul(z, e[k]), em), k);
    }
}
/* the Taylor coefficients at s of the columns of C ((D + 1) x p) by repeated synthetic division */
static void taylor(double *a, const double *C, int D, int p, double s)
{
    memcpy(a, C, (size_t) (D + 1) * p * sizeof(double));
    for (int i = 0; i < p; i++) { double *ai = a + (size_t) i * (D + 1); for (int k = 0; k < D; k++) for (int j = D - 1; j >= k; j--) ai[j] += s * ai[j + 1]; }
}

/* ---------------------------------------------------------------------------------------------------------------- */
det_fixt *det_fixt_alloc(int r)
{
    size_t rr = imax(r, 1);
    det_fixt *fx = malloc(sizeof(det_fixt) + (rr * rr + 5 * rr) * sizeof(cdouble));
    if (!fx) return NULL;
    fx->r = r;
    cdouble *b = (cdouble *) (fx + 1);
    fx->H = b; fx->w1 = b + rr * rr; fx->p1 = fx->w1 + rr; fx->w2 = fx->p1 + rr; fx->p2 = fx->w2 + rr; fx->mt = fx->p2 + rr;
    return fx;
}

int det_work_init(det_work *W, const det_model *Mo, void *buf, size_t bytes)
{
    invn_init();
    const det_prep *P = Mo->P;
    memset(W, 0, sizeof *W);
    W->Mo = Mo; W->N0 = P->N; W->N = Mo->N; W->r = Mo->r; W->deg = P->gdeg; W->p = P->p; W->np = P->np; W->nq = 24;
    W->zq = det_gl(24, &W->wq);
    int N0 = W->N0, N = W->N, r = imax(W->r, 1), d1 = W->deg + 1, pm = imax(W->p, 1), np = W->np;
    size_t nc = (size_t) N0 * N0 + (Mo->start == 2 ? (size_t) N * N : 0) + (size_t) N * N + N0 + (size_t) N0 * np + (size_t) pm * np + d1 + N0
              + (size_t) N * r + (size_t) r * r + 4 * (size_t) N + (size_t) r * (r + 2);
    size_t nd = (size_t) d1 * pm, need = nc * sizeof(cdouble) + nd * sizeof(double) + N * sizeof(int) + 64;
    char *m;
    if (buf && bytes >= need) m = buf; else { m = malloc(need); if (!m) return 1; W->heap = m; }
    m = (char *) (((size_t) m + 15) & ~(size_t) 15);
    cdouble *c = (cdouble *) m;
    W->S = c; c += (size_t) N0 * N0;
    if (Mo->start == 2) { W->Sx = c; c += (size_t) N * N; } else W->Sx = W->S;
    W->M = c; c += (size_t) N * N;
    W->F = c; c += N0; W->Rb = c; c += (size_t) N0 * np; W->Ib = c; c += (size_t) pm * np; W->e = c; c += d1; W->h = c; c += N0;
    W->X = c; c += (size_t) N * r; W->Hm = c; c += (size_t) r * r;
    W->v1 = c; c += N; W->v2 = c; c += N; W->v3 = c; c += N; W->v4 = c; c += N; W->Bf = c; c += (size_t) r * (r + 2);
    W->a = (double *) c; W->piv = (int *) (W->a + nd);
    return 0;
}
void det_work_free(det_work *W) { free(W->heap); W->heap = NULL; }

/* Sigma(theta) = int_0^1 F(u) F(u)' du (bilinear, no conjugation) into W->S, F(0) into W->h (sigma!) */
void det_sigma(det_work *W, cdouble th)
{
    const det_prep *P = W->Mo->P;
    int N = W->N0, p = W->p, nj = P->nj, np = W->np, D = W->deg;
    const double *br = P->br;
    cdouble *S = W->S, *e = W->e, *F = W->F, *Rb = W->Rb, *Ib = W->Ib;
    double *a = W->a;
    int pm = imax(p, 1);
    for (int i = 0; i < N * N; i++) S[i] = DZERO;
    for (int m = 0; m < np && p > 0; m++) {
        double am = br[m], Lm = br[m + 1] - am;
        ek(e, cscale(Lm, th), D);
        taylor(a, P->g + (size_t) m * (D + 1) * p, D, p, am - P->off[m]);
        for (int i = 0; i < p; i++) {
            cdouble acc = DZERO; double Lk = Lm;
            for (int k = 0; k <= D; k++) { acc = cadd(acc, cscale(a[k + i * (D + 1)] * Lk, e[k])); Lk *= Lm; }
            Ib[i + m * pm] = acc;
        }
    }
    for (int j = np - 1; j >= 0; j--) {
        double b = br[j + 1];
        Rb[0 + j * N] = cexp(cscale(-(1 - b), th));
        for (int i = 0; i < nj; i++) { double lam = P->jl[i]; Rb[1 + i + j * N] = lam >= b - 1e-14 ? cexp(cscale(-(lam - b), th)) : DZERO; }
        if (j == np - 1) { for (int i = 0; i < p; i++) Rb[1 + nj + i + j * N] = DZERO; }
        else {
            cdouble el = cexp(cscale(-(br[j + 2] - br[j + 1]), th));
            for (int i = 0; i < p; i++) Rb[1 + nj + i + j * N] = cadd(Ib[i + (j + 1) * pm], cmul(el, Rb[1 + nj + i + (j + 1) * N]));
        }
    }
    double at = cabs(th), rt = creal(th), h0 = at > 0 ? 1 / at : INFINITY;
    const double *zq = W->zq, *wq = W->wq;
    for (int j = 0; j < np; j++) {
        double b = br[j + 1], Lj = b - br[j], y0 = 0;
        const double *gj = P->g + (size_t) j * (D + 1) * p;
        while (y0 < Lj * (1 - 1e-14)) {
            double y1;
            if (at * Lj <= 2) y1 = Lj;
            else {
                double wdt = dmax(y0, h0);
                if (2 * rt * y0 < 80) wdt = dmin(wdt, 24 * h0);
                y1 = dmin(y0 + wdt, Lj);
                if ((Lj - y1) < 0.5 * wdt) y1 = Lj;
            }
            for (int q = 0; q < W->nq; q++) {
                double y = y0 + (y1 - y0) * zq[q], wqq = (y1 - y0) * wq[q];
                cdouble ey = cexp(cscale(-y, th));
                for (int i = 0; i < 1 + nj; i++) F[i] = cmul(ey, Rb[i + j * N]);
                if (p > 0) {
                    ek(e, cscale(y, th), D);
                    taylor(a, gj, D, p, b - y - P->off[j]);
                    for (int i = 0; i < p; i++) {
                        cdouble acc = DZERO; double yk = y;
                        for (int k = 0; k <= D; k++) { acc = cadd(acc, cscale(a[k + i * (D + 1)] * yk, e[k])); yk *= y; }
                        F[1 + nj + i] = cadd(acc, cmul(ey, Rb[1 + nj + i + j * N]));
                    }
                }
                for (int jj = 0; jj < N; jj++) {
                    cdouble fj = cscale(wqq, F[jj]);
                    for (int ii = 0; ii <= jj; ii++) S[ii + jj * N] = cadd(S[ii + jj * N], cmul(fj, F[ii]));
                }
            }
            y0 = y1;
        }
    }
    for (int jj = 0; jj < N; jj++) for (int ii = 0; ii < jj; ii++) S[jj + ii * N] = S[ii + jj * N];
    S[0] = e0c(cscale(2, th));                       /* Var J(1) in closed form */
    cdouble *h = W->h;
    h[0] = cexp(cneg(th));
    for (int i = 0; i < nj; i++) h[1 + i] = cexp(cscale(-P->jl[i], th));
    if (p > 0) {
        cdouble el = cexp(cscale(-(br[1] - br[0]), th));
        for (int i = 0; i < p; i++) h[1 + nj + i] = cadd(Ib[i], cmul(el, Rb[1 + nj + i]));
    }
}

/* M = I - 2 (s A + t B + gamma E11) Sigma with the (1,1) entry without cancellation (buildM!), into W->M */
static void buildM(det_work *W, double c, cdouble s, cdouble t, cdouble th)
{
    const det_model *Mo = W->Mo;
    int N = W->N0;
    const double *A = Mo->A, *B = Mo->B;
    cdouble *M = W->M, *S = W->S, ga = cscale(0.5, caddr(th, c));
    for (int i = 0; i < N * N; i++) M[i] = DZERO;
    for (int k = 0; k < N; k++) for (int i = 0; i < N; i++) {
        double ar = A[i + k * N], br = B[i + k * N];
        if (ar == 0 && br == 0) continue;
        cdouble q = cscale(-2, cadd(cscale(ar, s), cscale(br, t)));
        for (int j = 0; j < N; j++) M[i + j * N] = cadd(M[i + j * N], cmul(q, S[k + j * N]));
    }
    for (int i = 0; i < N; i++) M[i + i * N] = caddr(M[i + i * N], 1);
    cdouble g2 = cscale(2, ga);
    for (int j = 0; j < N; j++) M[0 + j * N] = csub(M[0 + j * N], cmul(g2, S[0 + j * N]));
    M[0] = cadd(M[0], csub(m011(c, t, th), csub(DONE, cmul(g2, S[0]))));
}
/* the stationary start: I - 2 (s A + t B + gamma (E11 - ENN)) Sigma_ext (buildMx!), with Sigma_ext in W->Sx */
static void buildMx(det_work *W, double c, cdouble s, cdouble t, cdouble th)
{
    const det_model *Mo = W->Mo;
    int N = W->N;
    cdouble *M = W->M, *S = W->Sx, ga = cscale(0.5, caddr(th, c));
    for (int i = 0; i < N * N; i++) M[i] = DZERO;
    for (int k = 0; k < N; k++) for (int i = 0; i < N; i++) {
        cdouble q = cadd(cscale(Mo->A[i + k * N], s), cscale(Mo->B[i + k * N], t));
        if (i == k && i == 0) q = cadd(q, ga);
        if (i == k && i == N - 1) q = csub(q, ga);
        if (creal(q) == 0 && cimag(q) == 0) continue;
        q = cscale(-2, q);
        for (int j = 0; j < N; j++) M[i + j * N] = cadd(M[i + j * N], cmul(q, S[k + j * N]));
    }
    for (int i = 0; i < N; i++) M[i + i * N] = caddr(M[i + i * N], 1);
}
/* the stationary covariance Sigma_ext = [S + v h h', v h; v h', v] into Sx */
static void statsigma(det_work *W, double v)
{
    int n = W->N0, N = n + 1;
    cdouble *Sx = W->Sx, *S = W->S, *h = W->h;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) Sx[i + j * N] = cadd(S[i + j * n], cscale(v, cmul(h[i], h[j])));
        Sx[n + j * N] = cscale(v, h[j]); Sx[j + n * N] = cscale(v, h[j]);
    }
    Sx[n + n * N] = CPX(v, 0);
}
/* H = 2 Dv L' S M0^{-1} L (r x r) into W->Hm, with M0's LU in W->M: X = M0^{-1} L */
static void hmat(det_work *W, const cdouble *S)
{
    const det_model *Mo = W->Mo;
    int N = W->N, r = W->r;
    for (int k = 0; k < r; k++) {
        cdouble *x = W->X + (size_t) k * N;
        for (int i = 0; i < N; i++) x[i] = CPX(Mo->L[i + (size_t) k * N], 0);
        det_csolve(N, W->M, W->piv, x);
    }
    /* SX = S X column by column into v1, then H[k, l] = 2 Dv[k] L[:, k]' SX[:, l] */
    for (int l = 0; l < r; l++) {
        const cdouble *x = W->X + (size_t) l * N;
        for (int i = 0; i < N; i++) { cdouble s = DZERO; for (int j = 0; j < N; j++) s = cadd(s, cmul(S[i + j * N], x[j])); W->v1[i] = s; }
        for (int k = 0; k < r; k++) {
            cdouble s = DZERO; for (int i = 0; i < N; i++) s = cadd(s, cscale(Mo->L[i + (size_t) k * N], W->v1[i]));
            W->Hm[k + l * r] = cscale(2 * Mo->Dv[k], s);
        }
    }
}

int det_mgfparts(det_work *W, double c, cdouble t, cdouble *thp, cdouble *ld, cdouble *mu, det_fixt *fx)
{
    const det_model *Mo = W->Mo;
    int N = W->N, r = W->r, odd;
    cdouble th = det_theta(c, t);
    *thp = th;
    det_sigma(W, th);
    if (Mo->start == 2) {
        statsigma(W, -1 / (2 * c));
        buildMx(W, c, DZERO, t, th);
        det_clu(N, W->M, W->piv, &odd);
        *ld = cadd(caddr(th, c), det_clogdet(N, W->M, odd));
        hmat(W, W->Sx);
        return det_ceigvals(r, W->Hm, mu);
    }
    buildM(W, c, DZERO, t, th);
    det_clu(N, W->M, W->piv, &odd);
    *ld = cadd(caddr(th, c), det_clogdet(N, W->M, odd));
    hmat(W, W->S);
    if (Mo->start == 1 && fx) {
        /* the fixed start (fixparts): with Q0 = t B + gamma E11, mt = h + yb, v0 = Q0 (h - yb), qt = -2 Q0 yb */
        cdouble ga = cscale(0.5, caddr(th, c));
        const double *B = Mo->B, *yb = Mo->yb, *L = Mo->L, *Dv = Mo->Dv;
        const cdouble *S = W->S, *h = W->h;
        cdouble *mt = W->v1, *v0 = W->v2, *qt = W->v3, *tmp = W->v4;
        /* (v1 is reused by hmat: H is complete here) */
        for (int i = 0; i < N; i++) {
            cdouble q0hy = DZERO, q0y = DZERO;
            for (int j = 0; j < N; j++) {
                cdouble qij = cscale(B[i + j * N], t);
                if (i == 0 && j == 0) qij = cadd(qij, ga);
                q0hy = cadd(q0hy, cmul(qij, caddr(h[j], -yb[j])));
                q0y = cadd(q0y, cscale(yb[j], qij));
            }
            v0[i] = q0hy; qt[i] = cscale(-2, q0y);
        }
        for (int i = 0; i < N; i++) mt[i] = caddr(h[i], yb[i]);
        /* cf = -gamma + yb' Q0 yb - (yb' qt)/2 ... computed as in Julia: yb' (Q0 yb) = -(yb' qt)/2 */
        cdouble yqy = DZERO; for (int i = 0; i < N; i++) yqy = cadd(yqy, cscale(yb[i], qt[i]));
        yqy = cscale(-0.5, yqy);
        cdouble *a0 = W->X, *aq = W->X + N;          /* (X is free once H is formed; it has N r >= N entries, and 2N where r >= 2) */
        cdouble a0b[64], aqb[64];
        if (r < 2) { a0 = N <= 64 ? a0b : malloc(N * sizeof(cdouble)); aq = N <= 64 ? aqb : malloc(N * sizeof(cdouble)); }
        for (int i = 0; i < N; i++) { a0[i] = v0[i]; aq[i] = qt[i]; }
        det_csolve(N, W->M, W->piv, a0); det_csolve(N, W->M, W->piv, aq);
        cdouble mta0 = DZERO, qsaq = DZERO;
        for (int i = 0; i < N; i++) mta0 = cadd(mta0, cmul(mt[i], a0[i]));
        /* S aq into tmp, qt' S aq */
        for (int i = 0; i < N; i++) { cdouble s = DZERO; for (int j = 0; j < N; j++) s = cadd(s, cmul(S[i + j * N], aq[j])); tmp[i] = s; }
        for (int i = 0; i < N; i++) qsaq = cadd(qsaq, cmul(qt[i], tmp[i]));
        fx->cf = cadd(cadd(cneg(ga), yqy), cadd(mta0, cscale(0.5, qsaq)));
        /* w2 = Dv .* (L' (S aq)) */
        for (int k = 0; k < r; k++) { cdouble s = DZERO; for (int i = 0; i < N; i++) s = cadd(s, cscale(L[i + (size_t) k * N], tmp[i])); fx->w2[k] = cscale(Dv[k], s); }
        /* w1 = 2 Dv .* (L' (S a0)) + Dv .* (L' mt) */
        for (int i = 0; i < N; i++) { cdouble s = DZERO; for (int j = 0; j < N; j++) s = cadd(s, cmul(S[i + j * N], a0[j])); tmp[i] = s; }
        for (int k = 0; k < r; k++) {
            cdouble s1 = DZERO, s2 = DZERO;
            for (int i = 0; i < N; i++) { s1 = cadd(s1, cscale(L[i + (size_t) k * N], tmp[i])); s2 = cadd(s2, cscale(L[i + (size_t) k * N], mt[i])); }
            fx->w1[k] = cadd(cscale(2 * Dv[k], s1), cscale(Dv[k], s2));
        }
        /* p1 = L' (M0^{-T} mt); p2 = L' (M0^{-T} S qt); mm = L' (M0^{-T} h + yb) */
        for (int i = 0; i < N; i++) tmp[i] = mt[i];
        det_csolvet(N, W->M, W->piv, tmp);
        for (int k = 0; k < r; k++) { cdouble s = DZERO; for (int i = 0; i < N; i++) s = cadd(s, cscale(L[i + (size_t) k * N], tmp[i])); fx->p1[k] = s; }
        for (int i = 0; i < N; i++) { cdouble s = DZERO; for (int j = 0; j < N; j++) s = cadd(s, cmul(S[i + j * N], qt[j])); tmp[i] = s; }
        det_csolvet(N, W->M, W->piv, tmp);
        for (int k = 0; k < r; k++) { cdouble s = DZERO; for (int i = 0; i < N; i++) s = cadd(s, cscale(L[i + (size_t) k * N], tmp[i])); fx->p2[k] = s; }
        for (int i = 0; i < N; i++) tmp[i] = h[i];
        det_csolvet(N, W->M, W->piv, tmp);
        for (int k = 0; k < r; k++) { cdouble s = DZERO; for (int i = 0; i < N; i++) s = cadd(s, cscale(L[i + (size_t) k * N], caddr(tmp[i], yb[i]))); fx->mt[k] = s; }
        if (r < 2 && N > 64) { free(a0); free(aq); }
        memcpy(fx->H, W->Hm, (size_t) r * r * sizeof(cdouble));
    }
    return det_ceigvals(r, W->Hm, mu);
}

/* E1(s) = s [p1' (I - sH)^{-1} w1 + p2' (I - sH)^{-1} w2] (fixE1): Gaussian elimination on (I - sH) x = s w (|s| <= 1) or
 * (I/s - H) x = w (|s| > 1); Bf r x (r + 2) */
cdouble det_fixE1(const det_fixt *fx, cdouble s, cdouble *Bf)
{
    int r = fx->r;
    if (r == 0) return DZERO;
    int big = cabs2(s) > 1;
    cdouble d = big ? cinv(s) : DONE, f = big ? DONE : s;
#define BF(i, j) Bf[(i) + (j) * r]
    for (int j = 0; j < r; j++) {
        for (int i = 0; i < r; i++) BF(i, j) = big ? cneg(fx->H[i + j * r]) : cneg(cmul(s, fx->H[i + j * r]));
        BF(j, j) = cadd(BF(j, j), d);
        BF(j, r) = cmul(f, fx->w1[j]); BF(j, r + 1) = cmul(f, fx->w2[j]);
    }
    for (int k = 0; k < r; k++) {
        int piv = k; double mx = cabs2(BF(k, k));
        for (int i = k + 1; i < r; i++) { double a2 = cabs2(BF(i, k)); if (a2 > mx) { mx = a2; piv = i; } }
        if (piv != k) for (int j = k; j < r + 2; j++) { cdouble tmp = BF(k, j); BF(k, j) = BF(piv, j); BF(piv, j) = tmp; }
        cdouble ip = cinv(BF(k, k)); BF(k, k) = ip;
        for (int i = k + 1; i < r; i++) {
            cdouble l = cmul(BF(i, k), ip);
            for (int j = k + 1; j < r + 2; j++) BF(i, j) = csub(BF(i, j), cmul(l, BF(k, j)));
        }
    }
    cdouble acc = DZERO;
    for (int col = r; col <= r + 1; col++) {
        const cdouble *pv = col == r ? fx->p1 : fx->p2;
        for (int k = r - 1; k >= 0; k--) {
            cdouble x = BF(k, col);
            for (int j = k + 1; j < r; j++) x = csub(x, cmul(BF(k, j), BF(j, col)));
            BF(k, col) = cmul(x, BF(k, k));
            acc = cadd(acc, cmul(pv[k], BF(k, col)));
        }
    }
#undef BF
    return acc;
}

cdouble det_dval_st(det_work *W, double c, cdouble s, cdouble t)
{
    int odd;
    cdouble th = det_theta(c, t);
    det_sigma(W, th);
    if (W->Mo->start == 2) {
        statsigma(W, -1 / (2 * c)); buildMx(W, c, s, t, th);
        det_clu(W->N, W->M, W->piv, &odd);
        return cmul(cexp(caddr(th, c)), det_cdet(W->N, W->M, odd));
    }
    buildM(W, c, s, t, th);
    det_clu(W->N0, W->M, W->piv, &odd);
    return cmul(cexp(caddr(th, c)), det_cdet(W->N0, W->M, odd));
}

int det_logmgf_tpart(det_work *W, double c, cdouble t, cdouble *th, det_fixt *fx)
{
    if (W->Mo->start == 1) {
        cdouble ld; cdouble mu[64], *m = W->r <= 64 ? mu : malloc(W->r * sizeof(cdouble));
        int rc = det_mgfparts(W, c, t, th, &ld, m, fx);
        if (m != mu) free(m);
        return rc;
    }
    *th = det_theta(c, t);
    det_sigma(W, *th);
    if (W->Mo->start == 2) statsigma(W, -1 / (2 * c));
    return 0;
}
void det_logmgf_spart(det_work *W, double c, cdouble s, cdouble t, cdouble th, const det_fixt *fx, cdouble *l0, cdouble *ld)
{
    const det_model *Mo = W->Mo;
    int odd;
    cdouble base = csub(cscale(-0.5, s), cscale(0.5, caddr(th, c)));
    if (Mo->start == 2) {
        buildMx(W, c, s, t, th);
        det_clu(W->N, W->M, W->piv, &odd);
        *ld = det_clogdet(W->N, W->M, odd); *l0 = base;
        return;
    }
    cdouble extra = DZERO;
    if (Mo->start == 1) extra = cscale(Mo->g0 * Mo->g0, csub(cadd(fx->cf, det_fixE1(fx, s, W->Bf)), cscale(Mo->kappa, s)));
    buildM(W, c, s, t, th);
    det_clu(W->N0, W->M, W->piv, &odd);
    *ld = det_clogdet(W->N0, W->M, odd); *l0 = cadd(base, extra);
}

/* =====================================================================================================================
 * t_* = sup{t: E e^{tV} < inf}, the first zero of D_V(t) on the positive real axis
 * ===================================================================================================================== */
/* the covariance Cov(J(r), J(u)) of the Ornstein-Uhlenbeck process (oukernel; v0 the variance of a stationary start) */
static double oukernel(double c, double r, double u, double v0)
{
    double mn = dmin(r, u);
    if (-2 * c * mn > 700) return (exp(c * fabs(r - u)) - exp(c * (r + u))) / (-2 * c) + exp(c * (r + u)) * v0;
    return exp(c * (r + u)) * (mn * det_e0r(2 * c * mn) + v0);
}
typedef struct { int n; const double *K; } dense_op;
static void dense_apply(int nb, const double *X, double *Y, void *v)
{
    const dense_op *o = v; int n = o->n;
    for (int b = 0; b < nb; b++) {
        const double *x = X + (size_t) b * n; double *y = Y + (size_t) b * n;
        for (int i = 0; i < n; i++) y[i] = 0;
        for (int k = 0; k < n; k++) { double xk = x[k]; if (xk == 0) continue; const double *col = o->K + (size_t) k * n; for (int i = 0; i < n; i++) y[i] += col[i] * xk; }
    }
}
/* the Nystrom nodes: 8 equal panels of m Gauss-Legendre nodes on every piece */
static int nystrom_nodes(const det_prep *P, int m, double **rs, double **ws, int **pj)
{
    const double *w, *z = det_gl(m, &w);
    int n = P->np * 8 * m;
    *rs = malloc(n * sizeof(double)); *ws = malloc(n * sizeof(double)); if (pj) *pj = malloc(n * sizeof(int));
    int i = 0;
    for (int j = 0; j < P->np; j++) {
        double a = P->br[j], b = P->br[j + 1];
        for (int kk = 0; kk < 8; kk++) for (int q = 0; q < m; q++) {
            double aa = a + (b - a) * kk / 8;
            (*rs)[i] = aa + (b - a) / 8 * z[q]; (*ws)[i] = (b - a) / 8 * w[q]; if (pj) (*pj)[i] = j; i++;
        }
    }
    return n;
}
int det_eigtop(const det_prep *P, double c, int m, int stat, double *l1, double *l2)
{
    double *rs, *ws; int n = nystrom_nodes(P, m, &rs, &ws, NULL);
    double v0 = stat ? -1 / (2 * c) : 0.0;
    double *sw = malloc(n * sizeof(double)), *Kw = malloc((size_t) n * n * sizeof(double));
    if (!Kw) { free(rs); free(ws); free(sw); return 1; }
    for (int i = 0; i < n; i++) sw[i] = sqrt(ws[i]);
    for (int k = 0; k < n; k++) for (int i = k; i < n; i++) { double v = sw[i] * oukernel(c, rs[i], rs[k], v0) * sw[k]; Kw[i + (size_t) k * n] = v; Kw[k + (size_t) i * n] = v; }
    if (P->K > 0) {
        int K = P->Kd;
        dfl_design D = {P->Kd, P->np, P->deg, P->br, P->C, P->loc, P->first};
        double *Q = malloc((size_t) n * K * sizeof(double)), *dv = malloc(K * sizeof(double));
        for (int i = 0; i < n; i++) { det_dval(&D, rs[i], dv); for (int a = 0; a < K; a++) Q[i + (size_t) a * n] = dv[a] * sw[i]; }
        det_orth(n, K, Q);
        /* Kw - Q (Q' Kw), then - (Kw Q) Q' */
        double *T = malloc((size_t) K * n * sizeof(double));
        for (int j = 0; j < n; j++) for (int a = 0; a < K; a++) { double s = 0; const double *q = Q + (size_t) a * n, *col = Kw + (size_t) j * n; for (int i = 0; i < n; i++) s += q[i] * col[i]; T[a + (size_t) j * K] = s; }
        for (int j = 0; j < n; j++) for (int a = 0; a < K; a++) { double t = T[a + (size_t) j * K]; const double *q = Q + (size_t) a * n; double *col = Kw + (size_t) j * n; for (int i = 0; i < n; i++) col[i] -= q[i] * t; }
        /* (Kw Q): n x K */
        double *U = malloc((size_t) n * K * sizeof(double));
        for (int a = 0; a < K; a++) for (int i = 0; i < n; i++) U[i + (size_t) a * n] = 0;
        for (int j = 0; j < n; j++) for (int a = 0; a < K; a++) { double qa = Q[j + (size_t) a * n]; const double *col = Kw + (size_t) j * n; double *u = U + (size_t) a * n; for (int i = 0; i < n; i++) u[i] += col[i] * qa; }
        for (int j = 0; j < n; j++) for (int a = 0; a < K; a++) { double qa = Q[j + (size_t) a * n]; const double *u = U + (size_t) a * n; double *col = Kw + (size_t) j * n; for (int i = 0; i < n; i++) col[i] -= u[i] * qa; }
        for (int k = 0; k < n; k++) for (int i = k + 1; i < n; i++) { double v = (Kw[i + (size_t) k * n] + Kw[k + (size_t) i * n]) / 2; Kw[i + (size_t) k * n] = Kw[k + (size_t) i * n] = v; }
        free(Q); free(dv); free(T); free(U);
    }
    dense_op op = {n, Kw};
    int rc = det_topeig(n, dense_apply, &op, l1, l2);
    free(rs); free(ws); free(sw); free(Kw);
    return rc;
}
/* eigtop_q: the eigenvalues of C^{1/2} (W + Lambda' B Lambda) C^{1/2}, here those of R' Q R with C = R R' (pivoted Cholesky;
 * the same nonzero eigenvalues as C Q) */
typedef struct { int M, rk; const double *R, *Q; double *tmp1, *tmp2; } q_op;
static void q_apply(int nb, const double *X, double *Y, void *v)
{
    const q_op *o = v; int M = o->M, rk = o->rk;
    for (int b = 0; b < nb; b++) {
        const double *x = X + (size_t) b * rk; double *y = Y + (size_t) b * rk;
        for (int i = 0; i < M; i++) { double s = 0; for (int k = 0; k < rk; k++) s += o->R[i + (size_t) k * M] * x[k]; o->tmp1[i] = s; }
        for (int i = 0; i < M; i++) o->tmp2[i] = 0;
        for (int j = 0; j < M; j++) { double t = o->tmp1[j]; const double *col = o->Q + (size_t) j * M; for (int i = 0; i < M; i++) o->tmp2[i] += col[i] * t; }
        for (int k = 0; k < rk; k++) { double s = 0; for (int i = 0; i < M; i++) s += o->R[i + (size_t) k * M] * o->tmp2[i]; y[k] = s; }
    }
}
int det_eigtop_q(const det_prep *P, double c, int m, double *l1, double *l2)
{
    double *rs, *ws; int *pj; int n = nystrom_nodes(P, m, &rs, &ws, &pj);
    int nj = P->nj, N = P->N, M = n + 1 + nj, d1 = P->gdeg + 1;
    double *pts = malloc(M * sizeof(double));
    memcpy(pts, rs, n * sizeof(double)); pts[n] = 1.0; for (int i = 0; i < nj; i++) pts[n + 1 + i] = P->jl[i];
    double *C = malloc((size_t) M * M * sizeof(double)), *Lam = calloc((size_t) N * M, sizeof(double));
    if (!C || !Lam) { free(rs); free(ws); free(pj); free(pts); free(C); free(Lam); return 1; }
    for (int j = 0; j < M; j++) for (int i = j; i < M; i++) { double v = oukernel(c, pts[i], pts[j], 0.0); C[i + (size_t) j * M] = C[j + (size_t) i * M] = v; }
    Lam[0 + (size_t) n * N] = 1;
    for (int i = 0; i < nj; i++) Lam[(1 + i) + (size_t) (n + 1 + i) * N] = 1;
    for (int i = 0; i < P->p; i++) for (int q = 0; q < n; q++) Lam[(1 + nj + i) + (size_t) q * N] = ws[q] * evalpoly(rs[q] - P->off[pj[q]], P->g + ((size_t) pj[q] * P->p + i) * d1, d1);
    /* Q = Lam' B Lam + diag(ws) */
    double *BL = malloc((size_t) N * M * sizeof(double)), *Q = malloc((size_t) M * M * sizeof(double));
    for (int j = 0; j < M; j++) for (int i = 0; i < N; i++) { double s = 0; for (int k = 0; k < N; k++) s += P->B[i + k * N] * Lam[k + (size_t) j * N]; BL[i + (size_t) j * N] = s; }
    for (int j = 0; j < M; j++) for (int i = 0; i < M; i++) { double s = 0; for (int k = 0; k < N; k++) s += Lam[k + (size_t) i * N] * BL[k + (size_t) j * N]; Q[i + (size_t) j * M] = s; }
    for (int q = 0; q < n; q++) Q[q + (size_t) q * M] += ws[q];
    for (int j = 0; j < M; j++) for (int i = j + 1; i < M; i++) { double v = (Q[i + (size_t) j * M] + Q[j + (size_t) i * M]) / 2; Q[i + (size_t) j * M] = Q[j + (size_t) i * M] = v; }
    /* pivoted Cholesky C = R R' (R M x rk), stopped where the remaining diagonal is below 1e-14 of the largest */
    double *R = calloc((size_t) M * M, sizeof(double)), *dg = malloc(M * sizeof(double)); int *pv = malloc(M * sizeof(int));
    double dmx = 0; for (int i = 0; i < M; i++) { dg[i] = C[i + (size_t) i * M]; pv[i] = i; dmx = dmax(dmx, dg[i]); }
    int rk = 0;
    for (int k = 0; k < M; k++) {
        int b = k; for (int i = k + 1; i < M; i++) if (dg[pv[i]] > dg[pv[b]]) b = i;
        int tmp = pv[k]; pv[k] = pv[b]; pv[b] = tmp;
        int piv = pv[k]; double d = dg[piv];
        if (!(d > 1e-14 * dmx)) break;
        double sd = sqrt(d);
        double *col = R + (size_t) k * M;
        for (int i = 0; i < M; i++) {
            double s = C[i + (size_t) piv * M];
            for (int l = 0; l < k; l++) s -= R[i + (size_t) l * M] * R[piv + (size_t) l * M];
            col[i] = s / sd;
        }
        col[piv] = sd;
        for (int i = k + 1; i < M; i++) { int ii = pv[i]; dg[ii] -= col[ii] * col[ii]; }
        dg[piv] = 0;
        for (int i = 0; i < k + 1; i++) if (pv[i] != piv) col[pv[i]] = 0;   /* the rows already eliminated */
        rk++;
    }
    double *t1 = malloc(M * sizeof(double)), *t2 = malloc(M * sizeof(double));
    q_op op = {M, rk, R, Q, t1, t2};
    int rc = det_topeig(rk, q_apply, &op, l1, l2);
    free(rs); free(ws); free(pj); free(pts); free(C); free(Lam); free(BL); free(Q); free(R); free(dg); free(pv); free(t1); free(t2);
    return rc;
}

/* D_V(t) at real t: real(exp(log D_V)) */
typedef struct { const det_model *Mo; double c; const double *t; double *f; det_status st; } fgrid_ctx;
static double fval(det_work *W, double c, double t, det_status *st)
{
    cdouble th, ld, mu[64], *m = W->r <= 64 ? mu : malloc(W->r * sizeof(cdouble));
    if (det_mgfparts(W, c, CPX(t, 0), &th, &ld, m, NULL) && st) det_fail(st, "the eigenvalues of the reciprocal roots did not converge at t = %g", t);
    if (m != mu) free(m);
    return creal(cexp(ld));
}
static void fgrid_body(int j, void *v)
{
    fgrid_ctx *o = v;
    char buf[DET_WORK_STACK]; det_work W;
    if (det_work_init(&W, o->Mo, buf, sizeof buf)) { det_fail(&o->st, "out of memory"); return; }
    o->f[j] = fval(&W, o->c, o->t[j], &o->st);
    det_work_free(&W);
}
int det_tstar(const det_model *Mo, double c, double *tsout, int *mult, char *err, size_t errlen)
{
    if (Mo->start == 2 && c >= 0) { snprintf(err, errlen, "the stationary start needs c < 0"); return 1; }
    const det_prep *P = Mo->P;
    double l1, l2;
    /* (eigtop projects J orthogonally off the span of the design; the V of GLS detrending and of the LM statistic (kind 3)
     * is no such norm, and eigtop_q takes its quadratic form: with eigtop the LM statistic's t_* came out up to 2.3 times
     * too large, above the unit root beyond twice the edge, where the path then failed) */
    int rc = isnan(P->cbar) && P->kind != 3 ? det_eigtop(P, c, 16, Mo->start == 2, &l1, &l2) : det_eigtop_q(P, c, 48, &l1, &l2);
    if (rc) { snprintf(err, errlen, "t_*: the Nystrom estimate failed"); return 1; }
    double te = 1 / (2 * l1), grid[81], vals[81];
    for (int i = 0; i < 81; i++) grid[i] = 0.98 * te + (1.02 * te - 0.98 * te) * i / 80;
    fgrid_ctx g = {Mo, c, grid, vals, {0, ""}};
    dfl_parfor(81, fgrid_body, &g);
    if (det_failed(&g.st)) { snprintf(err, errlen, "%s", g.st.msg); return 1; }
    char buf[DET_WORK_STACK]; det_work W;
    if (det_work_init(&W, Mo, buf, sizeof buf)) { snprintf(err, errlen, "out of memory"); return 1; }
    det_status st = {0, ""};
#define F_(tt) fval(&W, c, (tt), &st)
    double result = NAN; int mres = 1;
    int k = -1; for (int i = 0; i < 81; i++) if (vals[i] <= 0) { k = i; break; }
    double a = grid[0], b = grid[80];
    if (k > 0) {
        double lo = grid[k - 1], hi = grid[k];
        for (int it = 0; it < 100; it++) { double mid = (lo + hi) / 2; if (F_(mid) > 0) lo = mid; else hi = mid; if (hi - lo < 1e-13 * hi) break; }
        result = (lo + hi) / 2; mres = fabs(l1 - l2) < 1e-4 * l1 ? 2 : 1;
        goto done;
    }
    {
        int found = 0;
        /* far from the unit root: the scan in omega from the branch point, t = (c^2 + omega^2)/2 */
        if (c < 0 && F_(c * c / 2 + 5e-7) > 0) {
#define TW(om) ((c * c + (om) * (om)) / 2)
            double fprev = F_(TW(1e-3)), fmx = fabs(fprev), fpp = INFINITY;
            for (int j = 1; j <= 4000; j++) {
                double om = 0.05 * j, fj = F_(TW(om));
                if (fj <= 0) {
                    double lo = TW(om - 0.05), hi = TW(om);
                    for (int it = 0; it < 100; it++) { double mid = (lo + hi) / 2; if (F_(mid) > 0) lo = mid; else hi = mid; if (hi - lo < 1e-13 * hi) break; }
                    result = (lo + hi) / 2; mres = 1; goto done;
                }
                if (fprev < fpp && fprev <= fj && fprev < 1e-6 * fmx) { a = TW(dmax(om - 0.1, 1e-3)); b = TW(om); found = 1; break; }
                fmx = dmax(fmx, fj); fpp = fprev; fprev = fj;
            }
#undef TW
            if (!found) { a = grid[0]; b = grid[80]; }
        }
        if (!found) {
            int km = 0; for (int i = 1; i < 81; i++) if (vals[i] < vals[km]) km = i;
            a = grid[imax(km - 1, 0)]; b = grid[imin(km + 1, 80)];
        }
        double gg = (sqrt(5.0) - 1) / 2, x1 = b - gg * (b - a), x2 = a + gg * (b - a), f1 = F_(x1), f2 = F_(x2);
        for (int it = 0; it < 80; it++) {
            if (f1 < f2) { b = x2; x2 = x1; f2 = f1; x1 = b - gg * (b - a); f1 = F_(x1); }
            else { a = x1; x1 = x2; f1 = f2; x2 = a + gg * (b - a); f2 = F_(x2); }
        }
        result = (a + b) / 2; mres = 2;
    }
done:
#undef F_
    det_work_free(&W);
    if (det_failed(&st)) { snprintf(err, errlen, "%s", st.msg); return 1; }
    *tsout = result; if (mult) *mult = mres;
    return 0;
}
