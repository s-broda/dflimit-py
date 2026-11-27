/* dfl_det_design.c: the deterministic terms of the general engine and their preparation (a port of DFDet.jl): the
 * designs (trend, perron, fdesign, ltrend, design_from_matrix, detrend_first), their Gram matrices, the prepared design
 * of the one-step regression and of GLS detrending (prep, prep_gls: the orthonormal basis of the span of the regressors
 * and their derivatives, the matrices A and B of the quadratic forms of numerator and denominator in the Gaussian
 * vector ell), the model with its start (model, fixed_shift), and the cache of prepared designs. */
#include "dfl_det_int.h"

/* ---------------------------------------------------------------------------------------------------------------- */
void dfl_design_free(dfl_design *D) { if (!D) return; free(D->br); free(D->C); D->br = NULL; D->C = NULL; D->K = D->np = 0; }
static int design_alloc(dfl_design *D, int K, int np, int deg)
{
    D->K = K; D->np = np; D->deg = deg; D->loc = 0; D->first = 0;
    D->br = calloc(np + 1, sizeof(double)); D->C = calloc((size_t) np * (deg + 1) * imax(K, 1), sizeof(double));
    return !D->br || !D->C;
}
int dfl_design_copy(const dfl_design *D, dfl_design *out)
{
    if (design_alloc(out, D->K, D->np, D->deg)) return 1;
    memcpy(out->br, D->br, (D->np + 1) * sizeof(double));
    memcpy(out->C, D->C, (size_t) D->np * (D->deg + 1) * D->K * sizeof(double));
    out->loc = D->loc; out->first = D->first;
    return 0;
}
static inline const double *dcol(const dfl_design *D, int j, int i) { return D->C + ((size_t) j * D->K + i) * (D->deg + 1); }
static inline double poff(const dfl_design *D, int j) { return D->loc ? D->br[j] : 0.0; }
static int piece(const dfl_design *D, double r) { int j = 0; while (j < D->np - 1 && r > D->br[j + 1]) j++; return j; }
void det_dval(const dfl_design *D, double r, double *out)
{
    int j = piece(D, r);
    for (int i = 0; i < D->K; i++) out[i] = evalpoly(r - poff(D, j), dcol(D, j, i), D->deg + 1);
}
void dfl_design_value(const dfl_design *D, double r, double *out) { det_dval(D, r, out); }
void dfl_design_regmat(const dfl_design *D, int n, double *X)
{
    double *v = malloc((D->K + 1) * sizeof(double));
    for (int t = 1; t <= n; t++) { det_dval(D, (double) t / n, v); for (int i = 0; i < D->K; i++) X[(t - 1) + (size_t) i * n] = v[i]; }
    free(v);
}

int dfl_design_trend(int p, dfl_design *D, char *err, size_t errlen)
{
    if (p < -1) { snprintf(err, errlen, "trend: p >= -1"); return 1; }
    int K = p + 1, deg = p > 0 ? p : 0;
    if (design_alloc(D, K, 1, deg)) { snprintf(err, errlen, "out of memory"); return 1; }
    D->br[0] = 0; D->br[1] = 1;
    for (int i = 0; i < K; i++) D->C[i * (deg + 1) + i] = 1.0;
    return 0;
}

int dfl_design_perron(char model, int q, const double *lam0, dfl_design *D, char *err, size_t errlen)
{
    if (model >= 'a' && model <= 'c') model += 'A' - 'a';
    if (model != 'A' && model != 'B' && model != 'C') { snprintf(err, errlen, "perron: model A, B or C"); return 1; }
    if (q < 1) { snprintf(err, errlen, "perron: 0 < lambda_1 < ... < lambda_q < 1"); return 1; }
    double *lam = malloc(q * sizeof(double));
    memcpy(lam, lam0, q * sizeof(double));
    for (int i = 1; i < q; i++) { double v = lam[i]; int j = i - 1; while (j >= 0 && lam[j] > v) { lam[j + 1] = lam[j]; j--; } lam[j + 1] = v; }
    for (int i = 0; i < q; i++) if (!(lam[i] > 0 && lam[i] < 1) || (i > 0 && !(lam[i] > lam[i - 1]))) { free(lam); snprintf(err, errlen, "perron: 0 < lambda_1 < ... < lambda_q < 1"); return 1; }
    int K = 2 + (model == 'C' ? 2 * q : q);
    if (design_alloc(D, K, q + 1, 1)) { free(lam); snprintf(err, errlen, "out of memory"); return 1; }
    D->br[0] = 0; for (int i = 0; i < q; i++) D->br[i + 1] = lam[i]; D->br[q + 1] = 1;
    for (int j = 0; j <= q; j++) {                       /* piece j: after the first j breaks */
        double *Cj = D->C + (size_t) j * 2 * K;
        Cj[0 + 0 * 2] = 1; Cj[1 + 1 * 2] = 1;
        int col = 2;
        for (int i = 0; i < q; i++) {
            int on = i <= j - 1;
            if (model == 'A' || model == 'C') { if (on) Cj[0 + col * 2] = 1.0; col++; }
            if (model == 'B' || model == 'C') { if (on) { Cj[0 + col * 2] = -lam[i]; Cj[1 + col * 2] = 1.0; } col++; }
        }
    }
    free(lam);
    return 0;
}

int dfl_design_detrend_first(const dfl_design *D, dfl_design *out, char *err, size_t errlen)
{
    if (dfl_design_copy(D, out)) { snprintf(err, errlen, "out of memory"); return 1; }
    out->first = 1;
    return 0;
}

/* ---------- trend functions beyond piecewise polynomials: Chebyshev interpolants, local Taylor coefficients ---------- */
static double chebval(const double *c, int n, double x)
{
    double b1 = 0, b2 = 0;
    for (int k = n - 1; k >= 1; k--) { double t = 2 * x * b1 - b2 + c[k]; b2 = b1; b1 = t; }
    return x * b1 - b2 + c[0];
}
/* the derivative of a Chebyshev series of length n (into d, length max(n - 1, 1)) */
static int chebder(const double *c, int n, double *d)
{
    int m = n - 1;
    if (m <= 0) { d[0] = 0; return 1; }
    double *t = calloc(m + 2, sizeof(double));
    for (int k = m; k >= 1; k--) t[k - 1] = (k + 1 <= m ? t[k + 1] : 0.0) + 2 * k * c[k];
    t[0] /= 2;
    memcpy(d, t, m * sizeof(double)); free(t);
    return m;
}
static void cheb_taylor(const double *c, int nc, double a, double b, double r0, int n, double *out)
{
    double x0 = (2 * r0 - a - b) / (b - a), sc = 2 / (b - a), f = 1;
    double *cd = malloc((nc + 1) * sizeof(double)), *tmp = malloc((nc + 1) * sizeof(double));
    memcpy(cd, c, nc * sizeof(double)); int len = nc;
    for (int k = 0; k <= n; k++) {
        out[k] = chebval(cd, len, x0) * f;
        len = chebder(cd, len, tmp); memcpy(cd, tmp, len * sizeof(double));
        f *= sc / (k + 1);
    }
    free(cd); free(tmp);
}
static void chebcoef(void (*f)(double, double *, void *), void *ctx, double a, double b, int N, int K, double *Cc, double *V)
{
    for (int j = 0; j < N; j++) f((a + b) / 2 + (b - a) / 2 * cos(M_PI * (j + 0.5) / N), V + (size_t) j * K, ctx);
    for (int k = 0; k < N; k++) for (int i = 0; i < K; i++) {
        double acc = 0;
        for (int j = 1; j <= N; j++) acc += V[i + (size_t) (j - 1) * K] * cos(k * M_PI * (j - 0.5) / N);
        Cc[k + (size_t) i * N] = 2 * acc / N;
    }
    for (int i = 0; i < K; i++) Cc[(size_t) i * N] /= 2;
}

int dfl_design_fdesign(void (*f)(double r, double *out, void *ctx), void *ctx, int K, int nb, const double *breaks, double tol,
                       int maxdeg, int sub, dfl_design *D, char *err, size_t errlen)
{
    if (!(tol > 0)) tol = 1e-14;
    if (maxdeg <= 0) maxdeg = 96;
    if (K < 1) { snprintf(err, errlen, "fdesign: K >= 1"); return 1; }
    if (nb < 0) { snprintf(err, errlen, "fdesign: nb >= 0"); return 1; }
    int nk = nb + 2;
    double *bk = malloc(nk * sizeof(double));
    bk[0] = 0; for (int i = 0; i < nb; i++) bk[i + 1] = breaks[i]; bk[nk - 1] = 1;
    for (int i = 2; i < nk - 1; i++) { double v = bk[i]; int j = i - 1; while (j >= 1 && bk[j] > v) { bk[j + 1] = bk[j]; j--; } bk[j + 1] = v; }
    for (int i = 0; i < nk - 1; i++) if (!(bk[i + 1] > bk[i])) { free(bk); snprintf(err, errlen, "fdesign: breaks must lie strictly inside (0, 1) and be distinct"); return 1; }
    int npc = nk - 1;
    double **cs = calloc(npc, sizeof(double *)); int *degs = calloc(npc, sizeof(int));
    int Dmax = 0, rc = 0;
    for (int j = 0; j < npc && !rc; j++) {
        double a = bk[j], b = bk[j + 1];
        int N = 16, deg = 0;
        double *Cc = NULL;
        for (;;) {
            free(Cc);
            Cc = malloc((size_t) N * K * sizeof(double)); double *V = malloc((size_t) N * K * sizeof(double));
            chebcoef(f, ctx, a, b, N, K, Cc, V); free(V);
            double mx = 0; for (int i = 0; i < N * K; i++) mx = dmax(mx, fabs(Cc[i]));
            int last = 0;
            if (mx != 0) for (int k = 1; k <= N; k++) for (int i = 0; i < K; i++) if (fabs(Cc[(k - 1) + (size_t) i * N]) > tol * mx) last = k;
            if (last <= N - 6 || N > maxdeg) { deg = last - 1; break; }
            N *= 2;
        }
        if (N > maxdeg && deg > maxdeg - 6) { snprintf(err, errlen, "fdesign: on piece [%g, %g] the Chebyshev coefficients do not fall below %g by degree %d; the regressors must be smooth between the breaks", a, b, tol, maxdeg); rc = 1; }
        degs[j] = deg;
        cs[j] = malloc((size_t) (deg + 2) * K * sizeof(double));
        for (int i = 0; i < K; i++) for (int k = 0; k <= deg; k++) cs[j][k + (size_t) i * (deg + 1)] = Cc[k + (size_t) i * N];
        free(Cc);
        if (deg > Dmax) Dmax = deg;
    }
    if (!rc) {
        int ntot = 0;
        for (int j = 0; j < npc; j++) ntot += sub > 0 ? sub : imax(1, (int) ceil(degs[j] / 4.0));
        if (design_alloc(D, K, ntot, Dmax)) { snprintf(err, errlen, "out of memory"); rc = 1; }
        else {
            D->loc = 1;
            int jj = 0;
            for (int j = 0; j < npc; j++) {
                double a = bk[j], b = bk[j + 1]; int deg = degs[j], m = sub > 0 ? sub : imax(1, (int) ceil(deg / 4.0));
                for (int q = 0; q < m; q++, jj++) {
                    double r0 = a + (b - a) * q / m;
                    D->br[jj] = r0;
                    for (int i = 0; i < K; i++) if (deg >= 0) cheb_taylor(cs[j] + (size_t) i * (deg + 1), deg + 1, a, b, r0, deg, D->C + ((size_t) jj * K + i) * (Dmax + 1));
                }
            }
            D->br[ntot] = 1.0;
        }
    }
    for (int j = 0; j < npc; j++) free(cs[j]);
    free(cs); free(degs); free(bk);
    return rc;
}

static void legendre_f(double r, double *out, void *ctx)
{
    int p = *(int *) ctx; double x = 2 * r - 1, P0 = 1, P1 = x;
    out[0] = 1;
    if (p >= 1) out[1] = sqrt(3.0) * x;
    for (int k = 1; k <= p - 1; k++) { double P2 = ((2 * k + 1) * x * P1 - k * P0) / (k + 1); P0 = P1; P1 = P2; out[k + 1] = sqrt(2.0 * (k + 1) + 1) * P2; }
}
int dfl_design_ltrend(int p, dfl_design *D, char *err, size_t errlen)
{
    if (p < 0) { snprintf(err, errlen, "ltrend: p >= 0"); return 1; }
    return dfl_design_fdesign(legendre_f, &p, p + 1, 0, NULL, 1e-14, 96, 0, D, err, errlen);
}

/* ---------- the design from a regressor matrix ---------- */
/* the least squares fit of y (n points) by Chebyshev polynomials of degree deg at the equispaced points of [-1, 1]: the
 * coefficients in c (deg + 1) and the largest residual */
static double pfit(const double *y, int n, int deg, double *c)
{
    int m = deg + 1;
    double *V = malloc((size_t) n * m * sizeof(double)), *V0 = malloc((size_t) n * m * sizeof(double));
    for (int i = 0; i < n; i++) {
        double x = n == 1 ? -1.0 : -1.0 + 2.0 * i / (n - 1);
        double ac = acos(dclamp(x, -1, 1));
        for (int j = 0; j < m; j++) V[i + (size_t) j * n] = cos(j * ac);
    }
    memcpy(V0, V, (size_t) n * m * sizeof(double));
    det_lsq(n, m, V, y, c);
    double res = 0;
    for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < m; j++) s += V0[i + (size_t) j * n] * c[j]; res = dmax(res, fabs(s - y[i])); }
    free(V); free(V0);
    return res;
}
typedef struct { int n, K, nb; const int *bk; const double *lams; double **coefs; int *ncoef; } fm_ctx;
static void fm_f(double r, double *out, void *v)
{
    const fm_ctx *c = v;
    int j = 0; while (j < c->nb && r > c->lams[j]) j++;
    int t1 = c->bk[j] + 1, t2 = c->bk[j + 1];
    double x = t2 == t1 ? 0.0 : (2 * (c->n * r) - t1 - t2) / (t2 - t1);
    for (int k = 0; k < c->K; k++) out[k] = chebval(c->coefs[j * c->K + k], c->ncoef[j * c->K + k], x);
}
static int cmp_int(const void *a, const void *b) { int x = *(const int *) a, y = *(const int *) b; return (x > y) - (x < y); }
int dfl_design_from_matrix(int n, int K0, const double *X, int nb0, const int *TB0, double tol, int maxdeg, int minlen, dfl_design *D,
                           char *err, size_t errlen)
{
    det_warn_clear(); dfl_warning_clear();          /* (its warning, dfl_det_warning, is that of this call only) */
    if (!(tol > 0)) tol = 1e-8;
    if (maxdeg <= 0) maxdeg = 40;
    if (minlen <= 0) minlen = 10;
    if (n < 2 || K0 < 1) { snprintf(err, errlen, "design_from_matrix: X must have at least two rows and one column"); return 1; }
    double *Xs = malloc((size_t) n * K0 * sizeof(double));
    memcpy(Xs, X, (size_t) n * K0 * sizeof(double));
    for (int k = 0; k < K0; k++) {
        double m = 0; for (int t = 0; t < n; t++) m = dmax(m, fabs(Xs[t + (size_t) k * n]));
        if (m > 0) for (int t = 0; t < n; t++) Xs[t + (size_t) k * n] /= m;
    }
    /* dependent columns: QR with column pivoting */
    double *Q = malloc((size_t) n * K0 * sizeof(double)), *rd = malloc(K0 * sizeof(double)); int *perm = malloc(K0 * sizeof(int));
    memcpy(Q, Xs, (size_t) n * K0 * sizeof(double));
    det_qrp(n, K0, Q, perm, rd);
    int rk = 0; for (int j = 0; j < imin(K0, n); j++) if (fabs(rd[j]) > 1e-9 * fabs(rd[0])) rk++;
    int *keep = malloc(K0 * sizeof(int)); for (int j = 0; j < rk; j++) keep[j] = perm[j];
    qsort(keep, rk, sizeof(int), cmp_int);
    if (rk < K0) {
        char lst[256] = ""; int w = 0;
        for (int j = 0; j < K0; j++) { int in = 0; for (int l = 0; l < rk; l++) if (keep[l] == j) in = 1; if (!in && w < 240) w += snprintf(lst + w, sizeof lst - w, "%s%d", w ? ", " : "", j + 1); }
        det_warn("design_from_matrix: columns %s are linearly dependent on the others and are dropped", lst);
    }
    int K = rk;
    double *Xk = malloc((size_t) n * K * sizeof(double));
    for (int k = 0; k < K; k++) memcpy(Xk + (size_t) k * n, Xs + (size_t) keep[k] * n, n * sizeof(double));
    free(Q); free(rd); free(perm); free(Xs);
    int rc = 0; int *TB = NULL, nb = 0;
    double cbuf[64];
    if (nb0 < 0) {
        /* break detection: the sixth differences, clusters, the largest T at which polynomials of degree <= 8 fit 30
         * observations on either side */
        const int q = 6, w = 30;
        char *flagged = calloc(n + 1, 1);
        double *d = malloc(n * sizeof(double));
        for (int k = 0; k < K; k++) {
            memcpy(d, Xk + (size_t) k * n, n * sizeof(double)); int len = n;
            for (int r = 0; r < q; r++) { for (int t = 0; t < len - 1; t++) d[t] = d[t + 1] - d[t]; len--; }
            for (int t = 0; t < len; t++) if (fabs(d[t]) > 1e-7) flagged[t + 1] = 1;      /* (1-based t) */
        }
        int *cs = malloc((n + 1) * sizeof(int)), *ce = malloc((n + 1) * sizeof(int)), ncl = 0;
        for (int t = 1; t <= n; t++) if (flagged[t]) {
            if (ncl == 0 || t - ce[ncl - 1] > q) { cs[ncl] = t; ce[ncl] = t; ncl++; } else ce[ncl - 1] = t;
        }
        free(flagged); free(d);
        if (ncl > imax(10, n / 50)) { snprintf(err, errlen, "design_from_matrix: %d clusters of jumps or kinks: some column is not piecewise smooth at the sample's resolution (seasonal dummies?); such columns do not affect the limit and should be left out", ncl); rc = 1; }
        TB = malloc((ncl + 1) * sizeof(int));
        for (int ic = 0; ic < ncl && !rc; ic++) {
            int best = 0;
            for (int Tc = imax(cs[ic] - 1, 1); Tc <= imin(ce[ic] + q + 1, n - 1); Tc++) {
                int lo = imax(1, Tc - w + 1), hi = imin(n, Tc + w);
                if (Tc - lo + 1 < 3 || hi - Tc < 3) continue;
                int ok = 1;
                for (int k = 0; k < K && ok; k++) {
                    const double *x = Xk + (size_t) k * n;
                    ok = dmin(pfit(x + lo - 1, Tc - lo + 1, imin(8, Tc - lo), cbuf), 1.0) <= 1e-9 && dmin(pfit(x + Tc, hi - Tc, imin(8, hi - Tc - 1), cbuf), 1.0) <= 1e-9;
                }
                if (ok) best = Tc;
            }
            if (best == 0) { snprintf(err, errlen, "design_from_matrix: no break position fits the cluster of changes at t = %d..%d (a pulse, or a column that is not piecewise smooth?); such columns do not affect the limit and should be left out", cs[ic], ce[ic]); rc = 1; }
            TB[nb++] = best;
        }
        free(cs); free(ce);
    } else {
        TB = malloc((nb0 + 1) * sizeof(int)); nb = nb0;
        for (int i = 0; i < nb; i++) TB[i] = TB0[i];
    }
    if (rc) { free(TB); free(Xk); free(keep); return 1; }
    qsort(TB, nb, sizeof(int), cmp_int);
    int *bk = malloc((nb + 2) * sizeof(int)); bk[0] = 0; for (int i = 0; i < nb; i++) bk[i + 1] = TB[i]; bk[nb + 1] = n;
    for (int i = 0; i < nb + 1; i++) if (bk[i + 1] - bk[i] < minlen) {
        char lst[200] = ""; int w = 0; for (int l = 0; l < nb && w < 180; l++) w += snprintf(lst + w, sizeof lst - w, "%s%d", l ? ", " : "", TB[l]);
        snprintf(err, errlen, "design_from_matrix: a piece shorter than %d observations (breaks at [%s]): a pulse or a feature at the sample's resolution; such columns do not affect the limit and should be left out", minlen, lst);
        free(TB); free(Xk); free(keep); free(bk); return 1;
    }
    double **coefs = calloc((size_t) (nb + 1) * K, sizeof(double *)); int *ncoef = calloc((size_t) (nb + 1) * K, sizeof(int));
    for (int j = 0; j < nb + 1 && !rc; j++) {
        int t1 = bk[j] + 1, t2 = bk[j + 1];
        for (int k = 0; k < K && !rc; k++) {
            const double *y = Xk + (size_t) k * n + (t1 - 1); int got = 0;
            double *c = malloc((imin(maxdeg, t2 - t1) + 2) * sizeof(double));
            for (int deg = 0; deg <= imin(maxdeg, t2 - t1); deg++) {
                double res = pfit(y, t2 - t1 + 1, deg, c);
                if (res <= tol) { coefs[j * K + k] = c; ncoef[j * K + k] = deg + 1; got = 1; break; }
            }
            if (!got) { free(c); snprintf(err, errlen, "design_from_matrix: column %d is not smooth on observations %d..%d at the sample's resolution (no Chebyshev fit of degree <= %d within %g); such columns do not affect the limit and should be left out", keep[k] + 1, t1, t2, maxdeg, tol); rc = 1; }
        }
    }
    if (!rc) {
        double *lams = malloc((nb + 1) * sizeof(double));
        for (int i = 0; i < nb; i++) lams[i] = (double) TB[i] / n;
        fm_ctx ctx = {n, K, nb, bk, lams, coefs, ncoef};
        rc = dfl_design_fdesign(fm_f, &ctx, K, nb, lams, 1e-14, 96, 0, D, err, errlen);
        free(lams);
    }
    for (int i = 0; i < (nb + 1) * K; i++) free(coefs[i]);
    free(coefs); free(ncoef); free(TB); free(Xk); free(keep); free(bk);
    return rc;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the Gram matrix of the columns of the np blocks F ((deg + 1) x n each) */
void det_gram(int np, int deg, int n, const double *F, const double *br, int loc, double *G)
{
    for (int i = 0; i < n * n; i++) G[i] = 0;
    if (n == 0) return;
    if (loc) {
        /* local coefficients: Gauss-Legendre with deg + 2 nodes per piece (exact for the products), on the values */
        const double *w, *z = det_gl(deg + 2, &w);
        int nq = deg + 2;
        double *V = malloc((size_t) nq * n * sizeof(double));
        for (int j = 0; j < np; j++) {
            double L = br[j + 1] - br[j];
            for (int q = 0; q < nq; q++) for (int a = 0; a < n; a++) V[q + (size_t) a * nq] = evalpoly(L * z[q], F + ((size_t) j * n + a) * (deg + 1), deg + 1);
            for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) {
                double s = 0; for (int q = 0; q < nq; q++) s += V[q + (size_t) a * nq] * (V[q + (size_t) b * nq] * (L * w[q]));
                G[a + b * n] += s;
            }
        }
        free(V);
        for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) { double m = (G[a + b * n] + G[b + a * n]) / 2; G[a + b * n] = G[b + a * n] = m; }
        return;
    }
    for (int j = 0; j < np; j++) for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) for (int k = 0; k <= deg; k++) for (int l = 0; l <= deg; l++) {
        const double *Fa = F + ((size_t) j * n + a) * (deg + 1), *Fb = F + ((size_t) j * n + b) * (deg + 1);
        G[a + b * n] += Fa[k] * Fb[l] * (pow(br[j + 1], k + l + 1) - pow(br[j], k + l + 1)) / (k + l + 1);
    }
}
/* the derivative of the coefficient columns (rows: powers), the last row zero */
static void dcoef(int deg, int n, const double *C, double *Dm)
{
    for (int i = 0; i < n; i++) { for (int k = 0; k < deg; k++) Dm[k + i * (deg + 1)] = (k + 1) * C[k + 1 + i * (deg + 1)]; Dm[deg + i * (deg + 1)] = 0; }
}
/* the jumps of the columns of the blocks Cs (np blocks of (deg + 1) x n) at the breaks of D: the points with
 * max |jump| > thr into jl (and the jumps, n each, into Dl); returns their number */
static int jumps(const dfl_design *D, int n, const double *Cs, double thr, double *jl, double *Dl)
{
    int nj = 0, d1 = D->deg + 1;
    for (int j = 0; j < D->np - 1; j++) {
        double lam = D->br[j + 1], mx = 0, *dl = Dl ? Dl + (size_t) nj * n : NULL, tmp[1];
        for (int i = 0; i < n; i++) {
            double v = evalpoly(lam - poff(D, j + 1), Cs + ((size_t) (j + 1) * n + i) * d1, d1) - evalpoly(lam - poff(D, j), Cs + ((size_t) j * n + i) * d1, d1);
            mx = dmax(mx, fabs(v));
            if (dl) dl[i] = v; else tmp[0] = v;
        }
        if (mx > thr) { if (jl) jl[nj] = lam; nj++; }
    }
    return nj;
}
/* the orthonormal basis of the span of the columns of the blocks F (np blocks of (deg + 1) x nF): p, g (np blocks of
 * (deg + 1) x p) and Phi (nF x p, F = Phi g on the span) */
int det_ortho_basis(const dfl_design *D, int nF, const double *F, int *pp, double **gout, double **Phiout)
{
    int d1 = D->deg + 1, np = D->np;
    double *Gf = malloc((size_t) nF * nF * sizeof(double)), *w = malloc(nF * sizeof(double)), *V = malloc((size_t) nF * nF * sizeof(double));
    det_gram(np, D->deg, nF, F, D->br, D->loc, Gf);
    det_symeig(nF, Gf, w, V);
    double mx = -INFINITY; for (int i = 0; i < nF; i++) mx = dmax(mx, w[i]);
    int p = 0; for (int i = 0; i < nF; i++) if (w[i] > 1e-11 * mx) p++;
    double *g = calloc((size_t) np * d1 * imax(p, 1), sizeof(double)), *Phi = calloc((size_t) nF * imax(p, 1), sizeof(double));
    int k = 0;
    for (int i = 0; i < nF; i++) {
        if (!(w[i] > 1e-11 * mx)) continue;
        double sl = sqrt(w[i]);
        for (int a = 0; a < nF; a++) Phi[a + (size_t) k * nF] = V[a + (size_t) i * nF] * sl;
        for (int j = 0; j < np; j++) for (int r = 0; r < d1; r++) {
            double s = 0; for (int a = 0; a < nF; a++) s += F[((size_t) j * nF + a) * d1 + r] * V[a + (size_t) i * nF];
            g[((size_t) j * p + k) * d1 + r] = s / sl;
        }
        k++;
    }
    free(Gf); free(w); free(V);
    *pp = p; *gout = g; *Phiout = Phi;
    return 0;
}
/* the eigen-decomposition of the symmetric N x N matrix A with the eigenvalues |.| > 1e-10 max kept: Dv, L, r, rp, rm */
int det_keep_eig(int N, const double *A, double **Dv, double **L, int *r, int *rp, int *rm)
{
    double *w = malloc(N * sizeof(double)), *V = malloc((size_t) N * N * sizeof(double));
    det_symeig(N, A, w, V);
    double mx = 0; for (int i = 0; i < N; i++) mx = dmax(mx, fabs(w[i]));
    int k = 0; for (int i = 0; i < N; i++) if (fabs(w[i]) > 1e-10 * mx) k++;
    *Dv = malloc(imax(k, 1) * sizeof(double)); *L = malloc((size_t) N * imax(k, 1) * sizeof(double));
    int j = 0; *rp = *rm = 0;
    for (int i = 0; i < N; i++) if (fabs(w[i]) > 1e-10 * mx) {
        (*Dv)[j] = w[i]; memcpy(*L + (size_t) j * N, V + (size_t) i * N, N * sizeof(double)); j++;
        if (w[i] > 0) (*rp)++; else if (w[i] < 0) (*rm)++;
    }
    *r = k;
    free(w); free(V);
    return 0;
}
void det_prep_free(det_prep *P)
{
    if (!P) return;
    free(P->br); free(P->C); free(P->g); free(P->jl); free(P->A); free(P->B); free(P->L); free(P->Dv); free(P->b); free(P->off);
    free(P);
}
det_prep *det_prep_new(const dfl_design *D)
{
    det_prep *P = calloc(1, sizeof(det_prep));
    P->K = D->K; P->Kd = D->K; P->np = D->np; P->deg = D->deg; P->loc = D->loc; P->first = D->first;
    P->br = malloc((D->np + 1) * sizeof(double)); memcpy(P->br, D->br, (D->np + 1) * sizeof(double));
    P->C = malloc((size_t) imax(D->np * (D->deg + 1) * D->K, 1) * sizeof(double));
    memcpy(P->C, D->C, (size_t) D->np * (D->deg + 1) * D->K * sizeof(double));
    P->off = malloc(D->np * sizeof(double)); for (int j = 0; j < D->np; j++) P->off[j] = poff(D, j);
    P->gdeg = D->deg; P->cbar = NAN;
    return P;
}
/* the prepared design with a rank-one numerator 1/2 J(1)^2 only (no deterministic terms after GLS detrending) */
void det_prep_rank1(det_prep *P, double a11)
{
    P->p = 0; P->nj = 0; P->N = 1; P->K = 0;
    P->g = calloc(1, sizeof(double)); P->jl = calloc(1, sizeof(double));
    P->A = malloc(sizeof(double)); P->A[0] = a11; P->B = calloc(1, sizeof(double));
    P->L = malloc(sizeof(double)); P->L[0] = 1; P->Dv = malloc(sizeof(double)); P->Dv[0] = a11;
    P->r = 1; P->rp = a11 > 0; P->rm = a11 < 0; P->b = calloc(1, sizeof(double));
}

/* prep: the one-step regression (or, with D->first, the statistic that detrends first) */
static int prep_ols(const dfl_design *D, det_prep **out, char *err, size_t errlen)
{
    int K = D->K, np = D->np, d1 = D->deg + 1;
    det_prep *P = det_prep_new(D);
    double *jl = malloc(np * sizeof(double)), *Dl = malloc((size_t) np * imax(K, 1) * sizeof(double));
    int nj = K > 0 ? jumps(D, K, D->C, 1e-12, jl, Dl) : 0;
    if (K == 0) {
        det_prep_rank1(P, 0.5);
        free(jl); free(Dl); *out = P; return 0;
    }
    double *F = malloc((size_t) np * d1 * 2 * K * sizeof(double));
    for (int j = 0; j < np; j++) {
        memcpy(F + (size_t) j * d1 * 2 * K, D->C + (size_t) j * d1 * K, (size_t) d1 * K * sizeof(double));
        dcoef(D->deg, K, D->C + (size_t) j * d1 * K, F + (size_t) j * d1 * 2 * K + (size_t) d1 * K);
    }
    double *Gf = malloc((size_t) 4 * K * K * sizeof(double));
    det_gram(np, D->deg, 2 * K, F, D->br, D->loc, Gf);
    int p; double *g, *Phi;
    det_ortho_basis(D, 2 * K, F, &p, &g, &Phi);
    /* Phid = Phi[0:K, :], Phin = Phi[K:2K, :]; G = Gf[0:K, 0:K]; Pm = Phid' G^{-1} (p x K) */
    double *G = malloc((size_t) K * K * sizeof(double)), *X = malloc((size_t) K * imax(p, 1) * sizeof(double));
    for (int a = 0; a < K; a++) for (int b = 0; b < K; b++) G[a + b * K] = Gf[a + (size_t) b * 2 * K];
    for (int a = 0; a < K; a++) for (int k = 0; k < p; k++) X[a + k * K] = Phi[a + (size_t) k * 2 * K];
    double gm = 0, dev = 0;
    for (int a = 0; a < K; a++) for (int b = 0; b < K; b++) {
        gm = dmax(gm, fabs(G[a + b * K]));
        double s = 0; for (int k = 0; k < p; k++) s += Phi[a + (size_t) k * 2 * K] * Phi[b + (size_t) k * 2 * K];
        dev = dmax(dev, fabs(s - G[a + b * K]));
    }
    if (!(dev < 1e-10 * gm)) {
        snprintf(err, errlen, "prep: basis does not reproduce G");
        free(F); free(Gf); free(g); free(Phi); free(G); free(X); free(jl); free(Dl); det_prep_free(P); return 1;
    }
    det_rsolve(K, G, p, X);                   /* X = G^{-1} Phid; Pm = X' */
#define PM(k, a) X[(a) + (k) * K]
    int N = 1 + nj + p;
    P->p = p; P->nj = nj; P->N = N; P->g = g;
    P->jl = malloc(imax(nj, 1) * sizeof(double)); memcpy(P->jl, jl, nj * sizeof(double));
    double *A = calloc((size_t) N * N, sizeof(double)), *B = calloc((size_t) N * N, sizeof(double)), *bv = calloc(N, sizeof(double));
    double *d0 = malloc(K * sizeof(double)), *dd1 = malloc(K * sizeof(double));
    det_dval(D, 0.0, d0); det_dval(D, 1.0, dd1);
    A[0] = 0.5;
    int iz = 1 + nj;
    for (int k = 0; k < p; k++) {
        double s0 = 0, s1 = 0; for (int a = 0; a < K; a++) { s0 += PM(k, a) * d0[a]; s1 += PM(k, a) * dd1[a]; }
        bv[iz + k] = s0;
        A[(iz + k) + 0 * N] = A[0 + (iz + k) * N] = -0.5 * s1;
    }
    /* Pm Phin (p x p), Pm Phid (p x p) */
    double *Mn = malloc((size_t) p * p * sizeof(double)), *Md = malloc((size_t) p * p * sizeof(double)), *M = malloc((size_t) p * p * sizeof(double));
    for (int k = 0; k < p; k++) for (int l = 0; l < p; l++) {
        double sn = 0, sd = 0; for (int a = 0; a < K; a++) { sn += PM(k, a) * Phi[(K + a) + (size_t) l * 2 * K]; sd += PM(k, a) * Phi[a + (size_t) l * 2 * K]; }
        Mn[k + l * p] = sn; Md[k + l * p] = sd;
    }
    if (D->first) {
        for (int k = 0; k < p; k++) for (int l = 0; l < p; l++) { double s = 0; for (int m = 0; m < p; m++) s += Mn[k + m * p] * Md[m + l * p]; M[k + l * p] = s; }
    } else memcpy(M, Mn, (size_t) p * p * sizeof(double));
    for (int k = 0; k < p; k++) for (int l = 0; l < p; l++) A[(iz + k) + (iz + l) * N] = 0.5 * (M[k + l * p] + M[l + k * p]);
    for (int i = 0; i < nj; i++) for (int k = 0; k < p; k++) {
        double s = 0; for (int a = 0; a < K; a++) s += PM(k, a) * Dl[(size_t) i * K + a];
        A[(iz + k) + (1 + i) * N] = A[(1 + i) + (iz + k) * N] = 0.5 * s;
    }
    for (int k = 0; k < p; k++) for (int l = 0; l < p; l++) B[(iz + k) + (iz + l) * N] = -(Md[k + l * p] + Md[l + k * p]) / 2;
#undef PM
    P->A = A; P->B = B; P->b = bv;
    det_keep_eig(N, A, &P->Dv, &P->L, &P->r, &P->rp, &P->rm);
    free(F); free(Gf); free(Phi); free(G); free(X); free(jl); free(Dl); free(d0); free(dd1); free(Mn); free(Md); free(M);
    *out = P;
    return 0;
}

static const char *FIRST_GLS_MSG = "GLS detrending (gls = cbar) detrends first already: use the design itself, not detrend_first(D), with gls";
/* the directions b with d(0)' b = 0 and Delta_i' b = 0 (GLS detrending, the point-optimal and the LM statistic): k and Z */
int det_gls_directions(const dfl_design *D, double **Zout)
{
    int K = D->K;
    double *Dl = malloc((size_t) D->np * imax(K, 1) * sizeof(double));
    int nj = K > 0 ? jumps(D, K, D->C, 1e-12, NULL, Dl) : 0;
    int m = 1 + nj;
    double *Mr = malloc((size_t) m * imax(K, 1) * sizeof(double)), *d0 = malloc(imax(K, 1) * sizeof(double));
    det_dval(D, 0.0, d0);
    double mx = 0;
    for (int a = 0; a < K; a++) { Mr[0 + a * m] = d0[a]; for (int i = 0; i < nj; i++) Mr[(1 + i) + a * m] = Dl[(size_t) i * K + a]; }
    for (int i = 0; i < m * K; i++) mx = dmax(mx, fabs(Mr[i]));
    double *Z = malloc((size_t) imax(K * K, 1) * sizeof(double));
    int k = K > 0 ? det_nullspace(m, K, Mr, 1e-10 * dmax(1.0, mx), Z) : 0;
    free(Dl); free(Mr); free(d0);
    *Zout = Z;
    return k;
}
/* the blocks of delta = Z' d (np blocks of (deg + 1) x k), and its first and second derivatives */
void det_delta_blocks(const dfl_design *D, int k, const double *Z, double *Cd, double *Cd1, double *Cd2)
{
    int d1 = D->deg + 1, K = D->K;
    for (int j = 0; j < D->np; j++) {
        for (int i = 0; i < k; i++) for (int r = 0; r < d1; r++) {
            double s = 0; for (int a = 0; a < K; a++) s += D->C[((size_t) j * K + a) * d1 + r] * Z[a + (size_t) i * K];
            Cd[((size_t) j * k + i) * d1 + r] = s;
        }
        dcoef(D->deg, k, Cd + (size_t) j * d1 * k, Cd1 + (size_t) j * d1 * k);
        dcoef(D->deg, k, Cd1 + (size_t) j * d1 * k, Cd2 + (size_t) j * d1 * k);
    }
}
void det_evblk(const dfl_design *D, int k, const double *Cs, int j, double r, double *out)
{
    int d1 = D->deg + 1;
    for (int i = 0; i < k; i++) out[i] = evalpoly(r - poff(D, j), Cs + ((size_t) j * k + i) * d1, d1);
}

/* prep_gls: the design GLS-detrended at cbar < 0 */
static int prep_gls(const dfl_design *D, double cbar, det_prep **out, char *err, size_t errlen)
{
    if (!(cbar < 0 && isfinite(cbar))) { snprintf(err, errlen, "GLS detrending needs cbar < 0 (a finite number)"); return 1; }
    if (D->first) { snprintf(err, errlen, "%s", FIRST_GLS_MSG); return 1; }
    int np = D->np, d1 = D->deg + 1;
    det_prep *P = det_prep_new(D); P->cbar = cbar; P->kind = 1;
    double *Z; int k = det_gls_directions(D, &Z);
    if (k == 0) { free(Z); det_prep_rank1(P, 0.5); *out = P; return 0; }
    double *Cd = malloc((size_t) np * d1 * k * sizeof(double)), *Cd1 = malloc((size_t) np * d1 * k * sizeof(double)), *Cd2 = malloc((size_t) np * d1 * k * sizeof(double));
    det_delta_blocks(D, k, Z, Cd, Cd1, Cd2);
    double *T = malloc((size_t) np * d1 * k * sizeof(double));
    for (size_t i = 0; i < (size_t) np * d1 * k; i++) T[i] = Cd1[i] - cbar * Cd[i];
    double *Psi = malloc((size_t) k * k * sizeof(double)), *Gd = malloc((size_t) k * k * sizeof(double));
    det_gram(np, D->deg, k, T, D->br, D->loc, Psi); det_gram(np, D->deg, k, Cd, D->br, D->loc, Gd);
    double *psi1 = malloc(k * sizeof(double)), *delta1 = malloc(k * sizeof(double)), *e1 = malloc(k * sizeof(double)), *e2 = malloc(k * sizeof(double));
    det_evblk(D, k, Cd1, np - 1, 1.0, e1); det_evblk(D, k, Cd, np - 1, 1.0, delta1);
    for (int i = 0; i < k; i++) psi1[i] = e1[i] - cbar * delta1[i];
    /* the jumps of delta' (slope breaks); delta itself is continuous */
    double *jl = malloc(np * sizeof(double)), *vs = malloc((size_t) np * k * sizeof(double));
    int nj = 0, rc = 0;
    for (int j = 0; j < np - 1; j++) {
        double lam = D->br[j + 1], mxd = 0, mxe = 0, mxs = 0;
        det_evblk(D, k, Cd, j + 1, lam, e1); det_evblk(D, k, Cd, j, lam, e2);
        for (int i = 0; i < k; i++) { mxd = dmax(mxd, fabs(e1[i] - e2[i])); mxe = dmax(mxe, fabs(e2[i])); }
        if (!(mxd < 1e-9 * dmax(1.0, mxe))) { snprintf(err, errlen, "prep_gls: delta not continuous at %g", lam); rc = 1; break; }
        det_evblk(D, k, Cd1, j + 1, lam, e1); det_evblk(D, k, Cd1, j, lam, e2);
        for (int i = 0; i < k; i++) { vs[(size_t) nj * k + i] = e1[i] - e2[i]; mxs = dmax(mxs, fabs(e1[i] - e2[i])); }
        if (mxs > 1e-12) jl[nj++] = lam;
    }
    double *F = malloc((size_t) np * d1 * 2 * k * sizeof(double));
    for (int j = 0; j < np; j++) {
        memcpy(F + (size_t) j * d1 * 2 * k, Cd + (size_t) j * d1 * k, (size_t) d1 * k * sizeof(double));
        memcpy(F + (size_t) j * d1 * 2 * k + (size_t) d1 * k, Cd2 + (size_t) j * d1 * k, (size_t) d1 * k * sizeof(double));
    }
    int p = 0; double *g = NULL, *Phi = NULL;
    if (!rc) {
        det_ortho_basis(D, 2 * k, F, &p, &g, &Phi);
        double gm = 0, dev = 0;
        for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) {
            gm = dmax(gm, fabs(Gd[a + b * k]));
            double s = 0; for (int l = 0; l < p; l++) s += Phi[a + (size_t) l * 2 * k] * Phi[b + (size_t) l * 2 * k];
            dev = dmax(dev, fabs(s - Gd[a + b * k]));
        }
        if (!(dev < 1e-10 * gm)) { snprintf(err, errlen, "prep_gls: basis does not reproduce G_d"); rc = 1; }
    }
    if (!rc) {
        int N = 1 + nj + p, iz = 1 + nj;
        /* H (k x N), PiH = Psi \ H, av = e_1 - H' Psi^{-1} delta(1), A = av av'/2 */
        double *H = calloc((size_t) k * N, sizeof(double));
        for (int a = 0; a < k; a++) {
            H[a] = psi1[a];
            for (int i = 0; i < nj; i++) H[a + (size_t) (1 + i) * k] = -vs[(size_t) i * k + a];
            for (int l = 0; l < p; l++) H[a + (size_t) (iz + l) * k] = -(Phi[(k + a) + (size_t) l * 2 * k] - cbar * cbar * Phi[a + (size_t) l * 2 * k]);
        }
        double *PiH = malloc((size_t) k * N * sizeof(double)); memcpy(PiH, H, (size_t) k * N * sizeof(double));
        det_rsolve(k, Psi, N, PiH);
        double *pd = malloc(k * sizeof(double)); memcpy(pd, delta1, k * sizeof(double)); det_rsolve(k, Psi, 1, pd);
        double *av = calloc(N, sizeof(double)); av[0] = 1;
        for (int c = 0; c < N; c++) { double s = 0; for (int a = 0; a < k; a++) s += H[a + (size_t) c * k] * pd[a]; av[c] -= s; }
        double *A = malloc((size_t) N * N * sizeof(double)), *B = calloc((size_t) N * N, sizeof(double));
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) A[i + (size_t) j * N] = av[i] * av[j] / 2;
        /* B = -(H' Psi^{-1} Phid S + S' Phid' PiH) + PiH' Gd PiH, S the z block */
        double *PS = calloc((size_t) k * N, sizeof(double));      /* Phid S: k x N, zero but for the z block */
        for (int a = 0; a < k; a++) for (int l = 0; l < p; l++) PS[a + (size_t) (iz + l) * k] = Phi[a + (size_t) l * 2 * k];
        det_rsolve(k, Psi, N, PS);                                 /* Psi \ (Phid S) */
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double s1 = 0, s2 = 0, s3 = 0;
            for (int a = 0; a < k; a++) s1 += H[a + (size_t) i * k] * PS[a + (size_t) j * k];
            if (i >= iz) for (int a = 0; a < k; a++) s2 += Phi[a + (size_t) (i - iz) * 2 * k] * PiH[a + (size_t) j * k];
            for (int a = 0; a < k; a++) { double t = 0; for (int b = 0; b < k; b++) t += Gd[a + b * k] * PiH[b + (size_t) j * k]; s3 += PiH[a + (size_t) i * k] * t; }
            B[i + (size_t) j * N] = -(s1 + s2) + s3;
        }
        for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { double m = (B[i + (size_t) j * N] + B[j + (size_t) i * N]) / 2; B[i + (size_t) j * N] = B[j + (size_t) i * N] = m; }
        P->K = k; P->p = p; P->nj = nj; P->N = N; P->g = g;
        P->jl = malloc(imax(nj, 1) * sizeof(double)); memcpy(P->jl, jl, nj * sizeof(double));
        P->A = A; P->B = B; P->b = calloc(N, sizeof(double));
        det_keep_eig(N, A, &P->Dv, &P->L, &P->r, &P->rp, &P->rm);
        free(H); free(PiH); free(pd); free(av); free(PS);
    } else free(g);
    free(Phi); free(F); free(Z); free(Cd); free(Cd1); free(Cd2); free(T); free(Psi); free(Gd); free(psi1); free(delta1); free(e1); free(e2); free(jl); free(vs);
    if (rc) { det_prep_free(P); return 1; }
    *out = P;
    return 0;
}

int det_prep_make(const dfl_design *D, double cbar, det_prep **out, char *err, size_t errlen)
{
    if (D->np < 1 || !D->br || D->K < 0 || D->deg < 0 || (D->K > 0 && !D->C)) { snprintf(err, errlen, "the design is not valid"); return 1; }
    for (int j = 0; j < D->np; j++) if (!(D->br[j + 1] > D->br[j])) { snprintf(err, errlen, "the break points of the design must increase"); return 1; }
    if (D->br[0] != 0 || D->br[D->np] != 1) { snprintf(err, errlen, "the break points of the design must run from 0 to 1"); return 1; }
    int rc = isnan(cbar) ? prep_ols(D, out, err, errlen) : prep_gls(D, cbar, out, err, errlen);
    if (!rc) { (*out)->key = det_design_key(D, cbar); (*out)->refs = 1; }
    return rc;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the cache of prepared designs (prep_cached), FIFO with at most DET_CACHE_MAX entries; an entry is freed when it is
 * dropped and no user holds it */
unsigned long long det_design_key(const dfl_design *D, double cbar)
{
    unsigned long long h = 1469598103934665603ULL;
#define MIX(p, n) do { const unsigned char *b_ = (const unsigned char *) (p); for (size_t i_ = 0; i_ < (n); i_++) { h ^= b_[i_]; h *= 1099511628211ULL; } } while (0)
    int hdr[5] = {D->K, D->np, D->deg, D->loc, D->first};
    MIX(hdr, sizeof hdr);
    MIX(D->br, (D->np + 1) * sizeof(double));
    MIX(D->C, (size_t) D->np * (D->deg + 1) * D->K * sizeof(double));
    double cb = isnan(cbar) ? NAN : cbar; unsigned long long cbits; memcpy(&cbits, &cb, 8); if (isnan(cbar)) cbits = 0x7ff8000000000000ULL;
    MIX(&cbits, 8);
#undef MIX
    return h;
}

int det_model_make(const det_prep *P, int start, double g0, det_model *Mo, char *err, size_t errlen)
{
    memset(Mo, 0, sizeof *Mo);
    Mo->P = P;
    if (!isnan(P->cbar) && !(start == 0 || (start == 1 && g0 == 0))) { snprintf(err, errlen, "GLS detrending (gls = cbar) is implemented for the zero start only: the derivation of the GLS limit assumes u_0 = 0, and start = fixed (gamma0 != 0) and start = stationary are not supported with it"); return 1; }
    if (P->first && !(start == 0 || (start == 1 && g0 == 0))) { snprintf(err, errlen, "the statistic that detrends first (detrend_first) is implemented for the zero start only; start = fixed (gamma0 != 0) and start = stationary are not supported with it"); return 1; }
    int N = P->N;
    if (start == 0 || (start == 1 && g0 == 0)) {
        Mo->start = 0; Mo->g0 = 0; Mo->N = N; Mo->A = P->A; Mo->B = P->B; Mo->L = P->L; Mo->Dv = P->Dv;
        Mo->r = P->r; Mo->rp = P->rp; Mo->rm = P->rm; Mo->a = 0.5; Mo->kappa = 0.5; Mo->yb = calloc(N, sizeof(double));
        return 0;
    }
    if (start == 1) {
        /* the shift of the fixed start (fixed_shift): yb = A^+ b/2, kappa = 1/2 + yb' A yb, exactly 0 or 1/2 */
        int r = P->r;
        double *cb = malloc(r * sizeof(double)), res = 0, bm = 0;
        for (int k = 0; k < r; k++) { double s = 0; for (int i = 0; i < N; i++) s += P->L[i + (size_t) k * N] * P->b[i]; cb[k] = s; }
        for (int i = 0; i < N; i++) { double s = 0; for (int k = 0; k < r; k++) s += P->L[i + (size_t) k * N] * cb[k]; res = dmax(res, fabs(P->b[i] - s)); bm = dmax(bm, fabs(P->b[i])); }
        res /= dmax(1.0, bm);
        double *yb = calloc(N, sizeof(double));
        for (int i = 0; i < N; i++) { double s = 0; for (int k = 0; k < r; k++) s += P->L[i + (size_t) k * N] * cb[k] / P->Dv[k]; yb[i] = s / 2; }
        double kraw = 0.5;
        for (int i = 0; i < N; i++) { double s = 0; for (int j = 0; j < N; j++) s += P->A[i + (size_t) j * N] * yb[j]; kraw += yb[i] * s; }
        free(cb);
        if (!(res <= 1e-8)) { free(yb); snprintf(err, errlen, "model: internal error: b is not in the range of A to 1e-8 (residual %g), which contradicts the shift identity of the fixed start; the eigen-decomposition of A has failed for this design", res); return 1; }
        double kappa = fabs(kraw) < fabs(kraw - 0.5) ? 0.0 : 0.5;
        if (!(fabs(kraw - kappa) <= 1e-6)) { free(yb); snprintf(err, errlen, "model: internal error: kappa = %g is not 0 or 1/2 to 1e-6, which contradicts the shift identity of the fixed start; the eigen-decomposition of A has failed for this design", kraw); return 1; }
        Mo->start = 1; Mo->g0 = g0; Mo->N = N; Mo->A = P->A; Mo->B = P->B; Mo->L = P->L; Mo->Dv = P->Dv;
        Mo->r = P->r; Mo->rp = P->rp; Mo->rm = P->rm; Mo->a = 0.5 + g0 * g0 * kappa; Mo->kappa = kappa; Mo->yb = yb;
        return 0;
    }
    if (start == 2) {
        int n = N, M = N + 1;
        double *A = calloc((size_t) M * M, sizeof(double)), *B = calloc((size_t) M * M, sizeof(double));
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { A[i + (size_t) j * M] = P->A[i + (size_t) j * n]; B[i + (size_t) j * M] = P->B[i + (size_t) j * n]; }
        for (int i = 0; i < n; i++) { A[i + (size_t) n * M] = A[n + (size_t) i * M] = P->b[i] / 2; }
        A[n + (size_t) n * M] = -0.5;
        Mo->start = 2; Mo->g0 = 0; Mo->N = M; Mo->A = A; Mo->B = B; Mo->own = 1;
        det_keep_eig(M, A, &Mo->Dv, &Mo->L, &Mo->r, &Mo->rp, &Mo->rm);
        Mo->a = 0.5; Mo->kappa = 0.5; Mo->yb = calloc(M, sizeof(double));
        return 0;
    }
    snprintf(err, errlen, "start must be zero, fixed or stationary");
    return 1;
}
void det_model_free(det_model *Mo)
{
    if (!Mo) return;
    if (Mo->own) { free(Mo->A); free(Mo->B); free(Mo->L); free(Mo->Dv); }
    free(Mo->yb);
    memset(Mo, 0, sizeof *Mo);
}
