/* dfl_det_la.c: the small dense linear algebra of the general engine, without LAPACK: the symmetric eigenvalue problem
 * (cyclic Jacobi; the Julia engine calls LAPACK's eigen there), the null space (one-sided Jacobi SVD, as nullspace's
 * SVD), the complex LU factorization, its log-determinant and solves, and the eigenvalues of the small complex matrix
 * whose eigenvalues are the reciprocal roots (slu, slogdet, ssolve, ssolvet, seigvals of DFDet.jl, which the Julia
 * engine uses for designs of up to 12 terms and replaces by LAPACK beyond; here the same routines serve every size),
 * the two largest eigenvalues of the Nystrom matrices (subspace iteration; LAPACK's full spectrum in Julia), and
 * Householder factorizations for the designs given by a regressor matrix.  All matrices are column-major. */
#include "dfl_det_int.h"

/* ---------------------------------------------------------------------------------------------------------------- */
int det_symeig(int n, const double *A, double *w, double *V)
{
    if (n <= 0) return 0;
    double *a = malloc((size_t) n * n * sizeof(double)), *v = malloc((size_t) n * n * sizeof(double));
    if (!a || !v) { free(a); free(v); return 1; }
    memcpy(a, A, (size_t) n * n * sizeof(double));
    for (int i = 0; i < n * n; i++) v[i] = 0;
    for (int i = 0; i < n; i++) v[i + i * n] = 1;
    /* symmetrize (the callers pass symmetric matrices up to rounding: Symmetric() takes the upper triangle) */
    for (int j = 0; j < n; j++) for (int i = j + 1; i < n; i++) a[i + j * n] = a[j + i * n];
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0, dg = 0;
        for (int j = 0; j < n; j++) { dg += a[j + j * n] * a[j + j * n]; for (int i = 0; i < j; i++) off += a[i + j * n] * a[i + j * n]; }
        if (off == 0 || off <= 1e-36 * dg) break;
        for (int p = 0; p < n - 1; p++) for (int q = p + 1; q < n; q++) {
            double apq = a[p + q * n];
            if (apq == 0) continue;
            double app = a[p + p * n], aqq = a[q + q * n];
            if (fabs(apq) <= 1e-18 * (fabs(app) + fabs(aqq))) { a[p + q * n] = a[q + p * n] = 0; continue; }
            double th = (aqq - app) / (2 * apq);
            double t = (th >= 0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1));
            if (!isfinite(th)) t = 1 / (2 * th);
            double cs = 1 / sqrt(t * t + 1), sn = t * cs;
            for (int k = 0; k < n; k++) {               /* columns p, q */
                double akp = a[k + p * n], akq = a[k + q * n];
                a[k + p * n] = cs * akp - sn * akq; a[k + q * n] = sn * akp + cs * akq;
            }
            for (int k = 0; k < n; k++) {               /* rows p, q */
                double apk = a[p + k * n], aqk = a[q + k * n];
                a[p + k * n] = cs * apk - sn * aqk; a[q + k * n] = sn * apk + cs * aqk;
            }
            a[p + q * n] = a[q + p * n] = 0;
            for (int k = 0; k < n; k++) {
                double vkp = v[k + p * n], vkq = v[k + q * n];
                v[k + p * n] = cs * vkp - sn * vkq; v[k + q * n] = sn * vkp + cs * vkq;
            }
        }
    }
    /* ascending order (insertion sort of the indices) */
    int *ix = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) ix[i] = i;
    for (int i = 1; i < n; i++) { int k = ix[i], j = i - 1; while (j >= 0 && a[ix[j] + ix[j] * n] > a[k + k * n]) { ix[j + 1] = ix[j]; j--; } ix[j + 1] = k; }
    for (int i = 0; i < n; i++) w[i] = a[ix[i] + ix[i] * n];
    if (V) for (int i = 0; i < n; i++) memcpy(V + (size_t) i * n, v + (size_t) ix[i] * n, n * sizeof(double));
    free(ix); free(a); free(v);
    return 0;
}

/* the null space by the one-sided Jacobi SVD (Hestenes): the columns of A rotated until they are orthogonal, V the
 * product of the rotations; the columns of V whose rotated columns have norms <= atol span the null space */
int det_nullspace(int m, int n, const double *A, double atol, double *Z)
{
    if (n <= 0) return 0;
    double *U = malloc((size_t) imax(m, 1) * n * sizeof(double)), *V = malloc((size_t) n * n * sizeof(double));
    if (!U || !V) { free(U); free(V); return -1; }
    if (m > 0) memcpy(U, A, (size_t) m * n * sizeof(double));
    for (int i = 0; i < n * n; i++) V[i] = 0;
    for (int i = 0; i < n; i++) V[i + i * n] = 1;
    for (int sweep = 0; sweep < 80 && m > 0; sweep++) {
        int rot = 0;
        for (int i = 0; i < n - 1; i++) for (int j = i + 1; j < n; j++) {
            double al = 0, be = 0, ga = 0;
            for (int k = 0; k < m; k++) { double ui = U[k + i * m], uj = U[k + j * m]; al += ui * ui; be += uj * uj; ga += ui * uj; }
            if (ga == 0 || fabs(ga) <= 1e-15 * sqrt(al * be)) continue;
            rot = 1;
            double ze = (be - al) / (2 * ga), t = (ze >= 0 ? 1.0 : -1.0) / (fabs(ze) + sqrt(1 + ze * ze)), cs = 1 / sqrt(1 + t * t), sn = cs * t;
            for (int k = 0; k < m; k++) { double ui = U[k + i * m], uj = U[k + j * m]; U[k + i * m] = cs * ui - sn * uj; U[k + j * m] = sn * ui + cs * uj; }
            for (int k = 0; k < n; k++) { double vi = V[k + i * n], vj = V[k + j * n]; V[k + i * n] = cs * vi - sn * vj; V[k + j * n] = sn * vi + cs * vj; }
        }
        if (!rot) break;
    }
    int k = 0;
    for (int j = 0; j < n; j++) {
        double s = 0; for (int i = 0; i < m; i++) s += U[i + j * m] * U[i + j * m];
        if (!(sqrt(s) > atol)) { memcpy(Z + (size_t) k * n, V + (size_t) j * n, n * sizeof(double)); k++; }
    }
    free(U); free(V);
    return k;
}

int det_rsolve(int n, const double *A0, int m, double *B)
{
    double *A = malloc((size_t) n * n * sizeof(double));
    if (!A) return 1;
    memcpy(A, A0, (size_t) n * n * sizeof(double));
    int sing = 0;
    for (int k = 0; k < n; k++) {
        int piv = k; double mx = fabs(A[k + k * n]);
        for (int i = k + 1; i < n; i++) if (fabs(A[i + k * n]) > mx) { mx = fabs(A[i + k * n]); piv = i; }
        if (mx == 0) { sing = 1; continue; }
        if (piv != k) {
            for (int j = 0; j < n; j++) { double t = A[k + j * n]; A[k + j * n] = A[piv + j * n]; A[piv + j * n] = t; }
            for (int j = 0; j < m; j++) { double t = B[k + j * n]; B[k + j * n] = B[piv + j * n]; B[piv + j * n] = t; }
        }
        for (int i = k + 1; i < n; i++) {
            double l = A[i + k * n] / A[k + k * n];
            if (l == 0) continue;
            for (int j = k + 1; j < n; j++) A[i + j * n] -= l * A[k + j * n];
            for (int j = 0; j < m; j++) B[i + j * n] -= l * B[k + j * n];
        }
    }
    for (int j = 0; j < m; j++) for (int i = n - 1; i >= 0; i--) {
        double s = B[i + j * n];
        for (int l = i + 1; l < n; l++) s -= A[i + l * n] * B[l + j * n];
        B[i + j * n] = s / A[i + i * n];
    }
    free(A);
    return sing;
}

/* ---------------------------------------------------------------------------------------------------------------- */
void det_clu(int n, cdouble *M, int *p, int *odd)
{
    *odd = 0;
    for (int i = 0; i < n; i++) p[i] = i;
    for (int k = 0; k < n; k++) {
        int piv = k; double mx = cabs2(M[k + k * n]);
        for (int i = k + 1; i < n; i++) { double a = cabs2(M[i + k * n]); if (a > mx) { mx = a; piv = i; } }
        if (piv != k) {
            for (int j = 0; j < n; j++) { cdouble t = M[k + j * n]; M[k + j * n] = M[piv + j * n]; M[piv + j * n] = t; }
            int t = p[k]; p[k] = p[piv]; p[piv] = t; *odd = !*odd;
        }
        cdouble ip = cinv(M[k + k * n]);
        for (int i = k + 1; i < n; i++) {
            cdouble l = cmul(M[i + k * n], ip); M[i + k * n] = l;
            for (int j = k + 1; j < n; j++) M[i + j * n] = csub(M[i + j * n], cmul(l, M[k + j * n]));
        }
    }
}
cdouble det_clogdet(int n, const cdouble *F, int odd)
{
    double d = 0; cdouble s = CPX(odd ? -1.0 : 1.0, 0);
    for (int i = 0; i < n; i++) { cdouble u = F[i + i * n]; double a = cabs(u); d += log(a); s = cmul(s, cdivr(u, a)); }
    return cadd(CPX(d, 0), clog(s));
}
cdouble det_cdet(int n, const cdouble *F, int odd)
{
    cdouble s = CPX(odd ? -1.0 : 1.0, 0);
    for (int i = 0; i < n; i++) s = cmul(s, F[i + i * n]);
    return s;
}
void det_csolve(int n, const cdouble *F, const int *p, cdouble *b)
{
    cdouble x[64], *xx = n <= 64 ? x : malloc(n * sizeof(cdouble));
    for (int i = 0; i < n; i++) xx[i] = b[p[i]];
    for (int i = 1; i < n; i++) for (int j = 0; j < i; j++) xx[i] = csub(xx[i], cmul(F[i + j * n], xx[j]));
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) xx[i] = csub(xx[i], cmul(F[i + j * n], xx[j]));
        xx[i] = cdiv(xx[i], F[i + i * n]);
    }
    for (int i = 0; i < n; i++) b[i] = xx[i];
    if (xx != x) free(xx);
}
void det_csolvet(int n, const cdouble *F, const int *p, cdouble *b)
{
    cdouble y[64], *yy = n <= 64 ? y : malloc(n * sizeof(cdouble));
    for (int i = 0; i < n; i++) yy[i] = b[i];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) yy[i] = csub(yy[i], cmul(F[j + i * n], yy[j]));
        yy[i] = cdiv(yy[i], F[i + i * n]);
    }
    for (int i = n - 2; i >= 0; i--) for (int j = i + 1; j < n; j++) yy[i] = csub(yy[i], cmul(F[j + i * n], yy[j]));
    for (int i = 0; i < n; i++) b[p[i]] = yy[i];
    if (yy != y) free(yy);
}

/* ---------------------------------------------------------------------------------------------------------------- */
static inline double cabs1(cdouble z) { return fabs(creal(z)) + fabs(cimag(z)); }
/* the Givens rotation (c real, s complex) with c x + s y = r, -conj(s) x + c y = 0 */
static inline void givens_c(cdouble x, cdouble y, double *c, cdouble *s)
{
    if (creal(y) == 0 && cimag(y) == 0) { *c = 1; *s = DZERO; return; }
    double ax = cabs(x);
    if (ax == 0) { *c = 0; *s = cdivr(conj(y), cabs(y)); return; }
    double nrm = hypot(ax, cabs(y));
    *c = ax / nrm; *s = cdivr(cmul(cdivr(x, ax), conj(y)), nrm);
}
static void eig2(cdouble a, cdouble b, cdouble c, cdouble d, cdouble *e1, cdouble *e2)
{
    cdouble tr = cadd(a, d), dt = csub(cmul(a, d), cmul(b, c)), hd = cscale(0.5, csub(a, d));
    cdouble disc = csq(cadd(cmul(hd, hd), cmul(b, c)));
    cdouble m = cscale(0.5, tr);
    cdouble q = creal(cmul(conj(m), disc)) >= 0 ? cadd(m, disc) : csub(m, disc);
    if (creal(q) == 0 && cimag(q) == 0) { *e1 = m; *e2 = m; return; }
    *e1 = q; *e2 = cdiv(dt, q);
}
int det_ceigvals(int R, cdouble *H, cdouble *ev)
{
    if (R <= 0) return 0;
    if (R == 1) { ev[0] = H[0]; return 0; }
    if (R == 2) { eig2(H[0], H[2], H[1], H[3], &ev[0], &ev[1]); return 0; }
#define HH(i, j) H[(i) + (j) * R]
    const double ep = 2.220446049250313e-16;
    /* balancing (EISPACK balanc, radix 2, scaling only) */
    for (int it = 0; it < 20; it++) {
        int noconv = 0;
        for (int i = 0; i < R; i++) {
            double c = 0, r = 0;
            for (int j = 0; j < R; j++) { if (j == i) continue; c += cabs1(HH(j, i)); r += cabs1(HH(i, j)); }
            if (c == 0 || r == 0) continue;
            double g = r / 2, f = 1, s = c + r;
            while (c < g) { f *= 2; c *= 4; }
            g = r * 2;
            while (c >= g) { f /= 2; c /= 4; }
            if ((c + r) / f < 0.95 * s) {
                noconv = 1;
                for (int j = 0; j < R; j++) { HH(i, j) = cdivr(HH(i, j), f); HH(j, i) = cmulr(HH(j, i), f); }
            }
        }
        if (!noconv) break;
    }
    /* Hessenberg form by Givens rotations (similarity) */
    for (int k = 0; k < R - 2; k++) for (int i = R - 1; i >= k + 2; i--) {
        double cg; cdouble sg; givens_c(HH(i - 1, k), HH(i, k), &cg, &sg);
        if (creal(sg) == 0 && cimag(sg) == 0 && cg == 1) continue;
        for (int j = k; j < R; j++) {
            cdouble x = HH(i - 1, j), y = HH(i, j);
            HH(i - 1, j) = cadd(cscale(cg, x), cmul(sg, y)); HH(i, j) = cadd(cneg(cmul(conj(sg), x)), cscale(cg, y));
        }
        for (int j = 0; j < R; j++) {
            cdouble x = HH(j, i - 1), y = HH(j, i);
            HH(j, i - 1) = cadd(cscale(cg, x), cmul(conj(sg), y)); HH(j, i) = cadd(cneg(cmul(sg, x)), cscale(cg, y));
        }
        HH(i, k) = DZERO;
    }
    int ihi = R - 1, its = 0, total = 0;
    while (ihi >= 0) {
        if (ihi == 0) { ev[0] = HH(0, 0); break; }
        int l = 0;
        for (int k = ihi; k >= 1; k--) {
            if (cabs1(HH(k, k - 1)) <= ep * (cabs1(HH(k - 1, k - 1)) + cabs1(HH(k, k)))) { HH(k, k - 1) = DZERO; l = k; break; }
        }
        if (l == ihi) { ev[ihi] = HH(ihi, ihi); ihi--; its = 0; continue; }
        if (l == ihi - 1) { eig2(HH(l, l), HH(l, ihi), HH(ihi, l), HH(ihi, ihi), &ev[l], &ev[ihi]); ihi -= 2; its = 0; continue; }
        its++; total++;
        /* (the Julia engine gives up after 30 R iterations and takes LAPACK's eigenvalues; here the iteration continues) */
        if (total > 300 * R) return 1;
        cdouble mu;
        if (its % 10 == 0) mu = caddr(HH(ihi, ihi), 0.75 * fabs(creal(HH(ihi, ihi - 1))));
        else {
            cdouble a = HH(ihi - 1, ihi - 1), b = HH(ihi - 1, ihi), c = HH(ihi, ihi - 1), d = HH(ihi, ihi);
            cdouble hd = cscale(0.5, csub(a, d)), disc = csq(cadd(cmul(hd, hd), cmul(b, c)));
            cdouble m1 = cadd(cscale(0.5, cadd(a, d)), disc), m2 = csub(cscale(0.5, cadd(a, d)), disc);
            mu = cabs2(csub(m1, d)) <= cabs2(csub(m2, d)) ? m1 : m2;
        }
        cdouble x = csub(HH(l, l), mu), y = HH(l + 1, l);
        for (int k = l; k < ihi; k++) {
            double cg; cdouble sg; givens_c(x, y, &cg, &sg);
            for (int j = imax(l, k - 1); j <= ihi; j++) {
                cdouble u = HH(k, j), v = HH(k + 1, j);
                HH(k, j) = cadd(cscale(cg, u), cmul(sg, v)); HH(k + 1, j) = cadd(cneg(cmul(conj(sg), u)), cscale(cg, v));
            }
            for (int j = l; j <= imin(k + 2, ihi); j++) {
                cdouble u = HH(j, k), v = HH(j, k + 1);
                HH(j, k) = cadd(cscale(cg, u), cmul(conj(sg), v)); HH(j, k + 1) = cadd(cneg(cmul(sg, u)), cscale(cg, v));
            }
            if (k < ihi - 1) { x = HH(k + 1, k); y = HH(k + 2, k); }
        }
        for (int k = l + 2; k <= ihi; k++) HH(k, k - 2) = DZERO;
    }
#undef HH
    return 0;
}

/* ---------------------------------------------------------------------------------------------------------------- */
void det_orth(int n, int k, double *X)
{
    for (int pass = 0; pass < 2; pass++) for (int j = 0; j < k; j++) {
        double *x = X + (size_t) j * n;
        for (int i = 0; i < j; i++) {
            const double *q = X + (size_t) i * n; double d = 0;
            for (int l = 0; l < n; l++) d += q[l] * x[l];
            for (int l = 0; l < n; l++) x[l] -= d * q[l];
        }
        double s = 0; for (int l = 0; l < n; l++) s += x[l] * x[l];
        s = sqrt(s);
        if (s > 0) for (int l = 0; l < n; l++) x[l] /= s;
    }
}

int det_topeig(int n, void (*apply)(int nb, const double *X, double *Y, void *ctx), void *ctx, double *l1, double *l2)
{
    int nb = imin(n, 10);
    double *X = malloc((size_t) n * nb * sizeof(double)), *Y = malloc((size_t) n * nb * sizeof(double));
    double Hm[100], w[10], V[100], prev1 = NAN, prev2 = NAN;
    if (!X || !Y) { free(X); free(Y); return 1; }
    /* a deterministic start (a linear congruential generator) */
    unsigned long long s = 0x2545F4914F6CDD1DULL;
    for (int i = 0; i < n * nb; i++) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; X[i] = (double) (s >> 11) / 9007199254740992.0 - 0.5; }
    det_orth(n, nb, X);
    *l1 = *l2 = NAN;
    for (int it = 0; it < 400; it++) {
        apply(nb, X, Y, ctx);
        for (int a = 0; a < nb; a++) for (int b = 0; b < nb; b++) {
            double d = 0; for (int l = 0; l < n; l++) d += X[l + (size_t) a * n] * Y[l + (size_t) b * n];
            Hm[a + b * nb] = d;
        }
        for (int a = 0; a < nb; a++) for (int b = a + 1; b < nb; b++) { double m = (Hm[a + b * nb] + Hm[b + a * nb]) / 2; Hm[a + b * nb] = Hm[b + a * nb] = m; }
        det_symeig(nb, Hm, w, V);
        /* the Ritz values by modulus: the largest two (the operators are positive semidefinite up to rounding) */
        double e1 = w[nb - 1], e2 = nb > 1 ? w[nb - 2] : NAN;
        *l1 = e1; *l2 = e2;
        if (it > 2 && fabs(e1 - prev1) <= 1e-14 * fabs(e1) && (nb < 2 || fabs(e2 - prev2) <= 1e-14 * fabs(e1))) break;
        prev1 = e1; prev2 = e2;
        /* X <- orth(Y V) (the Ritz vectors times the operator) */
        for (int l = 0; l < n; l++) {
            double row[10];
            for (int b = 0; b < nb; b++) { double d = 0; for (int a = 0; a < nb; a++) d += Y[l + (size_t) a * n] * V[a + b * nb]; row[b] = d; }
            for (int b = 0; b < nb; b++) X[l + (size_t) b * n] = row[b];
        }
        det_orth(n, nb, X);
    }
    free(X); free(Y);
    return 0;
}

/* Householder QR with column pivoting (the largest remaining column norm first) */
void det_qrp(int n, int k, double *A, int *perm, double *rdiag)
{
    double *cn = malloc(k * sizeof(double));
    for (int j = 0; j < k; j++) { perm[j] = j; double s = 0; for (int i = 0; i < n; i++) s += A[i + (size_t) j * n] * A[i + (size_t) j * n]; cn[j] = s; }
    for (int j = 0; j < k && j < n; j++) {
        int pj = j; for (int l = j + 1; l < k; l++) if (cn[l] > cn[pj]) pj = l;
        if (pj != j) {
            for (int i = 0; i < n; i++) { double t = A[i + (size_t) j * n]; A[i + (size_t) j * n] = A[i + (size_t) pj * n]; A[i + (size_t) pj * n] = t; }
            double t = cn[j]; cn[j] = cn[pj]; cn[pj] = t; int ti = perm[j]; perm[j] = perm[pj]; perm[pj] = ti;
        }
        double s = 0; for (int i = j; i < n; i++) s += A[i + (size_t) j * n] * A[i + (size_t) j * n];
        double nrm = sqrt(s), x0 = A[j + (size_t) j * n];
        double al = x0 > 0 ? -nrm : nrm;
        rdiag[j] = al;
        if (nrm == 0) continue;
        double v0 = x0 - al;
        A[j + (size_t) j * n] = v0;
        double vn = v0 * v0 + (s - x0 * x0);
        for (int l = j + 1; l < k; l++) {
            double d = 0; for (int i = j; i < n; i++) d += A[i + (size_t) j * n] * A[i + (size_t) l * n];
            d = 2 * d / vn;
            for (int i = j; i < n; i++) A[i + (size_t) l * n] -= d * A[i + (size_t) j * n];
            cn[l] -= A[j + (size_t) l * n] * A[j + (size_t) l * n];
            if (cn[l] < 0) cn[l] = 0;
            /* (recompute the norm where cancellation set in) */
            double r = 0; for (int i = j + 1; i < n; i++) r += A[i + (size_t) l * n] * A[i + (size_t) l * n];
            cn[l] = r;
        }
    }
    free(cn);
}

int det_lsq(int n, int m, double *V, const double *y, double *c)
{
    double *b = malloc(n * sizeof(double)), *rd = malloc(m * sizeof(double));
    memcpy(b, y, n * sizeof(double));
    for (int j = 0; j < m; j++) {
        double s = 0; for (int i = j; i < n; i++) s += V[i + (size_t) j * n] * V[i + (size_t) j * n];
        double nrm = sqrt(s), x0 = V[j + (size_t) j * n], al = x0 > 0 ? -nrm : nrm;
        rd[j] = al;
        if (nrm == 0) continue;
        double v0 = x0 - al; V[j + (size_t) j * n] = v0;
        double vn = v0 * v0 + (s - x0 * x0);
        for (int l = j + 1; l < m; l++) {
            double d = 0; for (int i = j; i < n; i++) d += V[i + (size_t) j * n] * V[i + (size_t) l * n];
            d = 2 * d / vn; for (int i = j; i < n; i++) V[i + (size_t) l * n] -= d * V[i + (size_t) j * n];
        }
        double d = 0; for (int i = j; i < n; i++) d += V[i + (size_t) j * n] * b[i];
        d = 2 * d / vn; for (int i = j; i < n; i++) b[i] -= d * V[i + (size_t) j * n];
    }
    int sing = 0;
    for (int j = m - 1; j >= 0; j--) {
        double s = b[j]; for (int l = j + 1; l < m; l++) s -= V[j + (size_t) l * n] * c[l];
        if (rd[j] == 0) { c[j] = 0; sing = 1; } else c[j] = s / rd[j];
    }
    free(b); free(rd);
    return sing;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* errors and warnings */
static dfl_lock_t errlock = DFL_LOCK_INIT;
void det_fail(det_status *st, const char *fmt, ...)
{
    dfl_lock(&errlock);
    if (!st->err) {
        va_list ap; va_start(ap, fmt); vsnprintf(st->msg, sizeof st->msg, fmt, ap); va_end(ap);
        st->err = 1;
    }
    dfl_unlock(&errlock);
}
static _Thread_local char DETWARN[2048];
static _Thread_local int DETHOLD = 0;
/* (a composite function holds the warnings of its parts: the clearing at the entry of each part is then skipped) */
void det_warn_clear(void) { if (!DETHOLD) DETWARN[0] = 0; }
void det_warn_hold(int on) { DETHOLD += on ? 1 : -1; if (DETHOLD < 0) DETHOLD = 0; }
void det_warn(const char *fmt, ...)
{
    size_t k = strlen(DETWARN);
    if (k + 8 >= sizeof DETWARN) return;
    char buf[640]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    if (strstr(DETWARN, buf)) return;                 /* each message once per call */
    snprintf(DETWARN + k, sizeof DETWARN - k, "%s%s", k ? "\n" : "", buf);
}
const char *dfl_det_warning(void) { return DETWARN[0] ? DETWARN : NULL; }
