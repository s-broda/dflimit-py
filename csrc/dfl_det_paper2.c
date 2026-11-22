/* dfl_det_paper2.c: the statistics of the paper on general deterministic terms beyond the t-ratio (a port of
 * src/DFDet_paper2.jl of DFLimit.jl): the point-optimal statistic of Elliott, Rothenberg and Stock against cbar for any
 * design and the Gaussian power envelope (prep_pt, qgrid, ptF: one inversion of Gil-Pelaez of the quadratic form
 * Q = U + 1/2 + cbar^2 V), the LM statistic of Schmidt and Phillips with breaks (prep_lm: the law of -1/(2 sqrt V) by the
 * single inversion of the transform of V), and the companions of the GLS-detrended t-ratio, MZ_alpha (the coefficient
 * statistic with GLS detrending) and MSB (sqrt V_g). */
#include "dfl_det_int.h"

/* ---------------------------------------------------------------------------------------------------------------- */
/* prep_pt: Q = cbar^2 V_g + ell' A ell with A = C - cbar^2 B_g */
static int prep_pt(const dfl_design *D, double cbar, det_prep **out, char *err, size_t errlen)
{
    if (!(cbar < 0)) { snprintf(err, errlen, "the point-optimal statistic needs cbar < 0"); return 1; }
    int np = D->np, d1 = D->deg + 1;
    det_prep *P = det_prep_new(D); P->cbar = cbar; P->kind = 2;
    double *Z; int k = det_gls_directions(D, &Z);
    if (k == 0) { free(Z); det_prep_rank1(P, -cbar); P->key = det_design_key(D, cbar) ^ 0x5054ULL; P->refs = 1; *out = P; return 0; }
    double *Cd = malloc((size_t) np * d1 * k * sizeof(double)), *Cd1 = malloc((size_t) np * d1 * k * sizeof(double)), *Cd2 = malloc((size_t) np * d1 * k * sizeof(double));
    det_delta_blocks(D, k, Z, Cd, Cd1, Cd2);
    double *T = malloc((size_t) np * d1 * k * sizeof(double));
    for (size_t i = 0; i < (size_t) np * d1 * k; i++) T[i] = Cd1[i] - cbar * Cd[i];
    double *Xi = malloc((size_t) k * k * sizeof(double)), *Xi0 = malloc((size_t) k * k * sizeof(double));
    det_gram(np, D->deg, k, T, D->br, D->loc, Xi); det_gram(np, D->deg, k, Cd1, D->br, D->loc, Xi0);
    double *psi1 = malloc(k * sizeof(double)), *dot1 = malloc(k * sizeof(double)), *e1 = malloc(k * sizeof(double)), *e2 = malloc(k * sizeof(double));
    det_evblk(D, k, Cd1, np - 1, 1.0, dot1); det_evblk(D, k, Cd, np - 1, 1.0, e1);
    for (int i = 0; i < k; i++) psi1[i] = dot1[i] - cbar * e1[i];
    double *jl = malloc(np * sizeof(double)), *vs = malloc((size_t) np * k * sizeof(double)); int nj = 0;
    for (int j = 0; j < np - 1; j++) {
        double lam = D->br[j + 1], mxs = 0;
        det_evblk(D, k, Cd1, j + 1, lam, e1); det_evblk(D, k, Cd1, j, lam, e2);
        for (int i = 0; i < k; i++) { vs[(size_t) nj * k + i] = e1[i] - e2[i]; mxs = dmax(mxs, fabs(e1[i] - e2[i])); }
        if (mxs > 1e-12) jl[nj++] = lam;
    }
    double *F = malloc((size_t) np * d1 * 2 * k * sizeof(double));
    for (int j = 0; j < np; j++) {
        memcpy(F + (size_t) j * d1 * 2 * k, Cd + (size_t) j * d1 * k, (size_t) d1 * k * sizeof(double));
        memcpy(F + (size_t) j * d1 * 2 * k + (size_t) d1 * k, Cd2 + (size_t) j * d1 * k, (size_t) d1 * k * sizeof(double));
    }
    int p; double *g, *Phi;
    det_ortho_basis(D, 2 * k, F, &p, &g, &Phi);
    int N = 1 + nj + p, iz = 1 + nj;
    double *H = calloc((size_t) k * N, sizeof(double)), *H0 = calloc((size_t) k * N, sizeof(double));
    for (int a = 0; a < k; a++) {
        H[a] = psi1[a]; H0[a] = dot1[a];
        for (int i = 0; i < nj; i++) { H[a + (size_t) (1 + i) * k] = -vs[(size_t) i * k + a]; H0[a + (size_t) (1 + i) * k] = -vs[(size_t) i * k + a]; }
        for (int l = 0; l < p; l++) {
            double p2 = Phi[(k + a) + (size_t) l * 2 * k], pd = Phi[a + (size_t) l * 2 * k];
            H[a + (size_t) (iz + l) * k] = -(p2 - cbar * cbar * pd); H0[a + (size_t) (iz + l) * k] = -p2;
        }
    }
    double *XH = malloc((size_t) k * N * sizeof(double)), *XH0 = malloc((size_t) k * N * sizeof(double));
    memcpy(XH, H, (size_t) k * N * sizeof(double)); memcpy(XH0, H0, (size_t) k * N * sizeof(double));
    det_rsolve(k, Xi, N, XH); det_rsolve(k, Xi0, N, XH0);
    det_prep *Pg = NULL;
    int rc = det_prep_make(D, cbar, &Pg, err, errlen);
    if (!rc && Pg->N != N) { snprintf(err, errlen, "prep_pt: the GLS-detrended design has another basis"); rc = 1; }
    if (!rc) {
        double *C = malloc((size_t) N * N * sizeof(double));
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double s1 = 0, s2 = 0;
            for (int a = 0; a < k; a++) { s1 += H[a + (size_t) i * k] * XH[a + (size_t) j * k]; s2 += H0[a + (size_t) i * k] * XH0[a + (size_t) j * k]; }
            C[i + (size_t) j * N] = -s1 + s2;
        }
        C[0] -= cbar;
        for (int i = 0; i < N * N; i++) C[i] -= cbar * cbar * Pg->B[i];
        for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { double m = (C[i + (size_t) j * N] + C[j + (size_t) i * N]) / 2; C[i + (size_t) j * N] = C[j + (size_t) i * N] = m; }
        P->K = k; P->p = p; P->nj = nj; P->N = N; P->g = g; g = NULL;
        P->jl = malloc(imax(nj, 1) * sizeof(double)); memcpy(P->jl, jl, nj * sizeof(double));
        P->A = C; P->B = malloc((size_t) N * N * sizeof(double)); memcpy(P->B, Pg->B, (size_t) N * N * sizeof(double)); P->b = calloc(N, sizeof(double));
        det_keep_eig(N, C, &P->Dv, &P->L, &P->r, &P->rp, &P->rm);
        P->key = det_design_key(D, cbar) ^ 0x5054ULL; P->refs = 1;
    }
    det_prep_free(Pg);
    free(g); free(Phi); free(F); free(Z); free(Cd); free(Cd1); free(Cd2); free(T); free(Xi); free(Xi0); free(psi1); free(dot1); free(e1); free(e2); free(jl); free(vs);
    free(H); free(H0); free(XH); free(XH0);
    if (rc) { det_prep_free(P); return 1; }
    *out = P;
    return 0;
}

/* the characteristic function of Q under c on the grid of the Gil-Pelaez rule (qgrid): E exp(iuQ) = e^{iu/2} M(iu, iu cbar^2) */
typedef struct { int N; double *u, *w; cdouble *lM; } qgrid_t;
typedef struct { const det_model *Mo; double c, cb2; const double *u; cdouble *L0, *LD; det_status st; } qg_ctx;
static void qg_body(int k, void *v)
{
    qg_ctx *o = v;
    if (det_failed(&o->st)) return;
    char buf[DET_WORK_STACK]; det_work W;
    if (det_work_init(&W, o->Mo, buf, sizeof buf)) { det_fail(&o->st, "out of memory"); return; }
    cdouble s = CPX(0, o->u[k]), t = CPX(0, o->cb2 * o->u[k]), th;
    if (det_logmgf_tpart(&W, o->c, t, &th, NULL)) det_fail(&o->st, "the eigenvalues of the reciprocal roots did not converge");
    det_logmgf_spart(&W, o->c, s, t, th, NULL, &o->L0[k], &o->LD[k]);
    det_work_free(&W);
}
static int qgrid(const det_prep *P, double cbar, double c, int N, double umax, qgrid_t *G, char *err, size_t errlen)
{
    det_model Mo;
    if (det_model_make(P, 0, 0.0, &Mo, err, errlen)) return 1;
    G->N = N;
    G->u = malloc((N + 1) * sizeof(double)); G->w = malloc((N + 1) * sizeof(double)); G->lM = malloc((N + 1) * sizeof(cdouble));
    double h = asinh(umax) / N;
    for (int k = 0; k <= N; k++) { double v = k * h; G->u[k] = sinh(v); G->w[k] = cosh(v) * h; }
    G->u[0] = 1e-7; G->w[0] /= 2;
    cdouble *L0 = malloc((N + 1) * sizeof(cdouble)), *LD = malloc((N + 1) * sizeof(cdouble));
    qg_ctx g = {&Mo, c, cbar * cbar, G->u, L0, LD, {0, ""}};
    dfl_parfor(N + 1, qg_body, &g);
    int rc = 0;
    if (det_failed(&g.st)) { snprintf(err, errlen, "%s", g.st.msg); rc = 1; }
    else {
        double ph = cimag(LD[0]);
        for (int k = 0; k <= N; k++) {
            if (k > 0) { double d = cimag(LD[k]) - cimag(LD[k - 1]); ph += atan2(sin(d), cos(d)); }
            G->lM[k] = cadd(csub(L0[k], cscale(0.5, CPX(creal(LD[k]), ph))), CPX(0, G->u[k] / 2));
        }
    }
    free(L0); free(LD); det_model_free(&Mo);
    return rc;
}
static void qgrid_free(qgrid_t *G) { free(G->u); free(G->w); free(G->lM); }
static void ptF(const qgrid_t *G, double q, double *F, double *f)
{
    double a = 0, b = 0;
    for (int k = 0; k <= G->N; k++) {
        cdouble z = cexp(csub(G->lM[k], CPX(0, G->u[k] * q)));
        a += cimag(z) / G->u[k] * G->w[k]; b += creal(z) * G->w[k];
    }
    *F = 0.5 - a / M_PI; *f = b / M_PI;
}
static double ptquantile(const qgrid_t *G, double p, double q0)
{
    double lo = 1e-3, hi = 400, q = q0, F, f;
    for (int it = 0; it < 100; it++) {
        ptF(G, q, &F, &f);
        if (F > p) hi = q; else lo = q;
        if (fabs(F - p) < 1e-12) break;
        double qn = q - (F - p) / f;
        q = (lo < qn && qn < hi) ? qn : (lo + hi) / 2;
    }
    return q;
}
static void gp_defaults(int *N, double *umax) { if (*N <= 0) *N = 60000; if (!(*umax > 0)) *umax = 4000; }
int dfl_det_pt(int kind, const dfl_design *D, int n, const double *q, double cbar, double c, int N, double umax, double *out, char *err, size_t errlen)
{
    det_warn_clear(); gp_defaults(&N, &umax);
    if (kind != 1 && kind != 2) { snprintf(err, errlen, "kind must be 1 (density) or 2 (distribution function)"); return 1; }
    det_prep *P; if (prep_pt(D, cbar, &P, err, errlen)) return 1;
    qgrid_t G; int rc = qgrid(P, cbar, c, N, umax, &G, err, errlen);
    if (!rc) { for (int k = 0; k < n; k++) { double F, f; ptF(&G, q[k], &F, &f); out[k] = kind == 2 ? F : f; } qgrid_free(&G); }
    det_prep_free(P);
    return rc;
}
int dfl_det_pt_quantile(const dfl_design *D, double p, double cbar, int N, double umax, double *q, char *err, size_t errlen)
{
    det_warn_clear(); gp_defaults(&N, &umax);
    det_prep *P; if (prep_pt(D, cbar, &P, err, errlen)) return 1;
    qgrid_t G; int rc = qgrid(P, cbar, 0.0, N, umax, &G, err, errlen);
    if (!rc) { *q = ptquantile(&G, p, 6.0); qgrid_free(&G); }
    det_prep_free(P);
    return rc;
}
int dfl_det_pt_power(const dfl_design *D, double cbar, double c, double level, int N, double umax, double *b, double *power, char *err, size_t errlen)
{
    det_warn_clear(); gp_defaults(&N, &umax);
    det_prep *P; if (prep_pt(D, cbar, &P, err, errlen)) return 1;
    qgrid_t G; int rc = qgrid(P, cbar, 0.0, N, umax, &G, err, errlen);
    if (!rc) {
        *b = ptquantile(&G, level, 6.0); qgrid_free(&G);
        rc = qgrid(P, cbar, c, N, umax, &G, err, errlen);
        if (!rc) { double f; ptF(&G, *b, power, &f); qgrid_free(&G); }
    }
    det_prep_free(P);
    return rc;
}
int dfl_det_pt_envelope(const dfl_design *D, int n, const double *cs, double level, int N, double umax, double *out, char *err, size_t errlen)
{
    for (int k = 0; k < n; k++) {
        if (!(cs[k] < 0)) { out[k] = level; continue; }
        double b; if (dfl_det_pt_power(D, cs[k], cs[k], level, N, umax, &b, &out[k], err, errlen)) return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* prep_lm: the denominator V = int Sbar^2 of the LM statistic; |a1| (J_0(1) = a1' ell) into *a1norm */
int det_prep_lm(const dfl_design *D, det_prep **out, double *a1norm, char *err, size_t errlen)
{
    int np = D->np, d1 = D->deg + 1;
    double *Z; int k = det_gls_directions(D, &Z);
    if (k == 0) { free(Z); snprintf(err, errlen, "prep_lm: no trend function that vanishes at the origin and does not jump"); return 1; }
    det_prep *P = det_prep_new(D); P->kind = 3;
    double *Cd = malloc((size_t) np * d1 * k * sizeof(double)), *Cd1 = malloc((size_t) np * d1 * k * sizeof(double)), *Cd2 = malloc((size_t) np * d1 * k * sizeof(double));
    det_delta_blocks(D, k, Z, Cd, Cd1, Cd2);
    double *e1 = malloc(k * sizeof(double)), *e2 = malloc(k * sizeof(double)), *jl = malloc(np * sizeof(double)), *vs = malloc((size_t) np * k * sizeof(double)); int nj = 0;
    for (int j = 0; j < np - 1; j++) {
        double lam = D->br[j + 1], mxs = 0;
        det_evblk(D, k, Cd1, j + 1, lam, e1); det_evblk(D, k, Cd1, j, lam, e2);
        for (int i = 0; i < k; i++) { vs[(size_t) nj * k + i] = e1[i] - e2[i]; mxs = dmax(mxs, fabs(e1[i] - e2[i])); }
        if (mxs > 1e-12) jl[nj++] = lam;
    }
    double *F = malloc((size_t) np * d1 * 3 * k * sizeof(double));
    for (int j = 0; j < np; j++) {
        double *Fj = F + (size_t) j * d1 * 3 * k; size_t bl = (size_t) d1 * k;
        memcpy(Fj, Cd + (size_t) j * bl, bl * sizeof(double)); memcpy(Fj + bl, Cd1 + (size_t) j * bl, bl * sizeof(double)); memcpy(Fj + 2 * bl, Cd2 + (size_t) j * bl, bl * sizeof(double));
    }
    int p; double *g, *Phi;
    det_ortho_basis(D, 3 * k, F, &p, &g, &Phi);
    int N = 1 + nj + p, iz = 1 + nj, k3 = 3 * k;
#define PD(a, l) Phi[(a) + (size_t) (l) * k3]
#define P1(a, l) Phi[(k + (a)) + (size_t) (l) * k3]
#define P2(a, l) Phi[(2 * k + (a)) + (size_t) (l) * k3]
    double *H0 = calloc((size_t) k * N, sizeof(double));
    det_evblk(D, k, Cd1, np - 1, 1.0, e1);
    for (int a = 0; a < k; a++) {
        H0[a] = e1[a];
        for (int i = 0; i < nj; i++) H0[a + (size_t) (1 + i) * k] = -vs[(size_t) i * k + a];
        for (int l = 0; l < p; l++) H0[a + (size_t) (iz + l) * k] = -P2(a, l);
    }
    double *Xi0 = malloc((size_t) k * k * sizeof(double));
    for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) { double s = 0; for (int l = 0; l < p; l++) s += P1(a, l) * P1(b, l); Xi0[a + b * k] = s; }
    /* Bb = Xi0 \ H0; Ba = Xi0 \ (P1 S - (P1 Pd') Bb) */
    double *Bb = malloc((size_t) k * N * sizeof(double)); memcpy(Bb, H0, (size_t) k * N * sizeof(double)); det_rsolve(k, Xi0, N, Bb);
    double *Ba = calloc((size_t) k * N, sizeof(double)), *PP = malloc((size_t) k * k * sizeof(double));
    for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) { double s = 0; for (int l = 0; l < p; l++) s += P1(a, l) * PD(b, l); PP[a + b * k] = s; }
    for (int a = 0; a < k; a++) for (int j = 0; j < N; j++) {
        double s = j >= iz ? P1(a, j - iz) : 0.0;
        for (int b = 0; b < k; b++) s -= PP[a + b * k] * Bb[b + (size_t) j * k];
        Ba[a + (size_t) j * k] = s;
    }
    det_rsolve(k, Xi0, N, Ba);
    /* W = [Bb; Ba] (2k x N), T = [Pd; P1] (2k x p): B = -(S' T' W + W' T S) + W' T T' W */
    int k2 = 2 * k;
    double *Wm = malloc((size_t) k2 * N * sizeof(double)), *Tm = malloc((size_t) k2 * imax(p, 1) * sizeof(double));
    for (int j = 0; j < N; j++) for (int a = 0; a < k; a++) { Wm[a + (size_t) j * k2] = Bb[a + (size_t) j * k]; Wm[k + a + (size_t) j * k2] = Ba[a + (size_t) j * k]; }
    for (int l = 0; l < p; l++) for (int a = 0; a < k; a++) { Tm[a + (size_t) l * k2] = PD(a, l); Tm[k + a + (size_t) l * k2] = P1(a, l); }
    double *TW = malloc((size_t) p * N * sizeof(double));          /* T' W (p x N) */
    for (int l = 0; l < p; l++) for (int j = 0; j < N; j++) { double s = 0; for (int a = 0; a < k2; a++) s += Tm[a + (size_t) l * k2] * Wm[a + (size_t) j * k2]; TW[l + (size_t) j * p] = s; }
    double *B = calloc((size_t) N * N, sizeof(double));
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        double s = 0;
        if (i >= iz) s -= TW[(i - iz) + (size_t) j * p];             /* S' T' W */
        if (j >= iz) s -= TW[(j - iz) + (size_t) i * p];             /* W' T S */
        for (int l = 0; l < p; l++) s += TW[l + (size_t) i * p] * TW[l + (size_t) j * p];
        B[i + (size_t) j * N] = s;
    }
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { double m = (B[i + (size_t) j * N] + B[j + (size_t) i * N]) / 2; B[i + (size_t) j * N] = B[j + (size_t) i * N] = m; }
    /* a1 = e_1 - Bb' delta(1) */
    det_evblk(D, k, Cd, np - 1, 1.0, e1);
    double a1n = 0;
    for (int j = 0; j < N; j++) { double s = j == 0 ? 1.0 : 0.0; for (int a = 0; a < k; a++) s -= Bb[a + (size_t) j * k] * e1[a]; a1n = dmax(a1n, fabs(s)); }
#undef PD
#undef P1
#undef P2
    P->K = k; P->p = p; P->nj = nj; P->N = N; P->g = g;
    P->jl = malloc(imax(nj, 1) * sizeof(double)); memcpy(P->jl, jl, nj * sizeof(double));
    P->A = calloc((size_t) N * N, sizeof(double)); P->A[0] = 0.5; P->B = B; P->b = calloc(N, sizeof(double));
    P->L = calloc(N, sizeof(double)); P->L[0] = 1; P->Dv = malloc(sizeof(double)); P->Dv[0] = 0.5; P->r = 1; P->rp = 1; P->rm = 0;
    P->key = det_design_key(D, NAN) ^ 0x4c4dULL; P->refs = 1;
    free(Z); free(Cd); free(Cd1); free(Cd2); free(e1); free(e2); free(jl); free(vs); free(F); free(Phi); free(H0); free(Xi0); free(Bb); free(Ba); free(PP); free(Wm); free(Tm); free(TW);
    *a1norm = a1n; *out = P;
    return 0;
}
static int lmprep(const dfl_design *D, det_prep **P, char *err, size_t errlen)
{
    double a1;
    if (det_prep_lm(D, P, &a1, err, errlen)) return 1;
    if (!(a1 < 1e-9)) { det_prep_free(*P); snprintf(err, errlen, "the LM statistic: the law of -1/(2 sqrt V) needs a linear trend among the trend functions (J_0(1) does not vanish)"); return 1; }
    return 0;
}
static double lmV(const det_path *CP, double v) { return 1 - det_vsum(CP, v, 1); }
int dfl_det_lm_cdf(const dfl_design *D, int n, const double *x, double c, double *out, char *err, size_t errlen)
{
    det_warn_clear();
    det_prep *P; if (lmprep(D, &P, err, errlen)) return 1;
    det_path *CP = det_path_prep(P, 0, 0.0, c, err, errlen);
    det_prep_release(P);
    if (!CP) return 1;
    for (int k = 0; k < n; k++) out[k] = isnan(x[k]) ? NAN : x[k] >= 0 ? 1.0 : x[k] == -INFINITY ? 0.0 : lmV(CP, 1 / (4 * x[k] * x[k]));
    det_path_release(CP);
    return 0;
}
static double lm_q(const det_path *CP, double p)
{
    double lo = 1e-6, hi = 2.0;
    while (lmV(CP, hi) < p) hi *= 2;
    for (int i = 0; i < 70; i++) { double m = sqrt(lo * hi); if (lmV(CP, m) > p) hi = m; else lo = m; }
    return -0.5 / sqrt(sqrt(lo * hi));
}
int dfl_det_lm_quantile(const dfl_design *D, double p, double c, double *q, char *err, size_t errlen)
{
    det_warn_clear();
    if (!(p > 0 && p < 1)) { snprintf(err, errlen, "lm_quantile: p must lie in (0, 1)"); return 1; }
    det_prep *P; if (lmprep(D, &P, err, errlen)) return 1;
    det_path *CP = det_path_prep(P, 0, 0.0, c, err, errlen);
    det_prep_release(P);
    if (!CP) return 1;
    *q = lm_q(CP, p);
    det_path_release(CP);
    return 0;
}
int dfl_det_lm_power(const dfl_design *D, int n, const double *cs, double level, double *q, double *out, char *err, size_t errlen)
{
    if (dfl_det_lm_quantile(D, level, 0.0, q, err, errlen)) return 1;
    det_prep *P; if (lmprep(D, &P, err, errlen)) return 1;
    int rc = 0;
    for (int k = 0; k < n && !rc; k++) {
        det_path *CP = det_path_prep(P, 0, 0.0, cs[k], err, errlen);
        if (!CP) { rc = 1; break; }
        out[k] = lmV(CP, 1 / (4 * *q * *q));
        det_path_release(CP);
    }
    det_prep_release(P);
    return rc;
}

/* ---------------------------------------------------------------------------------------------------------------- */
int dfl_det_mza_cdf(const dfl_design *D, int n, const double *y, double c, double gls, const dfl_det_opts *o0, double *out, char *err, size_t errlen)
{
    dfl_det_opts o; if (o0) o = *o0; else dfl_det_defaults(&o);
    o.gls = isnan(gls) ? -13.5 : gls;
    return dfl_det_coef(2, D, n, y, c, &o, out, err, errlen);
}
int dfl_det_mza_quantile(const dfl_design *D, double p, double c, double gls, double y0, double tol, const dfl_det_opts *o0, double *q, char *err, size_t errlen)
{
    dfl_det_opts o; if (o0) o = *o0; else dfl_det_defaults(&o);
    o.gls = isnan(gls) ? -13.5 : gls;
    return dfl_det_coef_quantile(D, 1, &p, c, isnan(y0) ? -20.0 : y0, tol, &o, q, err, errlen);
}
int dfl_det_msb_cdf(const dfl_design *D, int n, const double *x, double c, double gls, double *out, char *err, size_t errlen)
{
    det_warn_clear();
    double cb = isnan(gls) ? -13.5 : gls;
    det_path *CP = det_v_path(D, cb, 0, 0.0, c, NAN, 1.0, err, errlen);
    if (!CP) return 1;
    for (int k = 0; k < n; k++) out[k] = x[k] <= 0 ? 0.0 : 1 - det_vsum(CP, x[k] * x[k], 1);
    det_path_release(CP);
    return 0;
}
int dfl_det_msb_quantile(const dfl_design *D, double p, double c, double gls, double *q, char *err, size_t errlen)
{
    det_warn_clear();
    if (!(p > 0 && p < 1)) { snprintf(err, errlen, "msb_quantile: p must lie in (0, 1)"); return 1; }
    double cb = isnan(gls) ? -13.5 : gls;
    det_path *CP = det_v_path(D, cb, 0, 0.0, c, NAN, 1.0, err, errlen);
    if (!CP) return 1;
    double lo = 1e-4, hi = 1.0;
    while (1 - det_vsum(CP, hi, 1) < p) hi *= 2;
    for (int i = 0; i < 60; i++) { double m = (lo + hi) / 2; if (1 - det_vsum(CP, m, 1) > p) hi = m; else lo = m; }
    *q = sqrt((lo + hi) / 2);
    det_path_release(CP);
    return 0;
}
