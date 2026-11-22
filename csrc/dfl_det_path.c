/* dfl_det_path.c: the outer contours of the general engine (a port of DFDet.jl): the roots in s and log P(t) continued
 * along a path in t (rootpath_sparse, match!), the hyperbola t = tau + a (cosh u - 1) + i b sinh u (cpath, hyperbola,
 * fixed_phi) and the vertical line of Gauss-Legendre panels (linecpath, line_refine), the caches of the prepared designs,
 * the outer paths and the refined lines (behind one lock), and the stored paths of the classical cases at the unit root
 * (the tables of data/paths, dfl_det_tables.c). */
#include "dfl_det_int.h"

static dfl_lock_t CLOCK_ = DFL_LOCK_INIT;
#define DET_CACHE_MAX 64

double det_default_rmax(double c) { return c < -100 ? dmax(1e5, 20 * c * c) : 1e5; }
cdouble det_logMtilde(const cdouble *rts, int r, int rp, cdouble lP, cdouble s)
{
    cdouble acc = lP;
    for (int j = 0; j < rp; j++) acc = csub(acc, cscale(0.5, clog(csub(rts[j], s))));
    for (int j = rp; j < r; j++) acc = csub(acc, cscale(0.5, clog(csub(s, rts[j]))));
    return acc;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the cache of prepared designs */
typedef struct { unsigned long long key; det_prep *P; int refs; } prep_entry;
static prep_entry PREPC[DET_CACHE_MAX]; static int NPREPC = 0;
static void prep_ref(det_prep *P) { P->refs++; }
static void prep_unref(det_prep *P) { if (--P->refs == 0) det_prep_free(P); }
det_prep *det_prep_cached(const dfl_design *D, double cbar, char *err, size_t errlen)
{
    unsigned long long key = det_design_key(D, cbar);
    dfl_lock(&CLOCK_);
    for (int i = 0; i < NPREPC; i++) if (PREPC[i].key == key) { det_prep *P = PREPC[i].P; prep_ref(P); dfl_unlock(&CLOCK_); return P; }
    dfl_unlock(&CLOCK_);
    det_prep *P = NULL;
    if (det_prep_make(D, cbar, &P, err, errlen)) return NULL;
    dfl_lock(&CLOCK_);
    for (int i = 0; i < NPREPC; i++) if (PREPC[i].key == key) { det_prep *Q = PREPC[i].P; prep_ref(Q); dfl_unlock(&CLOCK_); det_prep_free(P); return Q; }
    if (NPREPC == DET_CACHE_MAX) { prep_unref(PREPC[0].P); memmove(PREPC, PREPC + 1, (NPREPC - 1) * sizeof(prep_entry)); NPREPC--; }
    PREPC[NPREPC].key = key; PREPC[NPREPC].P = P; NPREPC++;
    prep_ref(P);     /* the cache's (det_prep_make gave the caller's) */
    dfl_unlock(&CLOCK_);
    return P;
}
void det_prep_release(det_prep *P) { if (!P) return; dfl_lock(&CLOCK_); prep_unref(P); dfl_unlock(&CLOCK_); }

/* the models: a prepared design with its start, shared by the paths built on it */
static det_model_ref *mref_new(det_prep *P, int start, double g0, char *err, size_t errlen)
{
    det_model_ref *m = calloc(1, sizeof *m);
    if (det_model_make(P, start, g0, &m->Mo, err, errlen)) { free(m); return NULL; }
    m->P = P; m->refs = 1;
    return m;
}
static void mref_unref_locked(det_model_ref *m)   /* (with the lock held) */
{
    if (--m->refs > 0) return;
    det_model_free(&m->Mo);
    prep_unref(m->P);
    free(m);
}

/* ---------------------------------------------------------------------------------------------------------------- */
static void path_free_locked(det_path *CP)
{
    if (CP->fx) for (int k = 0; k < CP->K; k++) free(CP->fx[k]);
    free(CP->fx); free(CP->t); free(CP->dt); free(CP->R); free(CP->lP); free(CP->wt); free(CP->cuts);
    if (CP->mref) mref_unref_locked(CP->mref);
    free(CP);
}
void det_path_release(det_path *CP)
{
    if (!CP) return;
    dfl_lock(&CLOCK_);
    if (--CP->refs == 0) path_free_locked(CP);
    dfl_unlock(&CLOCK_);
}
static void path_ref(det_path *CP) { dfl_lock(&CLOCK_); CP->refs++; dfl_unlock(&CLOCK_); }

typedef struct { unsigned long long key; int start; double g0, c, phi, hfac; det_path *CP; } path_entry;
static path_entry PATHC[DET_CACHE_MAX]; static int NPATHC = 0;
typedef struct { unsigned long long pkey; double xb; int nab; double ab[8]; det_path *CP; } line_entry;
static line_entry LINEC[DET_CACHE_MAX]; static int NLINEC = 0;

void dfl_det_empty_caches(void)
{
    dfl_lock(&CLOCK_);
    for (int i = 0; i < NPATHC; i++) if (--PATHC[i].CP->refs == 0) path_free_locked(PATHC[i].CP);
    for (int i = 0; i < NLINEC; i++) if (--LINEC[i].CP->refs == 0) path_free_locked(LINEC[i].CP);
    for (int i = 0; i < NPREPC; i++) prep_unref(PREPC[i].P);
    NPATHC = NLINEC = NPREPC = 0;
    dfl_unlock(&CLOCK_);
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the assignment of the new roots to the previous labels: the permutation of least total distance (exhaustive for r <= 6,
 * in the order of permutations_of), greedy beyond */
static int PERM[7][720][6], NPERM[7]; static volatile int PERM_DONE = 0;
static void perms_init(void)
{
    if (PERM_DONE) return;
    dfl_lock(&CLOCK_);
    if (!PERM_DONE) {
        NPERM[1] = 1; PERM[1][0][0] = 0;
        for (int n = 2; n <= 6; n++) {
            int c = 0;
            for (int i = 0; i < NPERM[n - 1]; i++) for (int k = 0; k < n; k++) {
                int q = 0;
                for (int j = 0; j < n; j++) PERM[n][c][j] = (j < k) ? PERM[n - 1][i][j] : (j == k ? n - 1 : PERM[n - 1][i][j - 1]);
                (void) q; c++;
            }
            NPERM[n] = c;
        }
        PERM_DONE = 1;
    }
    dfl_unlock(&CLOCK_);
}
static void match(cdouble *nw, const cdouble *prev, int r)
{
    cdouble tmp[64];
    if (r <= 6) {
        double best = INFINITY; int bp = 0;
        for (int p = 0; p < NPERM[r]; p++) {
            double d = 0; for (int k = 0; k < r; k++) d += cabs(csub(nw[PERM[r][p][k]], prev[k]));
            if (d < best) { best = d; bp = p; }
        }
        for (int k = 0; k < r; k++) tmp[k] = nw[PERM[r][bp][k]];
        memcpy(nw, tmp, r * sizeof(cdouble));
        return;
    }
    int left[64], nl = r;
    for (int l = 0; l < r; l++) left[l] = l;
    for (int k = 0; k < r; k++) {
        int j = 0; double bd = INFINITY;
        for (int l = 0; l < nl; l++) { double d = cabs(csub(nw[left[l]], prev[k])); if (d < bd) { bd = d; j = l; } }
        tmp[k] = nw[left[j]];
        for (int l = j; l < nl - 1; l++) left[l] = left[l + 1];
        nl--;
    }
    memcpy(nw, tmp, r * sizeof(cdouble));
}

/* mgfparts at the points of a path, on the workers */
typedef struct { const det_model *Mo; double c; const cdouble *t; const int *idx; cdouble *MU, *LD; det_fixt **FX; det_status st; } mp_ctx;
static void mp_body(int k, void *v)
{
    mp_ctx *o = v;
    if (det_failed(&o->st)) return;
    char buf[DET_WORK_STACK]; det_work W;
    if (det_work_init(&W, o->Mo, buf, sizeof buf)) { det_fail(&o->st, "out of memory"); return; }
    det_fixt *fx = NULL;
    if (o->FX && o->Mo->start == 1) { fx = det_fixt_alloc(o->Mo->r); o->FX[k] = fx; }
    cdouble th, ld;
    cdouble t = o->t[o->idx ? o->idx[k] : k];
    if (det_mgfparts(&W, o->c, t, &th, &ld, o->MU + (size_t) k * o->Mo->r, fx)) det_fail(&o->st, "the eigenvalues of the reciprocal roots did not converge at t = %g%+gi", creal(t), cimag(t));
    o->LD[k] = ld;
    det_work_free(&W);
}
static cdouble lr(cdouble ld, const cdouble *rts, int r, int rp)
{
    cdouble L = ld;
    for (int j = 0; j < rp; j++) L = cadd(L, clog(cinv(rts[j])));
    for (int j = rp; j < r; j++) L = cadd(L, clog(cneg(cinv(rts[j]))));
    return L;
}
static inline cdouble cont2pi(cdouble Lr, double target) { return cadd(Lr, CPX(0, 2 * M_PI * jround((target - cimag(Lr)) / (2 * M_PI)))); }

/* rootpath_sparse: the roots R (r x K) and log P (K) at the nodes tt[0], tt[sub], ..., the fixed start's data at the nodes
 * (FX, K entries, may be NULL), minpos and maxneg */
static int rootpath_sparse(const det_model *Mo, double c, const cdouble *tt, int nt, int sub, cdouble **Rout, cdouble **lPout, det_fixt ***FXout,
                           double *minpos, double *maxneg, char *err, size_t errlen)
{
    perms_init();
    int r = Mo->r, rp = Mo->rp, K = (nt - 1) / sub + 1;
    int *idx = malloc(K * sizeof(int)); for (int k = 0; k < K; k++) idx[k] = k * sub;
    cdouble *MU = malloc((size_t) r * K * sizeof(cdouble)), *LD = malloc(K * sizeof(cdouble));
    det_fixt **FX = Mo->start == 1 ? calloc(K, sizeof(det_fixt *)) : NULL;
    mp_ctx g = {Mo, c, tt, idx, MU, LD, FX, {0, ""}};
    dfl_parfor(K, mp_body, &g);
    if (det_failed(&g.st)) {
        snprintf(err, errlen, "%s", g.st.msg);
        free(idx); free(MU); free(LD); if (FX) { for (int k = 0; k < K; k++) free(FX[k]); free(FX); }
        return 1;
    }
    cdouble *R = malloc((size_t) r * K * sizeof(cdouble)), *lP = malloc(K * sizeof(cdouble));
    /* the start: real roots, rp positive and rm negative */
    double mxi = 0, mxa = 0; int npos = 0, nneg = 0;
    double pos[64], neg[64];
    for (int k = 0; k < r; k++) {
        cdouble s1 = cinv(MU[k]); mxi = dmax(mxi, fabs(cimag(s1))); mxa = dmax(mxa, cabs(s1));
        if (creal(s1) > 0) pos[npos++] = creal(s1); else if (creal(s1) < 0) neg[nneg++] = creal(s1);
    }
    if (!(mxi < 1e-8 * mxa)) det_warn("rootpath: roots not real at the start");
    int rc = 0;
    if (npos != Mo->rp || nneg != Mo->rm) { snprintf(err, errlen, "rootpath: %d positive and %d negative roots at the start, inertia of A %d, %d", npos, nneg, Mo->rp, Mo->rm); rc = 1; }
    if (!rc) {
        for (int i = 1; i < npos; i++) { double v = pos[i]; int j = i - 1; while (j >= 0 && pos[j] > v) { pos[j + 1] = pos[j]; j--; } pos[j + 1] = v; }
        for (int i = 1; i < nneg; i++) { double v = neg[i]; int j = i - 1; while (j >= 0 && neg[j] > v) { neg[j + 1] = neg[j]; j--; } neg[j + 1] = v; }
        for (int k = 0; k < npos; k++) R[k] = CPX(pos[k], 0);
        for (int k = 0; k < nneg; k++) R[npos + k] = CPX(neg[k], 0);
        cdouble Lr1 = lr(LD[0], R, r, rp);
        double m = jround(cimag(Lr1) / (2 * M_PI));
        if (!(fabs(cimag(Lr1) - 2 * M_PI * m) < 1e-7)) det_warn("rootpath: D_V prod mu not positive at the start");
        cdouble L1 = CPX(creal(Lr1), 0);
        lP[0] = cscale(-0.5, L1);
        /* the points between the nodes evaluated so far (index j of tt): mu and log D_V */
        cdouble *spMU = NULL, *spLD = NULL; int *spJ = NULL, nsp = 0;
        int *flagged = malloc(K * sizeof(int)), nfl = 0;
        double mnp = 0, mxn = 0;
        char wbuf[DET_WORK_STACK]; det_work W; det_work_init(&W, Mo, wbuf, sizeof wbuf);
        cdouble nw[64], Rw[64], mu1[64];
        for (int pass = 0; pass < 2 && !rc; pass++) {
            int dry = pass == 0;
            cdouble Lprev = L1; double dprev = NAN;
            nfl = 0;
            mnp = INFINITY; for (int k = 0; k < rp; k++) mnp = dmin(mnp, creal(R[k]));
            mxn = -INFINITY; for (int k = rp; k < r; k++) mxn = dmax(mxn, creal(R[k]));
            for (int k = 1; k < K && !rc; k++) {
                const cdouble *Rp = R + (size_t) (k - 1) * r;
                double dpred = isnan(dprev) ? NAN : dprev * cabs(csub(tt[idx[k]], tt[idx[k - 1]])) / cabs(csub(tt[idx[k - 1]], tt[idx[k - 2]]));
                for (int i = 0; i < r; i++) nw[i] = cinv(MU[(size_t) k * r + i]);
                match(nw, Rp, r);
                cdouble Lr = lr(LD[k], nw, r, rp);
                cdouble Lc = cont2pi(Lr, cimag(Lprev) + (isnan(dpred) ? 0.0 : dpred));
                int safe = !isnan(dpred) && fabs(cimag(Lc) - cimag(Lprev) - dpred) <= M_PI / 4;
                if (safe) for (int i = 0; i < r; i++) {
                    double dmn = INFINITY;
                    for (int j = 0; j < r; j++) { double dj = cabs(csub(Rp[j], Rp[i])); if (!(j == i || dj <= 1e-6 * cabs(Rp[i]))) dmn = dmin(dmn, dj); }
                    if (!(cabs(csub(nw[i], Rp[i])) < 0.45 * dmn)) { safe = 0; break; }
                }
                if (!safe) {
                    flagged[nfl++] = k;
                    if (!dry) {
                        for (int i = 0; i < r; i++) Rw[i] = Rp[i];
                        cdouble Lw = Lprev;
                        for (int j = idx[k - 1] + 1; j <= idx[k]; j++) {
                            const cdouble *mu; cdouble ldj;
                            if (j < idx[k]) {
                                int q = -1; for (int l = 0; l < nsp; l++) if (spJ[l] == j) { q = l; break; }
                                if (q >= 0) { mu = spMU + (size_t) q * r; ldj = spLD[q]; }
                                else { cdouble th; if (det_mgfparts(&W, c, tt[j], &th, &ldj, mu1, NULL)) { snprintf(err, errlen, "the eigenvalues of the reciprocal roots did not converge"); rc = 1; break; } mu = mu1; }
                            } else { mu = MU + (size_t) k * r; ldj = LD[k]; }
                            cdouble nn[64]; for (int i = 0; i < r; i++) nn[i] = cinv(mu[i]);
                            match(nn, Rw, r);
                            cdouble L2 = lr(ldj, nn, r, rp);
                            Lw = cont2pi(L2, cimag(Lw));
                            for (int i = 0; i < r; i++) Rw[i] = nn[i];
                            if (j < idx[k]) {
                                for (int i = 0; i < rp; i++) mnp = dmin(mnp, creal(Rw[i]));
                                for (int i = rp; i < r; i++) mxn = dmax(mxn, creal(Rw[i]));
                            }
                        }
                        for (int i = 0; i < r; i++) nw[i] = Rw[i];
                        Lc = Lw;
                    }
                }
                for (int i = 0; i < r; i++) R[(size_t) k * r + i] = nw[i];
                lP[k] = cscale(-0.5, Lc);
                dprev = cimag(Lc) - cimag(Lprev); Lprev = Lc;
                for (int i = 0; i < rp; i++) mnp = dmin(mnp, creal(nw[i]));
                for (int i = rp; i < r; i++) mxn = dmax(mxn, creal(nw[i]));
            }
            if (dry && !rc) {
                /* the points between the nodes of the flagged steps, on the workers */
                int npt = 0; for (int f = 0; f < nfl; f++) npt += idx[flagged[f]] - idx[flagged[f] - 1] - 1;
                if (npt > 0) {
                    spJ = malloc(npt * sizeof(int)); spMU = malloc((size_t) npt * r * sizeof(cdouble)); spLD = malloc(npt * sizeof(cdouble));
                    for (int f = 0; f < nfl; f++) for (int j = idx[flagged[f] - 1] + 1; j < idx[flagged[f]]; j++) spJ[nsp++] = j;
                    mp_ctx g2 = {Mo, c, tt, spJ, spMU, spLD, NULL, {0, ""}};
                    dfl_parfor(nsp, mp_body, &g2);
                    if (det_failed(&g2.st)) { snprintf(err, errlen, "%s", g2.st.msg); rc = 1; }
                }
            }
        }
        det_work_free(&W);
        free(spJ); free(spMU); free(spLD); free(flagged);
        *minpos = mnp; *maxneg = Mo->rm == 0 ? -INFINITY : mxn;
    }
    free(idx); free(MU); free(LD);
    if (rc) { free(R); free(lP); if (FX) { for (int k = 0; k < K; k++) free(FX[k]); free(FX); } return 1; }
    *Rout = R; *lPout = lP; *FXout = FX;
    return 0;
}

/* ---------------------------------------------------------------------------------------------------------------- */
static void hyperbola(double ts, double phi, double hfac, double rmax, double tau, double *a, double *b, double *h, int *K)
{
    double y = dmin(0.5, 0.9 * (M_PI / 2 - phi));
    *b = 0.9 * dmin(tau, ts - tau) / sin(y); *a = *b / tan(phi);
    *h = hfac * 2 * M_PI * y / (37 + 37 * tan(phi) * y);
    *K = (int) ceil(asinh(rmax / hypot(*a, *b)) / *h);
}
static const double PHI_FIXED[15] = {45, 50, 55, 60, 65, 70, 75, 78, 80, 82, 84, 85, 86, 87, 88};
typedef struct { const det_model *Mo; double c, tau, a, b, h, g2; int k0, k1; volatile int bad; det_status st; } fp_ctx;
static void fp_body(int k, void *v)
{
    fp_ctx *o = v;
    if (o->bad || det_failed(&o->st)) return;
    k += o->k0;
    char buf[DET_WORK_STACK]; det_work W;
    if (det_work_init(&W, o->Mo, buf, sizeof buf)) { det_fail(&o->st, "out of memory"); return; }
    det_fixt *fx = o->Mo->start == 1 ? det_fixt_alloc(o->Mo->r) : NULL;
    cdouble t = CPX(o->tau + o->a * (cosh(k * o->h) - 1), o->b * sinh(k * o->h)), dt = CPX(o->a * sinh(k * o->h), o->b * cosh(k * o->h)), th, ld, mu[64];
    if (det_mgfparts(&W, o->c, t, &th, &ld, mu, fx)) det_fail(&o->st, "the eigenvalues of the reciprocal roots did not converge");
    double val = exp(-creal(ld) / 2 + (fx ? o->g2 * creal(fx->cf) : 0.0)) * cabs(cdiv(dt, t)) * o->h / M_PI;
    if (!(isfinite(val) && val <= 2)) o->bad = 1;
    free(fx); det_work_free(&W);
}
/* the asymptotic angle with a fixed start (and below c = -100): the smallest angle of PHI_FIXED at which every term
 * |M_V(t_k) e^{g0^2 cf(t_k)} t'(u_k)/t_k| h/pi is at most 2 (fixed_phi) */
static int fixed_phi(const det_model *Mo, double c, double ts, double tau, double hfac, double rmax, double *phiout, char *err, size_t errlen)
{
    int nth = dfl_nworkers();
    for (int i = 0; i < 15; i++) {
        double phi = PHI_FIXED[i] * M_PI / 180, a, b, h; int K;
        hyperbola(ts, phi, hfac, rmax, tau, &a, &b, &h, &K);
        int ok = 1, k0 = 0, B = 16 * nth;
        while (ok && k0 <= K) {
            int k1 = imin(K, k0 + B - 1);
            fp_ctx g = {Mo, c, tau, a, b, h, Mo->g0 * Mo->g0, k0, k1, 0, {0, ""}};
            dfl_parfor(k1 - k0 + 1, fp_body, &g);
            if (det_failed(&g.st)) { snprintf(err, errlen, "%s", g.st.msg); return 1; }
            ok = !g.bad; k0 = k1 + 1;
        }
        if (ok) { *phiout = phi; return 0; }
    }
    *phiout = PHI_FIXED[14] * M_PI / 180;
    return 0;
}

static det_path *path_alloc(int K, int r)
{
    det_path *CP = calloc(1, sizeof *CP);
    CP->K = K; CP->r = r;
    CP->t = malloc(K * sizeof(cdouble)); CP->dt = malloc(K * sizeof(cdouble)); CP->lP = malloc(K * sizeof(cdouble));
    CP->wt = malloc(K * sizeof(double));
    CP->refs = 1;
    return CP;
}
/* linecpath: the vertical line t = tau + i w, Gauss-Legendre panels of nq nodes at cuts (NULL: the first of length
 * min(tau, 1)/4, doubling up to rmax), the roots continued on sub points between nodes */
static det_path *linecpath(det_model_ref *mr, double c, double ts, double tau, double rmax, int nq, const double *cuts0, int ncuts0,
                           char *err, size_t errlen)
{
    const det_model *Mo = &mr->Mo;
    const int sub = 8;
    const double *w, *z = det_gl(nq, &w);
    dvec cuts = {0};
    if (!cuts0) {
        dvec_push(&cuts, 0.0); double hh = dmin(tau, 1.0) / 4;
        while (cuts.v[cuts.n - 1] < rmax) { dvec_push(&cuts, cuts.v[cuts.n - 1] + hh); hh *= 2; }
    } else for (int i = 0; i < ncuts0; i++) dvec_push(&cuts, cuts0[i]);
    int npan = cuts.n - 1, nn = npan * nq;
    double *ws = malloc(nn * sizeof(double)), *wq = malloc(nn * sizeof(double));
    for (int k = 0; k < npan; k++) for (int q = 0; q < nq; q++) {
        ws[k * nq + q] = cuts.v[k] + (cuts.v[k + 1] - cuts.v[k]) * z[q]; wq[k * nq + q] = (cuts.v[k + 1] - cuts.v[k]) * w[q] / M_PI;
    }
    int nt = 1 + nn * sub;
    cdouble *tt = malloc(nt * sizeof(cdouble));
    tt[0] = CPX(tau, 0.0);
    double prev = 0; int j = 1;
    for (int k = 0; k < nn; k++) {
        for (int l = 1; l <= sub - 1; l++) tt[j++] = CPX(tau, prev + (ws[k] - prev) * l / sub);
        tt[j++] = CPX(tau, ws[k]); prev = ws[k];
    }
    cdouble *R, *lP; det_fixt **FX; double mnp, mxn;
    if (rootpath_sparse(Mo, c, tt, nt, sub, &R, &lP, &FX, &mnp, &mxn, err, errlen)) { free(ws); free(wq); free(tt); dvec_free(&cuts); return NULL; }
    det_path *CP = path_alloc(nn, Mo->r);
    CP->c = c; CP->tau = tau; CP->tstar = ts; CP->a = 0; CP->b = 1; CP->h = NAN; CP->phi = M_PI / 2;
    int r = Mo->r;
    CP->R = malloc((size_t) r * imax(nn, 1) * sizeof(cdouble));
    for (int k = 0; k < nn; k++) {
        CP->t[k] = tt[(k + 1) * sub]; CP->dt[k] = CPX(0, 1); CP->wt[k] = wq[k];
        memcpy(CP->R + (size_t) k * r, R + (size_t) (k + 1) * r, r * sizeof(cdouble)); CP->lP[k] = lP[k + 1];
    }
    if (FX) { CP->fx = calloc(nn, sizeof(det_fixt *)); for (int k = 0; k < nn; k++) CP->fx[k] = FX[k + 1]; free(FX[0]); free(FX); }
    CP->minpos = mnp; CP->maxneg = mxn;
    CP->line = 1; CP->nq = nq; CP->npan = npan; CP->cuts = cuts.v; CP->refined = cuts0 != NULL;
    CP->mref = mr; CP->Mo = &mr->Mo;
    dfl_lock(&CLOCK_); mr->refs++; dfl_unlock(&CLOCK_);
    free(ws); free(wq); free(tt); free(R); free(lP);
    return CP;
}

/* cpath: the hyperbola of the design with its start at c (phi NaN: pi/4, or fixed_phi with a fixed start and below c = -100,
 * steepened while the positive roots leave Re s > 0, the vertical line beyond 88 degrees) */
static det_path *cpath(det_model_ref *mr, double c, double phi, double hfac, double tau0, char *err, size_t errlen)
{
    const det_model *Mo = &mr->Mo;
    const int sub = 8;
    double rmax = det_default_rmax(c), ts;
    if (det_tstar(Mo, c, &ts, NULL, err, errlen)) return NULL;
    double tau = isnan(tau0) ? dmin(1.0, ts / 2) : tau0;
    int autophi = isnan(phi);
    if (autophi) {
        if (Mo->start == 1 || c < -100) { if (fixed_phi(Mo, c, ts, tau, hfac, rmax, &phi, err, errlen)) return NULL; }
        else phi = M_PI / 4;
    }
    double a, b, h, mnp, mxn; int K;
    cdouble *R = NULL, *lP = NULL; det_fixt **FX = NULL; cdouble *tt = NULL; int nt = 0;
    for (;;) {
        hyperbola(ts, phi, hfac, rmax, tau, &a, &b, &h, &K);
        nt = sub * K + 1;
        tt = malloc(nt * sizeof(cdouble));
        for (int k = 0; k < nt; k++) { double u = k * h / sub; tt[k] = CPX(tau + a * (cosh(u) - 1), b * sinh(u)); }
        int rc = rootpath_sparse(Mo, c, tt, nt, sub, &R, &lP, &FX, &mnp, &mxn, err, errlen);
        free(tt); tt = NULL;
        if (rc) return NULL;
        if (!(autophi && mnp <= 0 && phi < 88 * M_PI / 180)) break;
        free(R); free(lP); R = lP = NULL;
        if (FX) { for (int k = 0; k <= K; k++) free(FX[k]); free(FX); FX = NULL; }
        phi = dmin(phi + 5 * M_PI / 180, 88 * M_PI / 180);
    }
    if (autophi && mnp <= 0) {
        free(R); free(lP); if (FX) { for (int k = 0; k <= K; k++) free(FX[k]); free(FX); }
        return linecpath(mr, c, ts, tau, rmax, 24, NULL, 0, err, errlen);
    }
    det_path *CP = path_alloc(K + 1, Mo->r);
    CP->c = c; CP->tau = tau; CP->tstar = ts; CP->a = a; CP->b = b; CP->h = h; CP->phi = phi;
    for (int k = 0; k <= K; k++) {
        CP->t[k] = CPX(tau + a * (cosh(k * h) - 1), b * sinh(k * h));
        CP->dt[k] = CPX(a * sinh(k * h), b * cosh(k * h));
        CP->wt[k] = h / M_PI;
    }
    CP->wt[0] /= 2;
    CP->R = R; CP->lP = lP; CP->fx = FX; CP->minpos = mnp; CP->maxneg = mxn;
    CP->mref = mr; CP->Mo = &mr->Mo;
    dfl_lock(&CLOCK_); mr->refs++; dfl_unlock(&CLOCK_);
    return CP;
}

/* the classical cases at the unit root, whose paths are tables (table_case): p in -1, 0, 1, or -2 */
static int table_case(const det_prep *P, const det_model *Mo, double c)
{
    if (!(c == 0 && Mo->start == 0 && isnan(P->cbar))) return -2;
    if (P->first || P->loc || P->np != 1) return -2;
    for (int p = -1; p <= 1; p++) {
        int K = p + 1, deg = p > 0 ? p : 0;
        if (P->Kd != K || P->deg != deg) continue;
        int ok = 1;
        for (int i = 0; i < K && ok; i++) for (int k = 0; k <= deg; k++) if (P->C[i * (deg + 1) + k] != (k == i ? 1.0 : 0.0)) { ok = 0; break; }
        if (ok) return p;
    }
    return -2;
}
static det_path *table_path(det_model_ref *mr, int p)
{
    const det_path_table *T = &DET_PATH_TABLES[p + 1];
    if (T->r != mr->Mo.r || T->rp != mr->Mo.rp) return NULL;
    det_path *CP = path_alloc(T->K, T->r);
    CP->c = T->sc[0]; CP->tau = T->sc[1]; CP->tstar = T->sc[2]; CP->a = T->sc[3]; CP->b = T->sc[4]; CP->h = T->sc[5]; CP->phi = T->sc[6];
    CP->minpos = T->sc[7]; CP->maxneg = T->sc[8];
    if (CP->maxneg <= -1e300) CP->maxneg = -INFINITY;      /* (the tables store an infinite scalar as +-1e308) */
    if (CP->minpos >= 1e300) CP->minpos = INFINITY;
    CP->R = malloc((size_t) T->K * T->r * sizeof(cdouble));
    for (int k = 0; k < T->K; k++) {
        CP->t[k] = CPX(T->t[2 * k], T->t[2 * k + 1]); CP->dt[k] = CPX(T->dt[2 * k], T->dt[2 * k + 1]);
        CP->lP[k] = CPX(T->lP[2 * k], T->lP[2 * k + 1]); CP->wt[k] = T->wt[k];
    }
    for (int i = 0; i < T->K * T->r; i++) CP->R[i] = CPX(T->R[2 * i], T->R[2 * i + 1]);
    CP->mref = mr; CP->Mo = &mr->Mo;
    dfl_lock(&CLOCK_); mr->refs++; dfl_unlock(&CLOCK_);
    return CP;
}

int DET_USE_TABLES = 1;
/* an outer path of a prepared design that is not in the cache (the LM statistic), not cached: the caller keeps its own
 * reference to P */
det_path *det_path_prep(det_prep *P, int start, double g0, double c, char *err, size_t errlen)
{
    dfl_lock(&CLOCK_); prep_ref(P); dfl_unlock(&CLOCK_);
    det_model_ref *mr = mref_new(P, start, g0, err, errlen);
    if (!mr) { det_prep_release(P); return NULL; }
    det_path *CP = cpath(mr, c, NAN, 1.0, NAN, err, errlen);
    dfl_lock(&CLOCK_); mref_unref_locked(mr); dfl_unlock(&CLOCK_);
    return CP;
}
det_path *det_path_cached(const dfl_design *D, double cbar, int start, double g0, double c, double phi, double hfac, char *err, size_t errlen)
{
    if (start == 2 && !(c < 0)) { snprintf(err, errlen, "the stationary start needs c < 0"); return NULL; }
    if (start == 1 && g0 == 0) start = 0;
    if (start != 1) g0 = 0;
    unsigned long long key = det_design_key(D, cbar);
    double phik = isnan(phi) ? -1.0 : phi;
    dfl_lock(&CLOCK_);
    for (int i = 0; i < NPATHC; i++) {
        path_entry *e = &PATHC[i];
        if (e->key == key && e->start == start && e->g0 == g0 && e->c == c && e->phi == phik && e->hfac == hfac) { e->CP->refs++; det_path *CP = e->CP; dfl_unlock(&CLOCK_); return CP; }
    }
    dfl_unlock(&CLOCK_);
    det_prep *P = det_prep_cached(D, cbar, err, errlen);
    if (!P) return NULL;
    det_model_ref *mr = mref_new(P, start, g0, err, errlen);
    if (!mr) { det_prep_release(P); return NULL; }
    det_path *CP = NULL;
    int tp = isnan(phi) && hfac == 1 ? table_case(P, &mr->Mo, c) : -2;
    if (tp >= -1 && DET_USE_TABLES) CP = table_path(mr, tp);
    if (!CP) CP = cpath(mr, c, phi, hfac, NAN, err, errlen);
    dfl_lock(&CLOCK_);
    mref_unref_locked(mr);              /* (the path holds its own reference) */
    if (CP) {
        int dup = -1;
        for (int i = 0; i < NPATHC; i++) { path_entry *e = &PATHC[i]; if (e->key == key && e->start == start && e->g0 == g0 && e->c == c && e->phi == phik && e->hfac == hfac) dup = i; }
        if (dup >= 0) { if (--CP->refs == 0) path_free_locked(CP); CP = PATHC[dup].CP; CP->refs++; }
        else {
            if (NPATHC == DET_CACHE_MAX) { if (--PATHC[0].CP->refs == 0) path_free_locked(PATHC[0].CP); memmove(PATHC, PATHC + 1, (NPATHC - 1) * sizeof(path_entry)); NPATHC--; }
            path_entry e = {key, start, g0, c, phik, hfac, CP};
            PATHC[NPATHC++] = e;
            CP->refs++;                 /* the cache's */
        }
    }
    dfl_unlock(&CLOCK_);
    return CP;
}

/* a key for the contents of a path (the line cache) */
static unsigned long long path_pkey(const det_path *CP)
{
    unsigned long long h = 1469598103934665603ULL;
    const unsigned char *b = (const unsigned char *) CP->t;
    for (size_t i = 0; i < (size_t) CP->K * sizeof(cdouble); i++) { h ^= b[i]; h *= 1099511628211ULL; }
    double s[3] = {CP->c, CP->Mo->g0, (double) CP->Mo->start};
    b = (const unsigned char *) s; for (size_t i = 0; i < sizeof s; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    unsigned long long k = CP->Mo->P->key; b = (const unsigned char *) &k; for (int i = 0; i < 8; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
/* line_refine: the panels of the line cut to the oscillation of the outer integrand at |x| <= xmax (amax: the constants of
 * the kernels, the first the shift a of the model) */
static det_path *line_refine(det_path *CP, double xmax, int na, const double *amax, char *err, size_t errlen)
{
    double amx = -INFINITY; for (int i = 0; i < na; i++) amx = dmax(amx, amax[i]);
    if (!(isnan(CP->h) && (xmax > 0 || amx > CP->Mo->a)) || !CP->line || CP->refined) { path_ref(CP); return CP; }
    int nq = CP->nq, np = CP->npan, r = CP->r;
    double tau = CP->tau;
    dvec nc = {0}; dvec_push(&nc, 0.0); int changed = 0;
    for (int k = 0; k < np; k++) {
        double a = CP->cuts[k], b = CP->cuts[k + 1], Om = 0;
        int js[2] = {k * nq, k * nq + nq - 1};
        for (int jj = 0; jj < 2; jj++) {
            int j = js[jj]; double wj = cimag(CP->t[j]);
            if (!(wj > 0)) continue;
            const cdouble *Rj = CP->R + (size_t) j * r;
            double lmv = creal(det_logMtilde(Rj, r, CP->Mo->rp, CP->lP[j], DZERO));
            for (int i = 0; i < r; i++) for (int l = 0; l < na; l++) {
                double am = amax[l], kap = fabs(cimag(Rj[i])) / wj, ar = fabs(creal(Rj[i]));
                double nu = 40 + lmv - ar * am, de = tau * kap * kap / 4 + ar * kap / 2;
                if (!(nu > 0)) continue;
                double xu = de > 0 ? dmin(xmax, sqrt(nu / de)) : xmax;
                Om = dmax(Om, kap * kap * xu * xu / 4 + kap * am);
            }
        }
        int m = imax(1, (int) ceil(dmin((b - a) * Om / 30, 1e6)));
        if (m > 1) changed = 1;
        for (int i = 1; i <= m; i++) dvec_push(&nc, a + (b - a) * i / m);
    }
    if (!changed) { dvec_free(&nc); path_ref(CP); return CP; }
    long nn = (long) (nc.n - 1) * nq;
    if (nn > 400000) { snprintf(err, errlen, "line_refine: the line would need %ld nodes for |x| <= %g at c = %g, more than 400000 (the oscillation of the outer integrand grows with x; at c = 10 the bulk needs about 60000)", nn, xmax, CP->c); dvec_free(&nc); return NULL; }
    if (nc.n > 4000) det_warn("line_refine: the line needs %d panels for |x| <= %g at c = %g (the oscillation of the outer integrand grows with x); this is slow", nc.n - 1, xmax, CP->c);
    det_path *CPr = linecpath(CP->mref, CP->c, CP->tstar, tau, 1e5, nq, nc.v, nc.n, err, errlen);
    dvec_free(&nc);
    return CPr;
}
static int cmp_dbl(const void *a, const void *b) { double x = *(const double *) a, y = *(const double *) b; return (x > y) - (x < y); }
det_path *det_line_refined(det_path *CP, int nx, const double *xs, int na, const double *amax, char *err, size_t errlen)
{
    if (!isnan(CP->h)) { path_ref(CP); return CP; }
    double xm = 0; int any = 0;
    for (int i = 0; i < nx; i++) if (isfinite(xs[i])) { xm = any ? dmax(xm, fabs(xs[i])) : fabs(xs[i]); any = 1; }
    double xb = xm > 0 ? pow(2.0, ceil(log2(xm))) : 0.0;
    double ab[9]; int nab = 0;
    for (int i = 0; i < na; i++) if (isfinite(amax[i]) && amax[i] > CP->Mo->a) {
        double v = pow(2.0, ceil(log2(amax[i]))); int dup = 0;
        for (int l = 0; l < nab; l++) if (ab[l] == v) dup = 1;
        if (!dup && nab < 8) ab[nab++] = v;
    }
    qsort(ab, nab, sizeof(double), cmp_dbl);
    if (!(xb > 0 || nab > 0)) { path_ref(CP); return CP; }
    unsigned long long pk = path_pkey(CP);
    dfl_lock(&CLOCK_);
    for (int i = 0; i < NLINEC; i++) {
        line_entry *e = &LINEC[i];
        if (e->pkey == pk && e->xb == xb && e->nab == nab && !memcmp(e->ab, ab, nab * sizeof(double))) { e->CP->refs++; det_path *r = e->CP; dfl_unlock(&CLOCK_); return r; }
    }
    dfl_unlock(&CLOCK_);
    double am[9]; am[0] = CP->Mo->a; for (int l = 0; l < nab; l++) am[l + 1] = ab[l];
    det_path *CPr = line_refine(CP, xb, nab + 1, am, err, errlen);
    if (!CPr) return NULL;
    dfl_lock(&CLOCK_);
    if (NLINEC == DET_CACHE_MAX) { if (--LINEC[0].CP->refs == 0) path_free_locked(LINEC[0].CP); memmove(LINEC, LINEC + 1, (NLINEC - 1) * sizeof(line_entry)); NLINEC--; }
    line_entry e; e.pkey = pk; e.xb = xb; e.nab = nab; memcpy(e.ab, ab, sizeof ab); e.CP = CPr;
    LINEC[NLINEC++] = e; CPr->refs++;
    dfl_unlock(&CLOCK_);
    return CPr;
}

/* the outer contour of a variant rule of the checked computation (variant_path) */
det_path *det_variant_path(det_path *CP, int level, char *err, size_t errlen)
{
    det_model_ref *mr = CP->mref;
    if (CP->phi == M_PI / 2 && isnan(CP->h)) return linecpath(mr, CP->c, CP->tstar, CP->tau, 1e5, level == 2 ? 48 : 32, NULL, 0, err, errlen);
    /* (the variant hyperbolas are cached as in Julia: cpath_cached with an angle and a step factor) */
    double phi = level == 1 ? dmin(CP->phi + 7.5 * M_PI / 180, 88 * M_PI / 180) : CP->phi, hfac = level == 1 ? 0.8 : level == 0 ? 0.8 : 0.5;
    const det_prep *P = CP->Mo->P;
    dfl_design D = {P->Kd, P->np, P->deg, P->br, P->C, P->loc, P->first};
    return det_path_cached(&D, P->cbar, CP->Mo->start, CP->Mo->g0, CP->c, phi, hfac, err, errlen);
}

/* the outer contour of the law of V alone: the design's path, except where that is the vertical line (v_path) */
det_path *det_v_path(const dfl_design *D, double cbar, int start, double g0, double c, double phi, double hfac, char *err, size_t errlen)
{
    det_path *CP = det_path_cached(D, cbar, start, g0, c, phi, hfac, err, errlen);
    if (!CP) return NULL;
    if (!(isnan(phi) && isnan(CP->h))) return CP;
    const det_model *Mo = CP->Mo;
    double ph = M_PI / 4;
    if (Mo->start == 1 || c < -100) { if (fixed_phi(Mo, c, CP->tstar, CP->tau, hfac, det_default_rmax(c), &ph, err, errlen)) { det_path_release(CP); return NULL; } }
    int st = Mo->start; double gg = Mo->g0;
    det_path_release(CP);
    return det_path_cached(D, cbar, st, gg, c, ph, hfac, err, errlen);
}
