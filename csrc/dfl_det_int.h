/* dfl_det_int.h: the internals of the general engine (dfl_det_*.c), a port of src/DFDet.jl and src/DFDet_paper2.jl of
 * DFLimit.jl (the reference implementation): the prepared design, the model with its start, the work space of the
 * transform, the outer paths, the small dense linear algebra, and the helpers shared by the files.  Not part of the
 * public interface (dflimit.h); the test program dfl_test_det.c includes it for the checks of the layers. */
#ifndef DFL_DET_INT_H
#define DFL_DET_INT_H
#include "dflimit.h"
#include "dfl_lock.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <float.h>

/* ---- complex helpers (the operators of cdouble are functions, for MSVC) ---- */
static const cdouble DZERO = {0}, DONE = {1};
static inline double cabs2(cdouble z) { double a = creal(z), b = cimag(z); return a * a + b * b; }
static inline double cfabs(cdouble z) { return sqrt(cabs2(z)); }
static inline cdouble cmuladd(cdouble a, cdouble b, cdouble c) { return cadd(cmul(a, b), c); }
static inline cdouble cinv(cdouble z) { return cdiv(DONE, z); }
static inline cdouble cdivr(cdouble a, double b) { return CPX(creal(a) / b, cimag(a) / b); }
static inline cdouble caddr(cdouble a, double b) { return CPX(creal(a) + b, cimag(a)); }
static inline cdouble cis_(double x) { return CPX(cos(x), sin(x)); }
static inline cdouble cmulr(cdouble a, double b) { return CPX(creal(a) * b, cimag(a) * b); }
static inline cdouble cI(cdouble z) { return CPX(-cimag(z), creal(z)); }          /* i z */
static inline bool cisfinite(cdouble z) { return isfinite(creal(z)) && isfinite(cimag(z)); }
/* the principal square root (dfl_csqrt guards the libm csqrt at the edges of the range) */
static inline cdouble csq(cdouble z) { return dfl_csqrt(z); }
/* theta = sqrt(c^2 - 2t), formed as Julia forms complex(c * c - 2t) (the imaginary part -2 Im t, so that a real t beyond
 * the branch point gives the same signed zero, and the same branch, as there) */
static inline cdouble det_theta(double c, cdouble t) { return csq(CPX(c * c - 2 * creal(t), -2 * cimag(t))); }
/* expm1 for complex z (Julia's expm1(::Complex)): accurate for small |z| */
static inline cdouble cexpm1(cdouble z)
{
    double x = creal(z), y = cimag(z);
    if (fabs(x) < 0.5 && fabs(y) < 0.5) {
        double em = expm1(x), sy = sin(y), sh = sin(y / 2);
        return CPX(em * cos(y) - 2 * sh * sh, (em + 1) * sy);
    }
    return csub(cexp(z), DONE);
}
static inline double sgn1(double x) { return x >= 0 ? 1.0 : -1.0; }
static inline double dmax(double a, double b) { return a > b ? a : b; }
static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double dclamp(double x, double lo, double hi) { return x > hi ? hi : x < lo ? lo : x; }
static inline int imax(int a, int b) { return a > b ? a : b; }
static inline int imin(int a, int b) { return a < b ? a : b; }
/* Julia's round (half to even) */
static inline double jround(double x) { return nearbyint(x); }
/* the derivative of a column of coefficients of r^k: k c[k+1] (the last row zero), as dcoef */
/* evalpoly(x, c[0..n-1]) by Horner */
static inline double evalpoly(double x, const double *c, int n) { double s = 0; for (int k = n - 1; k >= 0; k--) s = s * x + c[k]; return s; }

/* ---- dynamic arrays ---- */
typedef struct { cdouble *v; int n, cap; } cvec;
typedef struct { double *v; int n, cap; } dvec;
static inline int cvec_push(cvec *a, cdouble x) { if (a->n == a->cap) { int nc = a->cap ? 2 * a->cap : 64; cdouble *p = realloc(a->v, nc * sizeof(cdouble)); if (!p) return 1; a->v = p; a->cap = nc; } a->v[a->n++] = x; return 0; }
static inline int dvec_push(dvec *a, double x) { if (a->n == a->cap) { int nc = a->cap ? 2 * a->cap : 64; double *p = realloc(a->v, nc * sizeof(double)); if (!p) return 1; a->v = p; a->cap = nc; } a->v[a->n++] = x; return 0; }
static inline void cvec_free(cvec *a) { free(a->v); a->v = NULL; a->n = a->cap = 0; }
static inline void dvec_free(dvec *a) { free(a->v); a->v = NULL; a->n = a->cap = 0; }

/* ---- errors and warnings ---- */
/* the status of a computation that may run on the workers: the first error is kept (under a lock), with its message */
typedef struct { volatile int err; char msg[512]; } det_status;
void det_fail(det_status *st, const char *fmt, ...);
static inline int det_failed(const det_status *st) { return *(const volatile int *) &st->err; }
/* the warning of the general engine on the calling thread (dfl_det_warning): the messages of one user-level call, joined */
void det_warn(const char *fmt, ...);
void det_warn_clear(void);
void det_warn_hold(int on);       /* on = 1: keep the warnings across the calls that follow, 0: release */
/* the classical engine's warning (dfl_warning, dfl_api.c) joined with msg, each message once: the router's composite calls */
void dfl_warning_join(const char *msg);

/* ---- small dense linear algebra (dfl_det_la.c); matrices column-major ---- */
/* symmetric eigen-decomposition by the cyclic Jacobi method: eigenvalues ascending in w, eigenvectors in the columns of V
 * (V may be NULL); A (n x n) is not changed */
int det_symeig(int n, const double *A, double *w, double *V);
/* the null space of the m x n matrix A to the tolerance atol (singular values <= atol, as nullspace(A; atol)): its
 * dimension k, the orthonormal basis in Z (n x k, Z holds n x n) */
int det_nullspace(int m, int n, const double *A, double atol, double *Z);
/* solve A X = B for X (A n x n real, B n x m, overwritten by X), partial pivoting; 1 if singular */
int det_rsolve(int n, const double *A, int m, double *B);
/* the complex LU factorization with partial pivoting (pivots by |.|^2, as slu): F (n x n, the unit lower factor below the
 * diagonal and U on and above), p the permutation (row i of PA is row p[i] of A), odd its parity */
void det_clu(int n, cdouble *F, int *p, int *odd);
cdouble det_clogdet(int n, const cdouble *F, int odd);       /* log det from the LU factors, as LinearAlgebra's logdet */
cdouble det_cdet(int n, const cdouble *F, int odd);
void det_csolve(int n, const cdouble *F, const int *p, cdouble *b);    /* A \ b in place */
void det_csolvet(int n, const cdouble *F, const int *p, cdouble *b);   /* transpose(A) \ b in place */
/* the eigenvalues of a small complex matrix (seigvals: the closed forms for sizes 1 and 2, balancing, Hessenberg form by
 * Givens rotations and the single-shift QR algorithm); H (r x r) is overwritten; 1 if it does not converge */
int det_ceigvals(int r, cdouble *H, cdouble *ev);
/* the largest eigenvalues of a symmetric operator by subspace iteration with Rayleigh-Ritz (blocks of nb vectors):
 * apply(x, y, ctx) sets y = A x for nb columns (n x nb); the two largest in l1 >= l2 */
int det_topeig(int n, void (*apply)(int nb, const double *X, double *Y, void *ctx), void *ctx, double *l1, double *l2);
/* the orthonormal basis of the columns of X (n x k) by modified Gram-Schmidt, twice (the thin factor Q of a QR); rank in *rk */
void det_orth(int n, int k, double *X);
/* Householder QR with column pivoting of the n x k matrix A (overwritten: R on and above the diagonal), perm the column
 * order; for the dependent columns of design_from_matrix */
void det_qrp(int n, int k, double *A, int *perm, double *rdiag);
/* least squares min |V c - y| (V n x m, n >= m, overwritten) by Householder QR; c (m) */
int det_lsq(int n, int m, double *V, const double *y, double *c);

/* ---- Gauss-Legendre rules on [0, 1] ---- */
static inline const double *det_gl(int n, const double **w) { return dfl_rule(n, w); }

/* ---- the prepared design (Prep of DFDet.jl) ---- */
typedef struct {
    int K, np, deg, loc, first;      /* K: that of the prepared design (k after GLS detrending); np pieces, coefficients of degree deg */
    int Kd;                          /* the number of regressors of the design (the columns of C) */
    double *br;                      /* np + 1 */
    double *C;                       /* np blocks of (deg + 1) x K */
    int p, nj, N;                    /* the basis of span{d, d'}, the jumps, the size of ell */
    int gdeg;                        /* the degree of the basis g (rows - 1) */
    double *g;                       /* np blocks of (gdeg + 1) x p */
    double *jl;                      /* nj */
    double *A, *B;                   /* N x N */
    double *L, *Dv;                  /* N x r, r */
    int r, rp, rm;
    double *b;                       /* N: the coefficient of J(0) */
    double *off;                     /* np */
    double cbar;                     /* NaN: least squares */
    unsigned long long key;          /* the hash of the design and cbar */
    int kind;                        /* 0 prep, 1 prep_gls, 2 prep_pt, 3 prep_lm */
    int refs;                        /* references (the cache, the models; det_prep_make gives one) */
} det_prep;
void det_prep_free(det_prep *P);
/* the prepared design of D (cbar NaN) or GLS-detrended at cbar < 0 */
int det_prep_make(const dfl_design *D, double cbar, det_prep **out, char *err, size_t errlen);
int det_prep_pt(const dfl_design *D, double cbar, det_prep **out, char *err, size_t errlen);
int det_prep_lm(const dfl_design *D, det_prep **out, double *a1norm, char *err, size_t errlen);
/* whether the constant function lies in the span of the regressors of D, decided on values at Chebyshev points of each
 * piece to tol (span_has_constant of DFDet.jl, tol 1e-6): the LM statistic's functions refuse a design without it, with
 * DET_LM_CONST_MSG followed by " does not contain it" */
int det_span_has_constant(const dfl_design *D, double tol);
/* the methods of the t-ratio's dist_cdf, dist_pdf and dist_quantile without a normal component (LINE_METHOD_MSG of
 * DFLimit.jl: also its refusal of 'lines' and 'mixture' there) */
#define DET_LINE_METHOD_MSG "method must be 'auto', 'line' (the line integral with the explicit kernel) or 'paths' (the inversion on paths around the roots)"
#define DET_LM_CONST_MSG "the LM statistic: the limit computed here holds only for trend functions whose span contains the constant (the LM regression estimates the level from the first observation, and the limit of the statistic is otherwise a different one); the design"
/* the cached prepared design (prep_cached); release with det_prep_release */
det_prep *det_prep_cached(const dfl_design *D, double cbar, char *err, size_t errlen);
unsigned long long det_design_key(const dfl_design *D, double cbar);
void det_prep_release(det_prep *P);
/* the helpers of the preparations (dfl_det_design.c) shared with dfl_det_paper2.c */
det_prep *det_prep_new(const dfl_design *D);
void det_prep_rank1(det_prep *P, double a11);
int det_ortho_basis(const dfl_design *D, int nF, const double *F, int *pp, double **gout, double **Phiout);
int det_keep_eig(int N, const double *A, double **Dv, double **L, int *r, int *rp, int *rm);
int det_gls_directions(const dfl_design *D, double **Zout);
void det_delta_blocks(const dfl_design *D, int k, const double *Z, double *Cd, double *Cd1, double *Cd2);
void det_evblk(const dfl_design *D, int k, const double *Cs, int j, double r, double *out);
/* the design's values at r */
void det_dval(const dfl_design *D, double r, double *out);
/* Gram matrix of the columns of the piecewise polynomials F[j] ((deg+1) x n each, np blocks) on [0, 1] */
void det_gram(int np, int deg, int n, const double *F, const double *br, int loc, double *G);

/* ---- the model (Model of DFDet.jl) ---- */
typedef struct {
    const det_prep *P;
    int start;                       /* 0 zero, 1 fixed, 2 stationary */
    double g0;
    int N;
    double *A, *B, *L, *Dv;          /* the zero and the fixed start: those of P (not owned) */
    int r, rp, rm;
    double a, kappa;
    double *yb;                      /* N */
    int own;                         /* A, B, L, Dv owned (the stationary start) */
} det_model;
int det_model_make(const det_prep *P, int start, double g0, det_model *Mo, char *err, size_t errlen);
void det_model_free(det_model *Mo);

/* ---- the transform (dfl_det_mgf.c) ---- */
typedef struct {                     /* the data of the fixed start at one t (FixT) */
    cdouble cf;
    int r;
    cdouble *H, *w1, *p1, *w2, *p2, *mt;   /* r x r, r, r, r, r, r (one block) */
} det_fixt;
det_fixt *det_fixt_alloc(int r);
typedef struct {
    const det_model *Mo;
    int N0, N, r, deg, p, np, nq;    /* P.N, the model's N */
    const double *zq, *wq;           /* the Gauss-Legendre rule of sigma (24 nodes) */
    cdouble *S, *Sx, *M, *F, *Rb, *Ib, *e, *h, *X, *Hm, *v1, *v2, *v3, *v4, *Bf;
    double *a;
    int *piv;
    void *heap;
} det_work;
/* the work space of the model: in the caller's buffer buf (bytes) where it fits, on the heap otherwise */
int det_work_init(det_work *W, const det_model *Mo, void *buf, size_t bytes);
void det_work_free(det_work *W);
#define DET_WORK_STACK 24576
void det_sigma(det_work *W, cdouble th);   /* Sigma(theta) into W->S (and h = F(0) into W->h) */
/* mgfparts: theta, log D_V(t), the r reciprocal roots mu (in mu), and with a fixed start its data into fx (may be NULL
 * otherwise); 1 if the eigenvalue routine fails */
int det_mgfparts(det_work *W, double c, cdouble t, cdouble *th, cdouble *ld, cdouble *mu, det_fixt *fx);
cdouble det_fixE1(const det_fixt *fx, cdouble s, cdouble *Bf);
cdouble det_dval_st(det_work *W, double c, cdouble s, cdouble t);   /* Dval of the model at (s, t) */
/* logmgf_point in two parts: the t part (theta, the fixed start's data) and the s part (log M without the determinant,
 * log det) */
int det_logmgf_tpart(det_work *W, double c, cdouble t, cdouble *th, det_fixt *fx);
void det_logmgf_spart(det_work *W, double c, cdouble s, cdouble t, cdouble th, const det_fixt *fx, cdouble *l0, cdouble *ld);
int det_tstar(const det_model *Mo, double c, double *ts, int *mult, char *err, size_t errlen);
/* log det(I - 2 Q Sigma) at (s, t) with its continuous argument from the pivots of K^T = Sigma_x^{-1} M^T and of Sigma (the
 * branch of Bao and Kan; W holds Sigma, and Sigma_x with a stationary start, for t: det_logmgf_tpart or det_mgfparts); K, Sl
 * of N x N, Sc of N0 x N0 complex entries and piv of N ints are the caller's work space; W->M is overwritten */
cdouble det_detlog_piv(det_work *W, double c, cdouble s, cdouble t, cdouble th, cdouble *K, cdouble *Sl, cdouble *Sc, int *piv);

/* the two largest eigenvalues of the Nystrom estimates (eigtop, eigtop_q; det_eigtop_k and det_eigtop_qk with the
 * covariance kernel kern(r, u, ctx) in place of the Ornstein-Uhlenbeck one, the keyword kern of DFDet.jl) */
int det_eigtop(const det_prep *P, double c, int m, int stat, double *l1, double *l2);
int det_eigtop_k(const det_prep *P, int m, double (*kern)(double r, double u, const void *ctx), const void *ctx, double *l1, double *l2);
int det_eigtop_q(const det_prep *P, double c, int m, double *l1, double *l2);
int det_eigtop_qk(const det_prep *P, int m, double (*kern)(double r, double u, const void *ctx), const void *ctx, double *l1, double *l2);
double det_e0r(double z);

/* ---- the volatility and persistence regimes (src/DFDet_vol.jl): the model (dfl_det_path.c) and its transform
 * (dfl_det_mgf.c) ----
 * The design refined at the regime boundaries (its prepared design P, by least squares or, with GLS detrending, by
 * prep_gls, and the engine's model of P with the start, the inner model of the paths, in mr), np pieces of lengths L with
 * omega^2 and c on each, the vector ell = (J(1), J at the jumps of d, z = int g J, J at the regime boundaries that are not
 * jumps of d) of dimension Ne, bpos[j] (j < np - 1) the position in ell of J at the boundary between pieces j and j + 1
 * (-1: not in ell), the padded matrices A, B (N x N) and Lx (N x r) of the design and the eigenvalues Dv.  The start: 0
 * zero (N = Ne), or 2 a normal start of variance v0 (the stationary start of the first regime unless det_vol_model is given
 * another), where J(0) is one more functional: N = Ne + 1 and A, B, Lx, Dv are the matrices of the engine's stationary
 * start (A bordered by b/2 and -1/2, B by zeros).  The path: nreg regimes, lam (nreg + 1) the boundaries, om the scaled
 * standard deviations, cr the c.  (The fixed start under regimes, vol_qf of DFDet.jl, is not ported: its distribution is
 * refused in every language.) */
struct det_model_ref;
typedef struct det_vmodel {
    struct det_model_ref *mr;
    const det_prep *P;
    int nreg; double *lam, *om, *cr;
    int np; double *L, *om2, *c;
    int Ne; int *bpos;
    int start, N; double v0;
    double *A, *B, *Lx, *Dv;
    int r, rp, rm;
    double cmin;
    unsigned long long key;
    int refs;
} det_vmodel;
/* the work space of the volatility transform at one t (VolWork): with a normal start h the mean of ell under the tilted
 * process started at one (vol_start!) and Sx the covariance of (ell, J(0)), N x N; otherwise Sx is S itself */
typedef struct {
    const det_vmodel *VM;
    int nq; const double *zq, *wq;
    cdouble *e, *F, *Rb, *Ib, *S, *M, *th, *v, *gam, *X, *Hm, *v1, *h, *Sx;
    double *a;
    int *piv;
    void *heap;
} det_vwork;
int det_vwork_init(det_vwork *W, const det_vmodel *VM, void *buf, size_t bytes);
void det_vwork_free(det_vwork *W);
/* the transform at t (mgfparts of a VolModel): log D_V(t) = sum_j (c_j + theta_j) L_j + log det{I - 2 (t B + Gamma) Sigma}
 * and the r reciprocal roots mu (with a normal start B, Gamma and Sigma those of (ell, J(0))); 1 if the eigenvalue routine
 * fails */
int det_vol_parts(det_vwork *W, cdouble t, cdouble *ld, cdouble *mu);
/* D(s, t) = exp{sum_j (c_j + theta_j) L_j} det{I - 2 (s A + t B + Gamma) Sigma} (vol_dval) */
cdouble det_vol_dval(const det_vmodel *VM, cdouble s, cdouble t);
/* t_* of the volatility model (vol_tstar) */
int det_vol_tstar(const det_vmodel *VM, double *ts, int *mult, char *err, size_t errlen);
/* the cached model (vol_model) of the design D (with D->first the statistic that detrends first) with the regimes of v at
 * the c of the regimes (cr, nreg values): start DFL_START_ZERO, or DFL_START_STAT (a normal start, its variance v0 that of
 * the stationary law of the first regime, om_1^2/(-2 c_1), where v0 is NaN), and GLS detrending at gls = cbar (NaN: least
 * squares; zero start); a reference (det_vol_release) */
det_vmodel *det_vol_model(const dfl_design *D, const dfl_vol *v, const double *cr, int start, double gls, double v0, char *err, size_t errlen);
void det_vol_release(det_vmodel *VM);

/* a transform model of the outer contour (TransformModel of DFDet.jl): the engine's model Mo (vm NULL) or the volatility
 * model vm; r, rp, rm, start, g0 the fields the contour reads */
typedef struct { const det_model *Mo; const det_vmodel *vm; int r, rp, rm, start; double g0; } det_tm;

/* ---- the outer paths (dfl_det_path.c) ---- */
typedef struct det_path {
    double c, tau, tstar, a, b, h, phi;       /* h NaN: the vertical line */
    int K, r;
    cdouble *t, *dt, *R, *lP;                 /* R: r x K (node k at R + k r) */
    double minpos, maxneg;
    det_model *Mo;                            /* shared with the cache entry of the model */
    det_fixt **fx;                            /* K, NULL entries for the zero and the stationary start */
    double *wt;
    int line, nq, npan, refined;              /* a line: nq nodes per panel, npan panels at cuts (npan + 1) */
    double *cuts;
    int refs;                                 /* references (the cache and the users) */
    struct det_model_ref *mref;
    det_vmodel *vm;                           /* the volatility model of the path (a reference), NULL for the engine's model */
} det_path;
typedef struct det_model_ref { det_prep *P; det_model Mo; int refs; } det_model_ref;
/* the cached outer path of the design with its start at c (cpath_cached): phi NaN and hfac 1 for the default */
det_path *det_path_cached(const dfl_design *D, double cbar, int start, double g0, double c, double phi, double hfac, char *err, size_t errlen);
void det_path_release(det_path *CP);
det_path *det_path_prep(det_prep *P, int start, double g0, double c, char *err, size_t errlen);
det_path *det_v_path_prep(det_prep *P, int start, double g0, double c, char *err, size_t errlen);   /* (v_own of such a path) */
/* the largest c of the LM statistic's functions (LM_CMAX of DFDet_paper2.jl: above it the inversion of its V moves between
 * contours by more than 1e-8, by 2e-7 at c = 12 and 3e-5 at c = 15) */
#define DET_LM_CMAX 10.0
/* a line refined for |x| <= max |xs| and the constants amax of the kernels (line_refined); the path itself (with a new
 * reference) where nothing is refined */
det_path *det_line_refined(det_path *CP, int nx, const double *xs, int na, const double *amax, int nc0, const double *c0s, char *err, size_t errlen);
det_path *det_variant_path(det_path *CP, int level, char *err, size_t errlen);
/* the cached outer contour of the volatility model (vol_path: phi NaN the default rule, hfac its step factor) */
det_path *det_vol_path(det_vmodel *VM, double phi, double hfac, char *err, size_t errlen);
det_path *det_v_path(const dfl_design *D, double cbar, int start, double g0, double c, double phi, double hfac, char *err, size_t errlen);
double det_default_rmax(double c);
/* the c below which the engine takes its rules far from the unit root (FAR_CMIN of DFDet.jl; until November 2026 c = -100):
 * the outer hyperbola the angle of fixed_phi with every start, and the U-shapes the caps 20/min(A, 40/|sigma|) of their
 * segments.  On the 45-degree hyperbola the terms grow far from the unit root, and from c = -70 to -100 the distribution
 * function was off by 1e-11 to 1.4e-9; with both rules from c = -60 on within 5e-13 (case DK 3e-11, as below c = -100) */
#define DET_FAR_CMIN (-60.0)
cdouble det_logMtilde(const cdouble *rts, int r, int rp, cdouble lP, cdouble s);

/* ---- the inner integrals and the outer sums (dfl_det_inner.c) ---- */
typedef struct {                      /* one term of an event (XSpec) */
    double c0, q, A, vmax, w, dc0, dq, dA;
    int bnd;
} det_xspec;
typedef struct { int n, ntail; double alpha, tol; int warn, filon; } det_eval_opts;
/* the largest number of nodes of the U-shapes of a node of the outer contour (ushapes stops with an error beyond it: with a
 * fixed start by the wide heights, with the zero and the stationary start by the segments of the narrow ones;
 * USHAPE_MAXNODES of DFDet.jl) */
#define DET_USHAPE_MAXNODES 1000000
void det_eval_defaults(const det_path *CP, det_eval_opts *o);
/* contour_eval: F and f of (U - c0 V)/sqrt(V) at xs (vpow 2; 3: the density kernel of the coefficient statistic); eF,
 * ef the rounding estimates (may be NULL) */
int det_contour_eval(const det_path *CP, int nx, const double *xs, double c0, int vpow, const det_eval_opts *o,
                     double *F, double *f, double *eF, double *ef, char *err, size_t errlen);
/* contour_eval_specs: the events specs[i] of nt[i] terms each (terms in one array, event i from off[i]) */
int det_contour_eval_specs(const det_path *CP, int ne, const int *nt, const det_xspec *specs, const det_eval_opts *o,
                           double *F, double *f, double *eF, double *ef, char *err, size_t errlen);
double det_vsum(const det_path *CP, double y, int cdf);

/* ---- the extensions of the limit (dfl_det_api.c, dfl_det_ext.c, dfl_det_inner.c) ---- */
/* the checks of the arguments with a volatility path (o->vol) and its model (*VM, a reference; NULL for one regime, which
 * the engine itself computes at *c1) */
int det_vol_args(const char *fn, const dfl_design *D, double c, const dfl_det_opts *o, det_vmodel **VM, double *c1, char *err, size_t errlen);
/* the t-ratio with a normal component (cov_eval, cov_quantile): o->rho != 1, o->vol NULL */
int det_cov_dist(int kind, const dfl_design *D, int n, const double *x, double c, const dfl_det_opts *o, double *out, char *err, size_t errlen);
int det_cov_quantile(const dfl_design *D, int n, const double *p, double c, double x0, double tol, const dfl_det_opts *o, double *q, char *err, size_t errlen);
/* the safeguarded Newton's method of the quantile searches (newton_safe): ff(x) -> (F, f) */
typedef int (*det_ff_t)(double x, void *ctx, double *F, double *f, char *err, size_t errlen);
int det_newton_safe(det_ff_t ff, void *ctx, double p, double x0, double step, double tol, double *xout, char *err, size_t errlen);
/* the inversion on two lines with the normal component in the kernel (lines_both) on the outer contour CP of the design
 * with its start (zero or stationary): F and f of rho tau_c0 + sqrt(1 - rho^2) Z, 0 < rho < 1, at the finite points x */
int det_lines_both(const det_path *CP, int nx, const double *x, double rho, double c0, double *F, double *f, char *err, size_t errlen);
/* the panels of [A, B] graded toward the points pos[i] with the smallest panels h[i] (graded_panels): *ab two ends per panel
 * (malloc'd), the number of panels returned */
int det_graded_panels(double A, double B, int nf, const double *pos, const double *h, double **ab);
/* the first seasonal t-ratio by the sum of the residues (t3_residues) on the outer contour CP (zero start) */
int det_t3_residues(const det_path *CP, int nx, const double *x, double c0, double *F, double *f, char *err, size_t errlen);

/* ---- the devices of Bao and Kan for the general engine (dfl_det_kan.c; src/DFKan_det.jl) ----
 * The transform of a model with a zero or stationary start at c with the branch of the square root of its determinant from
 * the pivots (det_detlog_piv): log E exp(i t1 A + i t2 B), A = U - cV, B = V, and the single inversion of the point-optimal
 * statistic (Q = U + 1/2 + cbar^2 V of the model of prep_pt) and of the coefficient statistic (coef_cdf_gp) with the
 * generic inversion of dfl_kan.c.  One work space: one per thread.  memo 1 keeps the values along one ray (the quantile
 * searches). */
typedef struct det_cf det_cf;
struct dfl_kcf;
det_cf *det_cf_new(const det_model *Mo, double c, int memo);
void det_cf_free(det_cf *F);
void det_cf_memo(det_cf *F, int on);
cdouble det_cf_logcf(det_cf *F, double t1, double t2);
void det_cf_scale(det_cf *F, double *out);
struct dfl_kcf *det_cf_kcf(det_cf *F);
int det_pt_cdf1(det_cf *F, double cbar, double q, double tol, double *P, double *e);
int det_pt_pdf1(det_cf *F, double cbar, double q, double tol, double *f, double *e);
int det_pt_quantile1(det_cf *F, double cbar, double p, double tol, double *q, double *e, char *err, size_t errlen);
int det_gcoef_cdf1(det_cf *F, double y, double c0, double tol, double *P, double *e);

/* ---- the line integral with the explicit kernel for the designs of rank one or two (dfl_det_line.c) ---- */
typedef struct dfl_prov dfl_prov;
/* method auto takes the line integral at c0 = 0 from this c on (the C timings: det_line_auto), except above the unit root
 * up to DET_LINE_PATHS_CMAX with deterministic terms, where it takes the paths */
#define DET_LINE_AUTO_CMIN (-20.0)
/* above this c the inversion on paths by one evaluation warns for designs beyond the classical cases and for a fixed start
 * with deterministic terms (far_above_warn of dfl_det_api.c) */
#define DET_FAR_ABOVE_CMIN 5.5
#define DET_LINE_PATHS_CMAX 0.5
int det_line_auto(const dfl_design *D, double c, double c0);
/* (and with c0 > 0 above the unit root on the vertical line of the paths, where the paths do not resolve the two roots of a
 * point x < 0: det_line_vline) */
int det_line_vline(const dfl_design *D, double c, const dfl_det_opts *o);
/* the provider when the method (o->method auto or line) takes the line integral (*out, a reference: det_line_release), NULL
 * when the paths compute; 1 with the error in err where method line is refused (the reason) or the provider fails */
int det_line_select(const dfl_design *D, double c, const dfl_det_opts *o, dfl_prov **out, char *err, size_t errlen);
void det_line_release(dfl_prov *pv);
/* one value (kind 1 density, 2 distribution function) at x with the null value c0: 0, the bound of the classical
 * integrator's paths (DFL_BOUND_*), or -1 where the provider failed */
int det_line_value(dfl_prov *pv, int kind, double x, double c0, double *out);
void det_line_point_error(char *err, size_t errlen, const char *fn, const dfl_prov *pv, double x, double c0, int rc);
void det_line_empty_cache(void);
int det_line_cache_size(void);
/* (for the tests) the transform at n points of the provider's line in one call: 0, and the points its memo then holds */
int det_line_raw_test(dfl_prov *pv, int n, size_t *nmemo);

/* the generated tables (dfl_det_tables.c) */
typedef struct { int K, r, rp; double sc[9]; const double *t, *dt, *R, *lP, *wt; } det_path_table;
extern const det_path_table DET_PATH_TABLES[3];          /* trend(-1), trend(0), trend(1) at c = 0, zero start */
typedef struct { int nq; const double *z, *lam, *D0, *D1; } det_filon_table;
extern const det_filon_table DET_FILON_TABLES[4];        /* nq = 16, 24, 32, 48 */

#endif
