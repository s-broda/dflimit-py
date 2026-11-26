/* dfl_kan.c: the devices of Yong Bao and Raymond Kan, "Exact Finite-Sample Distribution and Moments of the Least Squares
 * Estimator in First-Order Autoregressions: O(1) Algorithms", for the limits of the classical cases (a port of src/DFKan.jl
 * and src/DFKan_ratio.jl of DFLimit.jl; the method of every function here is theirs):
 *
 *   - the joint characteristic function of the limit pair (U, V) in closed forms that are free of cancellation on both
 *     sides of the unit root, with the branch of the square root settled at each point from factors whose arguments are
 *     known (no tracking along a grid), in Taylor-mode arithmetic (jets), so that derivatives of any order come out of the
 *     same assembly (dfl_kan_logcf.h, dfl_kan_jet.h);
 *   - the single inversion for statistics that reject on a half-plane (the coefficient statistic, the linear form U - aV,
 *     the power envelope): adaptive quadrature over decades and the tail completed by repeated integration by parts,
 *     the derivatives read off a jet of order eight, each value with an error estimate;
 *   - the moments from one jet per node of an adaptive quadrature in log s;
 *   - the distribution function of the t-ratio at zero in closed form (the chi-square representation).
 *
 * The pair is (U, V): the t-ratio is (U - c0 V)/sqrt(V) and the coefficient statistic (U - c0 V)/V.  Bao and Kan write
 * A = U - cV and B = V, so that E exp(i t1 A + i t2 B) = M(i t1, i (t2 - c t1)), M the joint moment generating function
 * of (U, V).  The quadrature is the algorithm of QuadGK.jl, so that the values are those of the Julia package to rounding. */
#include "dfl_kan_int.h"
#include "dfl_lock.h"
#include <float.h>

/* =====================================================================================================================
 * The series of the closed forms (the constants of src/DFKan.jl), computed once
 * ===================================================================================================================== */
#define DFL_KAN_KMS 26
static double FACT[171];
static double DFL_KAN_SER_COS[27], DFL_KAN_SER_SIN[27], DFL_KAN_SER_SH[27], DFL_KAN_SER_CD[27], DFL_KAN_SER_A[27];
static double DFL_KAN_SER_D1[40], DFL_KAN_SER_F1[42], DFL_KAN_SER_F2[42], DFL_KAN_SER_CH[41], DFL_KAN_SER_SHC[41];
static volatile int KAN_DONE = 0;
static dfl_lock_t KAN_LOCK = DFL_LOCK_INIT;
static double fact(int k) { return FACT[k]; }
static double pm1(int k) { return (k & 1) ? -1.0 : 1.0; }
static void kan_init(void)
{
    if (KAN_DONE) return;
    dfl_lock(&KAN_LOCK);
    if (!KAN_DONE) {
        for (int k = 0; k <= 170; k++) FACT[k] = exp(lgamma(k + 1.0));        /* (as exp(loggamma(k + 1)) in the Julia package) */
        for (int k = 0; k <= DFL_KAN_KMS; k++) {
            DFL_KAN_SER_COS[k] = pm1(k) / fact(2 * k);                          /* cos th = sum (-1)^k zeta^k/(2k)! */
            DFL_KAN_SER_SIN[k] = pm1(k) / fact(2 * k + 1);                      /* sin th/th */
            DFL_KAN_SER_SH[k] = pm1(k) / (ldexp(1.0, 2 * k) * fact(2 * k + 1)); /* sin(th/2)/(th/2) */
            DFL_KAN_SER_A[k] = 2 * pm1(k) / fact(2 * k + 2);                    /* Sh^2 = 2(1 - cos th)/zeta */
        }
        for (int k = 1; k <= DFL_KAN_KMS + 1; k++)                              /* (Sh - Ch)/zeta */
            DFL_KAN_SER_CD[k - 1] = pm1(k + 1) * (2 * k) / (ldexp(1.0, 2 * k) * fact(2 * k + 1));
        {   /* (tan(u)/u - 1)/th^2, u = th/2, in w = zeta/4 */
            enum { KM = 40 };
            double sn[KM + 1], cs[KM + 1], a[KM + 1];
            for (int k = 0; k <= KM; k++) { sn[k] = pm1(k) / fact(2 * k + 1); cs[k] = pm1(k) / fact(2 * k); }
            for (int i = 0; i <= KM; i++) { double acc = sn[i]; for (int j = 1; j <= i; j++) acc -= cs[j] * a[i - j]; a[i] = acc; }
            for (int i = 1; i <= KM; i++) DFL_KAN_SER_D1[i - 1] = a[i] / 4;
        }
        for (int j = 1; j <= 42; j++) DFL_KAN_SER_F1[j - 1] = (4.0 * j) / (ldexp(1.0, 2 * j) * fact(2 * j + 1));   /* e^{vt/2} h1/vt^3 */
        for (int j = 2; j <= 43; j++) DFL_KAN_SER_F2[j - 2] = (12 / fact(2 * j) - 4 / fact(2 * j - 1) - 12 / fact(2 * j + 1)) / ldexp(1.0, 2 * j);
        for (int k = 0; k <= 40; k++) { DFL_KAN_SER_CH[k] = 1 / (ldexp(1.0, 2 * k) * fact(2 * k)); DFL_KAN_SER_SHC[k] = 1 / (ldexp(1.0, 2 * k + 1) * fact(2 * k + 1)); }
        KAN_DONE = 1;
    }
    dfl_unlock(&KAN_LOCK);
}

/* =====================================================================================================================
 * Numbers (complex scalars) with the operations of the jets, so that the transform is written once for both
 * ===================================================================================================================== */
static inline cdouble s_const(int n, cdouble v) { (void) n; return v; }
static inline cdouble s_val(cdouble a) { return a; }
static inline cdouble s_setval(cdouble a, cdouble v) { (void) a; return v; }
static inline cdouble s_add(cdouble a, cdouble b) { return cadd(a, b); }
static inline cdouble s_sub(cdouble a, cdouble b) { return csub(a, b); }
static inline cdouble s_neg(cdouble a) { return cneg(a); }
static inline cdouble s_addc(cdouble a, cdouble z) { return cadd(a, z); }
static inline cdouble s_subc(cdouble a, cdouble z) { return csub(a, z); }
static inline cdouble s_csub(cdouble z, cdouble a) { return csub(z, a); }
static inline cdouble s_plusr(cdouble a, double r) { return CPX(creal(a) + r, cimag(a)); }
static inline cdouble s_subr(cdouble a, double r) { return CPX(creal(a) - r, cimag(a)); }
static inline cdouble s_rsub(double r, cdouble a) { return CPX(r - creal(a), -cimag(a)); }
static inline cdouble s_mulc(cdouble a, cdouble z) { return cmul(a, z); }
static inline cdouble s_mulr(cdouble a, double r) { return cscale(r, a); }
static inline cdouble s_divc(cdouble a, cdouble z) { return cdiv(a, z); }
static inline cdouble s_divr(cdouble a, double r) { return CPX(creal(a) / r, cimag(a) / r); }
static inline cdouble s_mul(cdouble a, cdouble b) { return cmul(a, b); }
static inline cdouble s_div(cdouble a, cdouble b) { return cdiv(a, b); }
static inline cdouble s_cdiv(cdouble z, cdouble b) { return cdiv(z, b); }
static inline cdouble s_sqrt(cdouble a) { return dfl_csqrt(a); }
static inline cdouble s_exp(cdouble a) { return cexp(a); }
static inline cdouble s_logp(cdouble a) { return clog(a); }
static inline cdouble s_logc(cdouble a, double ph) { return CPX(log(cabs(a)), ph); }
static inline cdouble s_ser(const double *cf, int ncf, cdouble z) { cdouble y = CPX(0, 0); for (int k = ncf - 1; k >= 0; k--) y = s_plusr(cmul(y, z), cf[k]); return y; }

/* jets of capacity 2 (the density's integrand), 10 (the tails, the scale, moments up to the ninth) and DFL_KAN_JMAX */
#define JT dfl_jet2
#define JCAP 2
#define JF(f) j2_##f
#include "dfl_kan_jet.h"
#undef JT
#undef JCAP
#undef JF
#define JT dfl_jet10
#define JCAP 10
#define JF(f) j10_##f
#include "dfl_kan_jet.h"
#undef JT
#undef JCAP
#undef JF
#define JT dfl_jet32
#define JCAP DFL_KAN_JMAX
#define JF(f) j32_##f
#include "dfl_kan_jet.h"
#undef JT
#undef JCAP
#undef JF

/* =====================================================================================================================
 * The phases of the closed forms: the arguments of factors that are known (Bao and Kan)
 * ===================================================================================================================== */
static double jsign(double x) { return x > 0 ? 1.0 : x < 0 ? -1.0 : x; }
/* the continuous argument of cos(theta) as a function of zeta = theta^2 (Re zeta <= 0): the principal argument shifted by
 * the multiple of 2 pi that brings it within pi/2 of -sign(Im theta) Re theta; equal to Im vartheta + Arg(1 + e^{-2
 * vartheta}), vartheta the principal root of -zeta */
static double dfl_kan_argcos(cdouble z)
{
    cdouble th = dfl_csqrt(z); double tr = creal(th), ti = cimag(th);
    double e = exp(-2 * fabs(ti));
    cdouble cz = CPX(cos(tr) * (1 + e), -sin(tr) * jsign(ti) * (1 - e));
    double a = carg(cz), target = -jsign(ti) * tr;
    return a + 2 * M_PI * nearbyint((target - a) / (2 * M_PI));
}
/* the continuous phase of the factor of the demeaned cases under the square root as the sum of the principal arguments of
 * three factors, each off the closed negative real axis: W = (theta/2) cot(theta/2), the limit of the rank-one factor and
 * the limit of the border Schur complement */
static double dfl_kan_argtau4(cdouble z0, cdouble zc0, cdouble D2, cdouble T0)
{
    cdouble D1 = cabs(z0) < 4 ? s_ser(DFL_KAN_SER_D1, 40, CPX(creal(z0) / 4, cimag(z0) / 4)) : cdiv(s_subr(cscale(2, D2), 1), z0);
    cdouble rho = s_plusr(cmul(zc0, D1), 1);
    cdouble Sinf = cdiv(cmul(T0, D2), cscale(2, rho));
    cdouble W = cdiv(CPX(1, 0), cscale(2, D2));
    return carg(W) + carg(rho) + carg(Sinf);
}

/* the transform: numbers, then the jets */
#define KT cdouble
#define KF(f) s_##f
#define KNAME(f) f##_s
#define KORD(a) 1
#include "dfl_kan_logcf.h"
#undef KT
#undef KF
#undef KNAME
#undef KORD
#define KT dfl_jet2
#define KF(f) j2_##f
#define KNAME(f) f##_j2
#define KORD(a) ((a).n)
#include "dfl_kan_logcf.h"
#undef KT
#undef KF
#undef KNAME
#define KT dfl_jet10
#define KF(f) j10_##f
#define KNAME(f) f##_j10
#include "dfl_kan_logcf.h"
#undef KT
#undef KF
#undef KNAME
#define KT dfl_jet32
#define KF(f) j32_##f
#define KNAME(f) f##_j32
#include "dfl_kan_logcf.h"
#undef KT
#undef KF
#undef KNAME
#undef KORD

/* =====================================================================================================================
 * The limit pair of a case
 * ===================================================================================================================== */
int dfl_limcf_init(dfl_limcf *F, int model, int stat, double c, double g0, char *err, size_t errlen)
{
    kan_init();
    memset(F, 0, sizeof *F);
    if (model < 1 || model > 3) { snprintf(err, errlen, "the closed-form transform of Bao and Kan: the model must be 1, 2 or 3 (cases DK, DU, DT, SK, SU, ST)"); return 1; }
    if (!isfinite(c) || !isfinite(g0)) { snprintf(err, errlen, "the closed-form transform of Bao and Kan: c and g0 must be finite"); return 1; }
    if (stat && c >= 0) { snprintf(err, errlen, "the closed-form transform of Bao and Kan: a stationary start requires c < 0"); return 1; }
    F->det = model; F->stat = stat != 0; F->c = c;
    double par = model == 1 ? g0 : model == 2 ? c * g0 : c * c * g0;
    F->par = stat ? 0.0 : par;
    F->w0 = (model == 1 && !stat) ? (1 + F->par * F->par) / 2 : 0.5;
    if (model == 3) {
        double c2 = c * c, c3 = c2 * c, c4 = c3 * c, c5 = c4 * c;
        double gam[5] = {24 * c2 * (c - 2) * (c2 - 3 * c + 6), -24 * c2 * (c - 2) * (c2 - 3 * c + 6), 8 * (c5 - 4 * c4 + 6 * c3 + 6 * c2 - 36 * c + 36),
                         -c5 + 24 * c3 - 120 * c2 + 288 * c - 288, c4 - 8 * c3 + 32 * c2 - 72 * c + 72};
        for (int k = 4; k <= 44; k++) { double s = 0; for (int j = 0; j <= 4; j++) s += 2 * gam[j] / fact(2 * k - j); F->ag[k - 4] = s; }
    }
    return 0;
}
cdouble dfl_kan_logcf(const dfl_limcf *F, cdouble t1, cdouble t2) { return lcf_s(F, t1, t2); }
void dfl_kan_logcf_jet(const dfl_limcf *F, int n, const cdouble *t1c, const cdouble *t2c, cdouble *out)
{
    if (n <= 2) {
        dfl_jet2 a, b; a.n = b.n = n; for (int k = 0; k < n; k++) { a.c[k] = t1c[k]; b.c[k] = t2c[k]; }
        dfl_jet2 L = lcf_j2(F, a, b); for (int k = 0; k < n; k++) out[k] = L.c[k];
    } else if (n <= 10) {
        dfl_jet10 a, b; a.n = b.n = n; for (int k = 0; k < n; k++) { a.c[k] = t1c[k]; b.c[k] = t2c[k]; }
        dfl_jet10 L = lcf_j10(F, a, b); for (int k = 0; k < n; k++) out[k] = L.c[k];
    } else {
        dfl_jet32 a, b; a.n = b.n = n; for (int k = 0; k < n; k++) { a.c[k] = t1c[k]; b.c[k] = t2c[k]; }
        dfl_jet32 L = lcf_j32(F, a, b); for (int k = 0; k < n; k++) out[k] = L.c[k];
    }
}
/* log M(s, t) = log E exp(sU + tV) = logcf at (-i s, -i (t + c s)) */
cdouble dfl_kan_logmgf(const dfl_limcf *F, cdouble s, cdouble t)
{
    return lcf_s(F, cmul(CPX(0.0, -1.0), s), cmul(CPX(0.0, -1.0), cadd(t, cscale(F->c, s))));
}
void dfl_kan_scale(const dfl_limcf *F, double *out)
{
    dfl_jet10 z = j10_const(3, CPX(0, 0)), x = j10_var(3, CPX(0, 0), CPX(1, 0));
    dfl_jet10 Lt = lcf_j10(F, z, x), Ls = lcf_j10(F, x, z);
    double Ev = creal(cdiv(Lt.c[1], CPX(0, 1))), Vv = creal(cscale(-2, Lt.c[2]));
    double Eu = creal(cdiv(Ls.c[1], CPX(0, 1))), Vu = creal(cscale(-2, Ls.c[2]));
    out[0] = Eu; out[1] = Ev; out[2] = isfinite(Vv) ? sqrt(fmax(Vv, 0.0)) : INFINITY; out[3] = Vu;
}
double dfl_kan_ray_scale(const double *sc, double alpha, double beta)
{
    double vv = alpha * alpha * sc[3] + beta * beta * (sc[2] * sc[2]);
    if (!(isfinite(vv) && vv > 0)) vv = sc[3];
    return 1 / sqrt(vv);
}

/* =====================================================================================================================
 * Adaptive Gauss-Kronrod quadrature (QuadGK.jl 2.11, order 7)
 * ===================================================================================================================== */
static const double GKX[8] = {-9.9145537112081263920685469752598e-01, -9.4910791234275852452618968404809e-01, -8.6486442335976907278971278864098e-01,
                              -7.415311855993944398638647732811e-01, -5.8608723546769113029414483825842e-01, -4.0584515137739716690660641207707e-01,
                              -2.0778495500789846760068940377309e-01, 0.0};
static const double GKW[8] = {2.2935322010529224963732008059913e-02, 6.3092092629978553290700663189093e-02, 1.0479001032225018383987632254189e-01,
                              1.4065325971552591874518959051021e-01, 1.6900472663926790282658342659795e-01, 1.9035057806478540991325640242055e-01,
                              2.0443294007529889241416199923466e-01, 2.0948214108472782801299917489173e-01};
static const double GKG[4] = {1.2948496616886969327061143267787e-01, 2.797053914892766679014677714229e-01, 3.8183005050511894495036977548818e-01,
                              4.1795918367346938775510204081658e-01};
typedef struct { double a, b, Iq, Eq; } qseg;
static int evalrule(dfl_qfun f, void *ctx, double a, double b, qseg *s)
{
    double h = 0.5 * (b - a);
    double f0 = f(a + h, ctx);
    double Ig = f0 * GKG[3];
    double Ik = f0 * GKW[7] + (f(a + (1 + GKX[6]) * h, ctx) + f(a + (1 - GKX[6]) * h, ctx)) * GKW[6];
    for (int i = 1; i <= 3; i++) {
        double fg = f(a + (1 + GKX[2 * i - 1]) * h, ctx) + f(a + (1 - GKX[2 * i - 1]) * h, ctx);
        double fk = f(a + (1 + GKX[2 * i - 2]) * h, ctx) + f(a + (1 - GKX[2 * i - 2]) * h, ctx);
        Ig += fg * GKG[i - 1];
        Ik += fg * GKW[2 * i - 1] + fk * GKW[2 * i - 2];
    }
    double Iks = Ik * h, Igs = Ig * h;
    s->a = a; s->b = b; s->Iq = Iks; s->Eq = fabs(Iks - Igs);
    return !isfinite(s->Eq);
}
/* the binary heap of the segments by their error estimate, largest first (the heap of DataStructures.jl with the
 * reverse ordering; 1-based indices as there) */
static void hp_down(qseg *xs, int i, qseg x, int len)
{
    int l;
    while ((l = 2 * i) <= len) {
        int r = l + 1, j = (r > len || xs[r - 1].Eq < xs[l - 1].Eq) ? l : r;
        if (!(x.Eq < xs[j - 1].Eq)) break;
        xs[i - 1] = xs[j - 1]; i = j;
    }
    xs[i - 1] = x;
}
static void hp_up(qseg *xs, int i, qseg x)
{
    int j;
    while ((j = i / 2) >= 1) {
        if (!(xs[j - 1].Eq < x.Eq)) break;
        xs[i - 1] = xs[j - 1]; i = j;
    }
    xs[i - 1] = x;
}
static int roundoff(double a, double b) { double c = 0.5 * (b - a); return a == a + (1 + GKX[0]) * c || b == a + (1 - GKX[0]) * c; }
int dfl_quadgk(dfl_qfun f, void *ctx, int np, const double *pts, double atol, double rtol, long maxevals, double *Iout, double *Eout, long *nev)
{
    int ns = np - 1, cap = ns + 64, len = ns, rc = 0;
    qseg *H = malloc(cap * sizeof(qseg));
    if (!H) return 1;
    for (int i = 0; i < ns; i++) rc |= evalrule(f, ctx, pts[i], pts[i + 1], &H[i]);
    long numevals = 15L * ns;
    double Is = H[0].Iq, Es = H[0].Eq;
    for (int i = 1; i < ns; i++) { Is += H[i].Iq; Es += H[i].Eq; }
    if (!rc && !(numevals >= maxevals || Es <= atol || Es <= rtol * fabs(Is))) {
        for (int i = len / 2; i >= 1; i--) hp_down(H, i, H[i - 1], len);
        while (Es > atol && Es > rtol * fabs(Is) && numevals < maxevals) {
            qseg s = H[0], y = H[len - 1]; len--;
            if (len > 0) hp_down(H, 1, y, len);
            double mid = (s.a + s.b) / 2;
            if (roundoff(s.a, mid) || roundoff(mid, s.b)) { H[len] = s; len++; hp_up(H, len, s); break; }
            qseg s1, s2;
            rc |= evalrule(f, ctx, s.a, mid, &s1) | evalrule(f, ctx, mid, s.b, &s2);
            if (rc) break;
            Is = (Is - s.Iq) + s1.Iq + s2.Iq; Es = (Es - s.Eq) + s1.Eq + s2.Eq; numevals += 30;
            if (len + 2 > cap) { int nc = 2 * cap; qseg *p = realloc(H, nc * sizeof(qseg)); if (!p) { rc = 1; break; } H = p; cap = nc; }
            H[len] = s1; len++; hp_up(H, len, s1);
            H[len] = s2; len++; hp_up(H, len, s2);
        }
        Is = H[0].Iq; Es = H[0].Eq;
        for (int i = 1; i < len; i++) { Is += H[i].Iq; Es += H[i].Eq; }
    }
    free(H);
    *Iout = Is; *Eout = Es; if (nev) *nev += numevals;
    return rc;
}

/* =====================================================================================================================
 * The single inversion with tail completion (half_cdf of DFKan)
 * ===================================================================================================================== */
int dfl_kan_decades(dfl_qfun f, void *ctx, double b, double tol, double rtol, double *Iout, double *eout, long *nev)
{
    double Is = 0, e = 0, a = 0; int quiet = 0, k = 1;
    while (a < b) {
        double hi = isinf(b) ? pow(10.0, k) : (pow(10.0, k + 0.5) >= b ? b : pow(10.0, k)), pts[2] = {a, hi}, Ik, ek;
        if (dfl_quadgk(f, ctx, 2, pts, tol / 10, rtol, 1000000L, &Ik, &ek, nev)) return 1;
        Is += Ik; e += ek; a = hi; k++;
        if (fabs(Ik) + ek <= tol / 20) quiet++; else quiet = 0;
        if (quiet >= 2 || k > 31) break;
    }
    *Iout = Is; *eout = e;
    return 0;
}
typedef struct { dfl_kcf *F; double alpha, beta, b, sc; } hc_ctx;
static double hc_f(double z, void *v)
{
    hc_ctx *h = v; double u = z * h->sc;
    cdouble L = h->F->lcf(h->F, h->alpha * u, h->beta * u);
    double y = exp(creal(L)) * sin(cimag(L) - h->b * u) / z;
    return isfinite(y) ? y : 0.0;
}
int dfl_kan_half_cdf(dfl_kcf *F, double alpha, double beta, double b, double tol, double w0, double *P, double *err)
{
    double sc4[4]; F->scale(F, sc4);
    hc_ctx h = {F, alpha, beta, b, dfl_kan_ray_scale(sc4, alpha, beta)};
    double w = w0 * alpha + b, T = 0, et = 0, S = INFINITY, Iv, e;
    if (fabs(w) > 1e-7) {
        S = 400 / fabs(w);
        for (int dbl = 0; dbl <= 5; dbl++) {
            cdouble Tc; F->tail(F, alpha, beta, b, w, S, 1, &Tc, &et); T = cimag(Tc);
            if (et <= tol / 10 || dbl == 5) break;
            S *= 2;
        }
    }
    long nev = 0;
    if (dfl_kan_decades(hc_f, &h, S / h.sc, tol, fmax(tol, 1e-14), &Iv, &e, &nev)) return 1;
    *P = 0.5 - (Iv + T) / M_PI; *err = (e + et) / M_PI;
    return 0;
}

/* the classical transform behind the interface */
static cdouble lim_lcf(dfl_kcf *K, double t1, double t2) { return lcf_s((const dfl_limcf *) K->ctx, CPX(t1, 0), CPX(t2, 0)); }
static void lim_scale(dfl_kcf *K, double *out) { dfl_kan_scale((const dfl_limcf *) K->ctx, out); }
/* the tail int_S^inf Im[phi(u alpha, u beta) e^{-i b u}]/u du by repeated integration by parts, the derivatives of the
 * slowly varying factor from a jet of order eight at S (tail_cdf of DFKan, N = 9) */
static void lim_tail(dfl_kcf *K, double alpha, double beta, double b, double w, double S, int cdf, cdouble *T, double *e)
{
    const dfl_limcf *F = K->ctx; enum { N = 9 };
    (void) cdf;
    dfl_jet10 sj = j10_var(N, CPX(S, 0), CPX(1, 0));
    dfl_jet10 L = lcf_j10(F, j10_mulr(sj, alpha), j10_mulr(sj, beta));
    dfl_jet10 ph = j10_exp(j10_sub(L, j10_mulc(sj, CPX(0, b))));
    dfl_jet10 g = j10_div(j10_mul(ph, j10_exp(j10_var(N, CPX(0, 0), CPX(0, w)))), sj);
    cdouble acc = CPX(0, 0), iw = cdiv(CPX(1, 0), CPX(0, w)), p = iw;
    for (int k = 0; k <= N - 2; k++) { acc = cadd(acc, cmul(cscale(fact(k), g.c[k]), p)); p = cmul(p, iw); }
    *T = acc; *e = cabs(cmul(cscale(fact(N - 1), g.c[N - 1]), p));
}
void dfl_kan_kcf(const dfl_limcf *F, dfl_kcf *K) { K->c = F->c; K->w0 = F->w0; K->lcf = lim_lcf; K->scale = lim_scale; K->tail = lim_tail; K->ctx = (void *) F; }

/* ---- the density of a ratio (ratio_pdf of DFKan) ---- */
typedef struct { const dfl_limcf *F; double r, sc; } rp_ctx;
static double rp_f(double z, void *v)
{
    rp_ctx *h = v; double u = z * h->sc;
    dfl_jet2 L = lcf_j2(h->F, j2_const(2, CPX(u, 0)), j2_var(2, CPX(-h->r * u, 0), CPX(1, 0)));
    double y = h->sc * cimag(cmul(cexp(L.c[0]), L.c[1]));
    return isfinite(y) ? y : 0.0;
}
/* the tail int_S^inf Im H(u) du, H(u) = d/dt2 phi(t1, t2) at (u, -r u): the derivatives of H along the ray are mixed
 * derivatives of phi, read off jets of x -> phi(S + x, -r (S + x) + xi x) at N complex xi on a circle of radius rho (the
 * coefficient of xi in the k-th derivative is k H^{(k-1)}(S), extracted by a discrete Fourier transform), as in RATIO_PDF of
 * Bao and Kan (tail_pdf of DFKan, N = 9) */
static void tail_pdf(const dfl_limcf *F, double r, double w, double S, double rho, double *T, double *last)
{
    enum { N = 9, K = N - 1 };
    cdouble a1[N];
    for (int k = 0; k < N; k++) a1[k] = CPX(0, 0);
    dfl_jet10 x = j10_var(N, CPX(0, 0), CPX(1, 0)), sj = j10_var(N, CPX(S, 0), CPX(1, 0));
    for (int m = 0; m <= K; m++) {
        double ang = 2 * M_PI * m / N;
        cdouble xi = cscale(rho, CPX(cos(ang), sin(ang)));
        dfl_jet10 ph = j10_exp(lcf_j10(F, sj, j10_sub(j10_mulc(x, xi), j10_mulr(sj, r))));
        for (int k = 0; k <= K; k++) a1[k] = cadd(a1[k], cdiv(cscale(fact(k), ph.c[k]), xi));
    }
    for (int k = 0; k < N; k++) a1[k] = CPX(creal(a1[k]) / N, cimag(a1[k]) / N);
    cdouble H[K], iwp[N + 1], Ts = CPX(0, 0); double lst = 0;
    for (int k = 1; k <= K; k++) H[k - 1] = CPX(creal(a1[k]) / k, cimag(a1[k]) / k);   /* H^{(0..K-1)}(S) */
    iwp[0] = CPX(1, 0); for (int k = 1; k <= N; k++) iwp[k] = cmul(iwp[k - 1], CPX(0, w));   /* (i w)^k */
    for (int k = 0; k <= K - 1; k++) {
        cdouble acc = CPX(0, 0); double bin = 1;
        for (int l = 0; l <= k; l++) { acc = cadd(acc, cmul(cscale(bin, iwp[k - l]), H[l])); bin = bin * (k - l) / (l + 1); }
        cdouble term = cdiv(acc, iwp[k + 1]);
        if (k < K - 1) Ts = cadd(Ts, term); else lst = cabs(term);
    }
    *T = cimag(Ts); *last = lst;
}
int dfl_kan_ratio_pdf(const dfl_limcf *F, double r, double tol, double *f, double *err, double *fs)
{
    double sc4[4]; dfl_kan_scale(F, sc4);
    double Ev = sc4[1], sdv = sc4[2], Vu = sc4[3];
    *fs = fmax(1.0, Ev / sqrt(fmax(Vu, DBL_MIN)));
    double vv = Vu + r * r * (sdv * sdv); if (!(isfinite(vv) && vv > 0)) vv = Vu;
    rp_ctx h = {F, r, 1 / sqrt(vv)};
    double w = F->w0, S = 400 / w, T = 0, et = 0, Iv, e;
    for (int dbl = 0; dbl <= 5; dbl++) {
        tail_pdf(F, r, w, S, 1 / Ev, &T, &et);
        if (et <= tol / 10 || dbl == 5) break;
        S *= 2;
    }
    long nev = 0;
    if (dfl_kan_decades(rp_f, &h, S / h.sc, tol * *fs, fmax(tol, 1e-14), &Iv, &e, &nev)) return 1;
    *f = (Iv + T) / M_PI; *err = (e + et) / M_PI;
    return 0;
}

/* ---- the root of an increasing function (illinois of DFKan) ---- */
int dfl_kan_illinois(double (*g)(double, void *), void *ctx, double lo, double hi, double xtol, double grow, double *xout, char *err, size_t errlen)
{
    double glo = g(lo, ctx), ghi = g(hi, ctx), d = grow > 0 ? grow : hi - lo; int k = 0;
    while (glo > 0 && k < 60) { hi = lo; ghi = glo; lo -= d; d *= 2; glo = g(lo, ctx); k++; }
    while (ghi < 0 && k < 120) { lo = hi; glo = ghi; hi += d; d *= 2; ghi = g(hi, ctx); k++; }
    if (!(isfinite(glo) && isfinite(ghi) && glo <= 0 && ghi >= 0)) {
        snprintf(err, errlen, "the quantile could not be bracketed (the distribution function is not finite or not monotone)"); return 1;
    }
    int side = 0; double x = (lo + hi) / 2;
    for (int it = 0; it < 200; it++) {
        x = (lo * ghi - hi * glo) / (ghi - glo);
        if (!(lo < x && x < hi)) x = (lo + hi) / 2;
        double gx = g(x, ctx);
        if (isnan(gx)) { snprintf(err, errlen, "the distribution function of the single inversion failed at %g", x); return 1; }
        if (gx == 0) break;
        if (gx < 0) { lo = x; glo = gx; if (side == -1) ghi /= 2; side = -1; }
        else { hi = x; ghi = gx; if (side == 1) glo /= 2; side = 1; }
        if (hi - lo < xtol) { x = (lo * ghi - hi * glo) / (ghi - glo); break; }
    }
    *xout = x;
    return 0;
}

/* ---- the moments from jets (ratio_mom, mom_one of DFKan) ---- */
typedef struct { const dfl_limcf *F; int N; double x0, a, cst, Ev; } mo_ctx;
static double mo_f(double y, void *v)
{
    mo_ctx *h = v; int N = h->N, j = N - 1;
    double s2 = exp(y) / h->Ev, ls = h->a * log(s2), val;
    if (N <= 10) {
        dfl_jet10 t1 = j10_var(N, CPX(0, 0), CPX(0, -1));
        dfl_jet10 L = lcf_j10(h->F, t1, j10_var(N, CPX(0, s2), CPX(0, h->x0)));
        cdouble l0 = L.c[0];
        dfl_jet10 Ex = j10_exp(j10_setval(L, CPX(0, 0)));
        val = h->cst * creal(cmul(cexp(CPX(ls + creal(l0), cimag(l0))), Ex.c[j]));
    } else {
        dfl_jet32 t1 = j32_var(N, CPX(0, 0), CPX(0, -1));
        dfl_jet32 L = lcf_j32(h->F, t1, j32_var(N, CPX(0, s2), CPX(0, h->x0)));
        cdouble l0 = L.c[0];
        dfl_jet32 Ex = j32_exp(j32_setval(L, CPX(0, 0)));
        val = h->cst * creal(cmul(cexp(CPX(ls + creal(l0), cimag(l0))), Ex.c[j]));
    }
    return isfinite(val) ? val : 0.0;
}
int dfl_kan_ratio_mom(const dfl_limcf *F, int k, double x0, double power, double tol, double *m, double *err, char *errmsg, size_t errlen)
{
    if (k < 1 || k > DFL_KAN_JMAX - 1) { snprintf(errmsg, errlen, "the moments from jets: at most %d moments (p = %d)", DFL_KAN_JMAX - 1, k); return 1; }
    double sc4[4]; dfl_kan_scale(F, sc4);
    static const double pts[12] = {-60.0, -40.0, -30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0, 45.0};
    for (int j = 1; j <= k; j++) {
        mo_ctx h = {F, j + 1, x0, j * power, 0, sc4[1]};
        h.cst = fact(j) / tgamma(h.a);
        if (dfl_quadgk(mo_f, &h, 12, pts, tol * 1e-2, tol, 1000000L, &m[j - 1], &err[j - 1], NULL)) {
            snprintf(errmsg, errlen, "the moments from jets: the quadrature failed for the moment %d", j); return 1;
        }
    }
    return 0;
}

/* =====================================================================================================================
 * The statistics of the classical cases by the single inversion (coef_cdf1 and so on of DFKan)
 * ===================================================================================================================== */
static double clamp01(double x) { return x < 0 ? 0.0 : x > 1 ? 1.0 : x; }
int dfl_kan_coef_cdf1(const dfl_limcf *F, double y, double c0, double tol, double *P, double *e)
{
    if (isnan(y)) { *P = NAN; *e = 0; return 0; }
    if (isinf(y)) { *P = y > 0 ? 1.0 : 0.0; *e = 0; return 0; }
    dfl_kcf K; dfl_kan_kcf(F, &K);
    double r = y + c0 - F->c;
    if (dfl_kan_half_cdf(&K, 1.0, -r, 0.0, tol, F->w0, P, e)) return 1;
    *P = clamp01(*P);
    return 0;
}
int dfl_kan_coef_pdf1(const dfl_limcf *F, double y, double c0, double tol, double *f, double *e, double *ftol)
{
    if (!isfinite(y)) { *f = isnan(y) ? NAN : 0.0; *e = 0; *ftol = tol; return 0; }
    double fs;
    if (dfl_kan_ratio_pdf(F, y + c0 - F->c, tol, f, e, &fs)) return 1;
    *f = fmax(*f, 0.0); *ftol = tol * fs;
    return 0;
}
typedef struct { const dfl_limcf *F; double p, tol, elast; int fail; } ci_ctx;
static double ci_g(double x, void *v)
{
    ci_ctx *h = v; dfl_kcf K; dfl_kan_kcf(h->F, &K); double P, e;
    if (dfl_kan_half_cdf(&K, 1.0, -x, 0.0, h->tol, h->F->w0, &P, &e)) { h->fail = 1; return NAN; }
    h->elast = e;
    return P - h->p;
}
int dfl_kan_coef_icdf1(const dfl_limcf *F, double p, double c0, double xtol, double tol, double *y, double *e, char *err, size_t errlen)
{
    double sc4[4]; dfl_kan_scale(F, sc4);
    double Eu = sc4[0], Ev = sc4[1], Vu = sc4[3], sd0 = sqrt(Vu) / Ev + fabs(Eu) / Ev, m0 = Eu / Ev, r;
    ci_ctx h = {F, p, tol, 0, 0};
    if (dfl_kan_illinois(ci_g, &h, m0 - 2 * sd0, m0 + 2 * sd0, xtol, 0, &r, err, errlen)) return 1;
    *y = r - (c0 - F->c); *e = h.elast;
    return 0;
}
int dfl_kan_lin_cdf1(const dfl_limcf *F, double b, double a, double tol, double *P, double *e)
{
    if (isnan(b)) { *P = NAN; *e = 0; return 0; }
    if (isinf(b)) { *P = b > 0 ? 1.0 : 0.0; *e = 0; return 0; }
    dfl_kcf K; dfl_kan_kcf(F, &K);
    if (dfl_kan_half_cdf(&K, 1.0, F->c - a, b, tol, F->w0, P, e)) return 1;
    *P = clamp01(*P);
    return 0;
}
typedef struct { dfl_kcf *K; double a, alpha, tol, elast; int fail; } en_ctx;
static double en_g(double x, void *v)
{
    en_ctx *h = v; double P, e;
    if (dfl_kan_half_cdf(h->K, 1.0, -h->a, x, h->tol, h->K->w0, &P, &e)) { h->fail = 1; return NAN; }
    h->elast = e;
    return P - h->alpha;
}
/* the asymptotic power envelope of Elliott, Rothenberg and Stock at c < 0 without deterministic terms (and with an
 * intercept, the same curve): the power at c of the test that rejects for small U - (c/2) V, its critical value the
 * alpha-quantile of U - (c/2) V at the unit root by the Illinois variant of regula falsi to 1e-12 from the bracket
 * [-0.45, 0.5] (grown in steps of one), both by the single inversion; err the sum of the two estimates at the end
 * (envelope1 of DFKan) */
int dfl_kan_envelope1(double c, double alpha, double tol, double *P, double *e, char *err, size_t errlen)
{
    if (isnan(c)) { *P = NAN; *e = 0; return 0; }
    if (!(c < 0)) { *P = alpha; *e = 0; return 0; }
    if (c == -INFINITY) { *P = 1.0; *e = 0; return 0; }
    double a = c / 2, b;
    dfl_limcf F0, Fc;
    if (dfl_limcf_init(&F0, 1, 0, 0.0, 0.0, err, errlen) || dfl_limcf_init(&Fc, 1, 0, c, 0.0, err, errlen)) return 1;
    dfl_kcf K0, Kc; dfl_kan_kcf(&F0, &K0); dfl_kan_kcf(&Fc, &Kc);
    en_ctx h = {&K0, a, alpha, tol, 0, 0};
    if (dfl_kan_illinois(en_g, &h, -0.45, 0.5, 1e-12, 1.0, &b, err, errlen)) return 1;
    double e1;
    if (dfl_kan_half_cdf(&Kc, 1.0, c - a, b, tol, Kc.w0, P, &e1)) { snprintf(err, errlen, "power_envelope: the single inversion failed at c = %g", c); return 1; }
    *P = clamp01(*P); *e = e1 + h.elast;
    return 0;
}
/* the region in which the single inversion is the default for the linear form P(U - aV <= b): a != 0, and -100 <= c <= 4 in
 * case DK with a zero start, -100 <= c <= 2 in the other cases (compared with the engines in November 2026: within 5.5e-11
 * of the classical code in case DK with a zero start, within 1e-12 of the general engine in the others, and right where
 * the general engine's one evaluation was off, by up to 6e-2 in case DK and 2.9e-3 in case DU with a fixed start in the
 * upper tail above the unit root).  At a = 0 (the law of U alone) the transform decays like a power, and b near the edge
 * -(1 + g0^2)/2 of the support makes the start S = 400/|w| of the tail run away; the engines have the closed form there.
 * At c = 7 in case DK the classical code differs from the single inversion by 2e-8 at the 99.9% point with a = 2, which no
 * third computation settled (lin_ok of DFKan) */
int dfl_kan_lin_ok(int model, int stat, double c, double g0, double a) { return a != 0 && c >= -100 && c <= ((model == 1 && !stat && g0 == 0) ? 4.0 : 2.0); }
/* the moments of the t-ratio from jets are the default up to c = DFL_INV_CMAX, any c0 and start (mom_ok of DFKan: compared
 * with the engines in November 2026 over the six cases, c from -480 to 20, c0 in {0, c, -2, 2}, g0 in {0, 1}: within 1e-11
 * relative near the unit root; with c0 = c far from it 1e-8 to 1.3e-7, where the engines' circles are not converged and
 * move towards these values under refinement; with a fixed start above the unit root and at c = 10, 20 with c0 = c the
 * engines' one evaluation was off by up to 0.65 relative) */
int dfl_kan_mom_ok(double c) { return c <= DFL_INV_CMAX; }
/* The first p moments of the t-ratio of a classical case for the start of a quantile search (mom_start of merged.jl,
 * November 2026): from one jet per node, the moments from jets of Bao and Kan (tratio_mom1: 0.2 to 1 ms for two moments,
 * where the circles of the engines, dfl_moments, take 10 to 150 ms), or 0 outside their region (dfl_kan_mom_ok; case DG),
 * where they fail or where an error estimate exceeds 1e-6 relative (to max(|m_j|, m_2^{j/2})); the search then starts from
 * the engines' circles.  The start only places the first step and the bracket of the search: the quantile is the root of
 * F(x) - p to the tolerance of the search either way.  DFL_MOM_START = 0 restores the start of before. */
int DFL_MOM_START = 1;
int dfl_mom_start(int p, int model, bool stat, double c, double c0, double g0, double *m)
{
    if (!DFL_MOM_START || p < 1 || p > DFL_KAN_JMAX - 1 || model < 1 || model > 3 || !dfl_kan_mom_ok(c)) return 0;
    dfl_limcf F; char e[256]; int k = p < 2 ? 2 : p;
    double mm[DFL_KAN_JMAX], ee[DFL_KAN_JMAX];
    if (dfl_limcf_init(&F, model, stat, c, stat ? 0.0 : g0, e, sizeof e) || dfl_kan_ratio_mom(&F, k, c0 - c, 0.5, 1e-10, mm, ee, e, sizeof e)) return 0;
    for (int j = 0; j < p; j++) {
        double sc = fmax(fabs(mm[j]), pow(fabs(mm[1]), (j + 1) / 2.0));
        if (!(isfinite(mm[j]) && ee[j] <= 1e-6 * sc)) return 0;
    }
    for (int j = 0; j < p; j++) m[j] = mm[j];
    return 1;
}
/* the single inversion is the default for the power envelope at c <= -1e-4 (within 2.7e-12 of the general engine's
 * envelope from c = -1e-4 to -500, November 2026; closer to zero the weight a = c/2 puts the critical value at the edge of
 * the support of U - aV), and at c >= 0 or NaN (the level, NaN) (envelope_ok of DFKan) */
int dfl_kan_envelope_ok(double c) { return !(c < 0) || c <= -1e-4; }

/* =====================================================================================================================
 * The distribution function of the t-ratio at zero in closed form (AR1LIM_RAY of Bao and Kan; ray_cdf of DFKan)
 * ===================================================================================================================== */
static double Phi_(double x) { return erfc(-x / sqrt(2.0)) / 2; }
static double npdf_(double x) { return exp(-x * x / 2) / sqrt(2 * M_PI); }
static double cexp1(double x) { return x == 0 ? 1.0 : expm1(x) / x; }
/* no deterministic terms, fixed start: U = [c_0 chi^2_1(delta) - (1 + g0^2)]/2 */
static double ray_k(double c, double g0)
{
    double c0 = cexp1(2 * c), dl = g0 * g0 * exp(2 * c) / c0, a = sqrt((1 + g0 * g0) / c0);
    return Phi_(a - sqrt(dl)) - Phi_(-a - sqrt(dl));
}
static const double RAY_C5[14] = {1.0 / 12, 1.0 / 12, 17.0 / 360, 7.0 / 360, 43.0 / 6720, 107.0 / 60480, 769.0 / 1814400, 163.0 / 1814400, 4097.0 / 239500800,
                                  709.0 / 239500800, 6827.0 / 14529715200, 15019.0 / 217945728000, 19661.0 / 2092278988800, 1139.0 / 951035904000};
typedef struct { double l1, l2, sd1, sd2; } ru_ctx;
static double ru_f(double z, void *v)
{
    ru_ctx *h = v; double zz = h->sd2 + z, y = (0.5 + h->l2 * (zz * zz)) / h->l1, r = sqrt(y);
    return (Phi_(r - h->sd1) - Phi_(-r - h->sd1)) * npdf_(z);
}
static int dcmp(const void *a, const void *b) { double x = *(const double *) a, y = *(const double *) b; return x < y ? -1 : x > y; }
/* an intercept: U = lambda_1 X_1 - lambda_2 X_2 - 1/2 with independent noncentral chi-square(1) variables; one integral of
 * the noncentral chi-square distribution function against the normal density */
static int ray_u(double c, double kap, double *out)
{
    double c0 = cexp1(2 * c), c1 = cexp1(c), c2, q;
    if (fabs(c) < 0.25) { q = 0; for (int k = 13; k >= 0; k--) q = q * c + RAY_C5[k]; c2 = c * c * q; }
    else { c2 = c0 - c1 * c1; q = c2 / (c * c); }
    double vs = sqrt(c2 * c2 + 4 * q), l1 = (vs + c2) / 4, l2 = q / (vs + c2);
    double d1 = c == 0 ? kap * kap / 2 : kap * kap / 2 * (1 - (c - 4) * c2 / (c * vs)), d2 = kap * kap - d1;
    ru_ctx h = {l1, l2, sqrt(fmax(d1, 0.0)), sqrt(fmax(d2, 0.0))};
    double wp[5]; int nw = 0;
    wp[nw++] = -12.0; wp[nw++] = 12.0;
    if (h.sd2 < 12) wp[nw++] = -h.sd2;
    double ys = (l1 * d1 - 0.5) / l2;
    if (ys > 0) for (int sg = -1; sg <= 1; sg += 2) { double zz = sg * sqrt(ys) - h.sd2; if (fabs(zz) < 12) wp[nw++] = zz; }
    qsort(wp, nw, sizeof(double), dcmp);
    int m = 0; for (int i = 0; i < nw; i++) if (m == 0 || wp[i] != wp[m - 1]) wp[m++] = wp[i];
    double Iv, Ev;
    if (dfl_quadgk(ru_f, &h, m, wp, 1e-14, 1e-12, 10000000L, &Iv, &Ev, NULL)) return 1;
    *out = Iv;
    return 0;
}
typedef struct { double a; } rk_ctx;
static double rsk_f(double z, void *v) { rk_ctx *h = v; return erf(sqrt(fmax(h->a + z * z, 0.0) / 2)) * npdf_(z); }
typedef struct { double c, sg; int fail; } rs_ctx;
static double rsu_f(double z, void *v) { rs_ctx *h = v; double r; if (ray_u(h->c, h->sg * z, &r)) { h->fail = 1; return 0; } return r * npdf_(z); }
int dfl_kan_ray_cdf(int model, int stat, double c, double g0, double *P, char *err, size_t errlen)
{
    if (!stat && model == 1) { *P = ray_k(c, g0); return 0; }
    if (!stat && model == 2) { if (ray_u(c, c * g0, P)) { snprintf(err, errlen, "ray_cdf: the quadrature failed"); return 1; } return 0; }
    if (!stat) { snprintf(err, errlen, "ray_cdf: cases DK, DU, SK, SU"); return 1; }
    if (!(c < 0)) { snprintf(err, errlen, "ray_cdf: a stationary start requires c < 0"); return 1; }
    double Iv, Ev, pts[3] = {-12.0, 0.0, 12.0};
    if (model == 1) {
        double lc = sqrt(-expm1(2 * c)) / (-4 * c); rk_ctx h = {1 / (2 * lc)};
        if (dfl_quadgk(rsk_f, &h, 3, pts, 1e-14, 1e-12, 10000000L, &Iv, &Ev, NULL)) { snprintf(err, errlen, "ray_cdf: the quadrature failed"); return 1; }
        *P = Iv; return 0;
    }
    if (model == 2) {
        rs_ctx h = {c, sqrt(-c / 2), 0};
        if (dfl_quadgk(rsu_f, &h, 3, pts, 1e-12, 1e-10, 10000000L, &Iv, &Ev, NULL) || h.fail) { snprintf(err, errlen, "ray_cdf: the quadrature failed"); return 1; }
        *P = Iv; return 0;
    }
    snprintf(err, errlen, "ray_cdf: cases DK, DU, SK, SU");
    return 1;
}
