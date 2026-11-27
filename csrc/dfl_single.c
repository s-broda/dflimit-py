/* dfl_single.c: the single integral (the line integral representation) at c0 = 0: the inner transforms of the
 * tilted density along paths in the complex plane (adaptive Gauss-Legendre rules, or the fixed node layouts of dfl_batch.c
 * where they serve: c <= 0, |q| above about 0.034, opts->batched), the outer integral along Re t = tau (trapezoidal rule in
 * u for the smooth part, Filon's rule for the edge part), zero and stationary start.  A port of outer / inner_density /
 * inner_cdf / density / cdf of DFExact.jl; the general kernel for c0 != 0 is in dfl_gen.c. */
#include "dflimit.h"
#include "dfl_lock.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const cdouble ZERO = {0}, ONE = {1};
#define EXPO DFL_EXPO
#define KAPPA DFL_KAPPA
#define WMAX DFL_WMAX
int DFL_NODEFAC = 1;                              /* multiplies every node count (2: convergence check) */
double DFL_FILONW = 300.0;

/* ---- quadrature primitives: f(y, ctx) complex ---- */
typedef cdouble (*fn_t)(double y, const void *ctx);
static const char RULE_WHAT[] = "a Gauss-Legendre rule of the inner integrals";
/* (a rule beyond DFL_RULEMAX, or a point already refused on this thread: nothing is evaluated, the bound is recorded) */
static cdouble glsum(fn_t f, const void *ctx, double a, double b, int n, int p)
{
    n *= DFL_NODEFAC;
    if (dfl_bound_hit()) return ZERO;
    const double *gw; const double *u = dfl_rule(n, &gw);
    if (u == NULL) { dfl_bound_set(DFL_BOUND_RULE, n, RULE_WHAT); return ZERO; }
    cdouble acc = ZERO; double L = b - a;
    for (int k = 0; k < n; k++) {
        double y, dy;
        if (p == 1) { y = a + L * u[k]; dy = L; }
        else if (p == 2) { y = a + L * u[k] * u[k]; dy = 2 * L * u[k]; }
        else { double u2 = u[k] * u[k]; y = a + L * u2 * u2; dy = 4 * L * u2 * u[k]; }
        acc = cadd(acc, cscale(dy * gw[k], f(y, ctx)));
    }
    return acc;
}
/* the node count of an outer piece of rayint, at most 0.85 rad per node; beyond DFL_RULEMAX the bound is recorded (glsum then
 * evaluates nothing) */
static int nodes(int n, double osc, double L)
{
    double mm = 1.2 * osc * L;
    if (mm > DFL_RULEMAX) { dfl_bound_set(DFL_BOUND_RULE, mm, RULE_WHAT); return n; }
    int m = 16 * (int) ceil(ceil(mm) / 16); return n > m ? n : m;
}
static cdouble rayint(fn_t f, const void *ctx, double R, double l, int n1, int n2, int p, double osc)
{
    double a = dfl_nosliver(fmin(R, 8 * l), R);
    cdouble acc = glsum(f, ctx, 0, a, n1, p);
    while (a < R) {                                  /* geometric panels of ratio 4 (no sliver at R: dfl_nosliver) */
        double b = dfl_nosliver(fmin(R, 4 * a), R);
        acc = cadd(acc, glsum(f, ctx, a, b, nodes(n2, osc, b - a), 1));
        a = b;
    }
    return acc;
}
static double cap(double x) { return x / sqrt(1 + x * x); }
static bool ray_ok(cdouble sd, cdouble e, double q, double at)
{
    cdouble sde = cmul(sd, e);
    double m = fmax(0, -creal(sde));
    return m * m * q * q / (4 * at) <= 2 && fabs(cimag(sde)) * q * sqrt(EXPO / at) <= 60;
}

/* ---- the inverse s-transform phi_t(X) ---- */
static cdouble phi_pos(int kern, cdouble X, cdouble s1, cdouble s2)
{
    if (kern == 1) return cdiv(cexp(cneg(cmul(s1, X))), csqrt(cscale(M_PI, X)));
    return cscale(1 / M_PI, cmul(dfl_k0x(cmul(cscale(0.5, csub(s2, s1)), X)), cexp(cneg(cmul(s2, X)))));
}
static cdouble phi_neg(int kern, cdouble X, cdouble s1, cdouble s2)
{
    if (kern == 1) return ZERO;
    return cscale(1 / M_PI, cmul(dfl_k0x(cneg(cmul(cscale(0.5, csub(s2, s1)), X))), cexp(cneg(cmul(s1, X)))));
}
static cdouble phi_alg(int kern, cdouble X, cdouble s1, cdouble s2)
{
    if (kern == 1) return cdiv(ONE, csqrt(cscale(M_PI, X)));
    return cscale(1 / M_PI, dfl_k0x(cmul(cscale(0.5, csub(s2, s1)), X)));
}

/* ---- gaussexp: int_0^inf A(x) exp(-t x^2/q^2 - sd x) dx along the saddle or the level-curve path ---- */
typedef cdouble (*cfn_t)(cdouble x, const void *ctx);
typedef struct { cfn_t A; const void *Actx; cdouble t, sd, x0, u0, e; double at, q, vmin; } gxc;
static cdouble gx_seg(double lam, const void *v) { const gxc *g = v; cdouble x = cscale(lam, g->x0);
    return cmul(cmul(g->A(x, g->Actx), cexp(cscale(-lam * (2 - lam), g->u0))), g->x0); }
static cdouble gx_line(double r, const void *v) { const gxc *g = v; cdouble x = cadd(g->x0, cscale(r, g->e));
    return cmul(cmul(g->A(x, g->Actx), cexp(csub(cneg(g->u0), CPX(g->at * r * r / (g->q * g->q), 0)))), g->e); }
static cdouble gx_level(double s, const void *v) { const gxc *g = v;
    double vv = g->vmin * sinh(s), dv = g->vmin * cosh(s);
    cdouble rt = cmul(g->sd, csqrt(cadd(ONE, cdiv(cscale(4 * vv * vv, g->t), cscale(g->q * g->q, cmul(g->sd, g->sd))))));
    cdouble x = cdiv(CPX(2 * vv * vv, 0), cadd(g->sd, rt));
    return cscale(exp(-vv * vv) * 2 * vv * dv, cdiv(g->A(x, g->Actx), rt)); }
static cdouble gaussexp(cfn_t A, const void *Actx, cdouble t, double q, cdouble sd, double lA, int force, int p)
{
    gxc g; g.A = A; g.Actx = Actx; g.t = t; g.sd = sd; g.q = q;
    g.at = cabs(t); g.e = cexp(CPX(0, -carg(t) / 2));
    g.x0 = cdiv(cscale(-q * q, sd), cscale(2, t)); g.u0 = cdiv(cscale(-q * q, cmul(sd, sd)), cscale(4, t));
    if (force == 1 || (force == 0 && creal(cmul(sd, g.e)) < 0)) {
        double ru = creal(g.u0), lam_max = ru > EXPO ? 1 - sqrt(1 - EXPO / ru) : 1.0;
        double l = fmin(lA / cabs(g.x0), 1 / (2 * cabs(g.u0) + 1));
        cdouble acc = rayint(gx_seg, &g, lam_max, l, 48, 32, p, 2 * fabs(cimag(g.u0)));
        if (ru > EXPO) return acc;
        return cadd(acc, rayint(gx_line, &g, q * sqrt(EXPO / g.at), lA, 48, 32, 1, 0));
    }
    double v0 = cabs(sd) * q / (2 * sqrt(g.at)), vA = sqrt(lA * cabs(sd));
    g.vmin = fmax(fmin(fmin(v0, vA), 1.0) / 2, 1e-4);
    double S = asinh(sqrt(EXPO) / g.vmin);
    if (p == 1) return glsum(gx_level, &g, 0, S, 16 * (int) ceil(ceil(13 * S) / 16), 1);
    double S1 = fmin(1.0, S);
    cdouble acc = glsum(gx_level, &g, 0, S1, 32, p);
    if (S > 1) acc = cadd(acc, glsum(gx_level, &g, 1, S, 16 * (int) ceil(ceil(13 * (S - 1)) / 16), 1));
    return acc;
}

/* ---- the inner integrals at c0 = 0 ---- */
typedef struct { int kern; cdouble tj, s1, s2, M0, e, sd; double q, at, th, w, vs, sg, kap, H, vs2, lA, fac; } ictx;
/* density: the ray from v = 0 (q > 0, or q < 0 with a negligible edge) */
static cdouble dens_ray(double r, const void *v) { const ictx *c = v; cdouble vv = cscale(r, c->e);
    return cscale(2 * exp(-c->at * r * r), cmul(cmul(vv, vv), cmul(phi_pos(c->kern, cadd(cscale(c->q, vv), CPX(0.5, 0)), c->s1, c->s2), c->e))); }
static cdouble dens_A(cdouble x, const void *v) { const ictx *c = v;
    return cmul(cscale(2 / (c->q * c->q * c->q), cmul(x, x)), cmul(phi_alg(c->kern, cadd(CPX(0.5, 0), x), c->s1, c->s2), cexp(cscale(-0.5, c->sd)))); }
/* the bulged segment, parametrized by d = v* - r */
static cdouble dens_pathd(double d, const ictx *c)
{
    double r = c->vs - d, aa = cap(r / c->H), bb = cap(d / c->H), g = c->H * aa * bb;
    double dg = bb / pow(sqrt(1 + (r / c->H) * (r / c->H)), 3) - aa / pow(sqrt(1 + (d / c->H) * (d / c->H)), 3);
    cdouble vv = CPX(r, -g * c->sg), dv = CPX(1, -dg * c->sg), X = cscale(fabs(c->q), CPX(d, g * c->sg));
    return cmul(cmul(cscale(2, cmul(vv, vv)), cexp(cneg(cmul(c->tj, cmul(vv, vv))))), cmul(phi_pos(c->kern, X, c->s1, c->s2), dv));
}
static cdouble dens_chi(double d, const ictx *c)
{
    cdouble arg = cscale(c->kap, csub(CPX(c->vs / 2 - d, 0), CPX(0, c->H * cap((c->vs - d) / c->H) * cap(d / c->H) * c->sg)));
    return cscale(0.5, cadd(ONE, dfl_erf(arg)));
}
static cdouble dens_fs(double r, const void *v) { const ictx *c = v; double d = c->vs - r; return cmul(dens_pathd(d, c), csub(ONE, dens_chi(d, c))); }
static cdouble dens_fe(double d, const void *v) { const ictx *c = v; return cmul(dens_pathd(d, c), dens_chi(d, c)); }
static cdouble dens_neg(double r, const void *v) { const ictx *c = v; cdouble vv = cadd(CPX(c->vs, 0), cscale(r, c->e)), X = cscale(-fabs(c->q) * r, c->e);
    return cmul(cmul(cscale(2, cmul(vv, vv)), cexp(cneg(cmul(c->tj, cmul(vv, vv))))), cmul(phi_neg(c->kern, X, c->s1, c->s2), c->e)); }

/* the first piece [0, a] of the edge segment, from the support edge, with the substitution u^4; with a bound e1 < a on its length
 * (the scale 1/b of K_0(b X) near X = 0 for c < -20, outer_c0) the rest of it in geometric panels of ratio 4 (edgehead in DFExact.jl) */
static cdouble edgehead(fn_t f, const void *ctx, double a, double e1)
{
    if (!(e1 > 0 && e1 < a)) return glsum(f, ctx, 0, a, 48, 4);
    cdouble acc = glsum(f, ctx, 0, e1, 48, 4); double lo = e1;
    while (lo < a) { double hi = dfl_nosliver(fmin(a, 4 * lo), a); acc = cadd(acc, glsum(f, ctx, lo, hi, 48, 1)); lo = hi; }
    return acc;
}
static void inner_density(int kern, cdouble tj, double q, cdouble s1, cdouble s2, int part, double e1, cdouble *Is, cdouble *Ie)
{
    ictx c; c.kern = kern; c.tj = tj; c.s1 = s1; c.s2 = s2; c.q = q;
    c.th = carg(tj) / 2; c.at = cabs(tj); c.e = cexp(CPX(0, -c.th)); c.w = fabs(cimag(tj));
    c.vs = 1 / (2 * fabs(q));
    *Is = ZERO; *Ie = ZERO;
    if (q == 0) { if (part != 2) *Is = cmul(cscale(sqrt(M_PI) / 2, cpow(tj, CPX(-1.5, 0))), phi_pos(kern, CPX(0.5, 0), s1, s2)); return; }   /* x = 0: I_f = (sqrt(pi)/2) t^(-3/2) phi_t(1/2), no edge part */
    if (q > 0 || 3 * creal(tj) * c.vs * c.vs / 16 > 40) {
        if (part == 2) return;
        c.sd = kern == 1 ? s1 : s2;
        c.lA = 1 / (cabs(kern == 1 ? s1 : cscale(0.5, csub(s2, s1))) + 2);
        if (q < 0 || ray_ok(c.sd, c.e, q, c.at)) {
            double R = q > 0 ? sqrt(EXPO / c.at) : fmin(c.vs, sqrt(EXPO / c.at));
            *Is = rayint(dens_ray, &c, R, c.lA / fabs(q), 48, 32, 1, fabs(q) * fabs(cimag(cmul(c.sd, c.e))));
            return;
        }
        *Is = gaussexp(dens_A, &c, tj, q, c.sd, c.lA, 0, 1);
        return;
    }
    c.sg = cimag(tj) >= 0 ? 1.0 : -1.0;
    c.kap = KAPPA / (c.vs / 2); c.H = 1.5 / c.kap;
    if (part != 2) {
        double r1 = dfl_nosliver(c.w > 0 ? fmin(c.vs, sqrt(EXPO / (1.9 * c.w))) : c.vs, c.vs);
        *Is = glsum(dens_fs, &c, 0, r1, 64, 1);
        if (r1 < c.vs) *Is = cadd(*Is, glsum(dens_fs, &c, r1, c.vs, 48, 1));
    }
    if (part == 1) return;
    double delta = c.w > 0 ? fmin(c.vs, EXPO / (1.9 * c.w * c.vs)) : c.vs;
    *Ie = cadd(edgehead(dens_fe, &c, delta / 8, e1 / fabs(q)), glsum(dens_fe, &c, delta / 8, delta, 64, 1));
    if (delta < c.vs) *Ie = cadd(*Ie, glsum(dens_fe, &c, delta, c.vs, 48, 1));
    if (kern != 1) {
        double rho = creal(cmul(cscale(0.5, csub(s2, cscale(3, s1))), c.e)) * fabs(q);
        double R = fmin(sqrt(EXPO / c.at), EXPO / (2 * c.at * c.vs * cos(c.th)));
        if (rho > 0) R = fmin(R, EXPO / rho);
        *Ie = cadd(*Ie, rayint(dens_neg, &c, R, 1 / (rho + 2 * c.at * c.vs * cos(c.th)), 64, 32, 4,
                               fabs(q) * fabs(cimag(cmul(s1, c.e))) + 2 * c.at * c.vs * fabs(sin(c.th))));
    }
}

/* cdf */
static cdouble cdf_const_pos(double r, const void *v) { const ictx *c = v; cdouble e2 = c->e;   /* e holds e2 here */
    return cmul(phi_pos(c->kern, cadd(CPX(0.5, 0), cscale(r, e2)), c->s1, c->s2), e2); }
static cdouble cdf_const_X(double X, const void *v) { const ictx *c = v; return phi_pos(c->kern, CPX(X, 0), c->s1, c->s2); }
static cdouble cdf_const_neg(double r, const void *v) { const ictx *c = v; cdouble e1 = c->e;
    return cmul(phi_neg(c->kern, cscale(-r, e1), c->s1, c->s2), e1); }
/* (the Gaussian factor e^{-|t| (r/q)^2} with r/q of order one: for |q| below about 1e-154, where the nodes r are of the order
 * of q, r^2 and q^2 underflowed, to 0/0 = NaN below 1e-162) */
static cdouble cdf_ray(double r, const void *v) { const ictx *c = v; double sq = c->q > 0 ? 1 : -1, rq = r / c->q;
    cdouble X = cadd(CPX(0.5, 0), cscale(sq * r, c->e));
    return cscale(sq * exp(-c->at * rq * rq), cmul(phi_pos(c->kern, X, c->s1, c->s2), c->e)); }
static cdouble cdf_A(cdouble x, const void *v) { const ictx *c = v; return cmul(phi_alg(c->kern, cadd(CPX(0.5, 0), x), c->s1, c->s2), cexp(cscale(-0.5, c->sd))); }
static cdouble cdf_path(double Xr, const ictx *c)
{
    double aa = cap(Xr / c->H), bb = cap((0.5 - Xr) / c->H), g = c->H * aa * bb;
    double dg = bb / pow(sqrt(1 + (Xr / c->H) * (Xr / c->H)), 3) - aa / pow(sqrt(1 + ((0.5 - Xr) / c->H) * ((0.5 - Xr) / c->H)), 3);
    cdouble X = CPX(Xr, g * c->sg), dX = CPX(1, dg * c->sg), x = csub(X, CPX(0.5, 0));
    return cneg(cmul(cmul(phi_pos(c->kern, X, c->s1, c->s2), cexp(cdiv(cneg(cmul(c->tj, cmul(x, x))), CPX(c->q * c->q, 0)))), dX));
}
static cdouble cdf_chi(double Xr, const ictx *c)
{
    cdouble arg = cscale(c->kap, csub(CPX(0.25 - Xr, 0), CPX(0, c->H * cap(Xr / c->H) * cap((0.5 - Xr) / c->H) * c->sg)));
    return cscale(0.5, cadd(ONE, dfl_erf(arg)));
}
static cdouble cdf_fs(double y, const void *v) { const ictx *c = v; return cmul(cdf_path(0.5 - y, c), csub(ONE, cdf_chi(0.5 - y, c))); }
static cdouble cdf_fe(double Xr, const void *v) { const ictx *c = v; return cmul(cdf_path(Xr, c), cdf_chi(Xr, c)); }
static cdouble cdf_neg(double r, const void *v) { const ictx *c = v; cdouble X = cscale(-r, c->e), x = csub(X, CPX(0.5, 0));
    return cneg(cmul(cmul(phi_neg(c->kern, X, c->s1, c->s2), cexp(cdiv(cneg(cmul(c->tj, cmul(x, x))), CPX(c->q * c->q, 0)))), c->e)); }

/* Psi_t(1/2): the mass of phi_t on (-inf, 1/2) */
static cdouble psi_half(int kern, cdouble s1, cdouble s2, cdouble M0)
{
    if (kern == 1) return cdiv(dfl_erf(csqrt(cscale(0.5, s1))), csqrt(s1));
    ictx c; c.kern = kern; c.s1 = s1; c.s2 = s2;
    double l = 1 / (cabs(csub(s2, s1)) + 2);
    if (cabs(s2) >= cabs(s1) / 4) {
        c.e = cexp(CPX(0, -carg(s2)));
        return csub(M0, rayint(cdf_const_pos, &c, 36 / cabs(s2), l, 40, 32, 1, 0));
    }
    cdouble a = rayint(cdf_const_X, &c, 0.5, l, 40, 32, 4, 0);
    c.e = cexp(CPX(0, -carg(cneg(s1))));
    return cadd(a, rayint(cdf_const_neg, &c, 36 / cabs(s1), l, 40, 32, 4, 0));
}
static void inner_cdf(int kern, cdouble tj, double q, cdouble s1, cdouble s2, cdouble M0, int part, double e1, cdouble *Is, cdouble *Ie)
{
    ictx c; c.kern = kern; c.tj = tj; c.s1 = s1; c.s2 = s2; c.q = q; c.M0 = M0;
    c.th = carg(tj) / 2; c.at = cabs(tj); c.e = cexp(CPX(0, -c.th)); c.w = fabs(cimag(tj));
    c.vs2 = 1 / (4 * q * q);
    *Is = ZERO; *Ie = ZERO;
    if (part != 2) *Is = psi_half(kern, s1, s2, M0);
    if (q == 0) return;   /* x = 0: I_F = Psi_t(1/2), the Gaussian part vanishes and there is no edge part */
    if (q > 0 || 3 * creal(tj) * c.vs2 / 16 > 40) {
        if (part == 2) return;
        c.sd = kern == 1 ? s1 : s2;
        c.lA = 1 / (cabs(kern == 1 ? s1 : cscale(0.5, csub(s2, s1))) + 2);
        if (q < 0 || ray_ok(c.sd, c.e, q, c.at)) {
            double R = q > 0 ? fabs(q) * sqrt(EXPO / c.at) : fmin(0.5, fabs(q) * sqrt(EXPO / c.at));
            *Is = cadd(*Is, rayint(cdf_ray, &c, R, c.lA, 48, 32, 1, fabs(cimag(cmul(c.sd, c.e)))));
            return;
        }
        *Is = cadd(*Is, gaussexp(cdf_A, &c, tj, q, c.sd, c.lA, 0, 1));
        return;
    }
    c.sg = cimag(tj) >= 0 ? 1.0 : -1.0;
    c.kap = KAPPA / 0.25; c.H = 1.5 / c.kap;
    if (part != 2) {
        double ep = dfl_nosliver(c.w > 0 ? fmin(0.5, fabs(q) * sqrt(EXPO / (1.9 * c.w))) : 0.5, 0.5);
        *Is = cadd(*Is, glsum(cdf_fs, &c, 0, ep, 64, 1));
        if (ep < 0.5) *Is = cadd(*Is, glsum(cdf_fs, &c, ep, 0.5, 48, 1));
    }
    if (part == 1) return;
    double delta = c.w > 0 ? fmin(0.5, EXPO * q * q / (0.95 * c.w)) : 0.5;
    *Ie = cadd(edgehead(cdf_fe, &c, delta / 8, e1), glsum(cdf_fe, &c, delta / 8, delta, 64, 1));
    if (delta < 0.5) *Ie = cadd(*Ie, glsum(cdf_fe, &c, delta, 0.5, 48, 1));
    if (kern != 1) {
        double rho = creal(cmul(cscale(0.5, csub(s2, cscale(3, s1))), c.e));
        double R = fmin(fabs(q) * sqrt(EXPO / c.at), EXPO * q * q / (c.at * cos(c.th)));
        if (rho > 0) R = fmin(R, EXPO / rho);
        *Ie = cadd(*Ie, rayint(cdf_neg, &c, R, 1 / (rho + c.at * cos(c.th) / (q * q)), 64, 32, 4,
                               fabs(cimag(cmul(s1, c.e))) + c.at * fabs(sin(c.th)) / (q * q)));
    }
}

/* ---- Filon's rule on a panel: int_a^b G(w) e^{-i y w} dw, G interpolated at the N Gauss-Legendre nodes ---- */
#define FILON_MAXN 64
static double *FILP[FILON_MAXN + 1], *FILW[FILON_MAXN + 1];
static dfl_lock_t fillock = DFL_LOCK_INIT;
static void filon_setup(int N)
{
    dfl_lock(&fillock);
    if (FILP[N] == NULL) {
        const double *gw; const double *u = dfl_rule(N, &gw);
        double *P = malloc(N * N * sizeof(double)), *ww = malloc(N * sizeof(double));
        for (int k = 0; k < N; k++) { double x = 2 * u[k] - 1; ww[k] = 2 * gw[k]; P[k] = 1; if (N > 1) P[N + k] = x;
            for (int n = 1; n < N - 1; n++) P[(n + 1) * N + k] = ((2 * n + 1) * x * P[n * N + k] - n * P[(n - 1) * N + k]) / (n + 1); }
        FILW[N] = ww; FILP[N] = P;
    }
    dfl_unlock(&fillock);
}
static cdouble filon_panel(const cdouble *G, double a, double b, double y, int N)
{
    filon_setup(N);
    const double *P = FILP[N], *ww = FILW[N];
    double L = b - a, Omega = y * L / 2, jn[FILON_MAXN];
    dfl_sphbessel(jn, N, Omega);
    cdouble acc = ZERO, mi = ONE;
    for (int n = 0; n < N; n++) {
        cdouble cn = ZERO;
        for (int k = 0; k < N; k++) cn = cadd(cn, cscale(ww[k] * P[n * N + k], G[k]));
        acc = cadd(acc, cscale((2 * n + 1) * jn[n], cmul(cn, mi)));
        mi = cmul(mi, CPX(0, -1));
    }
    return cscale(L / 2, cmul(cexp(CPX(0, -y * (a + b) / 2)), acc));
}

/* ---- the saddle factor's phase along the line (q > 0, c > 0) ---- */
static void saddle_phase(int kern, double q, int n, const cdouble *ts, const cdouble *s1, const cdouble *s2, double *psi)
{
    for (int j = 0; j < n; j++) {
        cdouble sd = kern == 1 ? s1[j] : s2[j], z = cdiv(cscale(q, sd), cscale(2, csqrt(ts[j])));
        psi[j] = NAN;
        if (creal(z) < 0 && creal(cmul(z, z)) > -30) psi[j] = cimag(cmul(z, z));
    }
}
static double max_phase_step(int n, const double *psi)
{
    double m = 0;
    for (int j = 0; j + 1 < n; j++) if (!isnan(psi[j]) && !isnan(psi[j + 1])) m = fmax(m, fabs(psi[j + 1] - psi[j]));
    return m;
}

/* ---- the bound of the rules up front (gauss_nmax, gauss_check of DFExact.jl) ----
 * The largest node count of a piece of the segment to the saddle that gaussexp takes for the inner integrals of the central
 * kernels at t and q > 0 (inner_density, inner_cdf), before rounding; 0 where they take another path.  Its phase 2 Im u0,
 * u0 = -sd^2 q^2/(4t), grows like q^2 and rayint gives every piece 1.2 nodes per radian: at c = 7 in case DT the largest
 * piece has more than DFL_RULEMAX nodes from q ~ 660.  outer_c0 refuses a line on which it exceeds DFL_RULEMAX before it
 * evaluates anything (the same steps as gaussexp and rayint, without the integrand). */
static double gauss_nmax(int kern, cdouble t, double q, cdouble s1, cdouble s2)
{
    cdouble sd = kern == 1 ? s1 : s2;
    double at = cabs(t); cdouble e = cexp(CPX(0, -carg(t) / 2));
    if (q <= 0 || ray_ok(sd, e, q, at) || creal(cmul(sd, e)) >= 0) return 0;
    double lA = 1 / (cabs(kern == 1 ? s1 : cscale(0.5, csub(s2, s1))) + 2);
    cdouble x0 = cdiv(cscale(-q * q, sd), cscale(2, t)), u0 = cdiv(cscale(-q * q, cmul(sd, sd)), cscale(4, t));
    double ru = creal(u0), R = ru > EXPO ? 1 - sqrt(1 - EXPO / ru) : 1.0;
    double l = fmin(lA / cabs(x0), 1 / (2 * cabs(u0) + 1)), osc = 2 * fabs(cimag(u0));
    double a = fmin(R, 8 * l), m = 0;
    while (a < R) { double b = fmin(R, 4 * a); m = fmax(m, 1.2 * osc * (b - a)); a = b; }
    return m;
}
static const char GAUSS_WHAT[] = "the segment to the saddle of the Gaussian factor of the inner integrals, whose phase grows like x^2,";
/* the margin of the early check of a provider's line on the coarse points of its outer panels (outer_c0; EARLY_MARGIN of
 * DFExact.jl: in a scan of 2172 configurations the check on the nodes refused wherever the coarse maximum exceeded twice
 * DFL_RULEMAX) */
#define EARLY_MARGIN 2.0
static int gauss_check(int kern, double q, int n, const cdouble *ts, const cdouble *s1, const cdouble *s2, dfl_bound_t *bd)
{
    if (!(q > 0)) return 0;
    double m = 0;
    for (int j = 0; j < n; j++) m = fmax(m, gauss_nmax(kern, ts[j], q, s1[j], s2[j]));
    if (!(m > DFL_RULEMAX)) return 0;
    bd->bound = DFL_BOUND_RULE; bd->need = m; bd->what = GAUSS_WHAT;
    return DFL_BOUND_RULE;
}

/* ---- the outer integral ---- */
typedef struct { int kern, kind, n; double q; const cdouble *ts, *s1, *s2, *M0, *P; const double *wt; double *g; cdouble *G; double y, cde; const double *us; double e1; dfl_bound_t *fail; } octx;
/* (a node whose inner integrals meet a bound of the engine records it in fail, and the other nodes are skipped) */
static void smooth_body(int j, void *v) { octx *o = v; cdouble Is, Ie;
    if (dfl_bound_failed(o->fail)) return;
    dfl_bound_clear();
    if (o->kind == 1) inner_density(o->kern, o->ts[j], o->q, o->s1[j], o->s2[j], 1, INFINITY, &Is, &Ie);
    else inner_cdf(o->kern, o->ts[j], o->q, o->s1[j], o->s2[j], o->M0[j], 1, INFINITY, &Is, &Ie);
    if (dfl_bound_note(o->fail)) return;
    cdouble val = cmul(o->P[j], Is); if (o->kind == 2) val = cdiv(val, o->ts[j]);
    o->g[j] = creal(val) * o->wt[j]; }
static void edge_body(int j, void *v) { octx *o = v; cdouble Is, Ie; cdouble t = o->ts[j + 1];
    if (dfl_bound_failed(o->fail)) return;
    dfl_bound_clear();
    if (o->kind == 1) inner_density(o->kern, t, o->q, o->s1[j + 1], o->s2[j + 1], 2, o->e1, &Is, &Ie);
    else inner_cdf(o->kern, t, o->q, o->s1[j + 1], o->s2[j + 1], o->M0[j + 1], 2, o->e1, &Is, &Ie);
    if (dfl_bound_note(o->fail)) return;
    cdouble val = cmul(o->P[j + 1], Ie); if (o->kind == 2) val = cdiv(val, t);
    o->G[j] = cmul(val, cexp(cscale(o->y, t))); }

/* the values of the smooth part from the fixed layouts bt: g[j] = Re(P_j v_j / t_j^(kind - 1)) w_j */
static void batch_smooth(const dfl_batch *bt, int kind, int n, const cdouble *ts, const cdouble *P, const cdouble *s1, const cdouble *s2, const cdouble *M0, const double *wt, double *g)
{
    cdouble *sv = malloc((n + 1) * sizeof(cdouble));
    dfl_batch_values(bt, 1, n, ts, s1, s2, M0, sv);
    for (int j = 0; j < n; j++) { cdouble val = cmul(P[j], sv[j]); if (kind == 2) val = cdiv(val, ts[j]); g[j] = creal(val) * wt[j]; }
    free(sv);
}
/* the value in *out; 0, or the bound of the engine that the line would exceed (recorded in *bd, *out NaN).  bt: the fixed
 * layouts of the inner integrals (NULL: the adaptive rules); e1: the cap of the first piece of the K_0 kernel's edge segment
 * for the adaptive rules (dfl_k0first) */
static int outer_c0(int kind, int model, double c, double q, double tau, bool stat, const dfl_single_opts *o, const dfl_batch *bt, double e1, double *out, dfl_bound_t *bd)
{
    *out = NAN;
    int kern = dfl_kern(model, stat);
    double hu = o->hu * dfl_hufac(c), cde = tau, Wm = dfl_wmaxc_m(model, c); int N = o->N;
    /* the smooth part: the trapezoidal rule in u, w = tau sinh u */
    double umax = asinh(Wm / cde); int nu = (int) floor(umax / hu) + 1;
    cdouble *ts = malloc(nu * sizeof(cdouble)), *P = malloc(nu * sizeof(cdouble)), *s1 = malloc(nu * sizeof(cdouble)), *s2 = malloc(nu * sizeof(cdouble)), *M0 = malloc(nu * sizeof(cdouble));
    double *wt = malloc(nu * sizeof(double)), *g = malloc(nu * sizeof(double)), *psi = malloc(nu * sizeof(double));
    for (int j = 0; j < nu; j++) { double u = j * hu; ts[j] = CPX(tau, cde * sinh(u)); wt[j] = cde * cosh(u); }
    dfl_tparts(model, c, nu, ts, stat, P, s1, s2, M0);
    saddle_phase(kern, q, nu, ts, s1, s2, psi);
    double val = NAN;
    dfl_bound_t fail = {0, 0, NULL}; *bd = fail;
    octx oc = {kern, kind, nu, q, ts, s1, s2, M0, P, wt, g, NULL, 0, cde, NULL, 0, &fail};
    if (q > 0 && max_phase_step(nu, psi) > 0.25) {
        /* Gauss-Legendre panels with node counts following the phase */
        double w0 = fmin(0.5, tau / 2); int ne = (int) ceil(log2(Wm / w0)) + 2;
        double *edges0 = malloc(ne * sizeof(double)); edges0[0] = 0; for (int p = 1; p < ne; p++) edges0[p] = w0 * pow(2, p - 1); edges0[ne - 1] = Wm;
        int K = 8, nk = (ne - 1) * K + 1;
        cdouble *te0 = malloc(nk * sizeof(cdouble)), *P0 = malloc(nk * sizeof(cdouble)), *s1e = malloc(nk * sizeof(cdouble)), *s2e = malloc(nk * sizeof(cdouble)), *M0e = malloc(nk * sizeof(cdouble));
        for (int p = 0; p < ne - 1; p++) for (int k = 0; k < K; k++) te0[p * K + k] = CPX(tau, edges0[p] + (edges0[p + 1] - edges0[p]) * k / K);
        te0[nk - 1] = CPX(tau, Wm);
        dfl_tparts(model, c, nk, te0, stat, P0, s1e, s2e, M0e);
        double *psie = malloc(nk * sizeof(double)); saddle_phase(kern, q, nk, te0, s1e, s2e, psie);
        /* (a provider of the general engine, whose transform costs much more than the case codes' closed forms: the bound of
         * the inner rules on these coarse points first, with the margin EARLY_MARGIN, before the transform at the nodes of the
         * panels, which grow in number with x: 67680 nodes in case DU at c = 10, x = 7860, where the bound then refused the
         * point anyway; early_bound of DFExact.jl, whose scan found the check on the nodes refusing wherever this one does) */
        if (model == DFL_MODEL_PROV && !bt) {
            double mx = 0;
            for (int j = 0; j < nk; j++) mx = fmax(mx, gauss_nmax(kern, te0[j], q, s1e[j], s2e[j]));
            if (mx > EARLY_MARGIN * DFL_RULEMAX) { fail.bound = DFL_BOUND_RULE; fail.need = mx; fail.what = GAUSS_WHAT; }
        }
        if (fail.bound) {
            free(edges0); free(te0); free(P0); free(s1e); free(s2e); free(M0e); free(psie);
            free(ts); free(P); free(s1); free(s2); free(M0); free(wt); free(g); free(psi);
            *bd = fail; return fail.bound;
        }
        int cap_n = 0; for (int p = 0; p < ne - 1; p++) { double dp = 0; for (int k = 0; k < K; k++) { int i = p * K + k; if (!isnan(psie[i]) && !isnan(psie[i + 1])) dp += fabs(psie[i + 1] - psie[i]); }
            int m = (int) fmax(1, ceil(dp / 50)); int n = (int) fmax(24, 16 * ceil(ceil(3 * dp / m) / 16)) * DFL_NODEFAC; cap_n += m * n; }
        cdouble *tn = malloc((cap_n + 1) * sizeof(cdouble)); double *wn = malloc(cap_n * sizeof(double)); int nn = 0;
        for (int p = 0; p < ne - 1; p++) {
            double a = edges0[p], b = edges0[p + 1], dp = 0;
            for (int k = 0; k < K; k++) { int i = p * K + k; if (!isnan(psie[i]) && !isnan(psie[i + 1])) dp += fabs(psie[i + 1] - psie[i]); }
            int m = (int) fmax(1, ceil(dp / 50)); int n = (int) fmax(24, 16 * ceil(ceil(3 * dp / m) / 16)) * DFL_NODEFAC;
            const double *gw; const double *u = dfl_rule(n, &gw);
            for (int k = 0; k < m; k++) { double aa = a + (b - a) * k / m, bb = a + (b - a) * (k + 1) / m;
                for (int i = 0; i < n; i++) { tn[nn + 1] = CPX(tau, aa + (bb - aa) * u[i]); wn[nn] = (bb - aa) * gw[i]; nn++; } }
        }
        tn[0] = CPX(tau, 0);
        cdouble *Pn = malloc((nn + 1) * sizeof(cdouble)), *s1n = malloc((nn + 1) * sizeof(cdouble)), *s2n = malloc((nn + 1) * sizeof(cdouble)), *M0n = malloc((nn + 1) * sizeof(cdouble));
        dfl_tparts(model, c, nn + 1, tn, stat, Pn, s1n, s2n, M0n);
        double *gn = malloc(nn * sizeof(double));
        octx on = {kern, kind, nn, q, tn + 1, s1n + 1, s2n + 1, M0n + 1, Pn + 1, wn, gn, NULL, 0, cde, NULL, 0, &fail};
        /* (a line on which the inner integrals need a rule beyond DFL_RULEMAX is refused before anything is evaluated) */
        if (bt) batch_smooth(bt, kind, nn, tn + 1, Pn + 1, s1n + 1, s2n + 1, M0n + 1, wn, gn);
        else if (!gauss_check(kern, q, nn + 1, tn, s1n, s2n, &fail)) dfl_parfor(nn, smooth_body, &on);
        if (!fail.bound) { val = 0; for (int j = 0; j < nn; j++) val += gn[j]; val /= M_PI; }
        free(edges0); free(te0); free(P0); free(s1e); free(s2e); free(M0e); free(psie); free(tn); free(wn); free(Pn); free(s1n); free(s2n); free(M0n); free(gn);
    } else {
        if (bt) batch_smooth(bt, kind, nu, ts, P, s1, s2, M0, wt, g);
        else if (!gauss_check(kern, q, nu, ts, s1, s2, &fail)) dfl_parfor(nu, smooth_body, &oc);
        if (!fail.bound) { g[0] *= 0.5; val = 0; for (int j = 0; j < nu; j++) val += g[j]; val *= hu / M_PI; }
    }
    free(ts); free(P); free(s1); free(s2); free(M0); free(wt); free(g); free(psi);
    if (fail.bound) { *bd = fail; return fail.bound; }
    /* the edge part (q < 0): Filon's rule on geometric panels */
    double y = 1 / (4 * q * q);
    if (q > 0 || 3 * tau * y / 16 > 40) { *out = val; return 0; }
    double w0 = fmin(fmin(0.5, 1 / y), tau / 2);
    int ne0 = (int) ceil(log2(Wm / w0)) + 2;
    double *edges0 = malloc(ne0 * sizeof(double)); edges0[0] = 0; for (int p = 1; p < ne0; p++) edges0[p] = w0 * pow(2, p - 1); edges0[ne0 - 1] = Wm;
    int cap_e = 1; for (int p = 0; p < ne0 - 1; p++) { double a = edges0[p], b = edges0[p + 1]; int m = dfl_filonm(a, b, y, c); cap_e += m * o->pfac; }
    double *edges = malloc((cap_e + 1) * sizeof(double)); int np = 0; edges[0] = 0;
    for (int p = 0; p < ne0 - 1; p++) {
        double a = edges0[p], b = edges0[p + 1]; int m = dfl_filonm(a, b, y, c);
        for (int k = 1; k <= m; k++) { double e = a + (b - a) * k / m;
            for (int r = 1; r <= o->pfac; r++) { edges[np + 1] = edges[np] + (e - edges[np]) * r / o->pfac; if (r == o->pfac) edges[np + 1] = e; np++; } }
    }
    /* (with pfac > 1 the sub-edges above are built from the running edge: equal subdivision of each panel) */
    if (o->pfac > 1) { double *e2 = malloc((np + 1) * sizeof(double)); int q2 = 0; e2[0] = 0;
        /* rebuild cleanly: subdivide each original panel into pfac */
        double *orig = malloc((cap_e + 1) * sizeof(double)); int no = 0; orig[0] = 0;
        for (int p = 0; p < ne0 - 1; p++) { double a = edges0[p], b = edges0[p + 1]; int m = dfl_filonm(a, b, y, c);
            for (int k = 1; k <= m; k++) orig[++no] = a + (b - a) * k / m; }
        for (int p = 0; p < no; p++) for (int r = 1; r <= o->pfac; r++) e2[++q2] = orig[p] + (orig[p + 1] - orig[p]) * r / o->pfac;
        memcpy(edges, e2, (q2 + 1) * sizeof(double)); np = q2; free(e2); free(orig); }
    const double *gw; const double *u = dfl_rule(N, &gw);
    int nw = np * N;
    cdouble *te = malloc((nw + 1) * sizeof(cdouble));
    te[0] = CPX(tau, 0);
    for (int p = 0; p < np; p++) for (int i = 0; i < N; i++) te[1 + p * N + i] = CPX(tau, edges[p] + (edges[p + 1] - edges[p]) * u[i]);
    cdouble *Pe = malloc((nw + 1) * sizeof(cdouble)), *s1e = malloc((nw + 1) * sizeof(cdouble)), *s2e = malloc((nw + 1) * sizeof(cdouble)), *M0e = malloc((nw + 1) * sizeof(cdouble));
    dfl_tparts(model, c, nw + 1, te, stat, Pe, s1e, s2e, M0e);
    cdouble *G = malloc(nw * sizeof(cdouble));
    octx oe = {kern, kind, nw, q, te, s1e, s2e, M0e, Pe, NULL, NULL, G, y, cde, NULL, INFINITY, &fail};
    /* the K_0 kernel for c < -20 (and for a provider where |s_1(tau)| is large: dfl_k0first): the first piece of the edge
     * segment is at most 1/b long, b = (s_2 - s_1)/2 at tau, the scale on which K_0(b X) turns from its logarithm to its
     * exponential decay (with a zero start b ~ c^2/2 in case U; the 48 nodes of a first piece of up to 1/16 lose 5e-10 in F at
     * c = -480); unchanged on [-20, inf) (k0first, edgehead in DFExact.jl) */
    if (kern == 2) oe.e1 = e1;
    if (bt) {
        cdouble *ev = malloc((nw + 1) * sizeof(cdouble));
        dfl_batch_values(bt, 2, nw, te + 1, s1e + 1, s2e + 1, M0e + 1, ev);
        for (int j = 0; j < nw; j++) { cdouble t = te[j + 1], val = cmul(Pe[j + 1], ev[j]); if (kind == 2) val = cdiv(val, t); G[j] = cmul(val, cexp(cscale(y, t))); }
        free(ev);
    } else dfl_parfor(nw, edge_body, &oe);
    if (fail.bound) { free(edges0); free(edges); free(te); free(Pe); free(s1e); free(s2e); free(M0e); free(G); *bd = fail; return fail.bound; }
    cdouble acc = ZERO;
    for (int p = 0; p < np; p++) acc = cadd(acc, filon_panel(G + p * N, edges[p], edges[p + 1], y, N));
    double ev = creal(cmul(cexp(CPX(-tau * y, 0)), acc)) / M_PI;
    free(edges0); free(edges); free(te); free(Pe); free(s1e); free(s2e); free(M0e); free(G);
    *out = val + ev;
    return 0;
}

/* ---- the uniform rule in w for the whole integrand (checking) ---- */
typedef struct { int kern, kind, n; double q; const cdouble *t, *s1, *s2, *M0, *P; double *g; dfl_bound_t *fail; } uctx;
static void uni_body(int j, void *v) { uctx *o = v; cdouble Is, Ie;
    if (dfl_bound_failed(o->fail)) return;
    dfl_bound_clear();
    if (o->kind == 1) inner_density(o->kern, o->t[j], o->q, o->s1[j], o->s2[j], 0, INFINITY, &Is, &Ie);
    else inner_cdf(o->kern, o->t[j], o->q, o->s1[j], o->s2[j], o->M0[j], 0, INFINITY, &Is, &Ie);
    if (dfl_bound_note(o->fail)) return;
    cdouble val = cmul(o->P[j], cadd(Is, Ie)); if (o->kind == 2) val = cdiv(val, o->t[j]); o->g[j] = creal(val); }
static int uniform_c0(int kind, int model, double c, double q, double tau, bool stat, double hw, double *out, dfl_bound_t *bd)
{
    int kern = dfl_kern(model, stat); int n = (int) floor(WMAX / hw) + 1;
    cdouble *t = malloc(n * sizeof(cdouble)), *P = malloc(n * sizeof(cdouble)), *s1 = malloc(n * sizeof(cdouble)), *s2 = malloc(n * sizeof(cdouble)), *M0 = malloc(n * sizeof(cdouble));
    double *g = malloc(n * sizeof(double));
    for (int j = 0; j < n; j++) t[j] = CPX(tau, j * hw);
    dfl_tparts(model, c, n, t, stat, P, s1, s2, M0);
    dfl_bound_t fail = {0, 0, NULL};
    uctx u = {kern, kind, n, q, t, s1, s2, M0, P, g, &fail};
    dfl_parfor(n, uni_body, &u);
    g[0] *= 0.5; double v = 0; for (int j = 0; j < n; j++) v += g[j]; v *= hw / M_PI;
    free(t); free(P); free(s1); free(s2); free(M0); free(g);
    *bd = fail; *out = fail.bound ? NAN : v;
    return fail.bound;
}

void dfl_single_defaults(dfl_single_opts *o)
{
    o->tau = 0; o->hu = 0.075; o->N = 20; o->nodefac = 1; o->pfac = 1; o->edge_hw = 0; o->uniform_hw = 0; o->batched = true; o->gen = false;
}
double dfl_default_tau(int model, double c, bool stat)
{
    if (model == DFL_MODEL_PROV) return dfl_prov_tau();
    if (c <= 0 && !stat) return 1.0;
    return fmin(1.0, dfl_tstar(model, c, stat) / 2);
}
/* one value through the adaptive single integral at c0 = 0 (dfl_gen.c adds the general kernel): 0, or the bound of the
 * engine that the line integral would exceed (*bd), with *out NaN */
int dfl_single_value_c0_b(int kind, double x, int model, double c, bool stat, const dfl_single_opts *o, double *out, dfl_bound_t *bd)
{
    double q = x;   /* x = 0 is taken exactly by inner_density and inner_cdf */
    double tau = o->tau > 0 ? o->tau : dfl_default_tau(model, c, stat);
    DFL_NODEFAC = o->nodefac > 0 ? o->nodefac : 1;
    dfl_bound_clear();
    if (o->uniform_hw > 0) return uniform_c0(kind, model, c, q, tau, stat, o->uniform_hw, out, bd);
    /* the fixed layouts (dfl_batch.c) where they serve: the chi-square kernel at tau = 1, the K_0 kernel's density at any tau
     * and its distribution function at tau = 1 (density, cdf of DFExact.jl) */
    int kern = dfl_kern(model, stat);
    double e1 = INFINITY, r1 = INFINITY;
    if (kern == 2) dfl_k0first(model, c, stat, tau, &e1, &r1);
    const dfl_batch *bt = NULL;
    if (dfl_usebatch(o->batched, c, q, tau)) {
        if (kern == 1) { if (tau == 1.0) bt = dfl_batch_get(1, kind, q, tau, e1, r1); }
        else if (DFL_BATCHK0 && (kind == 1 || tau == 1.0)) bt = dfl_batch_get(2, kind, q, tau, e1, r1);
    }
    int rc = outer_c0(kind, model, c, q, tau, stat, o, bt, e1, out, bd);
    dfl_batch_release(bt);
    return rc;
}
/* the caps of the first pieces of the K_0 kernel's edge segment and ray X < 0 (k0first of DFExact.jl; for a provider, also
 * where |s_1(tau)| exceeds DFL_K0FIRST_S1: the root scale of case DU at c = -20, which higher-order trends reach nearer the
 * unit root).  The scales of the kernel near X = 0 shrink as c decreases: the ray carries e^{-s_1 X}, and K_0(b X) turns
 * from its logarithm to its exponential decay at X ~ 1/b; for c < -20 the first panels are at most 1/b and 1e-2/|s_1| long */
void dfl_k0first(int model, double c, bool stat, double tau, double *e1, double *r1)
{
    *e1 = *r1 = INFINITY;
    if (c >= -20 && model != DFL_MODEL_PROV) return;
    cdouble t0 = CPX(tau, 0), P0, a1, a2, m0;
    dfl_tparts(model, c, 1, &t0, stat, &P0, &a1, &a2, &m0);
    if (c >= -20 && fabs(creal(a1)) <= DFL_K0FIRST_S1) return;
    *e1 = 1 / fabs(creal(csub(a2, a1)) / 2); *r1 = 1e-2 / fabs(creal(a1));
}
double dfl_single_value_c0(int kind, double x, int model, double c, bool stat, const dfl_single_opts *o)
{
    double v; dfl_bound_t bd;
    return dfl_single_value_c0_b(kind, x, model, c, stat, o, &v, &bd) ? NAN : v;
}

/* ---- exported for dfl_gen.c and dfl_double.c ---- */
cdouble dfl_phi_pos(int kern, cdouble X, cdouble s1, cdouble s2) { return phi_pos(kern, X, s1, s2); }
cdouble dfl_phi_neg(int kern, cdouble X, cdouble s1, cdouble s2) { return phi_neg(kern, X, s1, s2); }
cdouble dfl_gaussexp(cfn_t A, const void *Actx, cdouble t, double q, cdouble sd, double lA, int force, int p) { return gaussexp(A, Actx, t, q, sd, lA, force, p); }
bool dfl_ray_ok(cdouble sd, cdouble e, double q, double at) { return ray_ok(sd, e, q, at); }
cdouble dfl_psi_half(int kern, cdouble s1, cdouble s2, cdouble M0) { return psi_half(kern, s1, s2, M0); }
cdouble dfl_filon_panel(const cdouble *G, double a, double b, double y, int N) { return filon_panel(G, a, b, y, N); }
cdouble dfl_rayint(fn_t f, const void *ctx, double R, double l, int n1, int n2, int p, double osc) { return rayint(f, ctx, R, l, n1, n2, p, osc); }
