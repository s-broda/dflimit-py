/* dfl_kan_int.h: the internals of the devices of Bao and Kan (dfl_kan.c for the classical cases, dfl_det_kan.c for the
 * general engine): Yong Bao and Raymond Kan, "Exact Finite-Sample Distribution and Moments of the Least Squares Estimator
 * in First-Order Autoregressions: O(1) Algorithms", ported to the limits of this engine as in the module DFKan of
 * DFLimit.jl (src/DFKan.jl, src/DFKan_ratio.jl, src/DFKan_det.jl).  Not part of the public interface (dflimit.h); the test
 * program dfl_test_kan.c includes it for the checks of the layers. */
#ifndef DFL_KAN_INT_H
#define DFL_KAN_INT_H
#include "dflimit.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- adaptive Gauss-Kronrod quadrature (dfl_kan.c) ----
 * The algorithm of QuadGK.jl (2.11) that the Julia package uses: the 15-point Kronrod rule with its embedded 7-point
 * Gauss rule on each interval between consecutive points pts[0..np-1], then bisection of the segment of the largest error
 * estimate (kept in a binary heap) while the total estimate exceeds max(atol, rtol |I|) and fewer than maxevals
 * evaluations were made, so that the values are those of the Julia package to rounding.  Returns 0, or 1 where an
 * estimate is not finite. */
typedef double (*dfl_qfun)(double x, void *ctx);
int dfl_quadgk(dfl_qfun f, void *ctx, int np, const double *pts, double atol, double rtol, long maxevals, double *Iout, double *Eout, long *nev);

/* ---- the transform of a pair (A, B) for the single inversion ----
 * lcf(F, t1, t2) = log E exp(i t1 A + i t2 B) at real (t1, t2); scale(F, out): (E A, E B, sd B, Var A); tail(F, alpha, beta,
 * b, w, S, cdf, T, e): the tail int_S^inf psi(u) e^{-i w u} du of the single inversion along u (alpha, beta), psi the slowly
 * varying factor with (cdf 1) or without (cdf 0, the density) the factor 1/u, and the first omitted term of its expansion.
 * w0: the slope of the dominant linear phase along the rays. */
typedef struct dfl_kcf dfl_kcf;
struct dfl_kcf {
    double c, w0;
    cdouble (*lcf)(dfl_kcf *F, double t1, double t2);
    void (*scale)(dfl_kcf *F, double *out);
    void (*tail)(dfl_kcf *F, double alpha, double beta, double b, double w, double S, int cdf, cdouble *T, double *e);
    void *ctx;
};
/* int_0^b f by adaptive quadrature over [0, 10], [10, 100], ... (the last up to b), so that a power-law stretch of the
 * integrand is resolved on every scale; with b = Inf it stops after two consecutive negligible decades (decades of DFKan) */
int dfl_kan_decades(dfl_qfun f, void *ctx, double b, double tol, double rtol, double *Iout, double *eout, long *nev);
/* P[alpha A + beta B <= b] (alpha != 0): 1/2 - (1/pi) int_0^inf Im[phi(u alpha, u beta) e^{-i b u}]/u du, over [0, S] by
 * dfl_kan_decades and beyond S by the tail's expansion (S = 400/|w|, w = w0 alpha + b; doubled, at most five times, while
 * the first omitted term exceeds tol/10; with |w| <= 1e-7 the quadrature runs on); err the estimate of the quadrature plus
 * the first omitted term (half_cdf of DFKan) */
int dfl_kan_half_cdf(dfl_kcf *F, double alpha, double beta, double b, double tol, double w0, double *P, double *err);
/* the root of g (increasing) on a bracket grown from [lo, hi] in steps that double from grow (grow <= 0: hi - lo), by the
 * Illinois variant of regula falsi, to the width xtol (illinois of DFKan); 1 with the message where no bracket is found or
 * g fails (g returns NaN) */
int dfl_kan_illinois(double (*g)(double x, void *ctx), void *ctx, double lo, double hi, double xtol, double grow, double *x, char *err, size_t errlen);
/* the scale 1/sd of alpha A + beta B without the covariance (ray_scale of DFKan) from (E A, E B, sd B, Var A) */
double dfl_kan_ray_scale(const double *sc, double alpha, double beta);

/* ---- the closed-form transform of the classical cases (dfl_kan.c) ----
 * The limit pair of case model (1 no deterministic terms, 2 an intercept, 3 an intercept and a trend) at c, with a
 * stationary start (stat, c < 0) or the fixed start g0 (par = g0, c g0, c^2 g0 in models 1, 2, 3, the parameter of Bao and
 * Kan); w0 the slope of the dominant phase (1 + g0^2)/2 in model 1 with a fixed start, 1/2 otherwise; ag the series of
 * e^{vt} G/vt^8 of the trend cases. */
typedef struct { int det, stat; double c, par, w0; double ag[41]; } dfl_limcf;
int dfl_limcf_init(dfl_limcf *F, int model, int stat, double c, double g0, char *err, size_t errlen);
/* log E exp(i t1 A + i t2 B), A = U - cV, B = V, at complex (t1, t2) (complex arguments give the moment generating
 * function), and its jet of order n <= DFL_KAN_JMAX in the variable x of t1 = t1c[0] + t1c[1] x + ..., t2 likewise */
#define DFL_KAN_JMAX 32
cdouble dfl_kan_logcf(const dfl_limcf *F, cdouble t1, cdouble t2);
void dfl_kan_logcf_jet(const dfl_limcf *F, int n, const cdouble *t1c, const cdouble *t2c, cdouble *out);
/* log E exp(sU + tV) */
cdouble dfl_kan_logmgf(const dfl_limcf *F, cdouble s, cdouble t);
/* (E A, E B, sd B, Var A) from a jet of order three at the origin (ratio_scale of DFKan) */
void dfl_kan_scale(const dfl_limcf *F, double *out);
/* the transform interface of F (ctx = F) */
void dfl_kan_kcf(const dfl_limcf *F, dfl_kcf *K);
/* the density of A/B at r (ratio_pdf of DFKan): (1/pi) int_0^inf Im[d/dt2 phi(t1, t2) at t2 = -r t1] dt1, the derivative
 * from a jet of order one in t2, the tail by integration by parts from jets on a circle; tol relative to the scale
 * fs = max(1, E[B]/sd(A)) of the density (the absolute tolerance is tol fs) */
int dfl_kan_ratio_pdf(const dfl_limcf *F, double r, double tol, double *f, double *err, double *fs);
/* m[j] = E[(A - x0 B)^j / B^(j power)], j = 1..k (power 1: the moments of A/B about x0; 1/2: those of the t-ratio), from one
 * jet of order j per node of an adaptive quadrature in log s (ratio_mom of DFKan); k at most DFL_KAN_JMAX - 1 */
int dfl_kan_ratio_mom(const dfl_limcf *F, int k, double x0, double power, double tol, double *m, double *err, char *errmsg, size_t errlen);

/* ---- the statistics of the classical cases by the single inversion (coef_cdf1 and so on of DFKan) ----
 * Each value with its error estimate; the arguments as in the Julia package: y the point of the coefficient statistic
 * (U - c0 V)/V, b and a of the linear form U - aV, tol the absolute tolerance (coef_pdf1: relative to the density's scale,
 * *ftol the absolute tolerance it amounts to). */
int dfl_kan_coef_cdf1(const dfl_limcf *F, double y, double c0, double tol, double *P, double *e);
int dfl_kan_coef_pdf1(const dfl_limcf *F, double y, double c0, double tol, double *f, double *e, double *ftol);
int dfl_kan_coef_icdf1(const dfl_limcf *F, double p, double c0, double xtol, double tol, double *y, double *e, char *err, size_t errlen);
int dfl_kan_lin_cdf1(const dfl_limcf *F, double b, double a, double tol, double *P, double *e);
int dfl_kan_envelope1(double c, double alpha, double tol, double *P, double *e, char *err, size_t errlen);
/* the regions where the single inversion is the default (lin_ok, mom_ok, envelope_ok of DFKan; see the comments there) */
int dfl_kan_lin_ok(int model, int stat, double c, double g0, double a);
int dfl_kan_mom_ok(double c);
int dfl_kan_envelope_ok(double c);
/* P[U <= 0] in closed form, the distribution function at zero of the t-ratio (c0 = 0) and of the coefficient statistic
 * n(phi^ - 1) (ray_cdf of DFKan, AR1LIM_RAY of Bao and Kan): cases DK, DU (zero or fixed start), SK, SU */
int dfl_kan_ray_cdf(int model, int stat, double c, double g0, double *P, char *err, size_t errlen);

#endif
