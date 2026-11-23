/* dfl_kan_jet.inc: Taylor-mode arithmetic, after the jets of Bao and Kan: a truncated power series sum_k c[k] x^k,
 * k = 0..n-1 (n at most JCAP), with complex coefficients, the value of a quantity and its first n - 1 derivatives at a
 * point, scaled as Taylor coefficients (Jet of src/DFKan.jl).  Sums, products, quotients, sqrt, exp and the logarithms
 * (logp: the principal argument of the constant coefficient; logc: a prescribed one) are exact in the retained
 * coefficients; only the constant coefficient of a logarithm or a root carries a branch.  Included by dfl_kan.c once per
 * capacity, with JT the type, JCAP the capacity and JF(name) the names of the functions.  The operations follow those of
 * the Julia package term by term (a number times a jet is the complex product of every coefficient with it, a number
 * added changes the constant coefficient only), so that the values agree to rounding. */
typedef struct { int n; cdouble c[JCAP]; } JT;
static inline JT JF(const)(int n, cdouble v) { JT r; r.n = n; r.c[0] = v; for (int k = 1; k < n; k++) r.c[k] = CPX(0, 0); return r; }
/* the jet of the variable itself at x0, scaled by lam: x0 + lam x */
static inline JT JF(var)(int n, cdouble x0, cdouble lam) { JT r = JF(const)(n, x0); if (n > 1) r.c[1] = lam; return r; }
static inline cdouble JF(val)(JT a) { return a.c[0]; }
static inline JT JF(setval)(JT a, cdouble v) { a.c[0] = v; return a; }
static inline JT JF(add)(JT a, JT b) { for (int k = 0; k < a.n; k++) a.c[k] = cadd(a.c[k], b.c[k]); return a; }
static inline JT JF(sub)(JT a, JT b) { for (int k = 0; k < a.n; k++) a.c[k] = csub(a.c[k], b.c[k]); return a; }
static inline JT JF(neg)(JT a) { for (int k = 0; k < a.n; k++) a.c[k] = cneg(a.c[k]); return a; }
/* a + z, a - z, z - a (z complex), a + r, a - r, r - a (r real: the real part of the constant coefficient alone, as a real
 * number added to a complex one) */
static inline JT JF(addc)(JT a, cdouble z) { a.c[0] = cadd(a.c[0], z); return a; }
static inline JT JF(subc)(JT a, cdouble z) { a.c[0] = csub(a.c[0], z); return a; }
static inline JT JF(csub)(cdouble z, JT a) { cdouble v = csub(z, a.c[0]); a = JF(neg)(a); a.c[0] = v; return a; }
static inline JT JF(addr)(JT a, double r) { a.c[0] = CPX(creal(a.c[0]) + r, cimag(a.c[0])); return a; }
static inline JT JF(subr)(JT a, double r) { a.c[0] = CPX(creal(a.c[0]) - r, cimag(a.c[0])); return a; }
static inline JT JF(rsub)(double r, JT a) { cdouble v = CPX(r - creal(a.c[0]), -cimag(a.c[0])); a = JF(neg)(a); a.c[0] = v; return a; }
/* a z, a r (the complex product with r + 0i), a / z, a / r */
static inline JT JF(mulc)(JT a, cdouble z) { for (int k = 0; k < a.n; k++) a.c[k] = cmul(a.c[k], z); return a; }
static inline JT JF(mulr)(JT a, double r) { cdouble z = CPX(r, 0); for (int k = 0; k < a.n; k++) a.c[k] = cmul(a.c[k], z); return a; }
static inline JT JF(divc)(JT a, cdouble z) { for (int k = 0; k < a.n; k++) a.c[k] = cdiv(a.c[k], z); return a; }
static inline JT JF(divr)(JT a, double r) { cdouble z = CPX(r, 0); for (int k = 0; k < a.n; k++) a.c[k] = cdiv(a.c[k], z); return a; }
static inline JT JF(mul)(JT a, JT b)
{
    JT m; m.n = a.n;
    for (int k = 0; k < a.n; k++) {
        cdouble acc = CPX(0, 0);
        for (int j = 0; j <= k; j++) acc = cadd(acc, cmul(a.c[j], b.c[k - j]));
        m.c[k] = acc;
    }
    return m;
}
static inline JT JF(div)(JT a, JT b)
{
    JT m; m.n = a.n;
    cdouble ib = cdiv(CPX(1, 0), b.c[0]);
    for (int k = 0; k < a.n; k++) {
        cdouble acc = a.c[k];
        for (int j = 1; j <= k; j++) acc = csub(acc, cmul(b.c[j], m.c[k - j]));
        m.c[k] = cmul(acc, ib);
    }
    return m;
}
static inline JT JF(cdiv)(cdouble z, JT b) { return JF(div)(JF(const)(b.n, z), b); }
static inline JT JF(sqrt)(JT a)
{
    JT m; m.n = a.n;
    m.c[0] = dfl_csqrt(a.c[0]);
    cdouble h = cdiv(CPX(1, 0), cscale(2, m.c[0]));
    for (int k = 1; k < a.n; k++) {
        cdouble acc = a.c[k];
        for (int j = 1; j <= k - 1; j++) acc = csub(acc, cmul(m.c[j], m.c[k - j]));
        m.c[k] = cmul(acc, h);
    }
    return m;
}
static inline JT JF(exp)(JT a)
{
    JT m; m.n = a.n;
    m.c[0] = cexp(a.c[0]);
    for (int k = 1; k < a.n; k++) {
        cdouble acc = CPX(0, 0);
        for (int j = 1; j <= k; j++) acc = cadd(acc, cmul(cscale((double) j, a.c[j]), m.c[k - j]));
        m.c[k] = CPX(creal(acc) / k, cimag(acc) / k);
    }
    return m;
}
/* the logarithm with the constant coefficient v (the branch is the caller's) */
static inline JT JF(logv)(JT a, cdouble v)
{
    JT m; m.n = a.n;
    m.c[0] = v;
    cdouble ia = cdiv(CPX(1, 0), a.c[0]);
    for (int k = 1; k < a.n; k++) {
        cdouble acc = cscale((double) k, a.c[k]);
        for (int i = 1; i <= k - 1; i++) acc = csub(acc, cmul(cscale((double) i, m.c[i]), a.c[k - i]));
        cdouble q = cmul(acc, ia);
        m.c[k] = CPX(creal(q) / k, cimag(q) / k);
    }
    return m;
}
static inline JT JF(logp)(JT a) { return JF(logv)(a, clog(a.c[0])); }
static inline JT JF(logc)(JT a, double ph) { return JF(logv)(a, CPX(log(cabs(a.c[0])), ph)); }
/* sum_k cf[k] z^k, k = 0..ncf-1, by Horner's rule */
static inline JT JF(ser)(const double *cf, int ncf, JT z)
{
    JT y = JF(const)(z.n, CPX(0, 0));
    for (int k = ncf - 1; k >= 0; k--) y = JF(addr)(JF(mul)(y, z), cf[k]);
    return y;
}
