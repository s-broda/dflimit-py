/* dfl_kan_logcf.inc: the logarithm of the joint characteristic function of the limit pair of the classical cases, the
 * closed forms of Bao and Kan (their AR1LIM_LOGCF; _logcf and trendlim of src/DFKan.jl), written once for numbers and jets:
 * included by dfl_kan.c with KT the type (cdouble or a jet type), KF(name) its operations and KNAME(name) the names of the
 * functions defined here.  For a number the operations are those of complex arithmetic, so that the scalar and the jet
 * forms are the same text.
 *
 * With omega = c + i t1, theta^2 = 2 i t2 - 2 i c t1 - c^2 and tau_0 = t1^2 - 2 i t2, every function of theta is even, a
 * function of theta^2: for small |theta^2| by power series, elsewhere through vartheta = (-theta^2)^{1/2} and e^{-vartheta}
 * in forms in which nothing of order one is a difference of quantities of order e^{vartheta}, on either side of the unit
 * root.  The branch of the square root is fixed at the point: each argument under the root is a product of factors whose
 * continuous arguments are known (cos theta = e^{vartheta} (1 + e^{-2 vartheta})/2; the other factors are off the closed
 * negative real axis). */

/* vartheta - c without cancellation: for c >= 0 as -(theta^2 + c^2)/(vartheta + c), otherwise directly */
static KT KNAME(vtmc)(KT vt, KT zc, double c) { return c >= 0 ? KF(div)(KF(neg)(zc), KF(plusr)(vt, c)) : KF(subr)(vt, c); }
/* vartheta - omega without cancellation: on the explosive side as tau_0/(vartheta + omega) where that is the larger */
static KT KNAME(vtmo)(KT vt, KT zc, double c, KT om, KT t1, KT tau0)
{
    KT nu = KF(sub)(KNAME(vtmc)(vt, zc, c), KF(mulc)(t1, CPX(0, 1)));
    if (c > 0) {
        KT vpo = KF(add)(vt, om);
        if (cabs(KF(val)(vpo)) > cabs(KF(val)(nu))) nu = KF(div)(tau0, vpo);
    }
    return nu;
}
/* B0(vartheta) of the trend cases expanded about vartheta = c (sg = 1) or -c (sg = -1), u = vartheta -/+ c */
static KT KNAME(bpoly)(KT u, double c, int sg)
{
    double c2 = c * c, c3 = c2 * c, c4 = c3 * c, c5 = c4 * c, c6 = c5 * c, c7 = c6 * c, b0, b1, b2;
    double k4 = c4 - 8 * c3 + 32 * c2 - 72 * c + 72;
    if (sg > 0) {
        double q = c2 - 6 * c + 12;
        b0 = c * ((c - 2) * (c - 2)) * (q * q);
        b1 = 3 * c6 - 40 * c5 + 232 * c4 - 744 * c3 + 1344 * c2 - 1152 * c + 288;
        b2 = 3 * c5 - 32 * c4 + 152 * c3 - 408 * c2 + 576 * c - 288;
    } else {
        b0 = -c7;
        b1 = 3 * c6 - 8 * c5 + 8 * c4 + 24 * c3 - 96 * c2 + 288;
        b2 = -3 * c5 + 16 * c4 - 40 * c3 + 24 * c2 + 144 * c - 288;
    }
    return KF(plusr)(KF(mul)(KF(plusr)(KF(mul)(KF(plusr)(KF(mulr)(u, k4), b2), u), b1), u), b0);
}

static KT KNAME(lcf)(const dfl_limcf *F, KT t1, KT t2);
/* the cases with an intercept and a trend (the trend limit of Bao and Kan): the phase is that of the case without
 * deterministic terms corrected by the principal arguments of three factors that are off the closed negative real axis */
static KT KNAME(trendlim)(const dfl_limcf *F, KT t1, KT t2, KT om, KT zc, KT tau0)
{
    double c = F->c, c2 = c * c;
    KT Kb = KF(add)(KF(mulr)(tau0, -4 * (c2 - 3 * c + 3)), KF(mulr)(KF(mulc)(t1, CPX(0, 1)), c * (c - 2) * (c2 - 6 * c + 12)));
    KT zv = KF(rsub)(c2, zc);                               /* vartheta^2 */
    KT Lr, Q;
    cdouble g11, gll;
    if (cabs(KF(val)(KF(subr)(zc, c2))) < 16) {
        cdouble z0 = KF(val)(zv);
        KT f1 = KF(ser)(DFL_KAN_SER_F1, 42, zv), f2 = KF(ser)(DFL_KAN_SER_F2, 42, zv), Ge = KF(ser)(F->ag, 41, zv);
        KT f12 = KF(mul)(f1, f2);
        KT Psi = KF(add)(KF(mul)(f12, Kb), Ge);             /* e^{vartheta} D3/vartheta^8 */
        Lr = KF(plusr)(KF(sub)(KF(divr)(KF(neg)(om), 2), KF(divr)(KF(logp)(Psi), 2)), log(2.0) / 2);
        Q = KF(div)(KF(mul)(tau0, f12), KF(mulr)(Psi, 2));
        g11 = cdiv(KF(val)(f1), cscale(2, s_ser(DFL_KAN_SER_CH, 41, z0)));
        gll = cdiv(cneg(KF(val)(f2)), cscale(24, s_ser(DFL_KAN_SER_SHC, 41, z0)));
    } else {
        KT vt = KF(sqrt)(zv), E = KF(exp)(KF(neg)(vt)), E2 = KF(mul)(E, E);
        KT vmc, vpc, Bp, Bm;
        if (c >= 0) {
            vmc = KF(div)(KF(neg)(zc), KF(plusr)(vt, c)); vpc = KF(plusr)(vt, c);
            Bp = KNAME(bpoly)(vmc, c, 1); Bm = KNAME(bpoly)(KF(neg)(vmc), c, -1);
        } else {
            vpc = KF(div)(KF(neg)(zc), KF(subr)(vt, c)); vmc = KF(subr)(vt, c);
            Bp = KNAME(bpoly)(vpc, c, -1); Bm = KNAME(bpoly)(KF(neg)(vpc), c, 1);
        }
        KT B1 = KF(subr)(KF(mulr)(zv, (c - 3) * (c2 - 4 * c + 6)), 6 * (c2 * c - 5 * c2 + 12 * c - 12));
        KT h1 = KF(add)(KF(subr)(vt, 2), KF(mul)(E, KF(plusr)(vt, 2)));
        KT h2 = KF(add)(KF(neg)(KF(plusr)(KF(sub)(zv, KF(mulr)(vt, 6)), 12)), KF(mul)(E, KF(plusr)(KF(add)(zv, KF(mulr)(vt, 6)), 12)));
        KT h12 = KF(mul)(h1, h2), D3;
        if (c > 0) {                                       /* the explosive side: every term of the order of D3 itself */
            double K1 = c * (c - 2) * (c2 - 6 * c + 12);
            KT Pp = KF(plusr)(KF(mul)(vt, KF(plusr)(KF(add)(zv, KF(mulr)(vt, 8)), 24)), 24);
            KT Qc = KF(subr)(KF(mulr)(vt, c - 3), 3 * (c - 2));
            KT vQ = KF(mul)(vmc, Qc);
            KT t_a = KF(mulr)(KF(mul)(vQ, vQ), 8);
            KT t_b = KF(mulr)(KF(mul)(vmc, KF(add)(KF(mulr)(KF(mul)(KF(subr)(zv, 6), E), 8), KF(mul)(Pp, E2))), K1);
            KT t_c = KF(mul)(KF(mul)(tau0, h12), KF(plusr)(KF(cdiv)(CPX(K1, 0), KF(add)(vt, om)), 4 * (c2 - 3 * c + 3)));
            KT t_d = KF(mulr)(KF(mul)(zc, KF(mul)(B1, E)), 8);
            KT t_e = KF(mul)(vpc, KF(mul)(Bm, E2));
            D3 = KF(sub)(KF(add)(KF(sub)(KF(add)(t_a, t_b), t_c), t_d), t_e);
        } else {
            D3 = KF(add)(KF(mul)(h12, Kb), KF(sub)(KF(add)(KF(mul)(vmc, Bp), KF(mulr)(KF(mul)(zc, KF(mul)(B1, E)), 8)), KF(mul)(vpc, KF(mul)(Bm, E2))));
        }
        Lr = KF(plusr)(KF(sub)(KF(divr)(KF(neg)(KF(add)(om, vt)), 2), KF(divr)(KF(sub)(KF(logp)(D3), KF(mulr)(KF(logp)(vt), 8)), 2)), log(2.0) / 2);
        Q = KF(div)(KF(mul)(tau0, h12), KF(mulr)(D3, 2));
        cdouble v0 = KF(val)(vt), e0 = KF(val)(E), v2 = cmul(v0, v0);
        g11 = cdiv(KF(val)(h1), cmul(cmul(v2, v0), CPX(1 + creal(e0), cimag(e0))));
        gll = cdiv(cneg(KF(val)(h2)), cmul(cscale(12, cmul(v2, v2)), CPX(1 - creal(e0), -cimag(e0))));
    }
    /* the continuous phase from the factors */
    cdouble t10 = KF(val)(t1), t20 = KF(val)(t2);
    dfl_limcf FK = *F; FK.det = 1; FK.stat = 0; FK.par = 0; FK.w0 = 0.5;
    cdouble L0 = lcf_s(&FK, t10, t20), z1 = KF(val)(zc);
    cdouble tr = cmul(z1, cadd(g11, cscale(12, gll)));
    cdouble dt = cmul(cmul(cadd(cscale(12, cmul(z1, z1)), cscale(144, cmul(t10, t10))), g11), gll);
    cdouble sq = dfl_csqrt(csub(cmul(tr, tr), cscale(4, dt)));
    cdouble k1 = cscale(0.5, cadd(tr, sq)), k2 = cscale(0.5, csub(tr, sq));
    cdouble lr0 = KF(val)(Lr);
    cdouble rho = cdiv(cexp(cscale(2, csub(L0, lr0))), CPX(1 + creal(tr) + creal(dt), cimag(tr) + cimag(dt)));
    double ph = cimag(L0) - (carg(CPX(1 + creal(k1), cimag(k1))) + carg(CPX(1 + creal(k2), cimag(k2))) + carg(rho)) / 2;
    KT L = KF(setval)(Lr, CPX(creal(lr0), ph));
    if (F->stat) return KF(sub)(L, KF(divr)(KF(logp)(KF(plusr)(KF(mulr)(Q, c2 * c), 1)), 2));
    return F->par == 0 ? L : KF(add)(L, KF(mulr)(Q, F->par * F->par));
}

/* log E exp(i t1 A + i t2 B), A = U - cV, B = V (_logcf of src/DFKan.jl) */
static KT KNAME(lcf)(const dfl_limcf *F, KT t1, KT t2)
{
    double c = F->c, c2 = c * c;
    int n_ = KORD(t1);
    (void) n_;
    KT om = KF(plusr)(KF(mulc)(t1, CPX(0, 1)), c);                                /* omega = c + i t1 */
    KT zc = KF(sub)(KF(mulc)(t2, CPX(0, 2)), KF(mulc)(t1, CPX(0.0 * c, 2 * c)));  /* theta^2 + c^2 */
    KT zeta = KF(subr)(zc, c2);                                                  /* theta^2 */
    KT tau0 = KF(sub)(KF(mul)(t1, t1), KF(mulc)(t2, CPX(0, 2)));
    if (F->det == 3) return KNAME(trendlim)(F, t1, t2, om, zc, tau0);
    cdouble z0 = KF(val)(zeta);
    int below4 = cabs(z0) < 4;
    if (F->det == 1) {
        KT lC, lg, Kf;
        if (below4) {
            KT C = KF(ser)(DFL_KAN_SER_COS, 27, zeta), S = KF(ser)(DFL_KAN_SER_SIN, 27, zeta);
            KT f = KF(sub)(C, KF(mul)(om, S));
            KT g = KF(rsub)(1, KF(div)(KF(mul)(om, S), C));
            lC = KF(logc)(C, dfl_kan_argcos(z0)); lg = KF(logp)(g);
            Kf = KF(div)(KF(neg)(KF(mul)(tau0, S)), KF(mulr)(f, 2));
        } else {
            KT vt = KF(sqrt)(KF(neg)(zeta)), E2 = KF(exp)(KF(mulr)(vt, -2));
            KT nu = KNAME(vtmo)(vt, zc, c, om, t1, tau0);
            KT Bk = KF(add)(nu, KF(mul)(E2, KF(add)(vt, om)));
            KT oe = KF(plusr)(E2, 1);
            lC = KF(subr)(KF(add)(vt, KF(logp)(oe)), log(2.0));   /* log cos theta: Im vartheta + Arg(1 + e^{-2 vartheta}) */
            lg = KF(logp)(KF(div)(Bk, KF(mul)(vt, oe)));
            Kf = KF(div)(KF(neg)(KF(mul)(tau0, KF(rsub)(1, E2))), KF(mulr)(Bk, 2));
        }
        KT L = KF(sub)(KF(divr)(KF(neg)(om), 2), KF(divr)(KF(add)(lC, lg), 2));
        if (F->stat) return KF(sub)(L, KF(divr)(KF(logp)(KF(plusr)(KF(divr)(Kf, c), 1)), 2));
        return F->par == 0 ? L : KF(add)(L, KF(mulr)(Kf, F->par * F->par));
    }
    /* an intercept */
    KT tau1 = KF(mulr)(KF(sub)(KF(add)(KF(mul)(t1, t1), KF(mulc)(t1, CPX(0, c2))), KF(mulc)(t2, CPX(0.0 * c, 2 * c))), 2);
    KT tau2 = KF(plusr)(KF(sub)(KF(mulc)(t1, cscale(c, CPX(0, c - 2))), tau0), c2 * c);
    KT T0, lS, Q;
    cdouble D2;
    if (below4) {
        KT S = KF(ser)(DFL_KAN_SER_SIN, 27, zeta), Sh = KF(ser)(DFL_KAN_SER_SH, 27, zeta), ShmCh = KF(ser)(DFL_KAN_SER_CD, 27, zeta);
        KT tau10 = KF(add)(tau1, KF(mulr)(zeta, 2 * c)), tau20 = KF(sub)(tau2, zeta);
        KT Nz = KF(const)(n_, CPX(0, 0));
        for (int k = DFL_KAN_KMS; k >= 1; k--) {          /* N/zeta, N = 2 tau1 Sh^2 - 4 c^2 cos th + 4 tau2 sin th/th */
            double cst = -4 * c * DFL_KAN_SER_A[k - 1] - 4 * c2 * DFL_KAN_SER_COS[k] + 4 * DFL_KAN_SER_SIN[k - 1];
            KT term = KF(plusr)(KF(add)(KF(mulr)(tau10, 2 * DFL_KAN_SER_A[k]), KF(mulr)(tau20, 4 * DFL_KAN_SER_SIN[k])), cst);
            Nz = KF(add)(KF(mul)(Nz, zeta), term);
        }
        T0 = KF(div)(Nz, KF(mul)(Sh, Sh));
        lS = KF(logp)(Sh);
        Q = KF(div)(KF(mulr)(KF(mul)(KF(mul)(tau0, ShmCh), Sh), 2), Nz);
        cdouble sh0 = KF(val)(Sh);
        D2 = cdiv(cmul(sh0, sh0), cscale(2, KF(val)(S)));
    } else {
        KT vt = KF(sqrt)(KF(neg)(zeta)), E = KF(exp)(KF(neg)(vt)), E2 = KF(mul)(E, E);
        KT nu = KNAME(vtmo)(vt, zc, c, om, t1, tau0);
        KT q1 = KF(mulr)(KF(mul)(t1, t1), (2 - c) * (1 + c));
        KT q2 = KF(mulc)(t2, cscale(c - 1, CPX(0.0 * c, 2 * c)));
        KT Q0 = KF(sub)(KF(add)(KF(add)(q1, q2), KF(mul)(nu, tau2)), KF(mulc)(KF(mul)(t1, tau0), CPX(0, 1)));
        KT vt2 = KF(mul)(vt, vt);
        KT Den = KF(add)(KF(sub)(Q0, KF(mulr)(KF(mul)(E, tau1), 2)), KF(mul)(E2, KF(sub)(KF(sub)(tau1, KF(mulr)(vt2, c2)), KF(mul)(tau2, vt))));
        KT omE = KF(rsub)(1, E);
        T0 = KF(div)(KF(mulr)(Den, -2), KF(mul)(vt2, KF(mul)(omE, omE)));
        lS = KF(sub)(KF(add)(KF(divr)(vt, 2), KF(logp)(omE)), KF(logp)(vt));   /* sin(th/2)/(th/2) = e^{vt/2}(1 - E)/vt */
        KT br = KF(sub)(KF(rsub)(1, KF(divr)(vt, 2)), KF(mul)(E, KF(plusr)(KF(divr)(vt, 2), 1)));
        Q = KF(div)(KF(mul)(KF(mul)(tau0, br), omE), Den);
        cdouble e0 = KF(val)(E);
        D2 = cdiv(KF(val)(omE), cmul(CPX(1 + creal(e0), cimag(e0)), KF(val)(vt)));
    }
    KT lT = KF(logc)(T0, dfl_kan_argtau4(z0, KF(val)(zc), D2, KF(val)(T0)));
    KT L = KF(sub)(KF(sub)(KF(rsub)(log(2.0), KF(divr)(om, 2)), lS), KF(divr)(lT, 2));
    if (F->stat) return KF(sub)(L, KF(divr)(KF(logp)(KF(rsub)(1, KF(mulr)(Q, c))), 2));
    return F->par == 0 ? L : KF(sub)(L, KF(mulr)(Q, F->par * F->par));
}
