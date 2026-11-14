/* temporary diagnostic: one point of the line integral far from the unit root, on the Windows runner */
#include "dflimit.h"
#include <stdio.h>
int main(void)
{
    double xs[] = { -18.013226440199116, -17.5, -18.5, -16.0, -20.0 };
    double cs[] = { -480.0, -400.0, -300.0, -100.0 };
    int nw[] = { 1, 2, 4 };
    for (int w = 0; w < 3; w++) {
        dfl_set_nworkers(nw[w]);
        for (int j = 0; j < 4; j++) for (int m = 1; m <= 2; m++) {
            double F[5], f[5]; int ns = 0; char err[512] = "";
            int rc1 = dfl_dist(2, 5, xs, 1, cs[j], 0.0, 0.0, false, m, NULL, NULL, F, &ns, err, sizeof err);
            int rc2 = dfl_dist(1, 5, xs, 1, cs[j], 0.0, 0.0, false, m, NULL, NULL, f, &ns, err, sizeof err);
            printf("workers %d c %g method %d rc %d %d:", nw[w], cs[j], m, rc1, rc2);
            for (int i = 0; i < 5; i++) printf(" F=%.6e f=%.6e |", F[i], f[i]);
            printf(" %s\n", err);
        }
    }
    return 0;
}
