"""
Phase-count-guided length sweeps for the pumpkin-chain nodal statistics.
=======================================================================

Reproduces, from scratch, the generic nodal-surplus law and local-surplus
correlation of the vertex-touching binary two-cycle chains, for several
length vectors.  Nothing here reuses the manuscript's archived spectra.

Pipeline, per graph and per length vector:

 1. Secular function.  F_R(k) = Re[ det(I - T(k) S) * exp(-i k L) * c ], with c
    the constant unimodular factor that makes the expression real.  Entire in
    k, pole-free, no Dirichlet-eigenvalue artefacts.

 2. Root collection.  Sign-change sweep, but every window is checked against
    the exact phase-count of `phase_count_audit`, and windows that disagree are
    bisected until they do.  A plain sign-change sweep misses roots: a missing
    root shifts every later surplus by +1 and only becomes visible once the
    accumulated shift leaves [0, beta], often hundreds of eigenvalues later.

 3. Nodal count.  At each root, the vertex values come from the null vector of
    the all-vertex Dirichlet-to-Neumann matrix; on edge e = (u,v) the solution
    is P cos(ks) + Q sin(ks) with P = psi_u, Q = (psi_v - psi_u cos(k l))/sin(k l),
    and the interior zeros are floor((k l + delta)/pi) - floor(delta/pi) with
    delta = atan2(P, Q).

 4. Surplus.  sigma_n = phi_n - (n-1), n counted in the FULL spectrum with
    k_1 = 0 as the constant state.  Loop-supported states (k = 2 pi m / L_loop)
    are excluded from the statistics but still consume an index.

 5. Correlation.  With beta = 2 and symmetric marginals, C = 1 - 2 P(sigma=1).

Calibration: run with --calibrate to check the pipeline on a 3-star (sigma == 0
identically) and a tadpole (sigma ~ Bernoulli(1/2)).
"""

from __future__ import annotations

import argparse
import collections
import csv

import numpy as np
from scipy.optimize import brentq

from phase_count_audit import bond_scattering_matrix, count_increment, check_anchor


class Chain:
    def __init__(self, edges, degrees, lengths):
        self.edges, self.degrees, self.lengths = edges, degrees, list(lengths)
        self.L = float(sum(lengths))
        self.nV = len(degrees)
        self.beta = len(edges) - self.nV + 1
        self.S, self.ld = bond_scattering_matrix(edges, degrees, lengths)
        check_anchor(self.S, self.ld, self.beta)
        self.I = np.eye(len(self.ld))
        z = np.linalg.det(self.I - np.exp(1j * .7331 * self.ld)[:, None] * self.S)
        self.c = np.conj(z * np.exp(-1j * .7331 * self.L) / abs(z))

    def f(self, k):
        d = np.linalg.det(self.I - np.exp(1j * k * self.ld)[:, None] * self.S)
        return float(np.real(d * np.exp(-1j * k * self.L) * self.c))

    def fv(self, kk, chunk=25000):
        out = np.empty(len(kk))
        for a in range(0, len(kk), chunk):
            q = kk[a:a + chunk]
            M = self.I[None] - np.exp(1j * q[:, None] * self.ld[None])[:, :, None] * self.S[None]
            out[a:a + chunk] = np.real(np.linalg.det(M) * np.exp(-1j * q * self.L) * self.c)
        return out

    def _sign_change_roots(self, a, b, step):
        kk = np.arange(a, b, step)
        if len(kk) < 2:
            return []
        v = self.fv(kk)
        idx = np.where(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
        return [brentq(self.f, kk[i], kk[i + 1], xtol=1e-14, rtol=8.9e-16) for i in idx]

    def _collect(self, a, b, step, depth=0):
        n, _ = count_increment(self.S, self.ld, a, b, self.L,
                               beta=self.beta, strict=False)
        R = [x for x in self._sign_change_roots(a, b, step) if a < x < b]
        if len(R) == n:
            return R
        if depth > 60 or (b - a) < 1e-12:
            raise RuntimeError(
                f"unresolved window ({a!r}, {b!r}): phase count says {n}, sign changes "
                f"give {len(R)}. Refusing to invent a root -- rerun this window with a "
                f"finer step or higher precision.")
        m = 0.5 * (a + b)
        return (self._collect(a, m, min(step, (m - a) / 50), depth + 1)
                + self._collect(m, b, min(step, (b - m) / 50), depth + 1))

    def roots(self, K, step=0.002, window=5.0):
        out = []
        a = 0.0
        while a < K:
            b = min(a + window, K)
            out += self._collect(a, b, step)
            a = b
        return np.array(sorted(out))

    def phi(self, k, margins=None):
        """Interior zero count.  If `margins` is a list, append
        (pole margin, nodal PHASE margin, nodal METRIC margin).

        The two nodal margins differ by a factor k, and only the phase one is a
        meaningful uniform threshold: what has to be resolved is the distance of
        k*s from a multiple of pi, not a distance on the graph.  A metric margin
        of 6.8e-11 at k = 3658 is a phase margin of 2.5e-7 -- comfortable.  Both
        are recorded so that either convention can be quoted."""
        sines = np.array([np.sin(k * l) for l in self.lengths])
        M = np.zeros((self.nV, self.nV))
        for (u, v), l in zip(self.edges, self.lengths):
            A, B = -k / np.tan(k * l), k / np.sin(k * l)
            M[u, u] += A; M[v, v] += A; M[u, v] += B; M[v, u] += B
        psi = np.linalg.svd(M)[2][-1]
        total, nodal_margin = 0, np.inf
        for (u, v), l in zip(self.edges, self.lengths):
            P = psi[u]
            Q = (psi[v] - psi[u] * np.cos(k * l)) / np.sin(k * l)
            d = np.arctan2(P, Q)
            m0, m1 = np.floor(d / np.pi) + 1, np.floor((k * l + d) / np.pi)
            total += int(m1 - m0 + 1)
            for m in (m0, m1):
                if m0 <= m <= m1:
                    u0 = m * np.pi - d          # nodal point in PHASE units
                    nodal_margin = min(nodal_margin, u0, k * l - u0)
        if margins is not None:
            # nodal_margin is dimensionless (phase); divide by k for the metric one
            margins.append((float(np.min(np.abs(sines))), float(nodal_margin),
                            float(nodal_margin / k)))
        return total

    def surpluses(self, K, loop_lengths=(), step=0.002,
                  pole_tol=1e-8, nodal_tol=1e-9):
        R = self.roots(K, step)
        is_loop = np.zeros(len(R), bool)
        for Lo in loop_lengths:
            m = np.round(R * Lo / (2 * np.pi))
            is_loop |= np.abs(R - 2 * np.pi * m / Lo) < 1e-8
        margins, sig = [], []
        for j, k in enumerate(R):
            sig.append(np.nan if is_loop[j] else self.phi(k, margins) - (j + 1))
        sig = np.array(sig)
        mar = np.array(margins) if margins else np.zeros((1, 3))
        pole_m = float(mar[:, 0].min())
        nodal_phase_m = float(mar[:, 1].min())     # dimensionless clearance
        nodal_metric_m = float(mar[:, 2].min())    # metric distance to a vertex
        if pole_m < pole_tol:
            raise RuntimeError(f"a root came within {pole_m:.2e} of a DtN pole "
                               "(sin(k l_e) = 0): the vertex values are unreliable")
        if nodal_phase_m < nodal_tol:
            raise RuntimeError(f"nodal phase clearance {nodal_phase_m:.2e} below "
                               f"{nodal_tol:g}: the eigenpair is not generic at "
                               "working precision")
        bad = int(((sig[~np.isnan(sig)] < 0) | (sig[~np.isnan(sig)] > self.beta)).sum())
        if bad:
            raise RuntimeError(f"{bad} surpluses outside [0, {self.beta}] -- the root "
                               "list is incomplete or the nodal count is wrong")
        return R, sig, is_loop, pole_m, nodal_phase_m, nodal_metric_m


ETA = 0.6368335201743935294225288116763

TWO_TAIL = ([(0, 1), (1, 2), (1, 2), (2, 3), (2, 3), (3, 4)], [1, 3, 4, 3, 1])
ONE_TAIL = ([(0, 1), (1, 2), (1, 2), (2, 3), (2, 3)], [1, 3, 4, 2])
FIG_EIGHT = ([(0, 1), (0, 1), (1, 2), (1, 2)], [2, 4, 2])

# Every experiment carries its own cutoff, so the table is reproducible verbatim.
# Columns of the CSV: graph,label,K,r,N_generic,N_loop,P0,P1,P2,C,C_predicted,
#                     pole_margin,nodal_phase_margin,nodal_metric_margin
EXPERIMENTS = []


def _exp(graph, label, spec, lengths, K, loops, predicted, r=None):
    EXPERIMENTS.append(dict(graph=graph, label=label, spec=spec, lengths=lengths,
                            K=K, loops=loops, predicted=predicted, r=r))


_exp("[2,2]", "e,1,r2,r3", FIG_EIGHT, [np.e, 1.0, np.sqrt(2), np.sqrt(3)],
     2000.0, (np.e + 1.0, np.sqrt(2) + np.sqrt(3)), -1.0)

# One tail: seven ratios r = L_o / (2 L_B + L_o), for the C-versus-r figure.
_ONE_TAIL_TAILS = [np.pi / 17, np.pi / 5, 1.0, np.pi, 2 * np.pi, 4 * np.pi, 7 * np.pi]
for _t in _ONE_TAIL_TAILS:
    _lengths = [_t, np.e, 1.0, np.sqrt(2), np.sqrt(3)]
    _LB, _Lo = sum(_lengths[:3]), sum(_lengths[3:])
    _r = _Lo / (2 * _LB + _Lo)
    _exp("[1,2,2]", f"tail={_t:.4f}", ONE_TAIL, _lengths, 3000.0, (_Lo,),
         -ETA * _r, r=_r)

for _lab, _lengths, _K in [
        ("ABB lengths", [np.pi, np.e, 1.0, np.sqrt(2), np.sqrt(3), np.sqrt(5)], 4000.0),
        ("both tails tiny", [np.pi / 17, np.e, 1.0, np.sqrt(2), np.sqrt(3), np.sqrt(5) / 13], 3000.0),
        ("both tails long", [7 * np.pi, np.e, 1.0, np.sqrt(2), np.sqrt(3), 5 * np.sqrt(5)], 3000.0),
        ("one tiny, one long", [np.pi / 23, np.e, 1.0, np.sqrt(2), np.sqrt(3), 9 * np.sqrt(5)], 3000.0)]:
    _exp("[1,2,2,1]", _lab, TWO_TAIL, _lengths, _K, (), 0.0)


def run_all(csv_path="length_sweeps.csv"):
    rows = []
    print(f"{'graph':<11}{'label':<20}{'K':>7}{'N':>8}{'loops':>7}"
          f"{'P0':>8}{'P1':>8}{'P2':>8}{'C':>10}{'predicted':>11}")
    for e in EXPERIMENTS:
        ch = Chain(*e["spec"], e["lengths"])
        R, sig, is_loop, pole_m, ph_m, met_m = ch.surpluses(e["K"], e["loops"])
        gen = sig[~np.isnan(sig)]
        N = len(gen)
        c = collections.Counter(gen.tolist())
        p = [c.get(float(v), 0) / N for v in (0, 1, 2)]
        C = 1.0 - 2.0 * p[1]
        print(f"{e['graph']:<11}{e['label']:<20}{e['K']:>7.0f}{N:>8d}"
              f"{int(is_loop.sum()):>7d}{p[0]:>8.4f}{p[1]:>8.4f}{p[2]:>8.4f}"
              f"{C:>+10.5f}{e['predicted']:>+11.5f}")
        rows.append(dict(graph=e["graph"], label=e["label"], K=e["K"], r=e["r"],
                         N_generic=N, N_loop=int(is_loop.sum()),
                         P0=p[0], P1=p[1], P2=p[2], C=C,
                         C_predicted=e["predicted"],
                         pole_margin=pole_m, nodal_phase_margin=ph_m,
                         nodal_metric_margin=met_m))
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")
    return rows


def asymmetry_trend(cutoffs=(1000.0, 3000.0, 9000.0, 20000.0),
                    csv_path="asymmetry_trend.csv"):
    """P(sigma=0) - P(sigma=2) for [1,2,2] as the cutoff grows.  The limiting law is
    symmetric; the residual imbalance is a finite-cutoff discrepancy, not noise."""
    lengths = [np.pi, np.e, 1.0, np.sqrt(2), np.sqrt(3)]
    Lo = lengths[3] + lengths[4]
    ch = Chain(*ONE_TAIL, lengths)
    rows = []
    print(f"{'K':>8}{'N':>9}{'P0':>10}{'P2':>10}{'P0-P2':>11}")
    for K in cutoffs:
        _, sig, *_ = ch.surpluses(K, (Lo,))
        gen = sig[~np.isnan(sig)]
        N = len(gen)
        c = collections.Counter(gen.tolist())
        p0, p2 = c.get(0.0, 0) / N, c.get(2.0, 0) / N
        print(f"{K:>8.0f}{N:>9d}{p0:>10.5f}{p2:>10.5f}{p0 - p2:>+11.5f}")
        rows.append(dict(K=K, N_generic=N, P0=p0, P2=p2, imbalance=p0 - p2))
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")
    return rows


def calibrate():
    """Pipeline sanity checks, with assertions."""
    star = Chain([(0, 3), (1, 3), (2, 3)], [1, 1, 1, 3], [np.pi, np.e, 1.0])
    _, s, *_ = star.surpluses(200.0)
    assert set(s.tolist()) == {0}, f"tree must have sigma == 0, got {set(s.tolist())}"
    print(f"  3-star (tree, beta=0): {len(s)} states, sigma == 0 throughout   OK")

    # The degree-two end vertex makes the two parallel edges a bare loop of
    # length e+1; its loop-supported states must be excluded, or they contaminate
    # the statistics (they produce sigma = -1).
    Lo = np.e + 1.0
    tad = Chain([(0, 1), (1, 2), (1, 2)], [1, 3, 2], [np.pi, np.e, 1.0])
    _, s, is_loop, *_ = tad.surpluses(300.0, loop_lengths=(Lo,))
    gen = s[~np.isnan(s)]
    c = collections.Counter(gen.tolist())
    n0, n1 = c.get(0.0, 0), c.get(1.0, 0)
    assert set(c) <= {0.0, 1.0}, f"beta=1 must give sigma in {{0,1}}, got {set(c)}"
    assert abs(n0 - n1) <= 4 * np.sqrt(len(gen)), "tadpole is not Bernoulli(1/2)"
    print(f"  tadpole (beta=1):      {int(is_loop.sum())} loop states excluded, "
          f"{len(gen)} generic, split {n0}/{n1}   OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--asymmetry", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    elif a.asymmetry:
        asymmetry_trend()
    else:
        calibrate()
        print()
        run_all()
