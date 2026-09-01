"""
Root-independent phase-count audit for a quantum-graph secular sweep.
====================================================================

Identity
--------
Let U(k) = T(k) S, T(k) = diag(exp(i k l_d)) over directed edges, be the
unitary quantum map of a standard (Neumann-Kirchhoff) metric graph of total
length L.  The eigenphases theta_j(k) are strictly increasing in k, and
det U(k) = exp(2 i k L) det S, so the unwrapped total phase increases by
exactly 2kL.  Counting crossings of theta_j through 0 mod 2*pi gives

    N_+(K) = K L / pi - sum_j frac(theta_j(K)/2pi) + sum_j frac(theta_j(0)/2pi)

for the number of STRICTLY POSITIVE eigenvalues in (0, K], with multiplicity.
The k = 0 eigenvalue (the constant state on a connected NK graph) is NOT
counted.

The identity is exact; its floating-point evaluation is not.  Two hazards,
both guarded below.

Hazard 1 -- the branch convention at k = 0.
    The formula needs theta_j(0) in [0, 2*pi), i.e. arg(1) = 0.  At k = 0 the
    matrix is S, and 1 is an eigenvalue of S with multiplicity > 1 (the
    well-known k = 0 anomaly of the secular equation).  LAPACK's complex
    driver returns some of those as 1 - i*eps with eps ~ 1e-16, whose raw
    angle is a tiny NEGATIVE number.  A bare `theta % (2*pi)` then sends it to
    ~2*pi and inflates the count by exactly 1 -- silently, and only for some
    graphs.  For the [1,2,2,1] chain this cost +1; for [1,2,2] it happened not
    to fire.  `_phases` snaps such phases back to 0.

Hazard 2 -- a cutoff sitting on an eigenvalue.
    N_+ jumps there, and the fractional parts are ambiguous.  Every public
    routine reports (and by default refuses) a cutoff whose CLEARANCE
    min_j |exp(i theta_j(K)) - 1| is below `clearance_tol`.

Use `count_increment` on a window with a > 0 when only differences matter: the
anchor is then never evaluated and Hazard 1 cannot arise at all.  A window
starting at a = 0 (as the first one in `locate_gaps` does) still evaluates the
anchor, and relies on the snap.

Hazard 1 is also checkable rather than merely guarded.  For a connected graph
with standard conditions, the algebraic multiplicity of 0 as a root of the
secular determinant is the sum of the nullities of the two first-order factors
of the Laplacian (Fulling-Kuchment-Wilson, J. Phys. A 40 (2007) 14165, arXiv:
0708.3456; for Kirchhoff conditions this is 2 - V + E, a form the authors
attribute to Kurasov).  Since U(0) = S and the eigenphase velocities are
positive, that number is exactly the multiplicity of the eigenvalue 1 in S:

    mult_S(1) = beta + 1 .

`check_anchor` asserts it, and `count` calls `check_anchor` whenever `beta` is
supplied.

Verified against the ABB Appendix D.3 lengths (pi, e, 1, sqrt2, sqrt3, sqrt5):

    [1,2,2,1]:  N_+(40) = 155,  N_+(300) = 1168,  N_+(7775.579903981382) = 30299
    [1,2,2]  :  N_+(6342.123824304902) = 20198

both matching the cleaned spectra exactly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bond_scattering_matrix", "clearance", "phase_sum",
           "anchor_multiplicity", "check_anchor",
           "count", "count_increment", "audit", "locate_gaps", "nth_eigenvalue"]

_SNAP_TOL = 1e-9          # |theta| below this at the anchor is treated as exactly 0
_CLEARANCE_TOL = 1e-8     # refuse a cutoff closer than this to an eigenvalue


def bond_scattering_matrix(edges, degrees, lengths):
    """Neumann-Kirchhoff bond-scattering matrix S and directed-edge lengths.

    Directed edge 2i runs u -> v along edges[i] = (u, v); 2i+1 runs v -> u.
    """
    nd = 2 * len(edges)
    S = np.zeros((nd, nd))
    ld = np.empty(nd)
    origin, terminus = {}, {}
    for i, (u, v) in enumerate(edges):
        origin[2 * i], terminus[2 * i] = u, v
        origin[2 * i + 1], terminus[2 * i + 1] = v, u
        ld[2 * i] = ld[2 * i + 1] = lengths[i]
    for d in range(nd):
        w = terminus[d]
        for dp in range(nd):
            if origin[dp] == w:
                S[dp, d] = 2.0 / degrees[w] - (1.0 if dp == (d ^ 1) else 0.0)
    return S, ld


def _eigvals(S, ld, k):
    if k == 0.0:
        # S is REAL: LAPACK's real driver returns the degenerate eigenvalue 1 as
        # exactly 1+0j.  Casting to complex switches to the complex driver, which
        # returns some copies as 1 - i*eps and is what produced the +1 anchor bug.
        # The snap in _phases remains as a second line of defence.
        return np.linalg.eigvals(S)
    return np.linalg.eigvals(np.exp(1j * k * ld)[:, None] * S)


def anchor_multiplicity(S, ld=None, tol=1e-8):
    """Multiplicity of the eigenvalue 1 of S.

    For a connected standard (Neumann-Kirchhoff) metric graph this equals
    beta + 1: by Fulling-Kuchment-Wilson the secular determinant vanishes to
    order 1 + beta at k = 0, U(0) = S, and the eigenphase velocities are
    positive, so that order is exactly the multiplicity of 1 in S.  Checked on
    intervals, stars, circles, polygons, loop edges, tadpoles, theta graphs,
    the figure eight, [2,2], [1,2,2], [1,2,2,1], [3,2,1] and K4.
    """
    return int(np.sum(np.abs(np.linalg.eigvals(S) - 1.0) < tol))


def check_anchor(S, ld=None, beta=None, tol=1e-8, snap_tol=_SNAP_TOL):
    """Validate the k = 0 branch convention.  One eigendecomposition.

    Always checks that every copy of the eigenvalue 1 carries phase exactly 0
    (this is what the +1 anchor bug violated).  If `beta` is supplied, also
    checks the multiplicity against beta + 1.  Returns the multiplicity.
    """
    ev = np.linalg.eigvals(S)
    near = np.abs(ev - 1.0) < tol
    m = int(np.sum(near))
    if beta is not None and m != beta + 1:
        raise ValueError(f"anchor multiplicity {m} != beta+1 = {beta + 1}; the k=0 "
                         "branch convention is not established for this graph")
    th = np.angle(ev)
    th = np.where(th > -snap_tol, th, th + 2.0 * np.pi)
    th = np.where(np.abs(th) < snap_tol, 0.0, th)
    if np.any(th[near] != 0.0):
        raise ValueError("some copies of the eigenvalue 1 did not receive phase 0")
    return m


def clearance(S, ld, k):
    """min_j |exp(i theta_j(k)) - 1|.  Small <=> k is (near) an eigenvalue."""
    return float(np.min(np.abs(_eigvals(S, ld, k) - 1.0)))


def _phases(S, ld, k, snap_tol=_SNAP_TOL):
    """Eigenphases in [0, 2*pi), with arg(1) = 0 enforced."""
    th = np.angle(_eigvals(S, ld, k))            # in (-pi, pi]
    th = np.where(th > -snap_tol, th, th + 2.0 * np.pi)
    return np.where(np.abs(th) < snap_tol, 0.0, th)


def phase_sum(S, ld, k, snap_tol=_SNAP_TOL):
    return float(np.sum(_phases(S, ld, k, snap_tol))) / (2.0 * np.pi)


def _total_length(ld, total_length):
    return 0.5 * float(np.sum(ld)) if total_length is None else float(total_length)


def _check(S, ld, k, clearance_tol, strict, label):
    c = clearance(S, ld, k)
    if c < clearance_tol:
        msg = (f"cutoff {label}={k!r} has clearance {c:.3e} < {clearance_tol:g}: "
               "it sits on (or within rounding distance of) an eigenvalue")
        if strict:
            raise ValueError(msg)
        print("  WARNING:", msg)
    return c


def count(S, ld, K, total_length=None, beta=None, snap_tol=_SNAP_TOL,
          clearance_tol=_CLEARANCE_TOL, strict=True):
    """Number of eigenvalues in (0, K], k = 0 excluded.  Returns (n, residual, clearance).

    The anchor convention is validated on every call; pass `beta` to also assert
    the multiplicity invariant.
    """
    L = _total_length(ld, total_length)
    check_anchor(S, ld, beta, snap_tol=snap_tol)
    c = _check(S, ld, K, clearance_tol, strict, "K")
    val = K * L / np.pi - phase_sum(S, ld, K, snap_tol) + phase_sum(S, ld, 0.0, snap_tol)
    n = int(round(val))
    return n, abs(val - n), c


def count_increment(S, ld, a, b, total_length=None, beta=None,
                    clearance_tol=_CLEARANCE_TOL, strict=True):
    """Number of eigenvalues in (a, b].

    For 0 < a < b the anchor is never evaluated and Hazard 1 cannot arise.  For
    a = 0 the anchor IS evaluated (via phase_sum); this routine calls
    `check_anchor` internally, and the snap fixes the phase convention on the
    degenerate cluster.
    """
    L = _total_length(ld, total_length)
    if a == 0.0:
        check_anchor(S, ld, beta)
    for k, lab in ((a, "a"), (b, "b")):
        if k > 0.0:                       # k = 0 is the anchor: eigenvalue 1 is expected there
            _check(S, ld, k, clearance_tol, strict, lab)
    val = (b - a) * L / np.pi - phase_sum(S, ld, b) + phase_sum(S, ld, a)
    n = int(round(val))
    return n, abs(val - n)


def audit(S, ld, roots, K, total_length=None, verbose=True, beta=None, **kw):
    """Compare a root list against the exact count on (0, K].

    Pass `beta` to assert the anchor invariant before counting.
    """
    roots = np.asarray(roots, dtype=float)
    n_exact, resid, clr = count(S, ld, K, total_length, beta=beta, **kw)
    n_found = int(np.sum((roots > 0.0) & (roots <= K)))
    ok = (n_found == n_exact) and resid < 1e-6
    if verbose:
        print(f"  cutoff K               : {K!r}")
        print(f"  clearance at K         : {clr:.3e}")
        print(f"  exact N_+(K)           : {n_exact}   (rounding residual {resid:.2e})")
        print(f"  roots supplied in (0,K]: {n_found}")
        print("  status                 : "
              + ("COMPLETE" if ok else f"INCOMPLETE (missing {n_exact - n_found})"))
    return ok, n_exact, n_found


def locate_gaps(S, ld, roots, K, total_length=None, window=25.0, **kw):
    """Windows of width `window` where the root list disagrees with the exact count.

    Anchored at N_+(0) = 0.  The FIRST window starts at a = 0 and therefore does
    evaluate the anchor (validated by `check_anchor`); every later window uses
    two strictly positive endpoints, where the anchor plays no role at all.
    """
    roots = np.asarray(roots, dtype=float)
    bad, a = [], 0.0
    while a < K:
        b = min(a + window, K)
        expect, _ = count_increment(S, ld, a, b, total_length, **kw)
        got = int(np.sum((roots > a) & (roots <= b)))
        if got != expect:
            bad.append((a, b, expect, got))
        a = b
    return bad


def nth_eigenvalue(S, ld, n, total_length=None, tol=1e-13, hi=None):
    """The n-th positive eigenvalue k_n, by bisection on the exact count.

    NOTE this is the n-th eigenvalue itself, not a nominal Weyl cutoff n*pi/L.
    """
    L = _total_length(ld, total_length)
    lo, hi = 0.0, ((n + 8) * np.pi / L if hi is None else hi)
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        m, _, _ = count(S, ld, mid, L, strict=False, clearance_tol=0.0)
        if m >= n:
            hi = mid
        else:
            lo = mid
    return hi


if __name__ == "__main__":
    cases = [
        ("[1,2,2,1]", [(0, 1), (1, 2), (1, 2), (2, 3), (2, 3), (3, 4)], [1, 3, 4, 3, 1],
         [np.pi, np.e, 1.0, np.sqrt(2), np.sqrt(3), np.sqrt(5)],
         [0.1, 40.0, 300.0, 7775.579903981382]),
        ("[1,2,2]", [(0, 1), (1, 2), (1, 2), (2, 3), (2, 3)], [1, 3, 4, 2],
         [np.pi, np.e, 1.0, np.sqrt(2), np.sqrt(3)],
         [0.1, 40.0, 300.0, 6342.123824304902]),
    ]
    for name, E, D, lengths, Ks in cases:
        S, ld = bond_scattering_matrix(E, D, lengths)
        L = sum(lengths)
        print(f"\n{name}   L = {L!r}")
        beta = len(E) - len(D) + 1
        print(f"   beta = {beta};  mult(eigenvalue 1 of S) = "
              f"{check_anchor(S, ld, beta)}  (invariant beta+1 verified)")
        print(f"   phase_sum at anchor k=0           : {phase_sum(S, ld, 0.0)}")
        for K in Ks:
            n, r, c = count(S, ld, K, L, beta=beta)
            print(f"   N_+({K:<22}) = {n:<8d} residual {r:.1e}  clearance {c:.1e}")
