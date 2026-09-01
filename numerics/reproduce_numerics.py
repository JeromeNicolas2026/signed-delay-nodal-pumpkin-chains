#!/usr/bin/env python3
"""Reproduce the numerical checks reported in the manuscript.

The script starts from the archived spectral arrays.  It audits the raw-to-
cleaned correction for [1,2,2,1], reconstructs its nodal surplus directly
from the edge solutions, verifies the exact decompositions into local
surpluses, and compares both empirical laws with their analytic predictions.

The printed comparisons are deterministic finite-cutoff diagnostics.  No
independent-sampling model, standard error, or p-value is attached to the
ordered spectrum.

Requirements: Python 3.10+, NumPy, SciPy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad


PI = math.pi
TASK1_LENGTHS = np.array(
    [PI, math.e, 1.0, math.sqrt(2.0), math.sqrt(3.0), math.sqrt(5.0)]
)
TASK2_LENGTHS = TASK1_LENGTHS[:-1]


def load_array(path: Path, key: str) -> np.ndarray:
    """Load one named array without permitting pickled objects."""
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"{path}: missing key {key!r}; found {archive.files}")
        return np.asarray(archive[key])


def align_spectra(
    raw: np.ndarray, clean: np.ndarray, tolerance: float = 1.0e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monotonically align two spectra and return raw-only, clean-only, deltas."""
    i = j = 0
    raw_only: list[float] = []
    clean_only: list[float] = []
    corrections: list[float] = []
    while i < raw.size and j < clean.size:
        delta = float(clean[j] - raw[i])
        if abs(delta) <= tolerance:
            corrections.append(delta)
            i += 1
            j += 1
        elif raw[i] < clean[j]:
            raw_only.append(float(raw[i]))
            i += 1
        else:
            clean_only.append(float(clean[j]))
            j += 1
    raw_only.extend(map(float, raw[i:]))
    clean_only.extend(map(float, clean[j:]))
    return (
        np.asarray(raw_only),
        np.asarray(clean_only),
        np.asarray(corrections),
    )


def block_endpoint_and_sign(
    k: float, tail: float, parallel_a: float, parallel_b: float
) -> tuple[float, int]:
    """Return the remote vertex value q (central value 1) and local sign."""
    x, y, z = k * np.array([tail, parallel_a, parallel_b])
    a = math.tan(x) - 1.0 / math.tan(y) - 1.0 / math.tan(z)
    q = -(1.0 / math.sin(y) + 1.0 / math.sin(z)) / a
    eps = int(np.sign(a * math.sin(y) * math.sin(z)))
    if eps == 0:
        raise ArithmeticError(f"non-generic local sign at k={k:.17g}")
    return q, eps


def count_parallel_edge_zeros(f0: float, f1: float, phase: float) -> tuple[int, float]:
    """Count interior zeros for endpoint values f0,f1 and phase k*length."""
    numerator = -f0 * math.sin(phase)
    denominator = f1 - f0 * math.cos(phase)
    first = math.atan2(numerator, denominator) % PI
    margin = min(first, (phase - first) % PI)
    if first < 1.0e-13:
        return max(int(math.floor((phase - 1.0e-12) / PI)), 0), margin
    count = 0 if first >= phase else int(math.floor((phase - first) / PI)) + 1
    return count, margin


def task1_surpluses(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reconstruct global and local surpluses of [1,2,2,1]."""
    sigma = np.empty(eigenvalues.size, dtype=int)
    eps_left = np.empty(eigenvalues.size, dtype=int)
    eps_right = np.empty(eigenvalues.size, dtype=int)
    minimum_margin = math.inf

    x_left, y_left, z_left, y_right, z_right, x_right = TASK1_LENGTHS
    for index, k in enumerate(eigenvalues):
        q_left, eps_left[index] = block_endpoint_and_sign(
            float(k), x_left, y_left, z_left
        )
        q_right, eps_right[index] = block_endpoint_and_sign(
            float(k), x_right, y_right, z_right
        )

        zeros = int(math.floor(k * x_left / PI + 0.5))
        zeros += int(math.floor(k * x_right / PI + 0.5))
        for f0, f1, length in (
            (q_left, 1.0, y_left),
            (q_left, 1.0, z_left),
            (1.0, q_right, y_right),
            (1.0, q_right, z_right),
        ):
            edge_zeros, margin = count_parallel_edge_zeros(f0, f1, k * length)
            zeros += edge_zeros
            minimum_margin = min(minimum_margin, margin)

        # k=0 is the first eigenvalue; hence the positive root at zero-based
        # position index has n-1=index+1 in the nodal-surplus convention.
        sigma[index] = zeros - (index + 1)

    return sigma, eps_left, eps_right, minimum_margin


def sign_statistics(eps_left: np.ndarray, eps_right: np.ndarray) -> dict[str, float]:
    product = eps_left * eps_right
    return {
        "mean_left": float(np.mean(eps_left)),
        "mean_right": float(np.mean(eps_right)),
        "mean_product": float(np.mean(product)),
    }


def format_values(values: np.ndarray, digits: int = 15) -> str:
    return ", ".join(f"{float(value):.{digits}g}" for value in values)


def audit_task1(data_directory: Path) -> None:
    raw = load_array(data_directory / "task1_eigs_raw.npz", "eigs")
    clean = load_array(data_directory / "task1_eigs.npz", "eigs")
    assert np.all(np.diff(raw) > 0.0) and np.all(np.diff(clean) > 0.0)

    raw_only, clean_only, corrections = align_spectra(raw, clean)
    assert raw_only.shape == (2,) and np.allclose(raw_only, [232.0, 316.0], atol=1e-12)
    assert clean_only.shape == (1,) and abs(clean_only[0] - 315.9967814592119) < 1e-12
    assert corrections.size == 30298

    sigma, eps_left, eps_right, minimum_margin = task1_surpluses(clean)
    local_sum = (1 + eps_left) // 2 + (1 + eps_right) // 2
    assert np.array_equal(sigma, local_sum)
    assert np.all((0 <= sigma) & (sigma <= 2))

    counts = np.bincount(sigma, minlength=3)
    probabilities = counts / clean.size
    stats = sign_statistics(eps_left, eps_right)
    target_cutoff = 30300.0 * PI / float(np.sum(TASK1_LENGTHS))

    print("TASK 1 -- [1,2,2,1]")
    print(f"raw candidates / cleaned roots : {raw.size} / {clean.size}")
    print(f"raw-only pseudo-poles          : {format_values(raw_only)}")
    print(f"clean-only recovered root      : {format_values(clean_only)}")
    print(f"bitwise-adjusted matched roots : {np.count_nonzero(corrections)}")
    print(f"adjustments > 1e-12 / > 1e-10 : "
          f"{np.count_nonzero(np.abs(corrections) > 1e-12)} / "
          f"{np.count_nonzero(np.abs(corrections) > 1e-10)}")
    print(f"largest absolute adjustment    : {np.max(np.abs(corrections)):.12e}")
    print(f"Weyl target cutoff / last root : {target_cutoff:.12f} / {clean[-1]:.12f}")
    print(f"surplus counts                 : {counts.tolist()}")
    print(f"surplus frequencies            : {format_values(probabilities, 9)}")
    print(f"mean eps_L / eps_R             : {stats['mean_left']:.9f} / "
          f"{stats['mean_right']:.9f}")
    print(f"mean sign product              : {stats['mean_product']:.12f}")
    print(f"smallest nodal-zero margin     : {minimum_margin:.12e}")
    print("pointwise local decomposition  : verified for every cleaned root")


def audit_task2(data_directory: Path) -> None:
    path = data_directory / "task2.npz"
    with np.load(path, allow_pickle=False) as archive:
        expected = {"eigs2", "gen2", "loop2", "sig2g", "epsL2", "epsR2", "sig_loop"}
        if set(archive.files) != expected:
            raise KeyError(f"{path}: expected {sorted(expected)}, found {archive.files}")
        eigs = np.asarray(archive["eigs2"])
        generic = np.asarray(archive["gen2"])
        loops = np.asarray(archive["loop2"])
        sigma = np.asarray(archive["sig2g"], dtype=int)
        eps_left = np.asarray(archive["epsL2"], dtype=int)
        eps_right = np.asarray(archive["epsR2"], dtype=int)
        sigma_loop = np.asarray(archive["sig_loop"], dtype=int)

    assert np.array_equal(eigs, np.sort(np.concatenate((generic, loops))))
    assert np.intersect1d(generic, loops).size == 0
    assert generic.size == sigma.size == eps_left.size == eps_right.size

    mode = np.arange(1, loops.size + 1)
    loop_length = math.sqrt(2.0) + math.sqrt(3.0)
    exact_loops = 2.0 * PI * mode / loop_length
    loop_error = loops - exact_loops
    positions = np.searchsorted(eigs, loops)
    assert np.array_equal(eigs[positions], loops)
    assert np.array_equal(sigma_loop, (2 * mode - 1) - (positions + 1))

    local_sum = (1 + eps_left) // 2 + (1 + eps_right) // 2
    assert np.array_equal(sigma, local_sum)
    counts = np.bincount(sigma, minlength=3)
    probabilities = counts / generic.size
    stats = sign_statistics(eps_left, eps_right)

    eta = 4.0 / PI**2 * quad(
        lambda value: math.atan(2.0 * math.tan(value)),
        0.0,
        PI / 2.0,
        epsabs=2.0e-14,
        epsrel=2.0e-14,
        limit=200,
    )[0]
    block_length = PI + math.e + 1.0
    exact_correlation = -eta * loop_length / (2.0 * block_length + loop_length)
    exact_probabilities = np.array(
        [(1.0 + exact_correlation) / 4.0,
         (1.0 - exact_correlation) / 2.0,
         (1.0 + exact_correlation) / 4.0]
    )
    target_cutoff = 20200.0 * PI / float(np.sum(TASK2_LENGTHS))
    loop_density = loops.size / eigs.size
    exact_loop_density = loop_length / (2.0 * float(np.sum(TASK2_LENGTHS)))

    print("\nTASK 2 -- [1,2,2]")
    print(f"all / generic / loop roots     : {eigs.size} / {generic.size} / {loops.size}")
    print(f"Weyl target cutoff / last root : {target_cutoff:.12f} / {eigs[-1]:.12f}")
    print(f"max / RMS loop-root error      : {np.max(np.abs(loop_error)):.12e} / "
          f"{math.sqrt(float(np.mean(loop_error**2))):.12e}")
    print(f"loop density, data / theory    : {loop_density:.12f} / {exact_loop_density:.12f}")
    print(f"surplus counts                 : {counts.tolist()}")
    print(f"surplus frequencies            : {format_values(probabilities, 9)}")
    print(f"eta                            : {eta:.16f}")
    print(f"exact correlation              : {exact_correlation:.15f}")
    print(f"exact surplus probabilities    : {format_values(exact_probabilities, 12)}")
    print(f"mean eps_L / eps_R             : {stats['mean_left']:.9f} / "
          f"{stats['mean_right']:.9f}")
    print(f"mean sign product              : {stats['mean_product']:.12f}")
    print("pointwise local decomposition  : verified for every generic root")
    print("loop ordering and raw surplus  : verified exactly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing task1_eigs_raw.npz, task1_eigs.npz, and task2.npz",
    )
    arguments = parser.parse_args()
    audit_task1(arguments.data_directory)
    audit_task2(arguments.data_directory)


if __name__ == "__main__":
    main()
