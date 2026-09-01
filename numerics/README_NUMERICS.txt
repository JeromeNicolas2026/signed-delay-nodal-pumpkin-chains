NUMERICAL REPRODUCIBILITY PACKAGE
=================================

Files
-----
  task1_eigs_raw.npz   Raw root candidates for [1,2,2,1].
  task1_eigs.npz       Cleaned spectrum for [1,2,2,1].
  task2.npz             Full, generic, and loop spectra for [1,2,2], together
                        with the computed local signs and surpluses.
  reproduce_numerics.py
                        Deterministic audit and post-processing script.
  SHA256SUMS            Integrity checksums for the data and script.

Run
---
From this directory:

  python3 reproduce_numerics.py

Optional integrity check:

  sha256sum -c SHA256SUMS

Requirements: Python 3.10 or later, NumPy, and SciPy.

Scope
-----
The script reproduces every numerical value reported in the manuscript's
table for [1,2,2,1] and [1,2,2].  In particular, it:

  * audits the raw-to-cleaned [1,2,2,1] spectrum;
  * identifies the two false integer pseudo-poles and the recovered nearby
    genuine root;
  * reconstructs all [1,2,2,1] nodal surpluses from the exact edge solutions;
  * verifies the pointwise decomposition into the two local surpluses;
  * verifies the exact union of generic and loop-supported [1,2,2] states;
  * checks the exact loop sequence and its full-spectrum ordering;
  * computes the empirical correlations, the theoretical one-tail law, and
    the loop density.

The script deliberately reports no standard errors, Pearson p-values, or
chi-square p-values.  Its comparisons are deterministic finite-cutoff
diagnostics: a spectral sequence is ordered and is not an independent random
sample.

The archived arrays are the output of the original pole-cleared secular root
sweep.  This compact package reproduces and independently audits all reported
post-processing; it does not rerun that initial root search.

The parent directory additionally contains phase_count_audit.py, which
certifies the completeness of these archives independently of their root
search, and length_sweeps.py, which performs separate phase-count-guided
parameter sweeps.
