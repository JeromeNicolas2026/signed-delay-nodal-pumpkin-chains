# Signed scattering-delay balance: reproducibility package

This repository contains the numerical data and reproducibility code
accompanying the manuscript

> **Signed scattering-delay balance and exact nodal laws for adjacent
> edge-disjoint two-cycle pumpkin chains**

by Jérôme Nicolas.

The package supports the numerical checks only. The analytic proofs do not
depend on numerical data. The manuscript itself is not included here.

## Contents

| Path | Purpose |
| --- | --- |
| `phase_count_audit.py` | Root-independent completeness certificate based on the total eigenphase count. |
| `length_sweeps.py` | Phase-count-guided spectral sweeps for several length vectors, including calibration cases. |
| `length_sweeps.csv` | Deterministic output of the archived length sweeps. |
| `asymmetry_trend.csv` | Finite-cutoff asymmetry diagnostic for the one-tailed chain. |
| `make_length_figure.py` | Recreates the correlation-versus-length-ratio figure. |
| `correlation_vs_length_ratio.pdf` | Vector version of the generated figure. |
| `correlation_vs_length_ratio.png` | Raster version of the generated figure. |
| `numerics/` | Archived spectra and deterministic post-processing audit. |
| `SHA256SUMS` | Checksums for every versioned file except the manifest itself. |

The nested file `numerics/README_NUMERICS.txt` documents the archived arrays
and the raw-to-cleaned post-processing audit in more detail.

## Requirements

- Python 3.10 or later;
- NumPy;
- SciPy;
- Matplotlib, only for regenerating the figure.

To create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Quick verification

Run these commands from the repository root:

```bash
sha256sum -c SHA256SUMS
python3 phase_count_audit.py
python3 numerics/reproduce_numerics.py
python3 length_sweeps.py --calibrate
```

The phase-count audit must report the anchor invariant and, in particular,

```text
[1,2,2,1] : N_+(7775.579903981382) = 30299
[1,2,2]   : N_+(6342.123824304902) = 20198
```

The numerical audit must reproduce the exact one-tailed correlation

```text
-0.118797879168373
```

as well as the archived integer counts and local-surplus decompositions.

## Full spectral sweeps

The complete sweep regenerates `length_sweeps.csv`:

```bash
python3 length_sweeps.py
```

The finite-cutoff asymmetry table is regenerated separately:

```bash
python3 length_sweeps.py --asymmetry
```

The archived figure can be regenerated from `length_sweeps.csv` with

```bash
python3 make_length_figure.py
```

This command overwrites the archived PDF and PNG. PDF metadata written by
Matplotlib can vary between runs, so the regenerated PDF need not have the
same checksum even when its plotted data and appearance are unchanged. Run
the checksum verification before regenerating outputs, or use a disposable
clone for this step.

The complete sweep can take several tens of minutes, depending on the
machine. The `both tails long` experiment is normally the most expensive.

## Reproducibility notes

The published frequencies are ratios of integer counts and should reproduce
exactly. Individual floating-point root positions can differ in their last
one or two digits across numerical libraries or machines without changing
those counts.

The anchor check deserves particular attention. A previous implementation
exposed a LAPACK-driver-dependent phase convention at the degenerate
eigenvalue 1 of the bond-scattering matrix. The current
`check_anchor` routine verifies both the branch convention and the invariant
`mult_S(1) = beta + 1` before the anchored phase count is used.

The archived outputs were verified in the following environment:

- Python 3.12.13;
- NumPy 2.3.5;
- SciPy 1.17.0;
- Matplotlib 3.10.8;
- SciPy OpenBLAS 0.3.30, 64-bit integers;
- Linux x86_64, glibc 2.39.

The lower bounds in `requirements.txt` are intentionally less restrictive
than this tested environment.

## Integrity and archived version

Git's end-of-line conversion is disabled by `.gitattributes`, so the
checksums remain stable after cloning on different platforms. The definitive
archived release is the version-specific Zenodo record associated with the
corresponding GitHub release. Cite that version DOI rather than the concept
DOI, which follows the newest release.

## Licenses

The Python source code is released under the MIT License; see
`LICENSE-CODE`. The `.npz` and `.csv` data and the generated `.pdf` and `.png`
outputs are released under CC BY 4.0; see `LICENSE-DATA`. Repository
documentation is also covered by CC BY 4.0.

These licenses apply only to this repository. They are independent of any
copyright or publishing agreement governing the text of the associated
article.

After publication, a link to the journal version of the article will be
added here.
