"""Create the reproducibility figure for the one-tailed length sweeps."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ETA = 0.6368335201743935294225288116763


def load_one_tail_rows():
    with (ROOT / "length_sweeps.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["graph"] == "[1,2,2]"]
    rows.sort(key=lambda row: float(row["r"]))
    return rows


def main():
    rows = load_one_tail_rows()
    r = np.array([float(row["r"]) for row in rows])
    measured = np.array([float(row["C"]) for row in rows])

    line_r = np.linspace(0.04, 0.305, 300)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.7, 3.45), constrained_layout=True)
    ax.plot(line_r, -ETA * line_r, color="black", linewidth=1.25,
            label=r"exact law $-\eta r$")
    ax.scatter(r, measured, s=34, facecolors="white", edgecolors="black",
               linewidths=1.05, zorder=3, label="spectral sweeps")

    ax.set_xlabel(r"length ratio $r=L_\circ/(2L_B+L_\circ)$")
    ax.set_ylabel(r"local-surplus correlation $\mathcal{C}$")
    ax.set_xlim(0.04, 0.305)
    ax.set_ylim(-0.200, -0.020)
    ax.grid(True, color="0.86", linewidth=0.55)
    ax.legend(frameon=False, loc="lower left")

    fig.savefig(ROOT / "correlation_vs_length_ratio.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "correlation_vs_length_ratio.png", dpi=300,
                bbox_inches="tight")


if __name__ == "__main__":
    main()
