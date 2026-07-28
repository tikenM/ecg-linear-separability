#!/usr/bin/env python3
"""
Generate Fig. 2 — SVC confusion matrices for MIT-BIH DS2 and INCART.

Reads the confusion matrices directly from results.json so the figure
cannot drift from the numbers reported in the manuscript tables. The
resulting PNG is written to the working directory.

Requires: numpy, matplotlib.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_JSON = Path("/Users/miska/Library/CloudStorage/OneDrive-dibru.ac.in/Research/ECG/results/results.json")   # adjust to your local path
OUTPUT_PNG   = Path("/Users/miska/Library/CloudStorage/OneDrive-dibru.ac.in/Research/ECG/results/02_confusion_matrices.png")

# The manuscript uses the AAMI 5-class ordering
LABELS = ["N", "S", "V", "F", "Q"]

# Panel-specific styling — matches the manuscript figure conventions
PANEL_CONFIG = [
    dict(
        dataset_key   = "mitdb",
        panel_title   = "MIT-BIH SVC\n(Inter-patient DS2)",
        cmap          = "Blues",
        annot_switch  = 0.60,   # fraction of colorbar max above which text is white
    ),
    dict(
        dataset_key   = "incart",
        panel_title   = "INCART SVC\n(Patient-grouped)",
        cmap          = "Greens",
        annot_switch  = 0.60,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_confusion(results: dict, dataset_key: str) -> np.ndarray:
    """Fetch the SVC confusion matrix and reorder rows/cols to LABELS order."""
    m       = results[dataset_key]["interpatient"]["svc"]["metrics"]
    conf    = np.asarray(m["confusion"], dtype=int)
    src_lbl = m["labels"]
    # Reorder to canonical AAMI order N, S, V, F, Q
    idx = [src_lbl.index(l) for l in LABELS]
    return conf[np.ix_(idx, idx)]


def draw_heatmap(ax, conf: np.ndarray, cmap: str,
                 title: str, annot_switch: float,
                 show_ylabel: bool) -> None:
    """Draw one confusion-matrix panel with cell annotations and a colorbar."""
    vmax = conf.max()
    im   = ax.imshow(conf, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")

    # cell annotations — white text for dark cells, black otherwise
    threshold = annot_switch * vmax
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            v     = conf[i, j]
            color = "white" if v >= threshold else "black"
            ax.text(j, i, f"{v:,}" if v >= 1000 else str(v),
                    ha="center", va="center",
                    color=color, fontsize=11)

    # ticks, labels
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    if show_ylabel:
        ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)

    # colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("Count", fontsize=11)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrices — SVC (Primary Model)",
                 fontsize=15, fontweight="bold", y=1.02)

    for ax, cfg, show_y in zip(axes, PANEL_CONFIG, (True, False)):
        conf = load_confusion(results, cfg["dataset_key"])

        # Print per-class supports so the caller can sanity-check row sums
        row_sums = conf.sum(axis=1)
        print(f"{cfg['dataset_key']:>6} row sums (per-class test support):",
              dict(zip(LABELS, row_sums.tolist())))

        draw_heatmap(ax,
                     conf,
                     cmap        = cfg["cmap"],
                     title       = cfg["panel_title"],
                     annot_switch= cfg["annot_switch"],
                     show_ylabel = show_y)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\nWrote {OUTPUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
