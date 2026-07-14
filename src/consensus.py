"""Annotator-agreement analysis and consensus labels over the three rounds.

The three train_round_*.csv files are independent annotation passes over the
same 5000 images. This module quantifies their disagreement and builds the
labels used downstream:

    * pairwise_iou_table       - per-image union-IoU between every round pair.
    * plot_agreement_histogram - distribution of the mean pairwise IoU.
    * visualize_worst          - worst-K images with all 3 annotations overlaid.
    * medoid_annotations       - per image, the round that agrees best with the
                                 other two. Used as the single "ground truth"
                                 for validation instance metrics.
    * soft_mask / majority_mask- pixel-level consensus targets. Soft masks
                                 (values 0 / 1/3 / 2/3 / 1) are what
                                 data.SoftDocDataset feeds to BCE+Dice;
                                 majority vote is the hard fallback.
"""
import itertools
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M

PAIRS = list(itertools.combinations(D.ROUND_IDS, 2))  # (1,2), (1,3), (2,3)


# ----------------------------- agreement table ------------------------------
def pairwise_iou_table(rounds: Dict[int, D.Round], names=None) -> pd.DataFrame:
    """One row per image: exact union-IoU for each round pair + mean and min.
    Runs shapely on 3 x 5000 polygon pairs (~a minute)."""
    if names is None:
        names = sorted(rounds[1].keys())
    rows = []
    for name in names:
        row = {"image": name}
        for a, b in PAIRS:
            row[f"iou_{a}{b}"] = M.union_iou(rounds[a].get(name, []),
                                             rounds[b].get(name, []))
        ious = [row[f"iou_{a}{b}"] for a, b in PAIRS]
        row["iou_mean"] = float(np.mean(ious))
        row["iou_min"] = float(np.min(ious))
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_summary(df: pd.DataFrame) -> pd.Series:
    """Headline numbers: how noisy are the labels, and how often would the
    annotators fail the assignment's own 0.90 bar against each other?"""
    return pd.Series({
        "mean pairwise IoU": df["iou_mean"].mean(),
        "median pairwise IoU": df["iou_mean"].median(),
        "% images: min pair IoU < 0.90": 100.0 * (df["iou_min"] < 0.90).mean(),
        "% images: min pair IoU < 0.50": 100.0 * (df["iou_min"] < 0.50).mean(),
    })


# ----------------------------- visualization --------------------------------
def plot_agreement_histogram(df: pd.DataFrame, out_path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["iou_mean"], bins=50, color="steelblue")
    ax.axvline(0.90, color="red", ls="--", label="IoU 0.90 (primary metric bar)")
    ax.set_xlabel("mean pairwise IoU between the 3 annotation rounds")
    ax.set_ylabel("images")
    ax.set_title("Annotator agreement per image")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def visualize_worst(df: pd.DataFrame, rounds: Dict[int, D.Round], cfg: D.Config,
                    k: int = 5, out_path=None):
    """Worst-K agreement images with all three annotations overlaid
    (round 1 red, round 2 lime, round 3 cyan)."""
    import matplotlib.pyplot as plt
    colors = {1: "red", 2: "lime", 3: "cyan"}

    worst = df.nsmallest(k, "iou_mean")
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4.5))
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, worst.iterrows()):
        name = row["image"]
        image = D.load_image(cfg, name)
        H, W = image.shape[:2]
        ax.imshow(image)
        for r in D.ROUND_IDS:
            for poly in rounds[r].get(name, []):
                if poly and len(poly) >= 3:
                    pts = np.asarray(poly + [poly[0]], np.float32) * [W, H]
                    ax.plot(pts[:, 0], pts[:, 1], color=colors[r], lw=2,
                            label=f"round {r}")
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=8)
        ax.set_title(f"{name}\nmean IoU {row['iou_mean']:.3f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


# ----------------------------- consensus labels -----------------------------
def medoid_annotations(rounds: Dict[int, D.Round], names=None
                       ) -> Dict[str, D.Polygons]:
    """Per image, keep the annotation whose mean IoU to the other two rounds is
    highest (the medoid). This discards the outlier annotator per image and
    yields clean single-source polygons — used as validation ground truth."""
    if names is None:
        names = sorted(rounds[1].keys())
    out = {}
    for name in names:
        best_r, best_score = D.ROUND_IDS[0], -1.0
        for r in D.ROUND_IDS:
            others = [o for o in D.ROUND_IDS if o != r]
            score = np.mean([M.union_iou(rounds[r].get(name, []),
                                         rounds[o].get(name, []))
                             for o in others])
            if score > best_score:
                best_score, best_r = score, r
        out[name] = rounds[best_r].get(name, [])
    return out


def soft_mask(polys_per_round: List[D.Polygons], H: int, W: int) -> np.ndarray:
    """Mean of the per-round binary masks: float32 in {0, 1/3, 2/3, 1}."""
    return D.rasterize_counts(polys_per_round, H, W).astype(np.float32) / len(polys_per_round)


def majority_mask(polys_per_round: List[D.Polygons], H: int, W: int) -> np.ndarray:
    """Pixel-level majority vote (>= 2 of 3): uint8 {0, 1}."""
    n = len(polys_per_round)
    return (D.rasterize_counts(polys_per_round, H, W) > n / 2).astype(np.uint8)
