"""Inference, postprocess tuning on val, pred.csv generation + validation.

    * load_model / predict_val_probs - checkpoint -> cached prob maps.
    * sweep_thresholds               - F1@0.90 across thresholds 0.3..0.7.
    * compare_postprocess            - approxPolyDP eps values vs minAreaRect.
    * plot_overlays                  - predicted polygons drawn on ORIGINAL
                                       images (verifies coordinate mapping).
    * generate_pred_csv              - all 1000 test images -> pred.csv.
    * validate_pred_csv              - reload + json.loads every row, assert
                                       exactly the test filenames, plot 5
                                       random overlays.
"""
import json
import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch

from . import consensus as C
from . import data as D
from . import metrics as M
from . import postprocess as P
from .train import predict_probs


def load_model(ckpt_path: str, device=None):
    from seg_common import UNet
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def val_split_with_gt(cfg: D.Config):
    """(val_names, medoid GT polygons) for the fixed 4500/500 split."""
    rounds = D.load_all_rounds(cfg)
    _, val_names = D.train_val_split(sorted(rounds[1].keys()),
                                     cfg.val_size, cfg.seed)
    return val_names, C.medoid_annotations(rounds, names=val_names)


def _evaluate_probs(probs: Dict[str, np.ndarray], gts, threshold,
                    **post_kwargs) -> dict:
    preds = {name: P.prob_to_polygons(prob.astype(np.float32), threshold,
                                      **post_kwargs)
             for name, prob in probs.items()}
    return M.evaluate_polygons(preds, gts)


def _row(report: dict) -> dict:
    hi, lo = report["iou"][M.PRIMARY_IOU], report["iou"][M.SECONDARY_IOU]
    return {"F1@0.90": hi["F1"], "P@0.90": hi["P"], "R@0.90": hi["R"],
            "F1@0.50": lo["F1"], "dice": report["dice_mean"]}


def sweep_thresholds(probs: Dict[str, np.ndarray], gts,
                     thresholds=np.arange(0.30, 0.71, 0.05),
                     **post_kwargs) -> pd.DataFrame:
    """Binarization-threshold sweep on cached val probabilities."""
    rows = []
    for thr in thresholds:
        report = _evaluate_probs(probs, gts, float(thr), **post_kwargs)
        rows.append({"threshold": round(float(thr), 2), **_row(report)})
    df = pd.DataFrame(rows)
    best = df.loc[df["F1@0.90"].idxmax()]
    print(f"best threshold {best['threshold']}: F1@0.90 = {best['F1@0.90']:.4f}")
    return df


def compare_postprocess(probs: Dict[str, np.ndarray], gts, threshold: float,
                        eps_fracs=(0.005, 0.01, 0.015, 0.02),
                        **post_kwargs) -> pd.DataFrame:
    """approxPolyDP at several eps values vs minAreaRect, at a fixed threshold.
    Extra kwargs (mode, min_area_frac, min_mean_prob, ...) apply to every row."""
    rows = []
    for eps in eps_fracs:
        report = _evaluate_probs(probs, gts, threshold,
                                 method="approx", eps_frac=eps, **post_kwargs)
        rows.append({"method": f"approxPolyDP eps={eps}", **_row(report)})
    report = _evaluate_probs(probs, gts, threshold, method="minrect",
                             **post_kwargs)
    rows.append({"method": "minAreaRect", **_row(report)})
    return pd.DataFrame(rows)


def compare_modes(probs: Dict[str, np.ndarray], gts, threshold: float,
                  **post_kwargs) -> pd.DataFrame:
    """Old single-document behavior vs multi-instance extraction, side by side."""
    rows = []
    for mode in ("largest", "all"):
        report = _evaluate_probs(probs, gts, threshold, mode=mode, **post_kwargs)
        rows.append({"mode": mode, **_row(report)})
    return pd.DataFrame(rows)


# ----------------------------- visual checks --------------------------------
def plot_overlays(cfg: D.Config, preds: Dict[str, list], names,
                  gts=None, out_path=None):
    """Predicted polygons (yellow) — and GT (lime) if given — drawn on the
    ORIGINAL full-resolution images. The definitive check that normalized
    coordinates from the 384x384 mask map back correctly."""
    import matplotlib.pyplot as plt

    k = len(names)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4.5))
    axes = np.atleast_1d(axes)

    def draw(ax, polys, W, H, color, label):
        for poly in polys or []:
            if poly and len(poly) >= 3:
                pts = np.asarray(poly + [poly[0]], np.float32) * [W, H]
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, label=label)

    no_cache = D.Config(data_root=cfg.data_root)   # force original images
    for ax, name in zip(axes, names):
        image = D.load_image(no_cache, name)
        H, W = image.shape[:2]
        ax.imshow(image)
        draw(ax, preds.get(name, []), W, H, "yellow", "pred")
        if gts is not None:
            draw(ax, gts.get(name, []), W, H, "lime", "gt")
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


# ------------------------------- submission ---------------------------------
def generate_pred_csv(model, cfg: D.Config, out_csv: str, threshold: float,
                      device=None, **post_kwargs) -> pd.DataFrame:
    """Predict every test image and write pred.csv (image, polygon)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    test_names = D.list_test_images(cfg)
    print(f"predicting {len(test_names)} test images ...")
    probs = predict_probs(model, cfg, test_names, device)

    rows = []
    for name in test_names:
        polys = P.prob_to_polygons(probs[name].astype(np.float32), threshold,
                                   **post_kwargs)
        rows.append({"image": name, "polygon": json.dumps(polys)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} rows -> {out_csv}")
    return df


def validate_pred_csv(csv_path: str, cfg: D.Config, expected_rows: int = 1000,
                      plot_k: int = 5, seed: int = 0, out_path=None):
    """Reload the csv and verify the full submission contract; then overlay
    plot_k random predictions on their original test images."""
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["image", "polygon"], f"bad columns: {df.columns}"
    assert len(df) == expected_rows, f"{len(df)} rows, expected {expected_rows}"

    test_names = D.list_test_images(cfg)
    assert sorted(df["image"]) == sorted(test_names), \
        "image column does not match the test filenames exactly"

    n_empty = 0
    parsed = {}
    for row in df.itertuples():
        polys = json.loads(row.polygon)           # must be valid JSON
        assert isinstance(polys, list)
        for poly in polys:
            assert len(poly) >= 3, f"{row.image}: polygon with <3 points"
            for pt in poly:
                x, y = pt
                assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, \
                    f"{row.image}: point {pt} outside [0,1]"
        n_empty += not polys
        parsed[row.image] = polys
    print(f"OK: {len(df)} rows, all JSON valid, coords in [0,1], "
          f"{n_empty} empty predictions")

    sample = random.Random(seed).sample(test_names, plot_k)
    return plot_overlays(cfg, parsed, sample, out_path=out_path)
