"""Evaluation metrics: instance P/R/F1 @ IoU thresholds + pixel Dice.

All IoUs between polygon *instances* are computed exactly with shapely in
normalized [0,1] coordinate space — no rasterization error, and any bug in the
mask -> polygon coordinate mapping shows up here immediately instead of being
hidden by matching rasters against rasters.

Pixel Dice (the assignment's third criterion: unioned pred mask vs unioned GT
mask) is rasterized at a fixed resolution since it is a pixel metric.
"""
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

PRIMARY_IOU = 0.90
SECONDARY_IOU = 0.50
DICE_RASTER_SIZE = 512


# ------------------------------ geometry ------------------------------------
def polygon_to_shape(points) -> Optional[ShapelyPolygon]:
    """One [[x,y],...] ring -> valid shapely polygon, or None if degenerate.
    buffer(0) heals self-intersections annotators sometimes produce."""
    if points is None or len(points) < 3:
        return None
    shape = ShapelyPolygon(points)
    if not shape.is_valid:
        shape = shape.buffer(0)
    if shape.is_empty or shape.area <= 0:
        return None
    return shape


def shapes_from_polygons(polygons) -> list:
    shapes = [polygon_to_shape(p) for p in (polygons or [])]
    return [s for s in shapes if s is not None]


def geom_iou(a, b) -> float:
    union = a.union(b).area
    if union == 0:
        return 0.0
    return a.intersection(b).area / union


def union_iou(polygons_a, polygons_b) -> float:
    """IoU of the unioned geometry of two polygon lists. Both empty -> 1.0
    (perfect agreement that there is no document), one empty -> 0.0."""
    shapes_a = shapes_from_polygons(polygons_a)
    shapes_b = shapes_from_polygons(polygons_b)
    if not shapes_a and not shapes_b:
        return 1.0
    if not shapes_a or not shapes_b:
        return 0.0
    return geom_iou(unary_union(shapes_a), unary_union(shapes_b))


# --------------------------- instance matching ------------------------------
def match_counts(pred_polygons, gt_polygons, thr: float):
    """Greedy one-to-one matching (same protocol as seg_common.match_counts,
    but exact polygon IoU): each GT claims the best still-unused prediction
    with IoU >= thr. Returns (TP, FP, FN)."""
    preds = shapes_from_polygons(pred_polygons)
    gts = shapes_from_polygons(gt_polygons)

    matched = set()
    tp = 0
    for gt in gts:
        best_iou, best_j = -1.0, -1
        for j, pred in enumerate(preds):
            if j in matched:
                continue
            iou = geom_iou(pred, gt)
            if iou >= thr and iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            tp += 1
            matched.add(best_j)
    return tp, len(preds) - len(matched), len(gts) - tp


# ------------------------------ pixel dice ----------------------------------
def _raster_union(polygons, size: int) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    for poly in polygons or []:
        if poly and len(poly) >= 3:
            pts = (np.asarray(poly, np.float32) * size).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def union_dice(pred_polygons, gt_polygons, size: int = DICE_RASTER_SIZE) -> float:
    """Dice between the unioned pred and GT masks. Both empty -> 1.0."""
    pred = _raster_union(pred_polygons, size)
    gt = _raster_union(gt_polygons, size)
    denom = int(pred.sum()) + int(gt.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int(np.logical_and(pred, gt).sum()) / denom


# ------------------------------- reports ------------------------------------
def evaluate_polygons(preds: Dict[str, list], gts: Dict[str, list],
                      iou_thrs: Sequence[float] = (PRIMARY_IOU, SECONDARY_IOU)
                      ) -> dict:
    """Aggregate instance P/R/F1 at each IoU threshold plus mean pixel Dice.

    preds/gts: {image_name: list of normalized polygons}. Every GT image must
    have an entry in preds (missing counts as an empty prediction)."""
    names = sorted(gts.keys())
    report = {"images": len(names), "iou": {}}

    for thr in iou_thrs:
        total = np.zeros(3, np.int64)  # tp, fp, fn
        for name in names:
            total += match_counts(preds.get(name, []), gts[name], thr)
        tp, fp, fn = (int(v) for v in total)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report["iou"][thr] = {"P": precision, "R": recall, "F1": f1,
                              "TP": tp, "FP": fp, "FN": fn}

    report["dice_mean"] = float(np.mean(
        [union_dice(preds.get(name, []), gts[name]) for name in names]))
    report["gt_instances"] = sum(len(gts[n]) for n in names)
    report["pred_instances"] = sum(len(preds.get(n, [])) for n in names)
    return report


def report_score(report: dict) -> tuple:
    """Sortable model-selection score: primary F1, then secondary F1, then
    Dice (breaks ties in early epochs where F1@0.90 is still 0)."""
    return (report["iou"][PRIMARY_IOU]["F1"],
            report["iou"][SECONDARY_IOU]["F1"],
            report["dice_mean"])


def print_report(name: str, report: dict, primary: float = PRIMARY_IOU) -> None:
    print(f"\n=== {name}  (n={report['images']} imgs | GT {report['gt_instances']}"
          f" / pred {report['pred_instances']} instances) ===")
    for thr, m in report["iou"].items():
        star = "  <- primary" if abs(thr - primary) < 1e-9 else ""
        print(f"  IoU>={thr:.2f}   P {m['P']:.4f}   R {m['R']:.4f}   "
              f"F1 {m['F1']:.4f}   (TP {m['TP']}  FP {m['FP']}  FN {m['FN']}){star}")
    print(f"  pixel Dice (union): {report['dice_mean']:.4f}")
