"""Torch-free unit tests for metrics.py and postprocess.py (synthetic shapes).

Run:  python -m tests.test_core
"""
import numpy as np

from src import metrics as M
from src import postprocess as P


def square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_metrics():
    gt = [square(0.1, 0.1, 0.6, 0.6)]

    # identical polygons -> perfect at both thresholds, dice ~1
    rep = M.evaluate_polygons({"a": gt}, {"a": gt})
    assert rep["iou"][0.9]["F1"] == 1.0 and rep["iou"][0.5]["F1"] == 1.0
    assert rep["dice_mean"] > 0.99, rep["dice_mean"]

    # shift by 0.05: analytic IoU = 0.2025 / 0.2975 = 0.6807
    shifted = [square(0.15, 0.15, 0.65, 0.65)]
    iou = M.union_iou(gt, shifted)
    assert abs(iou - 0.2025 / 0.2975) < 1e-6, iou
    rep = M.evaluate_polygons({"a": shifted}, {"a": gt})
    assert rep["iou"][0.5] == {**rep["iou"][0.5], "TP": 1, "FP": 0, "FN": 0}
    assert rep["iou"][0.9] == {**rep["iou"][0.9], "TP": 0, "FP": 1, "FN": 1}

    # empty vs empty -> dice 1.0, no instance counts
    rep = M.evaluate_polygons({"a": []}, {"a": []})
    assert rep["dice_mean"] == 1.0 and rep["iou"][0.9]["F1"] == 0.0
    assert M.union_iou([], []) == 1.0 and M.union_iou(gt, []) == 0.0

    # two preds on one gt -> one TP one FP
    tp, fp, fn = M.match_counts(gt + [square(0.7, 0.7, 0.9, 0.9)], gt, 0.9)
    assert (tp, fp, fn) == (1, 1, 0)

    # self-intersecting bowtie is healed, not crashed
    bowtie = [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]]
    assert M.polygon_to_shape(bowtie) is not None

    # missing pred key counts as empty prediction (FN)
    rep = M.evaluate_polygons({}, {"a": gt})
    assert rep["iou"][0.9]["FN"] == 1
    print("test_metrics OK")


def test_postprocess_roundtrip():
    S = 384
    # known rectangle in pixel space -> mask -> polygons -> compare IoU
    x0, y0, x1, y1 = 50, 80, 300, 350
    mask = np.zeros((S, S), np.uint8)
    mask[y0:y1, x0:x1] = 1
    # pixels y0..y1-1 are set, so the true region spans [y0, y1) in continuous
    # coords -> normalized rect:
    expected = [square(x0 / S, y0 / S, x1 / S, y1 / S)]

    for method in ("approx", "minrect"):
        polys = P.extract_polygons(mask, method=method)
        assert len(polys) == 1
        iou = M.union_iou(polys, expected)
        assert iou > 0.98, (method, iou)
    assert len(P.extract_polygons(mask, method="minrect")[0]) == 4

    # prob map path + threshold
    prob = np.where(mask > 0, 0.9, 0.1).astype(np.float32)
    polys = P.prob_to_polygons(prob, threshold=0.5)
    assert M.union_iou(polys, expected) > 0.98

    # tiny blob dropped by min_area_frac; largest_only keeps one contour
    noisy = mask.copy()
    noisy[5:10, 5:10] = 1
    assert len(P.extract_polygons(noisy, largest_only=False)) == 1
    assert len(P.extract_polygons(noisy, largest_only=False,
                                  min_area_frac=0.0)) == 2
    assert len(P.extract_polygons(noisy, largest_only=True,
                                  min_area_frac=0.0)) == 1

    # empty mask -> empty list (serializes to "[]")
    assert P.extract_polygons(np.zeros((S, S), np.uint8)) == []
    print("test_postprocess_roundtrip OK")


if __name__ == "__main__":
    test_metrics()
    test_postprocess_roundtrip()
    print("ALL TESTS PASSED")
