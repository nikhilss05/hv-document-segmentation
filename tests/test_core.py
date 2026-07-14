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

    # ---- multi-instance matching (test set has ~1.5 docs/image) ----
    doc_a, doc_b = square(0.1, 0.1, 0.4, 0.4), square(0.6, 0.6, 0.9, 0.9)
    # 2 GT + 2 preds, both exact -> 2 TP
    assert M.match_counts([doc_a, doc_b], [doc_b, doc_a], 0.9) == (2, 0, 0)
    # 2 GT + 1 pred -> 1 TP, 1 FN
    assert M.match_counts([doc_a], [doc_a, doc_b], 0.9) == (1, 0, 1)
    # 1 GT + 2 preds -> 1 TP, 1 FP
    assert M.match_counts([doc_a, doc_b], [doc_a], 0.9) == (1, 1, 0)
    # each pred may claim at most one GT: duplicate preds on one GT -> 1 TP 1 FP
    assert M.match_counts([doc_a, doc_a], [doc_a], 0.9) == (1, 1, 0)
    # aggregate report over a multi-polygon image
    rep = M.evaluate_polygons({"a": [doc_a]}, {"a": [doc_a, doc_b]})
    assert rep["iou"][0.9] == {**rep["iou"][0.9], "TP": 1, "FP": 0, "FN": 1}
    assert rep["gt_instances"] == 2 and rep["pred_instances"] == 1

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

    # tiny blob dropped by min_area_frac; mode="largest" keeps one contour
    noisy = mask.copy()
    noisy[5:10, 5:10] = 1
    assert len(P.extract_polygons(noisy, mode="all")) == 1
    assert len(P.extract_polygons(noisy, mode="all", min_area_frac=0.0)) == 2
    assert len(P.extract_polygons(noisy, mode="largest", min_area_frac=0.0)) == 1

    # empty mask -> empty list (serializes to "[]")
    assert P.extract_polygons(np.zeros((S, S), np.uint8)) == []
    print("test_postprocess_roundtrip OK")


def test_postprocess_multi_instance():
    S = 384
    # two real documents + one low-confidence blob + one speckle
    prob = np.full((S, S), 0.05, np.float32)
    prob[40:180, 40:180] = 0.95          # doc A (large, confident)
    prob[220:340, 200:360] = 0.90        # doc B (smaller, confident)
    prob[300:330, 40:90] = 0.42          # noise: passes thr 0.35, mean < 0.5
    prob[10:14, 350:354] = 0.99          # speckle: confident but ~16 px

    expected = [square(40 / S, 40 / S, 180 / S, 180 / S),
                square(200 / S, 220 / S, 360 / S, 340 / S)]

    # new default pipeline: both documents, nothing else
    polys = P.prob_to_polygons(prob, threshold=0.35, mode="all",
                               min_area_frac=0.005, min_mean_prob=0.5)
    assert len(polys) == 2, len(polys)
    rep = M.evaluate_polygons({"a": polys}, {"a": expected})
    assert rep["iou"][0.9] == {**rep["iou"][0.9], "TP": 2, "FP": 0, "FN": 0}

    # legacy mode: recall capped at the single largest document
    largest = P.prob_to_polygons(prob, threshold=0.35, mode="largest",
                                 min_area_frac=0.005, min_mean_prob=0.5)
    assert len(largest) == 1
    assert M.match_counts(largest, expected, 0.9) == (1, 0, 1)

    # disabling the mean-prob filter lets the low-confidence blob through
    loose = P.prob_to_polygons(prob, threshold=0.35, mode="all",
                               min_area_frac=0.005, min_mean_prob=0.0)
    assert len(loose) == 3
    # the speckle only appears once the area floor is dropped too
    assert len(P.prob_to_polygons(prob, threshold=0.35, mode="all",
                                  min_area_frac=0.0, min_mean_prob=0.0)) == 4
    print("test_postprocess_multi_instance OK")


if __name__ == "__main__":
    test_metrics()
    test_postprocess_roundtrip()
    test_postprocess_multi_instance()
    print("ALL TESTS PASSED")
