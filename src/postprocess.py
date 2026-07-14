"""Probability mask -> normalized submission polygons (multi-instance).

The test set averages ~1.5 documents per image, so the default pipeline keeps
EVERY connected component that passes two filters, instead of only the largest
contour:

    * area  >= min_area_frac of the mask       (drops speckles)
    * mean predicted probability inside the
      blob >= min_mean_prob                    (drops low-confidence noise that
                                                survives a low threshold,
                                                without killing small real
                                                documents the model is sure of)

mode="largest" restores the old single-document behavior for comparison.

Coordinate mapping (the critical pitfall): contour points from a
`img_size x img_size` mask are pixel indices; we add 0.5 (pixel center) and
divide by the mask size, landing in [0, 1]. Because the network input was the
*whole* image resized square (no letterboxing / cropping), normalized
coordinates transfer directly back onto the original image regardless of its
aspect ratio. infer.plot_overlays draws the result on original images to
verify exactly this.
"""
from typing import List, Optional

import cv2
import numpy as np

# list of polygons, each a list of normalized [x, y] points (same as data.Polygons;
# not imported from there so this module stays torch-free and unit-testable)
Polygons = List[List[List[float]]]

MODES = ("all", "largest")


def prob_to_mask(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prob > threshold).astype(np.uint8)


def component_masks(prob: np.ndarray, threshold: float = 0.5,
                    min_area_frac: float = 0.005,
                    min_mean_prob: float = 0.5,
                    mode: str = "all") -> List[np.ndarray]:
    """Connected components of the thresholded prob map that pass the area and
    mean-probability filters, as boolean masks. mode="largest" keeps only the
    biggest survivor (legacy single-document behavior)."""
    assert mode in MODES, mode
    binary = prob_to_mask(prob, threshold)
    H, W = binary.shape
    num_labels, labels = cv2.connectedComponents(binary)

    min_area = min_area_frac * H * W
    kept = []
    for label in range(1, num_labels):                    # 0 is background
        component = labels == label
        area = int(component.sum())
        if area < min_area:
            continue
        if min_mean_prob > 0 and float(prob[component].mean()) < min_mean_prob:
            continue
        kept.append((area, component))

    if mode == "largest" and kept:
        kept = [max(kept, key=lambda t: t[0])]
    return [component for _, component in kept]


def _contour_to_polygon(contour, H: int, W: int, method: str, eps_frac: float,
                        pixel_offset: float) -> Optional[list]:
    """One pixel-space contour -> one normalized polygon (or None if degenerate).

    method       : "approx"  - cv2.approxPolyDP with eps = eps_frac * perimeter
                   "minrect" - cv2.minAreaRect (forces a 4-point quad)
    pixel_offset : added to contour coords before normalizing (0.5 = pixel
                   centers; symmetric, avoids the 1-px shrink of raw indices).
    """
    assert method in ("approx", "minrect"), method
    if method == "approx":
        eps = eps_frac * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, eps, closed=True)
        if len(approx) < 3:
            return None
        pts = approx.reshape(-1, 2).astype(np.float32)
    else:
        pts = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)

    pts = (pts + pixel_offset) / np.array([W, H], np.float32)
    pts = np.clip(pts, 0.0, 1.0)
    return [[round(float(x), 5), round(float(y), 5)] for x, y in pts]


def prob_to_polygons(prob: np.ndarray, threshold: float = 0.5,
                     mode: str = "all",
                     min_area_frac: float = 0.005,
                     min_mean_prob: float = 0.5,
                     method: str = "approx",
                     eps_frac: float = 0.005,
                     pixel_offset: float = 0.5) -> Polygons:
    """Probability map -> list of normalized polygons, one per kept component
    (possibly empty; serializes to "[]"). Each component's outer contour is
    simplified separately."""
    H, W = prob.shape
    polygons = []
    for component in component_masks(prob, threshold, min_area_frac,
                                     min_mean_prob, mode):
        contours, _ = cv2.findContours(component.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        poly = _contour_to_polygon(contour, H, W, method, eps_frac, pixel_offset)
        if poly:
            polygons.append(poly)
    return polygons


def extract_polygons(binary: np.ndarray, mode: str = "all",
                     min_area_frac: float = 0.005,
                     method: str = "approx", eps_frac: float = 0.005,
                     pixel_offset: float = 0.5) -> Polygons:
    """Binary-mask entry point: same geometry pipeline, no probability filter
    (a {0,1} mask carries no confidence, so min_mean_prob is moot)."""
    return prob_to_polygons(binary.astype(np.float32), threshold=0.5,
                            mode=mode, min_area_frac=min_area_frac,
                            min_mean_prob=0.0, method=method,
                            eps_frac=eps_frac, pixel_offset=pixel_offset)
