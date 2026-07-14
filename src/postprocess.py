"""Probability mask -> normalized submission polygons.

Coordinate mapping (the critical pitfall): contour points from a
`img_size x img_size` mask are pixel indices; we add 0.5 (pixel center) and
divide by the mask size, landing in [0, 1]. Because the network input was the
*whole* image resized square (no letterboxing / cropping), normalized
coordinates transfer directly back onto the original image regardless of its
aspect ratio. infer.plot_overlays draws the result on original images to
verify exactly this.
"""
from typing import List

import cv2
import numpy as np

# list of polygons, each a list of normalized [x, y] points (same as data.Polygons;
# not imported from there so this module stays torch-free and unit-testable)
Polygons = List[List[List[float]]]


def prob_to_mask(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prob > threshold).astype(np.uint8)


def extract_polygons(binary: np.ndarray,
                     largest_only: bool = True,
                     min_area_frac: float = 0.01,
                     method: str = "approx",
                     eps_frac: float = 0.01,
                     pixel_offset: float = 0.5) -> Polygons:
    """Binary mask -> list of normalized polygons (possibly empty).

    largest_only  : keep only the biggest contour (one document per photo).
    min_area_frac : drop blobs smaller than this fraction of the mask area.
    method        : "approx"  - cv2.approxPolyDP with eps = eps_frac * perimeter
                    "minrect" - cv2.minAreaRect (forces a 4-point quad)
    pixel_offset  : added to contour coords before normalizing (0.5 = pixel
                    centers; symmetric, avoids the 1-px shrink of raw indices).
    """
    assert method in ("approx", "minrect")
    H, W = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = min_area_frac * H * W
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if largest_only and contours:
        contours = [max(contours, key=cv2.contourArea)]

    polygons = []
    for contour in contours:
        if method == "approx":
            eps = eps_frac * cv2.arcLength(contour, closed=True)
            approx = cv2.approxPolyDP(contour, eps, closed=True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2).astype(np.float32)
        else:
            pts = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)

        pts = (pts + pixel_offset) / np.array([W, H], np.float32)
        pts = np.clip(pts, 0.0, 1.0)
        polygons.append([[round(float(x), 5), round(float(y), 5)] for x, y in pts])
    return polygons


def prob_to_polygons(prob: np.ndarray, threshold: float = 0.5,
                     **extract_kwargs) -> Polygons:
    """Convenience: threshold + extract in one call."""
    return extract_polygons(prob_to_mask(prob, threshold), **extract_kwargs)
