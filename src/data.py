"""Data access for the document-segmentation task.

Nothing here assumes local data: every path derives from a Config whose
`data_root` points at the folder that contains `images/` and `labels/`
(on Colab this comes from kagglehub, see colab/runner.py).

Provides:
    * Config / resolve_data_root      - path handling.
    * load_round / load_all_rounds    - parse the three annotator CSVs.
    * train_val_split                 - fixed-seed 4500/500 split.
    * rasterize_counts                - per-pixel annotator vote counts (0..3).
    * prepare_image_cache             - one-time downscale of photos (speed).
    * SoftDocDataset                  - training targets: soft (mean-of-rounds)
                                        or majority-vote masks.
    * InferenceDataset                - image tensor + name, no targets.
    * build_transforms                - albumentations pipelines (aug is
                                        identical for image+mask; masks use
                                        NEAREST interpolation by default).
"""
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

ROUND_IDS = (1, 2, 3)

# A "polygons" value is the parsed content of one CSV `polygon` cell:
# a list of polygons, each a list of [x, y] points normalized to [0, 1].
Polygons = List[List[List[float]]]
Round = Dict[str, Polygons]  # image_name -> polygons


@dataclass
class Config:
    data_root: str                    # contains images/{train,test} and labels/
    img_size: int = 384               # square working resolution (divisible by 8)
    seed: int = 42
    val_size: int = 500
    cache_dir: Optional[str] = None   # optional resized-image cache (much faster epochs)
    cache_max_side: int = 768

    @property
    def labels_dir(self) -> str:
        return os.path.join(self.data_root, "labels")

    def images_dir(self, split: str) -> str:
        return os.path.join(self.data_root, "images", split)


def resolve_data_root(base: str) -> str:
    """kagglehub returns the dataset top folder; the data lives under
    student/student. Accept any of the three levels."""
    for cand in (base,
                 os.path.join(base, "student"),
                 os.path.join(base, "student", "student")):
        if os.path.isdir(os.path.join(cand, "images", "train")):
            return cand
    raise FileNotFoundError(f"no images/train under {base} (or its student/ subfolders)")


# ------------------------------- labels -------------------------------------
def load_round(csv_path: str) -> Round:
    """One annotator CSV -> {image_name: polygons}."""
    df = pd.read_csv(csv_path)
    return {row.image: json.loads(row.polygon) for row in df.itertuples()}


def load_all_rounds(cfg: Config) -> Dict[int, Round]:
    """All three annotation passes, keyed by round id (1, 2, 3)."""
    return {r: load_round(os.path.join(cfg.labels_dir, f"train_round_{r}.csv"))
            for r in ROUND_IDS}


def train_val_split(names: List[str], val_size: int = 500, seed: int = 42):
    """Deterministic split of image names -> (train_names, val_names)."""
    names = sorted(names)
    rng = random.Random(seed)
    rng.shuffle(names)
    val = sorted(names[:val_size])
    train = sorted(names[val_size:])
    return train, val


# ------------------------------- images -------------------------------------
def split_of(name: str) -> str:
    return "train" if name.startswith("train_") else "test"


def image_path(cfg: Config, name: str) -> str:
    """Prefer the resized cache copy when it exists."""
    if cfg.cache_dir:
        cached = os.path.join(cfg.cache_dir, split_of(name), name)
        if os.path.exists(cached):
            return cached
    return os.path.join(cfg.images_dir(split_of(name)), name)


def load_image(cfg: Config, name: str) -> np.ndarray:
    """RGB uint8 image."""
    path = image_path(cfg, name)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def prepare_image_cache(cfg: Config, splits=("train", "test"), jpeg_quality=92,
                        log_every=1000) -> None:
    """One-time downscale of every photo to `cache_max_side` on the long edge.

    Masks are rasterized from *normalized* polygons at whatever resolution the
    loaded image has, so downscaling is label-safe. Cuts JPEG-decode time per
    epoch by roughly an order of magnitude on Colab."""
    assert cfg.cache_dir, "set cfg.cache_dir first"
    for split in splits:
        src_dir = cfg.images_dir(split)
        dst_dir = os.path.join(cfg.cache_dir, split)
        os.makedirs(dst_dir, exist_ok=True)
        names = sorted(os.listdir(src_dir))
        done = 0
        for name in names:
            dst = os.path.join(dst_dir, name)
            if not os.path.exists(dst):
                img = cv2.imread(os.path.join(src_dir, name))
                h, w = img.shape[:2]
                scale = cfg.cache_max_side / max(h, w)
                if scale < 1.0:
                    img = cv2.resize(img, (round(w * scale), round(h * scale)),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            done += 1
            if log_every and done % log_every == 0:
                print(f"  cache {split}: {done}/{len(names)}")


# ------------------------------- masks --------------------------------------
def polys_to_mask(polygons: Polygons, H: int, W: int, value: int = 1) -> np.ndarray:
    """Union of normalized polygons -> uint8 mask of `value` on H x W."""
    mask = np.zeros((H, W), np.uint8)
    for poly in polygons:
        if poly and len(poly) >= 3:
            pts = (np.asarray(poly, np.float32) * np.array([W, H])).astype(np.int32)
            cv2.fillPoly(mask, [pts], value)
    return mask


def rasterize_counts(polys_per_round: List[Polygons], H: int, W: int) -> np.ndarray:
    """Per-pixel count of annotators that marked the pixel as document,
    uint8 in {0..len(polys_per_round)}. Divide by the round count to get the
    soft consensus mask (0 / 0.33 / 0.67 / 1.0 for three rounds)."""
    counts = np.zeros((H, W), np.uint8)
    for polys in polys_per_round:
        counts += polys_to_mask(polys, H, W, value=1)
    return counts


# ----------------------------- transforms -----------------------------------
def build_transforms(img_size: int, train: bool):
    """Albumentations pipelines. Geometric transforms apply the *same* warp to
    image and mask, with NEAREST interpolation on the mask (albumentations
    default) so soft-label levels stay discrete. Pinned to albumentations
    1.4.x in colab/runner.py (arg names changed in 2.x)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if train:
        steps = [
            A.Resize(img_size, img_size),
            A.Perspective(scale=(0.02, 0.08), p=0.5),
            A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.25,
                                       contrast_limit=0.25, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                 val_shift_limit=10, p=0.3),
        ]
    else:
        steps = [A.Resize(img_size, img_size)]
    steps += [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    return A.Compose(steps)


# ------------------------------ datasets ------------------------------------
class SoftDocDataset(Dataset):
    """Training/val dataset with configurable targets built from one or more
    label sets over the same images.

    label_sets: list of {image_name: polygons} dicts (e.g. the 3 rounds, or a
                single consensus dict).
    label_mode:
        "soft"     - target = fraction of sets marking the pixel (soft mask).
        "majority" - target = 1 where more than half of the sets agree.
    With a single label set both modes reduce to the plain binary mask."""

    def __init__(self, cfg: Config, names: List[str], label_sets: List[Round],
                 transform, label_mode: str = "soft"):
        assert label_mode in ("soft", "majority")
        self.cfg = cfg
        self.names = list(names)
        self.label_sets = label_sets
        self.transform = transform
        self.label_mode = label_mode

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        image = load_image(self.cfg, name)
        H, W = image.shape[:2]
        polys_per_set = [ls.get(name, []) for ls in self.label_sets]
        counts = rasterize_counts(polys_per_set, H, W)

        out = self.transform(image=image, mask=counts)
        counts_t = out["mask"].float()
        n = len(self.label_sets)
        if self.label_mode == "soft":
            target = counts_t / n
        else:
            target = (counts_t > n / 2).float()
        return out["image"], target.unsqueeze(0)


class InferenceDataset(Dataset):
    """Image tensor + image name; no targets. For val/test prediction."""

    def __init__(self, cfg: Config, names: List[str], transform):
        self.cfg = cfg
        self.names = list(names)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        image = load_image(self.cfg, name)
        return self.transform(image=image)["image"], name


def list_test_images(cfg: Config) -> List[str]:
    return sorted(n for n in os.listdir(cfg.images_dir("test"))
                  if n.lower().endswith(".jpg"))
