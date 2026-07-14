"""Shared code for the document-segmentation U-Net task.

Goal: given a photo of a document, predict a binary mask that marks the
document pixels (foreground) against the background.

What lives in this module:
    * DoubleConv / UNet        - the segmentation network.
    * Dataset classes          - load an image + its target mask and run them
                                 through an albumentations transform (which
                                 must end in Normalize + ToTensorV2).
    * dice_coeff / dice_loss   - the overlap metric and its differentiable loss.
    * preprocess / save_*      - inference helpers and prediction export.

Data layout:
    <DATA_ROOT>/train/images/*.jpg  + seg_maps/*.png   (train image + its mask)
    <DATA_ROOT>/test/images/*.jpg
Train targets come from one of train_round_1.csv / train_round_2.csv /
train_round_3.csv - pick one round to train on.

Prediction "standard format": <PRED_ROOT>/<model>/<image_stem>.png, uint8
{0,255}, IMG_SIZE x IMG_SIZE.  Students submit a pred.csv over the test images.
"""
import csv
import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --- absolute paths so the notebooks/scripts work from any working directory --
DATA_ROOT = "/media/beegfs/users/yashvardhan.g/HIRING_TASK/dataset"          # images + seg_maps
TRAIN_CSV = "/media/beegfs/users/yashvardhan.g/HIRING_TASK/prepared_dataset/train_round_1.csv"  # pick round 1, 2, or 3
PRED_ROOT = "/media/beegfs/users/yashvardhan.g/HIRING_TASK/predictions"      # where mask PNGs are written
CKPT_ROOT = "/media/beegfs/users/yashvardhan.g/HIRING_TASK/training/checkpoints"  # saved model weights

IMG_SIZE = 256                       # square working resolution (divisible by 8: the U-Net pools 3x)
IOU_THR  = 0.75                      # overlap ratio to count a predicted instance as a match
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = (0.485, 0.456, 0.406)         # ImageNet channel means used to normalise inputs
STD  = (0.229, 0.224, 0.225)         # ImageNet channel std-devs


# ------------------------------- model --------------------------------------
class DoubleConv(nn.Module):
    """The basic U-Net block: (3x3 conv -> BatchNorm -> ReLU) applied twice.

    `padding=1` keeps the height/width unchanged, so the block only changes the
    channel count (from `in_ch` to `out_ch`) while mixing in local context."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            # first conv: in_ch -> out_ch
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # second conv: out_ch -> out_ch
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small 4-level U-Net (base=32 -> ~1.9M parameters).

    Encoder-decoder with skip connections:
      * the encoder halves the resolution and doubles the channels four times;
      * the decoder upsamples back to full resolution, and at each step
        concatenates the matching encoder feature map (the "skip") so the fine
        spatial detail lost during pooling is recovered.
    The output is a single-channel logit map (before sigmoid) at input size."""

    def __init__(self, base=32):
        super().__init__()

        # ---- encoder (contracting path) ----
        self.d1 = DoubleConv(3, base)               # RGB in     -> base channels
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)     # deepest level (bottleneck)
        self.pool = nn.MaxPool2d(2)                  # halves height and width

        # ---- decoder (expanding path) ----
        # each ConvTranspose doubles the resolution; the following DoubleConv
        # then fuses the upsampled features with the concatenated encoder skip,
        # so its input channel count is (upsampled + skip).
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.u3 = DoubleConv(base * 8, base * 4)     # base*8 = up3 (base*4) + skip enc3 (base*4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.u1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, kernel_size=1)  # 1x1 conv -> one logit per pixel

    def forward(self, x):
        # ---- encoder: keep each level's output so the decoder can use it ----
        enc1 = self.d1(x)                       # full resolution
        enc2 = self.d2(self.pool(enc1))         # 1/2
        enc3 = self.d3(self.pool(enc2))         # 1/4
        bottleneck = self.d4(self.pool(enc3))   # 1/8 (deepest features)

        # ---- decoder: upsample, concatenate the matching skip, then fuse ----
        dec3 = self.u3(torch.cat([self.up3(bottleneck), enc3], dim=1))
        dec2 = self.u2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.u1(torch.cat([self.up1(dec2), enc1], dim=1))

        return self.out(dec1)                   # (N, 1, H, W) logits


# ------------------------------- data ---------------------------------------
def list_train(limit=None):
    """Return (image_path, mask_path) pairs for the train split, keeping only the
    images that have a matching seg-map on disk. `limit` truncates the list."""
    image_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "train", "images", "*.jpg")))

    pairs = []
    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(DATA_ROOT, "train", "seg_maps", stem + ".png")
        if os.path.exists(mask_path):
            pairs.append((image_path, mask_path))

    if limit:
        return pairs[:limit]
    return pairs


class SegDataset(Dataset):
    """Train dataset backed by (image, seg-map) pairs on disk.

    Each item is read from disk, passed through the albumentations `transform`
    (which must end in Normalize + ToTensorV2), and the mask is binarised to a
    float {0,1} tensor with a leading channel axis, so it matches the model's
    (N, 1, H, W) output."""

    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        image_path, mask_path = self.pairs[i]

        # OpenCV loads images as BGR, so convert to RGB.
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=image, mask=mask)

        # mask: (H, W) uint8 -> (1, H, W) float tensor of 0.0 / 1.0
        mask_tensor = (transformed["mask"] > 127).float().unsqueeze(0)
        return transformed["image"], mask_tensor






# ------------------------- CSV-driven data (train.csv) ----------------------
def load_csv_records(csv_path):
    """Read a prepared_dataset CSV into a list of records, each a dict:
        {"image_name", "image_path", "polygons"}
    `polygons` is the list of normalized document polygons for the image; the
    split (train/test) is inferred from the image-name prefix."""
    import csv as _csv

    records = []
    for row in _csv.DictReader(open(csv_path)):
        image_name = row["image"]
        split = "train" if image_name.startswith("train_") else "test"
        records.append({
            "image_name": image_name,
            "image_path": os.path.join(DATA_ROOT, split, "images", image_name),
            "polygons": json.loads(row["polygon"]),       # the polygon cell is a JSON list
        })
    return records


def polys_to_mask(polygons, H, W):
    """Rasterise a list of normalized polygons into one union binary (0/255)
    mask of size H x W. Each polygon's [x, y] in [0, 1] is scaled to pixels."""
    mask = np.zeros((H, W), np.uint8)
    for poly in polygons:
        if poly and len(poly) >= 3:                       # need at least 3 points for a polygon
            pts = (np.array(poly, np.float32) * np.array([W, H])).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


class CsvSegDataset(Dataset):
    """Train/val dataset whose target masks are rasterised on the fly from the
    polygon column of the CSV (one image per record)."""

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def _raw(self, i):
        """Load the raw (RGB image, rasterised grayscale mask) for record i."""
        record = self.records[i]
        image = cv2.cvtColor(cv2.imread(record["image_path"]), cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        mask = polys_to_mask(record["polygons"], H, W)    # mask at the image's own resolution
        return image, mask

    def __getitem__(self, i):
        image, mask = self._raw(i)
        transformed = self.transform(image=image, mask=mask)
        mask_tensor = (transformed["mask"] > 127).float().unsqueeze(0)
        return transformed["image"], mask_tensor




# --------------- instance Precision/Recall/F1 @ IoU (shared metric) ----------
def polys_to_instances(polygons, size=None):
    """Rasterise polygons to a LIST of per-instance boolean masks (size x size),
    one mask per polygon. Unlike polys_to_mask these are kept separate, so each
    instance can be matched one-to-one during scoring."""
    if size is None:
        size = IMG_SIZE

    instances = []
    for poly in polygons:
        if poly and len(poly) >= 3:
            mask = np.zeros((size, size), np.uint8)
            pts = (np.array(poly, np.float32) * size).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
            instances.append(mask.astype(bool))
    return instances


def mask_to_instances(binary, min_area=80):
    """Split a predicted binary union mask into separate instances via connected
    components, dropping blobs smaller than `min_area` pixels (noise)."""
    num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))

    instances = []
    for label in range(1, num_labels):                    # label 0 is the background
        component = labels == label
        if int(component.sum()) >= min_area:
            instances.append(component)
    return instances


def _iou(a, b):
    """Intersection-over-union of two boolean masks (0 if the union is empty)."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def match_counts(preds, gts, thr):
    """Greedy one-to-one matching between predicted and GT instances: each GT
    claims the best still-unused prediction with IoU >= thr.

    Returns (TP, FP, FN) = (matched, unmatched predictions, unmatched GTs)."""
    matched = set()
    true_positives = 0

    for gt in gts:
        best_iou = -1.0
        best_j = -1
        for j, pred in enumerate(preds):
            if j in matched:
                continue                                  # a prediction matches at most one GT
            iou = _iou(pred, gt)
            if iou >= thr and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            true_positives += 1
            matched.add(best_j)

    false_positives = len(preds) - len(matched)           # predictions that matched nothing
    false_negatives = len(gts) - true_positives           # GTs that were missed
    return true_positives, false_positives, false_negatives


def seg_prf_report(preds_list, gts_list, iou_thrs=(IOU_THR, 0.5)):
    """Aggregate instance Precision/Recall/F1 at each IoU threshold across all
    images, plus the pixel-union Dice. `preds_list` / `gts_list` are per-image
    lists of boolean instance masks."""
    report = {"images": len(gts_list), "iou": {}}

    # ---- instance-level Precision / Recall / F1, summed over all images ----
    for thr in iou_thrs:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for preds, gts in zip(preds_list, gts_list):
            tp, fp, fn = match_counts(preds, gts, thr)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        report["iou"][thr] = {
            "P": precision, "R": recall, "F1": f1,
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
        }

    # ---- pixel Dice on the union of all instances per image (instance-agnostic) ----
    dices = []
    for preds, gts in zip(preds_list, gts_list):
        if not preds and not gts:
            dices.append(1.0)                             # both empty -> perfect agreement
            continue

        pred_union = np.zeros((IMG_SIZE, IMG_SIZE), bool)
        for pred in preds:
            pred_union |= pred

        gt_union = np.zeros((IMG_SIZE, IMG_SIZE), bool)
        for gt in gts:
            gt_union |= gt

        intersection = np.logical_and(pred_union, gt_union).sum()
        dice = (2 * intersection + 1e-6) / (pred_union.sum() + gt_union.sum() + 1e-6)
        dices.append(dice)

    report["dice_mean"] = float(np.mean(dices))
    report["gt_instances"] = int(sum(len(g) for g in gts_list))
    report["pred_instances"] = int(sum(len(p) for p in preds_list))
    return report


def print_report(name, rep, primary=IOU_THR):
    """Pretty-print a seg_prf_report dict, flagging the primary IoU threshold."""
    print(f"\n=== {name}  (n={rep['images']} imgs | GT {rep['gt_instances']} / "
          f"pred {rep['pred_instances']} instances) ===")
    for thr, m in rep["iou"].items():
        star = "  <- primary" if abs(thr - primary) < 1e-9 else ""
        print(f"  IoU>={thr:.2f}   Precision {m['P']:.4f}   Recall {m['R']:.4f}   "
              f"F1 {m['F1']:.4f}   (TP {m['TP']}  FP {m['FP']}  FN {m['FN']}){star}")
    print(f"  pixel Dice (union): {rep['dice_mean']:.4f}")


# ------------------------------- metrics ------------------------------------
def dice_coeff(pred_bin, gt_bin, eps=1e-6):
    """Dice overlap of two binary arrays (or tensors): 2|A n B| / (|A| + |B|).
    1.0 = identical, 0.0 = disjoint. `eps` avoids 0/0 on empty masks."""
    pred = np.asarray(pred_bin).astype(bool)
    gt = np.asarray(gt_bin).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return float((2 * intersection + eps) / (pred.sum() + gt.sum() + eps))


def dice_loss(logits, target, eps=1e-6):
    """Differentiable (soft) Dice loss = 1 - Dice, computed per sample on the
    sigmoid probabilities and averaged over the batch. It is combined with BCE
    during training to pair a pixel-wise term with an overlap term."""
    prob = torch.sigmoid(logits)                          # logits -> probabilities in [0, 1]
    numerator = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps       # soft intersection, per sample
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps  # soft area sum
    dice_per_sample = numerator / denominator
    return (1 - dice_per_sample).mean()


# ------------------------- inference / predictions --------------------------
def preprocess(img_rgb):
    """Turn an RGB image into the model's input tensor: resize to IMG_SIZE,
    scale to [0, 1], apply ImageNet Normalize, and reorder to (1, 3, H, W)."""
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    normalised = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    chw = normalised.transpose(2, 0, 1)                   # HWC -> CHW
    return torch.from_numpy(chw).unsqueeze(0).float()     # add the batch axis -> 1CHW


@torch.no_grad()
def save_test_predictions(model, pred_dir, thresh=0.5):
    """Run the model over every test image and save one binary mask PNG per image
    (IMG_SIZE, values {0, 255}) into `pred_dir` - the standard format test.py
    reads. Returns the number of images written."""
    os.makedirs(pred_dir, exist_ok=True)
    model.eval().to(DEVICE)

    image_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "test", "images", "*.jpg")))
    for image_path in image_paths:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        x = preprocess(image).to(DEVICE)
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        mask = (prob > thresh).astype(np.uint8) * 255     # {0,1} -> {0,255} for a viewable PNG

        stem = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(pred_dir, stem + ".png"), mask)

    return len(image_paths)


# ------------------------------ submission format ---------------------------
def mask_to_polygons(binary, size=None, min_area=80, epsilon_frac=0.005):
    """Convert a binary mask into a list of normalized polygons - the format the
    submission CSV expects in its `polygon` column.

    One polygon per external contour: contours smaller than `min_area` pixels are
    dropped as noise, each contour is simplified with approxPolyDP (tolerance =
    `epsilon_frac` of its perimeter), and its points are divided by `size` so the
    coordinates land in [0, 1]. Returns a list of polygons, each itself a list of
    [x, y] points."""
    if size is None:
        size = IMG_SIZE

    contours, _ = cv2.findContours(binary.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue                                      # skip tiny noise blobs
        epsilon = epsilon_frac * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) < 3:
            continue                                      # need at least 3 points for a polygon
        points = approx.reshape(-1, 2).astype(np.float32) / size   # pixels -> normalized [0, 1]
        polygons.append([[round(float(x), 5), round(float(y), 5)] for x, y in points])
    return polygons
