"""Training: U-Net (from seg_common) on soft consensus masks.

    * loss        = BCE-with-logits + soft Dice (both accept soft targets,
                    so the 0/0.33/0.67/1.0 consensus masks work unchanged).
    * schedule    = AdamW + per-step cosine annealing, AMP on CUDA.
    * checkpoint  = best val instance F1 @ IoU 0.90 (tie-broken by F1@0.50
                    then Dice), evaluated with exact shapely polygon IoU
                    against the per-image medoid annotation.
    * sanity mode = --overfit N trains and validates on the same N images
                    with augmentation off; F1@0.90 should approach 1.0.

Run (see colab/runner.py for the Colab wiring):
    python -m src.train --data-root <root> [--cache-dir <dir>] [--overfit 10]
"""
import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import consensus as C
from . import data as D
from . import metrics as M
from . import postprocess as P

# seg_common.py ships in the dataset and is vendored at the repo root; its
# module-level path constants are unused here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seg_common import UNet, dice_loss  # noqa: E402


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def predict_probs(model, cfg: D.Config, names, device, batch_size=32,
                  num_workers=2) -> Dict[str, np.ndarray]:
    """Sigmoid probability maps (img_size x img_size, float16) per image."""
    ds = D.InferenceDataset(cfg, names, D.build_transforms(cfg.img_size, train=False))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    probs = {}
    for images, batch_names in loader:
        logits = model(images.to(device))
        p = torch.sigmoid(logits)[:, 0].cpu().numpy().astype(np.float16)
        for name, prob in zip(batch_names, p):
            probs[name] = prob
    return probs


def evaluate_model(model, cfg: D.Config, names, gts, device, threshold=0.5,
                   **post_kwargs) -> dict:
    """Predict -> postprocess to polygons -> shapely instance metrics."""
    probs = predict_probs(model, cfg, names, device)
    preds = {name: P.prob_to_polygons(prob.astype(np.float32), threshold,
                                      **post_kwargs)
             for name, prob in probs.items()}
    return M.evaluate_polygons(preds, {n: gts[n] for n in names})


def train(cfg: D.Config, epochs=12, batch_size=16, lr=3e-4, weight_decay=1e-4,
          label_mode="soft", overfit=0, out_dir="outputs", num_workers=2,
          val_threshold=0.5, amp=True, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    # ---- labels & split ----
    rounds = D.load_all_rounds(cfg)
    train_names, val_names = D.train_val_split(sorted(rounds[1].keys()),
                                               cfg.val_size, cfg.seed)
    if overfit:
        train_names = train_names[:overfit]
        val_names = train_names            # sanity: memorize and re-score
    print(f"train {len(train_names)} / val {len(val_names)} images | "
          f"labels={label_mode} | device={device}")

    print("building medoid validation GT ...")
    val_gt = C.medoid_annotations(rounds, names=val_names)

    if label_mode in ("soft", "majority"):
        label_sets = [rounds[r] for r in D.ROUND_IDS]
        ds_mode = label_mode
    elif label_mode == "medoid":
        label_sets = [C.medoid_annotations(rounds, names=train_names)]
        ds_mode = "soft"
    elif label_mode.startswith("round"):
        label_sets = [rounds[int(label_mode[-1])]]
        ds_mode = "soft"
    else:
        raise ValueError(f"unknown label_mode {label_mode}")

    train_tf = D.build_transforms(cfg.img_size, train=not overfit)
    train_ds = D.SoftDocDataset(cfg, train_names, label_sets, train_tf, ds_mode)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=len(train_ds) > batch_size)

    # ---- model / optim ----
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(1, len(train_loader)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device == "cuda")

    best_score, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                loss = bce_dice_loss(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

        report = evaluate_model(model, cfg, val_names, val_gt, device,
                                threshold=val_threshold)
        score = M.report_score(report)
        f1_hi = report["iou"][M.PRIMARY_IOU]["F1"]
        f1_lo = report["iou"][M.SECONDARY_IOU]["F1"]
        print(f"epoch {epoch:3d}/{epochs}  loss {np.mean(losses):.4f}  "
              f"val F1@0.90 {f1_hi:.4f}  F1@0.50 {f1_lo:.4f}  "
              f"dice {report['dice_mean']:.4f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}  "
              f"({time.time() - t0:.0f}s)")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "f1_090": f1_hi, "f1_050": f1_lo,
                        "dice": report["dice_mean"]})

        ckpt = {"model": model.state_dict(), "img_size": cfg.img_size,
                "epoch": epoch, "report": report, "label_mode": label_mode}
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        if best_score is None or score > best_score:
            best_score = score
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
            print(f"  -> new best (F1@0.90 {f1_hi:.4f}), saved best.pt")

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return model, history


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label-mode", default="soft",
                    choices=["soft", "majority", "medoid",
                             "round1", "round2", "round3"])
    ap.add_argument("--overfit", type=int, default=0,
                    help="sanity mode: train+val on the first N images, no aug")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args(argv)

    cfg = D.Config(data_root=D.resolve_data_root(args.data_root),
                   img_size=args.img_size, cache_dir=args.cache_dir)
    train(cfg, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          label_mode=args.label_mode, overfit=args.overfit,
          out_dir=args.out_dir, num_workers=args.workers, amp=not args.no_amp)


if __name__ == "__main__":
    main()
