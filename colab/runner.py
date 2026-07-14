# Colab runner — copy each "# %%" cell into a Colab notebook cell (or open this
# file in Colab via jupytext/VS Code, which understand the markers).
# Runtime: GPU (T4). Cells are ordered; each is verifiable before the next.

# %% [1] Install dependencies
# albumentations pinned to 1.4.x: the 2.x line renamed transform arguments.
# !pip -q install kagglehub shapely "albumentations==1.4.24"

# %% [2] Clone the pipeline repo  (EDIT THE URL after you push)
REPO_URL = "https://github.com/<your-user>/hv-doc-sg.git"
# !git clone $REPO_URL /content/hv-doc-sg

# %% [3] Download the dataset from Kaggle
import kagglehub
DATASET_BASE = kagglehub.dataset_download("yashvardhangera/hv-doc-data")
print("dataset at:", DATASET_BASE)

# %% [4] Paths + imports
import sys
REPO = "/content/hv-doc-sg"
sys.path.insert(0, REPO)                  # makes `src` and `seg_common` importable

from src import consensus as C
from src import data as D
from src import infer as I
from src import metrics as M
from src import train as T

cfg = D.Config(
    data_root=D.resolve_data_root(DATASET_BASE),
    img_size=384,
    cache_dir="/content/img_cache",       # resized copies; big epoch speedup
)
print("data root:", cfg.data_root)

# %% [5] Build the resized image cache (one-time, ~5-8 min, CPU only)
D.prepare_image_cache(cfg)

# %% [6] STEP 2 — annotator-agreement analysis (three rounds, same 5000 images)
rounds = D.load_all_rounds(cfg)
train_names, val_names = D.train_val_split(sorted(rounds[1].keys()),
                                           cfg.val_size, cfg.seed)
print(f"{len(train_names)} train / {len(val_names)} val")

iou_df = C.pairwise_iou_table(rounds)           # exact shapely IoU, ~1 min
print(C.agreement_summary(iou_df))
C.plot_agreement_histogram(iou_df)
C.visualize_worst(iou_df, rounds, cfg, k=5)

# %% [7] STEP 3a — overfit-10 sanity check (~2-3 min; F1@0.90 should -> 1.0)
T.train(cfg, epochs=40, batch_size=4, overfit=10, out_dir="/content/overfit",
        num_workers=2)

# %% [8] STEP 3b — full training on soft consensus masks
# 12 epochs / batch 16 @ 384 on a T4 is roughly 25-35 min with the cache.
model, history = T.train(cfg, epochs=12, batch_size=16, label_mode="soft",
                         out_dir="/content/outputs", num_workers=2)

# %% [9] STEP 4 — cache val probabilities, sweep threshold + postprocess
import torch
model, ckpt = I.load_model("/content/outputs/best.pt")
print("best epoch:", ckpt["epoch"])
M.print_report("best ckpt (thr 0.5)", ckpt["report"])

val_names, val_gt = I.val_split_with_gt(cfg)
device = "cuda" if torch.cuda.is_available() else "cpu"
val_probs = T.predict_probs(model, cfg, val_names, device)

sweep = I.sweep_thresholds(val_probs, val_gt)
display(sweep)
BEST_THR = float(sweep.loc[sweep["F1@0.90"].idxmax(), "threshold"])

post_cmp = I.compare_postprocess(val_probs, val_gt, BEST_THR)
display(post_cmp)
# pick the winning postprocess config from the table:
POST = dict(method="approx", eps_frac=0.01)

# %% [10] Verify coordinate mapping: predictions on ORIGINAL val images
from src import postprocess as P
import numpy as np
val_preds = {n: P.prob_to_polygons(p.astype(np.float32), BEST_THR, **POST)
             for n, p in val_probs.items()}
I.plot_overlays(cfg, val_preds, val_names[:5], gts=val_gt)

# %% [11] STEP 5 — generate + validate pred.csv
I.generate_pred_csv(model, cfg, "/content/pred.csv", BEST_THR, **POST)
I.validate_pred_csv("/content/pred.csv", cfg)   # asserts + 5 random overlays

# %% [12] Download
# from google.colab import files; files.download("/content/pred.csv")
