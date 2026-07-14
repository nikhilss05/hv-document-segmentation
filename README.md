# hv-doc-sg — document segmentation pipeline

Predict the polygon marking a document's boundary in a photo. Primary metric:
instance P/R/F1 @ IoU ≥ 0.90. See [assignment.md](assignment.md).

## Layout

| file | what |
|---|---|
| `src/data.py` | Config/paths, label CSV loading, fixed 4500/500 split, soft-mask datasets, augmentation, image cache |
| `src/metrics.py` | instance P/R/F1 @ IoU 0.90 & 0.50 via exact shapely polygon IoU, pixel Dice |
| `src/consensus.py` | pairwise agreement between the 3 annotation rounds, worst-case visualization, medoid GT, soft/majority consensus masks |
| `src/train.py` | U-Net (`seg_common.py`), BCE+Dice on soft targets, cosine LR, AMP, checkpoint on val F1@0.90, `--overfit N` sanity mode |
| `src/postprocess.py` | prob mask → connected components filtered by area + mean inside-probability (multi-document; `mode="largest"` = legacy) → approxPolyDP / minAreaRect → normalized polygons |
| `src/infer.py` | threshold sweep, postprocess comparison, overlay checks, `pred.csv` generation + validation |
| `colab/runner.py` | ordered Colab cells: installs → kagglehub → cache → consensus → sanity → train → tune → pred.csv |

## Label-noise handling

The three `train_round_*.csv` are independent annotators over the same images.

- **Training targets**: soft masks = mean of the three rasterized annotations
  (pixel values 0 / ⅓ / ⅔ / 1), consumed directly by BCE+Dice. `--label-mode`
  also supports `majority`, `medoid`, `round{1,2,3}` for ablations.
- **Validation GT**: per image, the *medoid* annotation (highest mean IoU to
  the other two) — drops the outlier annotator, keeps clean polygon instances
  for F1@0.90.

## Run on Colab

Push this repo to GitHub, then follow the cells in `colab/runner.py`
(clone → `kagglehub.dataset_download("yashvardhangera/hv-doc-data")` → run).
All paths flow through `src.data.Config`; nothing assumes local data.
