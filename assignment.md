# 📄 Document Segmentation Assignment

Hello, ML Explorers! 👋

Welcome to the Document Segmentation challenge. You'll be building a real segmentation pipeline end-to-end

---

## 📚 The Resources at Your Disposal

- **Dataset (Kaggle):** yashvardhangera/hv-doc-data
- **Starter Colab notebook (HERE ):** provided alongside this page — it downloads the dataset straight from Kaggle via `kagglehub`, imports the `seg_common.py` that ships inside the dataset, and runs a baseline U-Net training loop end-to-end.
- `seg_common.py` (included in the dataset) contains the model definition, dataset classes, loss functions, and inference helpers you can take reference from.

(Psst — you can run multiple Colab notebooks in parallel under different Google accounts to try a few approaches at once while one trains 🤫)

---

## 🧠 Assignment Overview

This is a **single-stage** assignment — with each result considered as a submission, no separate phases.Although you can make multiple submission 

You are tasked with building a **document segmentation model**: given a image containing document, predict the polygon marking the document's boundary against the background.

- Train on the provided **labeled training images** (`images/train`, 5,000 images) using the polygon annotations in `train_round_1.csv /` `train_round_2.csv` / `train_round_3.csv`
- Each annotation file is a version of the training set labels created by a seperate annotator
- Use the **U-Net architecture provided in `seg_common.py`** (a small 4-level U-Net, ~1.9M params) as your sample model for training
- Generate predictions for every image in `images/test` (1,000 unlabeled images) and submit them as `pred.csv`

---

## 📁 Data Folder Structure

The Kaggle dataset unpacks to:

```
hv-doc-data/
└── student/
    └── student/
        ├── labels/
        │   ├── train_round_1.csv
        │   ├── train_round_2.csv
        │   └── train_round_3.csv
        ├── images/
        │   ├── train/
        │   │   ├── train_00000.jpg
        │   │   ├── train_00001.jpg
        │   │   └── ... (5,000 files total)
        │   └── test/
        │       ├── test_00000.jpg
        │       ├── test_00001.jpg
        │       └── ... (1,000 files total)
        ├── seg_common.py
        └── sample_train.ipynb
```

✅ Label CSV format (`image`, `polygon`, `corners`):

```
image,polygon,corners
train_00000.jpg,"[[[0.12,0.08],[0.91,0.10],[0.89,0.95],[0.10,0.93]]]",...
```

- `image` — the training image filename
- `polygon` — a JSON list of polygons, each a list of `[x, y]` points normalized to `[0, 1]`. One polygon per document instance in the image (rasterize this to get your training mask)
- `corners` — an additional annotation column, not required for the baseline pipeline

---

## 🧹 A Note on the Labels

The three `train_round_*.csv` files aren't three different data splits — they're three **independent annotation passes over the same 5,000 training images**, each done by a different human annotator.

Manual annotation isn't perfect. Deciding how to handle the version,  is something the research scientist has to figure out 🔭

---

---

## 🎯 Objective

Build a pipeline that segments the document region in a photo — predicting polygon per image marking document's boundary. Your final output is:

- A **CSV prediction file** (`pred.csv`) in the format below
- A **well-documented Colab Notebook**

---

## 📤 Submission Format

You must submit one file: `pred.csv`

```
image,polygon
test_00000.jpg,"[[[0.15,0.10],[0.88,0.12],[0.86,0.90],[0.13,0.89]]]"
test_00001.jpg,"[]"
...
```

📌 Ensure:

- You make predictions for **all 1,000 test images**
- `image` matches the exact filenames in `images/test`
- `polygon` is a JSON-encoded list of polygons (each a list of normalized `[x, y]` points) — use an empty list `[]` if no document is detected

---

## 📊 Evaluation Criteria

| Criteria | Description |
| --- | --- |
| ✅ **Instance Precision / Recall / F1 @ IoU ≥ 0.90** | Primary metric — how well predicted polygons match ground-truth document instances |
| ⚖️ **Instance Precision / Recall / F1 @ IoU ≥ 0.50** | Secondary, looser matching threshold |
| 🎯 **Pixel Dice** | Overlap between the unioned predicted mask and the ground-truth mask |

The evaluation labels for `images/test` are **not distributed** — scoring happens only on the leaderboard.

---

## ✅ Final Submission Checklist

- [ ]  `pred.csv` uploaded to the API
- [ ]  Colab notebook (or GitHub repo) shared with **view access**
- [ ]  Code is clean and modular
- [ ]  Predictions cover all 1,000 test images

---

## 📝 How to Submit

1. Train your model using the provided Colab notebook (or your own)
2. Generate `pred.csv` following the format above
3. Upload `pred.csv` on the API 
4. Share your Colab notebook link via *(submission form link to be added)*

---

## 🤝 Questions or Support?

If you hit any doubts or issues, reach out to the organizing team — we're here to help you succeed.

---

## 🚀 Now Go Build Something Awesome!

This assignment is your playground to demonstrate your comuter vision and problem-solving skills to build from raw pixels to a trained model,  Show us your creativity, clarity, and code!

**All the best! May your masks be tight and your polygons be sharp 🤖📐**