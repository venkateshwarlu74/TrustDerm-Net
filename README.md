# TrustDerm-Net

Code for **TrustDerm-Net: Image–Metadata Fusion with Evidential Distillation for Reliable Skin Cancer Triage** (manuscript in preparation).

TrustDerm-Net is a 5.6 M-parameter MobileViT student distilled from a ConvNeXt-B teacher. It combines a benign/malignant hierarchy, gated cross-attention fusion of image and patient metadata, evidential uncertainty for referral, and a concept head for the seven-point checklist.

**Research code only. It is not a medical device and must not be used for patient care.** The melanoma score must not be used on smartphone images: melanoma AUROC on PAD-UFES-20 is 0.432, below chance.

## Results (five-fold models, HAM10000 lesion-disjoint test set, 1,470 images)

| Measure | Value |
|---|---|
| Balanced accuracy | 71.4 ± 4.5% (95% CI 67.6–74.7%) |
| Macro-AUROC | 0.947 ± 0.015 |
| Triage AUROC | 0.920 ± 0.004 |
| Accuracy after referring the 20% least certain cases | 82.3% → 89.4% |
| Triage AUROC on Derm7pt test / PAD-UFES-20 | 0.865 / 0.812 |

## Layout

| Folder | Contents |
|---|---|
| `code/` | Data manifest, preprocessing, model, losses, training, evaluation, efficiency measurement, and `frozen_splits.csv` (the split of every image) |
| `_tools/` | Kaggle launcher and five-fold queue, fold merging, statistics, figures, Grad-CAM faithfulness |
| `app/` | Tkinter desktop demo (see `app/README_GUI.md`) |

## Setup

```
pip install -r requirements.txt
```

Versions listed are those used for local analysis. Training ran on Kaggle GPUs; see `code/README_KAGGLE.md`.

## Reproducing the results

1. Obtain the datasets below. No images are included here.
2. Train the teacher, then the student, for folds 0–4 (`code/README_KAGGLE.md`). The queue in `_tools/five_fold_queue.py` shows the flags for the student, the eight ablations and the six baselines. All runs use seed 42.
3. Merge and analyse: `_tools/merge_folds.py`, `_tools/step5_analysis.py`, `_tools/text_numbers.py`, `_tools/figs_results.py`.
4. Grad-CAM: `_tools/gradcam_fig8.py`, then `_tools/faith_summary.py`. Set `HAM_ZIP` to the path of your HAM10000 zip.

For the Kaggle launcher, set `KAGGLE_USERNAME` and upload your own copy of Derm7pt as a private Kaggle dataset named `derm7pt-release-v0`.

The app needs trained weights in `results/student_v1/ckpt/` and threshold files in `results/student_eval_v2/results/`. These are not part of this repository.

## Data

| Dataset | Source | Note |
|---|---|---|
| HAM10000 | Harvard Dataverse, doi:10.7910/DVN/DBW86T | Dataset's own terms of use |
| Derm7pt | From its authors, after registration | May not be redistributed; only image identifiers and split assignments are included here |
| PAD-UFES-20 | Mendeley Data, doi:10.17632/zr7vgbcyr2.1 | CC BY 4.0 |

## Licence

[To be chosen by the authors before this repository is made public.]

## Citation

[Add once the paper is published.]