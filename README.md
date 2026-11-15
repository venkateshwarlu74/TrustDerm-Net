# TrustDerm-Net

Code for **TrustDerm-Net: Image–Metadata Fusion with Evidential Distillation for Uncertainty-Aware Skin Cancer Triage** (submitted to *Computerized Medical Imaging and Graphics*).

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
| Melanoma AUROC on ISIC 2020 (33,126 dermoscopic images, post hoc) | 0.759 [95% CI 0.736–0.779] |

## Layout

| Folder | Contents |
|---|---|
| `code/` | Data manifest, preprocessing, model, losses, training, evaluation, efficiency measurement, and `frozen_splits.csv` (the split of every image) |
| `_tools/` | Kaggle launcher and five-fold queue, fold merging, statistics, figures, Grad-CAM faithfulness, ISIC 2020 and distillation-temperature analyses, Zenodo deposit builder |
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
5. ISIC 2020 (post hoc): `code/verify_dups.py` (second-stage NCC check of dHash candidates), `code/eval_external.py`, then `_tools/isic2020_analysis.py` (add `--ci` for the patient-level bootstrap). Distillation temperature: `_tools/tau_analysis.py`.

For the Kaggle launcher, set `KAGGLE_USERNAME` and upload your own copy of Derm7pt as a private Kaggle dataset named `derm7pt-release-v0`.

## Weights, splits and predictions

The five trained student models, their validation-fixed thresholds, the lesion-level splits, the near-duplicate lists and all prediction files behind the paper's tables are on Zenodo: https://doi.org/10.5281/zenodo.24216402 (CC BY-NC 4.0). The app expects the weights in `results/student_v1/ckpt/` and the threshold files in `results/student_eval_v2/results/`.

## Data

| Dataset | Source | Note |
|---|---|---|
| HAM10000 | Harvard Dataverse, doi:10.7910/DVN/DBW86T | Dataset's own terms of use |
| Derm7pt | From its authors, after registration | May not be redistributed; only image identifiers and split assignments are included here |
| PAD-UFES-20 | Mendeley Data, doi:10.17632/zr7vgbcyr2.1 | CC BY 4.0 |
| ISIC 2020 | ISIC, doi:10.34970/2020-ds01 | CC BY-NC 4.0 |

## Licence

Code: MIT (see `LICENSE`). Weights and data files on Zenodo: CC BY-NC 4.0, because the training data allow non-commercial use only.

## Citation

Katineedi, V., Mudigonda, M. TrustDerm-Net: Image–Metadata Fusion with Evidential Distillation for Uncertainty-Aware Skin Cancer Triage. Submitted to *Computerized Medical Imaging and Graphics*, 2026. Full reference to be added on publication.