# TrustDerm-Net: splits, near-duplicate lists, predictions and weights

This record holds the research data behind the manuscript "TrustDerm-Net: Image–Metadata Fusion with Evidential Distillation for Uncertainty-Aware Skin Cancer Triage" (submitted to *Computerized Medical Imaging and Graphics*). With it, every number in Tables 4–8 and Supplementary Tables S1–S7 can be recomputed without retraining any model. Code: https://github.com/venkateshwarlu74/TrustDerm-Net

**No images are included.** Obtain them from the original providers:

| Dataset | Source | Licence |
|---|---|---|
| HAM10000 | Harvard Dataverse, doi:10.7910/DVN/DBW86T | Dataset terms of use |
| Derm7pt | From its authors after registration | CC BY-NC-ND (no redistribution) |
| PAD-UFES-20 | Mendeley Data, doi:10.17632/zr7vgbcyr2.1 | CC BY 4.0 |
| ISIC 2020 | ISIC, doi:10.34970/2020-ds01 | CC BY-NC 4.0 |

Because the Derm7pt licence forbids redistribution, Derm7pt rows carry only the image identifier (`D7_<case_num>`, from the `case_num` column of Derm7pt `meta.csv`), the split and the model outputs. Their diagnosis and seven-point labels are left blank; join them from your own copy of Derm7pt.

## Contents

| Path | What it is |
|---|---|
| `splits/lesion_level_splits.csv` | Split of every image: `fold0`–`fold4` (development folds), `test` (HAM10000 lesion-disjoint test set), `derm7pt_test`, `pad_ufes`, `isic2020_external`, or `excluded_duplicate`. `lesion_or_patient_group` is the HAM10000 lesion ID, the PAD-UFES-20 patient ID or the ISIC 2020 patient ID used to keep groups within one split. |
| `near_duplicates/excluded_near_duplicates_derm7pt_pad.csv` | The 56 Derm7pt and PAD-UFES-20 images removed as near-duplicates of HAM10000 images (64-bit dHash, Hamming distance ≤ 3). |
| `near_duplicates/isic2020_dhash_candidates_ncc.csv` | The 12,060 ISIC 2020 images flagged by dHash, their closest development image and the maximum normalised cross-correlation (64 × 64 greyscale, 8 flips/rotations). None reached NCC ≥ 0.98, so all 33,126 ISIC 2020 images were kept. |
| `predictions/internal_external/{model}_fold{k}.csv` | Outputs of each of the 5 fold models on its validation fold (`eval_set = val`), the HAM10000 test set (`test`), the Derm7pt test split (`derm7pt_test`) and PAD-UFES-20 (`pad_ufes`). |
| `predictions/isic2020/{model}_isic2020_fold{k}.csv` | Outputs on all 33,126 ISIC 2020 images (`eval_set = isic2020`), plus the validation fold again as a reproducibility check. `orig_split` keeps the dHash flag. |
| `results_tables/` | The tables computed from these predictions (CSV/JSON), named after the manuscript tables. |
| `weights/student_fold{k}.pt` | PyTorch state of the five TrustDerm-Net student models (MobileViT-S, 5.6 M parameters). `student_fold{k}_thresholds.json` holds the validation-fixed triage threshold `t_mal` and the referral thresholds `tau_u` for coverage 0.7–1.0. |
| `MD5SUMS.txt` | Checksums of every file. |

## Models

`student` = TrustDerm-Net; `teacher` = ConvNeXt-B teacher; `abl_nometa`, `abl_concat`, `abl_pmd0` (no modality dropout), `abl_flat`, `abl_softmax`, `abl_noconcepts`, `abl_kdnone`, `abl_kdplain` = ablations (i)–(viii); `abl_tau1`, `abl_tau2` = distillation temperature 1 and 2 (the main model uses 4); `base_resnet50`, `base_effb3`, `base_mnv2`, `base_deits`, `base_swint`, `base_convnextt` = image-only baselines.

## Columns of the prediction files

| Column | Meaning |
|---|---|
| `image_id`, `dataset`, `split`, `eval_set` | Image, source dataset, split, evaluation set |
| `y` | True class index: 0 NV, 1 BKL, 2 DF, 3 VASC, 4 MEL, 5 BCC, 6 AKIEC; −1 = no fine label (most of ISIC 2020) |
| `z` | True coarse label: 1 malignant (MEL, BCC, AKIEC), 0 benign; for ISIC 2020, 1 = melanoma |
| `skin_tone` | PAD-UFES-20 Fitzpatrick group (12 = I–II, 34 = III–IV, 56 = V–VI); −1 = not recorded |
| seven concept columns (`atypical_network` … `regression`) | True seven-point labels (Derm7pt only, blanked here); −1 = not available |
| `p_NV` … `p_AKIEC` | Predicted class probabilities |
| `p_mal` | Predicted probability of malignancy: Stage 1 output for hierarchical models, sum of the MEL, BCC and AKIEC probabilities for flat models |
| `u` | Uncertainty score used for referral: u = K/S for evidential heads, 1 − max p for softmax heads; for hierarchical models the larger of the Stage 1 and selected Stage 2 values |
| `chat_*` | Predicted concept probabilities (student and ablations with a concept head) |

## Licence

Prediction files, splits, tables and weights: CC BY-NC 4.0, consistent with the non-commercial terms of the training data. The weights were trained on HAM10000 and Derm7pt images and are for research use only. They are not a medical device.

## Citation

Please cite the article (details to be added on publication) and this record.
