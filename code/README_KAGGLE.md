# Running TrustDerm-Net on Kaggle

## 1. One-time setup

1. **Upload the code.** On Kaggle go to *Datasets → New Dataset*, upload this `code` folder and name the dataset `trustderm-code`. Keep it **Private**.
2. **Add the data** as inputs to your notebook (*Add Input*). The scripts find each dataset by its files, so the dataset names do not matter.

| Dataset | Where to get it | Required file the script looks for |
|---|---|---|
| HAM10000 | Kaggle: search "Skin Cancer MNIST: HAM10000" (the same files as your `archive (1).zip`) | `HAM10000_metadata.csv` |
| ISIC 2019 | Kaggle mirror (search "ISIC 2019") or challenge.isic-archive.com | `ISIC_2019_Training_GroundTruth.csv`, `ISIC_2019_Training_Metadata.csv` |
| Derm7pt | derm.cs.sfu.ca (free registration) → upload as a **private** Kaggle dataset | `meta/meta.csv`, `meta/*_indexes.csv`, `images/` |
| ISIC 2020 | Kaggle competition "SIIM-ISIC Melanoma Classification" (join it to get access) | `train.csv`, `jpeg/train/` |
| PAD-UFES-20 | Mendeley Data (CC BY 4.0) → upload as a private Kaggle dataset | `metadata.csv`, images |
| DDI | Stanford AIMI (signed data-use agreement) → private upload only if the agreement allows it | `ddi_metadata.csv`, images |

You can start with **HAM10000 alone**. Datasets that are missing are skipped.

3. **Notebook settings:**
   - *Accelerator* → **GPU T4 ×2** or **P100**.
   - *Internet* → **On**. It is needed once, to download the pretrained ImageNet weights through `timm`.

## 2. Notebook cells (run in order)

```
# Cell 1 – copy the code into the writable folder
!cp -r /kaggle/input/trustderm-code/code /kaggle/working/code
%cd /kaggle/working/code
!python -c "import torch, timm; print(torch.__version__, timm.__version__, torch.cuda.get_device_name(0))"
```
```
# Cell 2 – build the manifest, preprocess (hair removal + colour constancy), make lesion-level splits
!python build_manifest.py
!python preprocess.py --workers 4
!python make_splits.py          # prints the class-by-split table -> Table II and Fig. 4
```
```
# Cell 3 – 2-minute smoke test (must finish without errors before anything else)
!python train.py --role student --kd none --fold 0 --epochs 1 --max_train 512 --tag smoke
```
```
# Cell 4 – teacher, then student, for one fold
!python train.py --role teacher --fold 0
!python train.py --role student --fold 0 --teacher_ckpt /kaggle/working/ckpt/teacher_fold0.pt
```
```
# Cell 5 – evaluation + efficiency (after all folds of a run are trained)
!python evaluate.py --tag student
!python efficiency.py --ckpt /kaggle/working/ckpt/student_fold0.pt
```

**Saving your work.**
- Use *Save Version → Save & Run All (Commit)*. Long runs then keep going in the background for up to 12 h, and everything in `/kaggle/working` (cache, checkpoints, predictions) is saved as the notebook's output.
- In the next session, add that output as an input and copy it back before you continue:
  `!cp -r /kaggle/input/<your-notebook-output>/* /kaggle/working/`

## 3. Experiment plan and GPU budget

Kaggle gives about **30 GPU-hours per week**. The times below are rough estimates for about 21k training images on one T4. Measure your own after fold 0 and adjust.

| Run | Command flags | Folds | Est. GPU h |
|---|---|---|---|
| Teacher (ConvNeXt-B, 320 px, 20 ep) | `--role teacher` | 5 | ~12 |
| **TrustDerm-Net (full)** | `--role student --teacher_ckpt …` | 5 | ~5 |
| Ablation: no metadata | `--no_meta --tag abl_nometa` | 3 | ~2 |
| Ablation: concat fusion | `--fusion concat --tag abl_concat` | 3 | ~2 |
| Ablation: no modality dropout | `--p_md 0 --tag abl_nomd` | 3 | ~2 |
| Ablation: flat 7-class head | `--flat --tag abl_flat` | 3 | ~2 |
| Ablation: softmax instead of evidential | `--softmax --tag abl_softmax` | 3 | ~2 |
| Ablation: no concepts | `--no_concepts --tag abl_nocon` | 3 | ~2 |
| Ablation: no distillation | `--kd none --tag abl_nokd` | 3 | ~2 |
| Ablation: plain (unweighted) distillation | `--kd plain --tag abl_kdplain` | 3 | ~2 |
| Baselines (same pipeline, different backbone) | `--kd none --no_meta --no_concepts --flat --softmax --backbone resnet50 --tag base_resnet50` etc. | 3–5 | ~1–2 each |

Timm backbone names for the baselines: `resnet50`, `efficientnet_b3`, `mobilenetv2_100`, `swin_tiny_patch4_window7_224`, `convnext_tiny`, `deit_small_patch16_224`.

**Order of work:** smoke test → fold 0 teacher and student (check that the numbers are sensible) → remaining folds → ablations → baselines.

## 4. What to send back for Step 5

Download `/kaggle/working/results/` and `/kaggle/working/preds/`, plus the `*_log.csv` files in `ckpt/`, and put them in `Desktop\Paper2\results\`. Step 5 builds Tables IV–VIII and Figs. 4–8 directly from these files.
