# TrustDerm-Net Desktop Demo

A Windows desktop app that runs the trained TrustDerm-Net student (the 5.6 M-parameter MobileViT-S model from the paper) on a single dermoscopic image.

> **Research prototype, not a medical device.** It must not be used to diagnose patients. The model was trained on dermoscopic images (HAM10000). On ordinary smartphone photos (PAD-UFES-20) its melanoma AUROC was 0.43, so do not present it as a phone-camera screener. On external dermoscopic images (ISIC 2020, 33,126 images) melanoma AUROC was 0.76 and triage AUROC 0.75.

## How to start

- **Double-click** `app\Run_TrustDerm_GUI.bat`, or
- from the repository folder run: `python app\trustderm_gui.py`
- **For a demo**, which opens an image and analyses it straight away:
  `python app\trustderm_gui.py --open results\step5\fig8\raw\ISIC_0027263.jpg`
- **Without a window** (prints a text report):
  `python app\trustderm_gui.py --cli IMAGE --age 60 --sex male --site torso --model ens`

The models take about 10–20 s to load on this laptop. Each analysis then takes a few seconds on the CPU, because Grad-CAM is computed for all five fold models.

## What you do

1. Click **Open dermoscopic image…** and choose a JPG or PNG.
2. Optionally enter **age, sex and lesion site**. Leave them blank or "Unknown" if they aren't known; the model uses its learned "missing" tokens, as in the paper.
3. Choose the **model**: the ensemble of the 5 fold students (the default), or a single fold.
4. Click **Analyse**. **Save report…** writes a text report and the Grad-CAM image.

## What the app shows

| Panel | Meaning | Source in the paper |
|---|---|---|
| Model input | The image after hair removal, Shades-of-Gray colour correction and centre crop, exactly as in training | Section 3.3 |
| Triage: screen-positive / screen-negative | Stage-1 P(malignant) compared with the threshold t_mal, which was set on validation for 95% sensitivity. On the HAM10000 test set it caught 95.5% of malignant lesions but also flagged about 40% of benign ones (specificity 59.8%). So "screen-positive" means "needs review", not "cancer". | Eq. (21), Section 3.10, Table 4 |
| Model is confident / Refer to a dermatologist | Evidential uncertainty u = K/S compared with τ_u, the threshold that keeps 80% of validation cases (κ = 0.8) | Eqs. (14a–b), (22); Section 3.10; Table 7(a) |
| Diagnosis probabilities | 7-class composite probabilities; red bars are malignant classes | Eq. (13) |
| Seven-point checklist concepts | Concept-head probabilities; values ≥ 0.5 (dashed line) are shown in bold as "present" | Eq. (10); Section 3.6 |
| Grad-CAM | The regions that raised the predicted class, averaged over the selected models | Section 5.6, Fig. 8 |

The thresholds come from `results\student_eval_v2\results\student_fold{0..4}.json`. For the ensemble, the app averages the five folds' probabilities, uncertainties and thresholds. That is a convenience for the demo and is not the protocol used in the paper, where each fold is evaluated separately.

## Checks done (2026-10-15; rechecked 2026-11-16)

- **Same predictions as the paper.** On 109 HAM10000 test images, the app (fold 0) agrees with the saved fold-0 predictions on the predicted class for 109/109 images. The largest probability difference is 0.013, which comes from JPEG re-encoding in the training cache.
- **GUI.** It was opened and tested on a melanoma test image (ISIC_0027263): screen-positive, confident, melanoma 89% (five-fold ensemble).
- **Same weights as the published record (2026-11-16).** The five checkpoints are identical to those in the Zenodo deposit (doi:10.5281/zenodo.24216402). Section, equation, table and figure numbers above refer to the manuscript submitted to *Computerized Medical Imaging and Graphics*.

## Files

- `trustderm_gui.py`: the app. It uses `code\model.py`, `code\preprocess.py` and `code\data.py` unchanged.
- `Run_TrustDerm_GUI.bat`: double-click launcher.
- It needs Python with torch, timm, torchvision, OpenCV and Pillow. All are already installed on this PC; nothing new was installed.
- The model weights are in `results\student_v1\ckpt\student_fold{0..4}.pt` (about 21 MB each).

## Next steps (optional)

- **Android app.** Export the student to ONNX, quantize it, and run it with ONNX Runtime Mobile. This needs the `onnx` and `onnxruntime` packages, so ask before installing them.
- **Browser version.** A Gradio interface on Hugging Face Spaces. This needs `gradio`.
