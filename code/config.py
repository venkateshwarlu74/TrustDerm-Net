"""Shared constants and default paths for TrustDerm-Net experiments (Kaggle layout)."""
import os

INPUT_ROOT = os.environ.get("TD_INPUT", "/kaggle/input")
WORK = os.environ.get("TD_WORK", "/kaggle/working")
CACHE = os.path.join(WORK, "cache")          # preprocessed images
MANIFEST = os.path.join(WORK, "manifest.csv")
CKPT = os.path.join(WORK, "ckpt")
PREDS = os.path.join(WORK, "preds")
RESULTS = os.path.join(WORK, "results")

# Class order: benign branch first (indices 0-3), malignant branch second (4-6)
CLASSES = ["NV", "BKL", "DF", "VASC", "MEL", "BCC", "AKIEC"]
BENIGN, MALIGNANT = [0, 1, 2, 3], [4, 5, 6]
MEL = CLASSES.index("MEL")

SITES = ["head/neck", "upper extremity", "lower extremity", "torso", "palms/soles", "oral/genital"]
CONCEPTS = ["atypical_network", "blue_whitish_veil", "atypical_vascular", "irregular_streaks",
            "irregular_dots_globules", "irregular_blotches", "regression"]

# Roles in the manifest
DEV = "dev"                                   # HAM10000 + ISIC 2019 (+ Derm7pt train/val)
EXTERNAL = ["isic2020", "pad_ufes", "ddi"]    # fully external, test only
DERM7PT_TEST = "derm7pt_test"                 # concept-supervised domain, reported separately

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
