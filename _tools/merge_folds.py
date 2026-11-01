"""Merge the fold 3-4 outputs (results/fivefold/<tag>) into the run folders that step5_analysis.py reads.

The fold 0-2 tables are first kept as results/step5_3fold. Existing fold 0-2 files are never overwritten.

Usage: python _tools/merge_folds.py
"""
import shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_tools"))
SRC = ROOT / "results" / "fivefold"
FOLDER = {"teacher": "teacher_eval", "student": "student_eval_v2"}


def main():
    step5, keep = ROOT / "results" / "step5", ROOT / "results" / "step5_3fold"
    if not keep.exists():
        shutil.copytree(step5, keep)
        print("kept", keep)
    for d in sorted(p for p in SRC.iterdir() if p.is_dir()):
        tag = d.name
        dst = ROOT / "results" / FOLDER.get(tag, tag)
        for sub in ("preds", "results"):
            (dst / sub).mkdir(exist_ok=True)
            for k in (3, 4):
                for name in (f"{tag}_fold{k}.csv", f"{tag}_fold{k}.json"):
                    f = d / sub / name
                    if f.exists():
                        shutil.copy2(f, dst / sub / name)
        n = len(list((dst / "preds").glob(f"{tag}_fold*.csv")))
        print(f"{tag:16s} -> {dst.name:18s} folds with predictions: {n}")
        ck = d / "ckpt"
        if tag == "student" and ck.exists():        # Grad-CAM needs the fold 3-4 student weights
            for f in ck.glob("student_fold[34]*"):
                shutil.copy2(f, ROOT / "results" / "student_v1" / "ckpt" / f.name)


if __name__ == "__main__":
    main()
