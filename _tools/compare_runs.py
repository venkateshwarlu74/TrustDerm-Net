"""Compare runs side by side (mean ± SD over folds) from their per-fold result files.

Usage: python compare_runs.py student=results/student_eval_v2 abl_kdnone=results/abl_kdnone ...
       (each folder must contain results/{tag}_per_fold.csv or {tag}_per_fold.csv)
"""
import os, sys
import pandas as pd

ROWS = [("test", "bacc"), ("test", "macro_f1"), ("test", "macro_auroc"), ("test", "mel_auroc"), ("test", "ece"),
        ("test", "aurc"), ("test", "bin_auroc"), ("test", "bin_spec"), ("test", "acc_accepted@0.8"),
        ("derm7pt_test", "bacc"), ("derm7pt_test", "concept_mean_auroc"),
        ("pad_ufes", "bacc"), ("pad_ufes", "bin_auroc"), ("pad_ufes", "bin_auroc_gap")]


def load(tag, folder):
    for p in (os.path.join(folder, "results", f"{tag}_per_fold.csv"), os.path.join(folder, f"{tag}_per_fold.csv")):
        if os.path.exists(p):
            return pd.read_csv(p)
    sys.exit(f"no per-fold results for {tag} in {folder}")


def main():
    cols = {}
    for arg in sys.argv[1:]:
        tag, folder = arg.split("=", 1)
        d = load(tag, folder)
        g = d.groupby(["set", "metric"]).value
        mean, sd = g.mean(), g.std()
        cols[tag] = [f"{mean[k]:.3f} ± {sd[k]:.3f}" if k in mean.index else "–" for k in ROWS]
    out = pd.DataFrame(cols, index=[f"{s}: {m}" for s, m in ROWS])
    print(out.to_string())


if __name__ == "__main__":
    main()
