"""Post hoc KD-temperature experiment (tau = 1, 2 vs the main tau = 4): ablation-style table + PAD melanoma diagnostics.

Usage: python _tools/tau_analysis.py   -> results/step5/tau_experiment.csv, tau_pad_mel_vs.csv, tau_experiment.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
import step5_analysis as S

S.RUNS = {"student": ("results/student_eval_v2", "TrustDerm-Net (tau = 4, main)"),
          "abl_tau2": ("results/abl_tau2", "KD tau = 2"),
          "abl_tau1": ("results/abl_tau1", "KD tau = 1"),
          "abl_kdnone": ("results/abl_kdnone", "w/o distillation")}
runs = {t: S.load(t, f) for t, (f, _) in S.RUNS.items()}
runs["teacher"] = S.load("teacher", "results/teacher_eval")
for t, r in runs.items():
    print(t, "folds", sorted(r))
t6 = S.table6({k: v for k, v in runs.items() if k in S.RUNS})
t6.to_csv(S.OUT / "tau_experiment.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(t6.drop(columns=["variant"]).to_string(index=False))

MEL = S.MEL
rows, extra = [], {}
for t, r in runs.items():
    sens, tri, auc_bin, mel = [], [], [], []
    for k, pr in r.items():
        d = pr["pad_ufes"]; t_mal, _ = S.thresholds(pr["val"])
        mel.append(roc_auc_score(d.y == MEL, d.p_MEL))
        sens.append(((d.p_mal >= t_mal) & (d.y == MEL)).sum() / (d.y == MEL).sum())
        auc_bin.append(roc_auc_score(d.z, d.p_mal))
        for other in ("NV", "BKL", "BCC", "AKIEC"):
            g = d[d.y.isin([MEL, S.CLASSES.index(other)])]
            rows.append(dict(run=t, fold=k, vs=other, auroc=roc_auc_score(g.y == MEL, g.p_MEL)))
    extra[t] = dict(pad_mel_auroc=S.ms(mel), pad_mel_flagged_by_tmal=S.ms(sens), pad_triage_auroc=S.ms(auc_bin))
pv = pd.DataFrame(rows).groupby(["run", "vs"]).auroc.agg(["mean", "std"]).round(3).unstack("vs")
pv.to_csv(S.OUT / "tau_pad_mel_vs.csv")
print("\nPAD melanoma diagnostics:"); print(pd.DataFrame(extra).T.to_string())
print("\nPAD melanoma-vs-class AUROC (mean):"); print(pv["mean"].to_string())
# paired bootstrap on PAD MEL AUROC is not run; fold SDs are reported instead
(S.OUT / "tau_experiment.json").write_text(json.dumps(extra, indent=1))
