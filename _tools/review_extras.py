"""Extra analyses requested by the guide's review (B4, B5, B6, D6), computed from the saved five-fold
prediction files only (no retraining). Output: results/step5/review_extras.json + printed summary.

B4  out-of-distribution detection: AUROC for separating HAM10000 test (in-domain) from Derm7pt test and
    PAD-UFES-20 images, using u = K/S and using 1 - max p (exploratory).
B5  triage workload at t_mal: benign lesions flagged per malignant lesion caught, PPV, NPV (HAM10000 test);
    decision-curve net benefit at the operating point.
B6  per-fold bAcc, macro-F1, DF recall for the full model.
D6  per-fold thresholds t_mal and tau_u (kappa = 0.8).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "results" / "student_eval_v2" / "preds"
CLASSES = ["NV", "BKL", "DF", "VASC", "MEL", "BCC", "AKIEC"]
P7 = [f"p_{c}" for c in CLASSES]
out = {"per_fold": [], "ood": [], "workload": []}
for k in range(5):
    p = pd.read_csv(PREDS / f"student_fold{k}.csv")
    val, te = p[p.eval_set == "val"], p[p.eval_set == "test"]
    t_mal = float(np.quantile(val.p_mal[val.z == 1].to_numpy(), 0.05, method="lower"))
    tau_u = float(np.quantile(val.u, 0.8))
    y, pred = te.y.to_numpy(), te[P7].to_numpy().argmax(1)
    out["per_fold"].append(dict(fold=k, bacc=balanced_accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
                                df_recall=recall_score(y, pred, labels=[2], average="macro"),
                                df_found=int(((y == 2) & (pred == 2)).sum()), t_mal=t_mal, tau_u=tau_u))
    # B5 workload on HAM10000 test
    flag = te.p_mal.to_numpy() >= t_mal
    z = te.z.to_numpy()
    tp, fp, fn, tn = int((flag & (z == 1)).sum()), int((flag & (z == 0)).sum()), int((~flag & (z == 1)).sum()), int((~flag & (z == 0)).sum())
    n = len(z)
    nb = tp / n - fp / n * (t_mal / (1 - t_mal))                    # decision-curve net benefit at p_t = t_mal
    nb_all = (z == 1).mean() - (z == 0).mean() * (t_mal / (1 - t_mal))
    out["workload"].append(dict(fold=k, tp=tp, fp=fp, fn=fn, tn=tn, benign_per_cancer=fp / tp, ppv=tp / (tp + fp),
                                npv=tn / (tn + fn), net_benefit=nb, net_benefit_treat_all=nb_all))
    # B4 OOD detection
    for s in ("derm7pt_test", "pad_ufes"):
        o = p[p.eval_set == s]
        lab = np.r_[np.zeros(len(te)), np.ones(len(o))]
        u = np.r_[te.u, o.u]
        mp = np.r_[1 - te[P7].max(1), 1 - o[P7].max(1)]
        out["ood"].append(dict(fold=k, shifted=s, auroc_u=roc_auc_score(lab, u), auroc_1mp=roc_auc_score(lab, mp)))

ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
pf, wl, od = pd.DataFrame(out["per_fold"]), pd.DataFrame(out["workload"]), pd.DataFrame(out["ood"])
print("B6 per fold:\n", pf.round(4).to_string(index=False))
print("median bAcc:", round(pf.bacc.median(), 4))
print("D6 t_mal:", ms(pf.t_mal), " tau_u(k=0.8):", ms(pf.tau_u))
print("B5 workload:\n", wl.round(4).to_string(index=False))
print("  benign flagged per cancer:", ms(wl.benign_per_cancer), " PPV:", ms(wl.ppv), " NPV:", ms(wl.npv),
      " net benefit:", ms(wl.net_benefit), " vs treat-all:", ms(wl.net_benefit_treat_all))
for s, g in od.groupby("shifted"):
    print(f"B4 OOD HAM test vs {s}: AUROC u = {ms(g.auroc_u)}, 1-max p = {ms(g.auroc_1mp)}")
summary = dict(median_bacc=float(pf.bacc.median()), t_mal=[float(pf.t_mal.mean()), float(pf.t_mal.std(ddof=1))],
               tau_u=[float(pf.tau_u.mean()), float(pf.tau_u.std(ddof=1))],
               benign_per_cancer=[float(wl.benign_per_cancer.mean()), float(wl.benign_per_cancer.std(ddof=1))],
               ppv=[float(wl.ppv.mean()), float(wl.ppv.std(ddof=1))], npv=[float(wl.npv.mean()), float(wl.npv.std(ddof=1))],
               ood={s: dict(u=[float(g.auroc_u.mean()), float(g.auroc_u.std(ddof=1))],
                            one_minus_maxp=[float(g.auroc_1mp.mean()), float(g.auroc_1mp.std(ddof=1))]) for s, g in od.groupby("shifted")})
out["summary"] = summary
json.dump(out, open(ROOT / "results" / "step5" / "review_extras.json", "w"), indent=1)
