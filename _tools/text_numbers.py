"""Numbers quoted in the running text of Sections 5-6 that are not in a table. Reads the saved prediction files.

Usage: python _tools/text_numbers.py        -> results/step5/text_numbers.json
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from step5_analysis import (BASELINES, CLASSES, CONCEPTS, FULL, MEL, OUT, P7, RUNS, aurc, binary_metrics, boot_bacc,
                            boot_ci, boot_p, calib_row, concept_metrics, fold_table, load, mcnemar_folds,
                            multiclass_metrics, thresholds)

sd = lambda x: [round(float(np.mean(x)), 4), round(float(np.std(x, ddof=1)), 4)]
pred = lambda d: d[P7].to_numpy().argmax(1)


def per_run(r):
    t = {s: fold_table(r, lambda pr, a, b: multiclass_metrics(pr[s], a, b)) for s in ("test", "derm7pt_test", "pad_ufes")}
    b = {s: fold_table(r, lambda pr, a, b_: binary_metrics(pr[s], a, b_)) for s in ("test", "derm7pt_test", "pad_ufes")}
    return dict(acc_per_fold=[round(float((pred(pr["test"]) == pr["test"].y).mean()), 4) for pr in r.values()],
                acc=sd([(pred(pr["test"]) == pr["test"].y).mean() for pr in r.values()]),
                bacc_per_fold=[round(float(x), 4) for x in t["test"].bacc],
                sens_NV=sd(t["test"].sens_NV), sens_MEL=sd(t["test"].sens_MEL), macro_f1=sd(t["test"].macro_f1),
                ece=sd(t["test"].ece), bacc=sd(t["test"].bacc), d7_bacc=sd(t["derm7pt_test"].bacc),
                pad_triage=sd(b["pad_ufes"].auroc), mel_auroc=sd(t["test"].mel_auroc),
                pad_mel_auroc=sd([roc_auc_score(pr["pad_ufes"].y == MEL, pr["pad_ufes"].p_MEL) for pr in r.values()]),
                d7_triage=sd(b["derm7pt_test"].auroc))


def flagged(r, s, cls):
    out = []
    for pr in r.values():
        t, _ = thresholds(pr["val"])
        g = pr[s][pr[s].y == CLASSES.index(cls)]
        out.append((g.p_mal >= t).mean())
    return sd(out)


def conf(r, s, a, b):
    """share of true class a predicted as class b, one value per fold"""
    return sd([(pred(pr[s][pr[s].y == CLASSES.index(a)]) == CLASSES.index(b)).mean() for pr in r.values()])


def main():
    runs = {t: load(t, f) for t, (f, _) in RUNS.items()}
    runs.update({t: load(t, v[0]) for t, v in BASELINES.items()})
    runs["teacher"] = load("teacher", "results/teacher_eval")
    full, teacher = runs[FULL], runs["teacher"]
    o = dict(runs={t: per_run(r) for t, r in runs.items()})

    # student vs teacher
    bo = boot_bacc([full, teacher], "test")
    delta = bo[:, 0] - bo[:, 1]
    o["student_vs_teacher"] = dict(delta_ci=[round(float(x), 4) for x in boot_ci(delta)], p=float(boot_p(delta)),
                                   delta=round(float(np.mean(o["runs"][FULL]["bacc"][0] - o["runs"]["teacher"]["bacc"][0])), 4))
    bd = boot_bacc([full, teacher], "derm7pt_test")
    o["student_vs_teacher_d7"] = dict(delta_ci=[round(float(x), 4) for x in boot_ci(bd[:, 0] - bd[:, 1])])
    o["teacher_ece"] = sd([calib_row(dict_d)["ece"] for dict_d in
                           [pr["test"].assign(u=1 - pr["test"][P7].to_numpy().max(1)) for pr in teacher.values()]])
    o["teacher_pad"] = dict(mel_vs_bcc=sd([roc_auc_score(g.y == MEL, g.p_MEL) for g in
                                           [pr["pad_ufes"][pr["pad_ufes"].y.isin([MEL, CLASSES.index("BCC")])] for pr in teacher.values()]]),
                            flagged_mel=flagged(teacher, "pad_ufes", "MEL"))

    # baselines: unadjusted detail
    all_ps, exc = [], []
    for t in BASELINES:
        ps = mcnemar_folds(full, runs[t])
        all_ps += ps
        exc += [(t, k, round(p, 3)) for k, p in enumerate(ps) if p >= 0.05]
    # direction matters: a small p against the student would not be a win
    wins = 0
    for t in BASELINES:
        for k in full:
            d, e = full[k]["test"], runs[t][k]["test"]
            wins += (pred(d) == d.y).mean() > (pred(e) == e.y).mean()
    o["mcnemar"] = dict(n=len(all_ps), sig=sum(p < 0.05 for p in all_ps), exceptions=exc, student_acc_higher=int(wins))
    o["best_acc_each_fold"] = {k: max(((pred(r[k]["test"]) == r[k]["test"].y).mean(), t) for t, r in
                                      {FULL: full, **{t: runs[t] for t in BASELINES}}.items())[1] for k in full}

    # triage per class
    o["flagged"] = {f"{s}_{c}": flagged(full, s, c) for s, c in
                    [("test", "MEL"), ("derm7pt_test", "MEL"), ("pad_ufes", "MEL"), ("pad_ufes", "BCC"), ("pad_ufes", "AKIEC")]}

    # confusions
    o["conf"] = dict(ham_mel_nv=conf(full, "test", "MEL", "NV"), ham_mel_bkl=conf(full, "test", "MEL", "BKL"),
                     d7_bkl_nv=conf(full, "derm7pt_test", "BKL", "NV"), pad_mel_nv=conf(full, "pad_ufes", "MEL", "NV"))
    o["ham_mel_benign"] = sd([(pred(pr["test"][pr["test"].y == MEL]) < 4).mean() for pr in full.values()])

    # routing with the triage threshold
    o["bacc_routed"] = sd([json.load(open(Path("results/student_eval_v2/results") / f"student_fold{k}.json"))["test"]["bacc_routed"]
                           for k in full])

    # concepts
    cm = fold_table(full, lambda pr, a, b: concept_metrics(pr["derm7pt_test"]))
    o["concepts"] = {c: sd(cm[c]) for c in cm.columns if "auroc" in c}

    # score ranking on every validation fold and test set
    rows = {}
    for s in ("val", "test", "derm7pt_test"):
        rows[s] = []
        for pr in full.values():
            d = pr[s]
            c = (pred(d) == d.y).astype(float).to_numpy()
            rows[s].append((round(aurc(d.u.to_numpy(), c), 4), round(aurc(1 - d[P7].to_numpy().max(1), c), 4)))
    o["aurc_u_vs_maxp"] = rows

    # accepted-vs-overall accuracy on PAD, kept for 5.4
    o["accepted_vs_referred_pad"] = []
    for pr in full.values():
        _, tau = thresholds(pr["val"])
        d = pr["pad_ufes"]
        keep = d.u <= tau[0.8]
        c = pred(d) == d.y
        o["accepted_vs_referred_pad"].append((round(float(c[keep].mean()), 4), round(float(c[~keep].mean()), 4)))

    # per-fold half-width of paired CI, used for the power statement
    tags = [t for t in RUNS if t != FULL]
    bt = boot_bacc([full] + [runs[t] for t in tags], "test")
    o["ci_halfwidth_abl"] = {t: round(float((boot_ci(bt[:, j + 1] - bt[:, 0])[1] - boot_ci(bt[:, j + 1] - bt[:, 0])[0]) / 2), 4)
                             for j, t in enumerate(tags)}
    with open(OUT / "text_numbers.json", "w") as f:
        json.dump(o, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in o.items()}, indent=0, default=str))


if __name__ == "__main__":
    main()
