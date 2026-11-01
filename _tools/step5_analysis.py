"""Step 5 analysis: Tables IV-VIII (and the numbers quoted in Section V) from saved prediction files.

Everything is recomputed from results/<run>/preds/<tag>_fold<k>.csv, using the metric functions of
code/evaluate.py, so the paper and the per-run summaries cannot drift apart.
Statistics (Section IV-F): paired bootstrap over test images (B = 2000) of the fold-averaged metric,
exact McNemar test per fold, Holm adjustment within each table.

Usage: python _tools/step5_analysis.py            -> results/step5/*.csv + step5_numbers.md
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CLASSES, CONCEPTS, MEL                                       # noqa: E402
from evaluate import KAPPAS, aurc, binary_metrics, concept_metrics, ece, multiclass_metrics   # noqa: E402

OUT = ROOT / "results" / "step5"
B, SEED, S_STAR = 2000, 0, 0.95
SETS = ["test", "derm7pt_test", "pad_ufes"]
SET_NAME = {"test": "HAM10000 internal test", "derm7pt_test": "Derm7pt test", "pad_ufes": "PAD-UFES-20"}

# tag -> (folder, label). Ablation numbering follows Section IV-D.
FULL = "student"
RUNS = {
    "student": ("results/student_eval_v2", "TrustDerm-Net (full)"),
    "abl_nometa": ("results/abl_nometa", "(i) w/o metadata"),
    "abl_concat": ("results/abl_concat", "(ii) concatenation fusion"),
    "abl_pmd0": ("results/abl_pmd0", "(iii) w/o modality dropout"),
    "abl_flat": ("results/abl_flat", "(iv) flat 7-class head"),
    "abl_softmax": ("results/abl_softmax", "(v) softmax heads"),
    "abl_noconcepts": ("results/abl_noconcepts", "(vi) w/o concept head"),
    "abl_kdnone": ("results/abl_kdnone", "(vii) w/o distillation"),
    "abl_kdplain": ("results/abl_kdplain", "(viii) KD w/o confidence weights"),
}
BASELINES = {   # tag -> (folder, label, efficiency-table name)
    "base_resnet50": ("results/base_resnet50", "ResNet-50", "ResNet-50"),
    "base_effb3": ("results/base_effb3", "EfficientNet-B3", "EfficientNet-B3"),
    "base_mnv2": ("results/base_mnv2", "MobileNetV2", "MobileNetV2"),
    "base_deits": ("results/base_deits", "DeiT-S", "DeiT-S"),
    "base_swint": ("results/base_swint", "Swin-T", "Swin-T"),
    "base_convnextt": ("results/base_convnextt", "ConvNeXt-T", "ConvNeXt-T"),
}
P7 = [f"p_{c}" for c in CLASSES]


# ----------------------------------------------------------------------------------------------- loading
def load(tag, folder):
    """fold -> {eval_set: DataFrame sorted by image_id}; None if the run is not available yet."""
    d = ROOT / folder / "preds"
    files = sorted(d.glob(f"{tag}_fold*.csv")) if d.exists() else []
    if not files:
        return None
    out = {}
    for f in files:
        k = int(f.stem.rsplit("_fold", 1)[1])
        p = pd.read_csv(f)
        out[k] = {n: g.drop(columns="eval_set").sort_values("image_id").reset_index(drop=True)
                  for n, g in p.groupby("eval_set", sort=False)}
    return out


def thresholds(val):
    pos = val.p_mal[val.z == 1].to_numpy()
    t_mal = float(np.quantile(pos, 1 - S_STAR, method="lower"))
    return t_mal, {k: float(np.quantile(val.u, k)) for k in KAPPAS}


def fold_table(run, fn):
    """Apply fn(fold_preds, t_mal, taus) to every fold; return DataFrame (fold x metric)."""
    rows = []
    for k, pr in run.items():
        t_mal, taus = thresholds(pr["val"])
        rows.append(dict(fold=k, **fn(pr, t_mal, taus)))
    return pd.DataFrame(rows).set_index("fold")


def ms(x, digits=3):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return "–"
    return f"{x.mean():.{digits}f} ± {x.std(ddof=1):.{digits}f}" if len(x) > 1 else f"{x.mean():.{digits}f}"


# ----------------------------------------------------------------------------------------------- statistics
def bacc_vec(y, correct, idx):
    """Balanced accuracy for each bootstrap resample (rows of idx)."""
    K = 7
    out = np.zeros(len(idx))
    present = np.zeros(len(idx))
    for k in range(K):
        mk = (y[idx] == k)
        nk = mk.sum(1)
        ck = (mk & correct[idx]).sum(1)
        ok = nk > 0
        out[ok] += ck[ok] / nk[ok]
        present += ok
    return out / present


def boot_idx(n, rng):
    return rng.integers(0, n, size=(B, n))


def boot_bacc(runs_preds, eval_set, rng_seed=SEED):
    """runs_preds: list of runs; returns (B x len(runs)) fold-averaged bAcc on the same resamples."""
    y = runs_preds[0][0][eval_set].y.to_numpy().astype(int)
    idx = boot_idx(len(y), np.random.default_rng(rng_seed))
    res = []
    for run in runs_preds:
        per_fold = []
        for k in sorted(run):
            d = run[k][eval_set]
            assert (d.y.to_numpy() == y).all()
            per_fold.append(bacc_vec(y, d[P7].to_numpy().argmax(1) == y, idx))
        res.append(np.mean(per_fold, 0))
    return np.stack(res, 1)


def boot_ci(v):
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def boot_p(delta):
    p = 2 * min((delta <= 0).mean(), (delta >= 0).mean())
    return max(p, 1 / B)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar_folds(a, b, eval_set="test"):
    """Exact McNemar per shared fold; returns list of p-values."""
    ps = []
    for k in sorted(set(a) & set(b)):
        da, db = a[k][eval_set], b[k][eval_set]
        y = da.y.to_numpy().astype(int)
        ca, cb = da[P7].to_numpy().argmax(1) == y, db[P7].to_numpy().argmax(1) == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        ps.append(binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)
    return ps


# ----------------------------------------------------------------------------------------------- post-hoc calibration
def fit_temperature(p, y):
    logit = np.log(np.clip(p, 1e-8, 1))
    def nll(t):
        z = logit / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def apply_temperature(p, t):
    z = np.log(np.clip(p, 1e-8, 1)) / t
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def with_probs(d, p7, u=None):
    d = d.copy()
    d[P7] = p7
    d["p_mal"] = p7[:, 4:].sum(1)
    d["u"] = 1 - p7.max(1) if u is None else u
    return d


def calib_row(d):
    p7, y = d[P7].to_numpy(), d.y.to_numpy().astype(int)
    correct = (p7.argmax(1) == y).astype(float)
    return dict(ece=ece(p7.max(1), correct), brier=np.mean(((p7 - np.eye(7)[y]) ** 2).sum(1)),
                aurc=aurc(d.u.to_numpy(), correct))


# ----------------------------------------------------------------------------------------------- tables
def table4(run):
    """Internal test, per class (one-vs-rest) + macro + triage."""
    rows = []
    for c_i, c in enumerate(CLASSES):
        f = fold_table(run, lambda pr, t, ta: dict(
            n=int((pr["test"].y == c_i).sum()),
            sens=multiclass_metrics(pr["test"], t, ta)[f"sens_{c}"],
            spec=multiclass_metrics(pr["test"], t, ta)[f"spec_{c}"],
            auroc=roc_auc_score(pr["test"].y == c_i, pr["test"][f"p_{c}"]),
            auprc=average_precision_score(pr["test"].y == c_i, pr["test"][f"p_{c}"])))
        rows.append(dict(cls=c, n=int(f.n.iloc[0]), sens=ms(f.sens), spec=ms(f.spec), auroc=ms(f.auroc), auprc=ms(f.auprc)))
    t4 = pd.DataFrame(rows)
    m = fold_table(run, lambda pr, t, ta: multiclass_metrics(pr["test"], t, ta))
    bm = fold_table(run, lambda pr, t, ta: binary_metrics(pr["test"], t, ta))
    bb = boot_bacc([run], "test")[:, 0]
    macro = dict(bacc=ms(m.bacc), bacc_ci=boot_ci(bb), macro_f1=ms(m.macro_f1), macro_auroc=ms(m.macro_auroc),
                 mel_auroc=ms(m.mel_auroc), mel_auprc=ms(m.mel_auprc), acc=ms(m["acc_accepted@1.0"]),
                 triage_auroc=ms(bm.auroc), triage_auprc=ms(bm.auprc), triage_sens=ms(bm.sens), triage_spec=ms(bm.spec),
                 t_mal=ms([thresholds(pr["val"])[0] for pr in run.values()]))
    return t4, macro


def table5(run):
    rows = []
    for s in SETS:
        m = fold_table(run, lambda pr, t, ta: multiclass_metrics(pr[s], t, ta))
        bm = fold_table(run, lambda pr, t, ta: binary_metrics(pr[s], t, ta))
        d0 = run[0][s]
        rows.append(dict(set=SET_NAME[s], n=len(d0), n_mal=int(d0.z.sum()),
                         classes="/".join(CLASSES[k] for k in sorted(d0.y.unique())),
                         bacc=ms(m.bacc), macro_f1=ms(m.macro_f1), macro_auroc=ms(m.macro_auroc),
                         mel_auroc=ms(m.mel_auroc), mel_auprc=ms(m.mel_auprc), ece=ms(m.ece), brier=ms(m.brier),
                         bin_auroc=ms(bm.auroc), bin_auprc=ms(bm.auprc), bin_sens=ms(bm.sens), bin_spec=ms(bm.spec),
                         bin_brier=ms(bm.brier)))
    per_class = []
    for s in SETS:
        m = fold_table(run, lambda pr, t, ta: multiclass_metrics(pr[s], t, ta))
        r = dict(set=SET_NAME[s])
        for c in CLASSES:
            r[c] = ms(m[f"sens_{c}"]) if f"sens_{c}" in m else "–"
        per_class.append(r)
    return pd.DataFrame(rows), pd.DataFrame(per_class)


def table6(runs):
    full = runs[FULL]
    tags = [t for t in RUNS if runs.get(t) is not None]
    rows = []
    bo = {s: boot_bacc([runs[t] for t in tags], s) for s in ("test", "derm7pt_test")}
    for j, t in enumerate(tags):
        r = runs[t]
        m = {s: fold_table(r, lambda pr, a, b: multiclass_metrics(pr[s], a, b)) for s in SETS}
        bm = fold_table(r, lambda pr, a, b: binary_metrics(pr["pad_ufes"], a, b))
        cm = fold_table(r, lambda pr, a, b: concept_metrics(pr["derm7pt_test"]))
        row = dict(tag=t, variant=RUNS[t][1], bacc=ms(m["test"].bacc), macro_f1=ms(m["test"].macro_f1),
                   mel_auroc=ms(m["test"].mel_auroc), ece=ms(m["test"].ece), aurc=ms(m["test"].aurc),
                   d7_bacc=ms(m["derm7pt_test"].bacc),
                   concept_auroc=ms(cm.concept_mean_auroc) if "concept_mean_auroc" in cm else "–",
                   pad_bacc=ms(m["pad_ufes"].bacc), pad_bin_auroc=ms(bm.auroc), pad_mel_auroc=ms(m["pad_ufes"].mel_auroc))
        if t != FULL:
            for s, key in (("test", ""), ("derm7pt_test", "d7_")):
                delta = bo[s][:, j] - bo[s][:, 0]           # variant - full
                lo, hi = boot_ci(delta)
                row[f"{key}delta_bacc"] = f"{(m[s].bacc.mean() - m_full(full, s)):+.3f} [{lo:+.3f}, {hi:+.3f}]"
                row[f"{key}p_boot"] = boot_p(delta)
            row["mcnemar_p_median"] = float(np.median(mcnemar_folds(full, r)))
        rows.append(row)
    df = pd.DataFrame(rows)
    for key in ("", "d7_"):
        mask = df[f"{key}p_boot"].notna()
        df.loc[mask, f"{key}p_holm"] = holm(df.loc[mask, f"{key}p_boot"])
    return df


def m_full(full, s):
    return fold_table(full, lambda pr, a, b: multiclass_metrics(pr[s], a, b)).bacc.mean()


def table7_calibration(runs):
    """Native, temperature-scaled and all-fold-ensemble versions of the full model and the softmax variant."""
    rows = []
    for t in (FULL, "abl_softmax"):
        r = runs[t]
        label = RUNS[t][1]
        temps = {k: fit_temperature(pr["val"][P7].to_numpy(), pr["val"].y.to_numpy().astype(int)) for k, pr in r.items()}
        for s in SETS:
            nat = pd.DataFrame([calib_row(pr[s]) for pr in r.values()])
            ts = pd.DataFrame([calib_row(with_probs(pr[s], apply_temperature(pr[s][P7].to_numpy(), temps[k])))
                               for k, pr in r.items()])
            maxp = pd.DataFrame([calib_row(with_probs(pr[s], pr[s][P7].to_numpy())) for pr in r.values()])
            ens_p = np.mean([pr[s][P7].to_numpy() for pr in r.values()], 0)
            ens = calib_row(with_probs(r[0][s], ens_p))
            score = "u = K/S (Eq. 14)" if t == FULL else "1 − max p"
            rows += [dict(model=label, variant=f"native, score {score}", set=SET_NAME[s], **{k: ms(nat[k]) for k in nat}),
                     dict(model=label, variant="+ temperature scaling", set=SET_NAME[s], **{k: ms(ts[k]) for k in ts},
                          T=ms(list(temps.values()), 2))]
            if t == FULL:
                rows.append(dict(model=label, variant="exploratory: score 1 − max p7", set=SET_NAME[s],
                                 **{k: ms(maxp[k]) for k in maxp}))
            rows.append(dict(model=label, variant=f"{len(r)}-model ensemble ({len(r)}× cost)", set=SET_NAME[s],
                             **{k: f"{v:.3f}" for k, v in ens.items()}))
    return pd.DataFrame(rows)


def table7_selective(run):
    rows = []
    for s in SETS:
        m = fold_table(run, lambda pr, t, ta: multiclass_metrics(pr[s], t, ta))
        bm = fold_table(run, lambda pr, t, ta: binary_metrics(pr[s], t, ta))
        for kap in KAPPAS:
            rows.append(dict(set=SET_NAME[s], kappa=kap, coverage=ms(m[f"coverage@{kap}"]),
                             acc_accepted=ms(m[f"acc_accepted@{kap}"]),
                             system_mal_sens=ms(bm[f"system_mal_sens@{kap}"]) if f"system_mal_sens@{kap}" in bm else "–"))
    return pd.DataFrame(rows)


def table7_fairness(runs):
    rows = []
    for t in (FULL, "abl_softmax", "base_resnet50"):
        r = runs.get(t)
        if r is None:
            continue
        label = RUNS[t][1] if t in RUNS else BASELINES[t][1]
        d0 = r[0]["pad_ufes"]
        for g in sorted(d0.skin_tone[d0.skin_tone > 0].unique()):
            g_rows = []
            for k, pr in r.items():
                t_mal, _ = thresholds(pr["val"])
                d = pr["pad_ufes"][pr["pad_ufes"].skin_tone == g]
                z, pm = d.z.to_numpy(), d.p_mal.to_numpy()
                g_rows.append(dict(auroc=roc_auc_score(z, pm) if len(set(z)) > 1 else np.nan,
                                   sens=((pm >= t_mal) & (z == 1)).sum() / max(1, z.sum()),
                                   spec=((pm < t_mal) & (z == 0)).sum() / max(1, (z == 0).sum())))
            gd = pd.DataFrame(g_rows)
            d = d0[d0.skin_tone == g]
            rows.append(dict(model=label, group=f"FST {str(g)[0]}–{str(g)[1]}" if g > 9 else str(g), n=len(d),
                             n_mal=int(d.z.sum()), n_benign=int((d.z == 0).sum()),
                             auroc=ms(gd.auroc), sens=ms(gd.sens), spec=ms(gd.spec)))
    n_unknown = int((runs[FULL][0]["pad_ufes"].skin_tone <= 0).sum())
    return pd.DataFrame(rows), n_unknown


def table8(runs, eff):
    full = runs[FULL]
    avail = [t for t in BASELINES if runs.get(t) is not None]
    bo = boot_bacc([full] + [runs[t] for t in avail], "test") if avail else None
    rows = []

    def perf(r):
        m = fold_table(r, lambda pr, a, b: multiclass_metrics(pr["test"], a, b))
        d7 = fold_table(r, lambda pr, a, b: multiclass_metrics(pr["derm7pt_test"], a, b))
        pad = fold_table(r, lambda pr, a, b: binary_metrics(pr["pad_ufes"], a, b))
        return dict(bacc=ms(m.bacc), macro_f1=ms(m.macro_f1), macro_auroc=ms(m.macro_auroc), mel_auroc=ms(m.mel_auroc),
                    ece=ms(m.ece), d7_bacc=ms(d7.bacc), pad_bin_auroc=ms(pad.auroc))

    def effrow(name):
        if eff is None or name not in set(eff.model):
            return {}
        e = eff[eff.model == name].iloc[0]
        return dict(params_M=f"{e.params_M:.2f}", GMACs=f"{e.GMACs:.2f}", size_MB=f"{e.size_MB_fp32:.1f}",
                    cpu_ms=f"{e.latency_ms_median:.1f}")

    rows.append(dict(model="TrustDerm-Net (ours)", **perf(full), **effrow("TrustDerm-Net (student)")))
    for j, t in enumerate(avail, start=1):
        delta = bo[:, 0] - bo[:, j]                               # ours - baseline
        lo, hi = boot_ci(delta)
        r = runs[t]
        rows.append(dict(model=BASELINES[t][1], **perf(r), **effrow(BASELINES[t][2]),
                         delta_bacc=f"[{lo:+.3f}, {hi:+.3f}]", p_boot=boot_p(delta),
                         mcnemar_p=";".join(f"{p:.2g}" for p in mcnemar_folds(full, r))))
    for t in BASELINES:
        if t not in avail:
            rows.append(dict(model=BASELINES[t][1] + " (pending)", **effrow(BASELINES[t][2])))
    rows.append(dict(model="Teacher ConvNeXt-B (reference)", **effrow("Teacher (ConvNeXt-B)")))
    df = pd.DataFrame(rows)
    if "p_boot" in df:
        mask = df.p_boot.notna()
        df.loc[mask, "p_holm"] = holm(df.loc[mask, "p_boot"])
    return df


# ----------------------------------------------------------------------------------------------- diagnostics
def pad_diagnostics(runs):
    """Why is melanoma AUROC below 0.5 on PAD-UFES-20?"""
    out = {}
    r = runs[FULL]
    rows = []
    for k, pr in r.items():
        d = pr["pad_ufes"]
        for ci in sorted(d.y.unique()):
            g = d[d.y == ci]
            rows.append(dict(fold=k, true=CLASSES[ci], n=len(g), mean_p_mel=g.p_MEL.mean(),
                             pred_as_mel=(g[P7].to_numpy().argmax(1) == MEL).mean()))
    out["p_mel_by_true_class"] = pd.DataFrame(rows).groupby("true")[["n", "mean_p_mel", "pred_as_mel"]].mean().round(3)
    pair = []
    for k, pr in r.items():
        d = pr["pad_ufes"]
        for other in ("NV", "BKL", "BCC", "AKIEC"):
            g = d[d.y.isin([MEL, CLASSES.index(other)])]
            pair.append(dict(fold=k, vs=other, auroc=roc_auc_score(g.y == MEL, g.p_MEL)))
    out["mel_vs_class_auroc"] = pd.DataFrame(pair).groupby("vs").auroc.agg(["mean", "std"]).round(3)
    mel_runs = {}
    for t in list(RUNS) + list(BASELINES):
        rr = runs.get(t)
        if rr is None:
            continue
        mel_runs[t] = ms([roc_auc_score(pr["pad_ufes"].y == MEL, pr["pad_ufes"].p_MEL) for pr in rr.values()])
    out["pad_mel_auroc_all_runs"] = mel_runs
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {t: load(t, f) for t, (f, _) in RUNS.items()}
    runs.update({t: load(t, v[0]) for t, v in BASELINES.items()})
    missing = [t for t, r in runs.items() if r is None]
    print("runs loaded:", [t for t in runs if runs[t] is not None], "| missing:", missing)
    eff_path = OUT / "efficiency.csv"
    eff = pd.read_csv(eff_path) if eff_path.exists() else None

    t4, macro = table4(runs[FULL])
    t5, t5c = table5(runs[FULL])
    t6 = table6(runs)
    t7c = table7_calibration(runs)
    t7s = table7_selective(runs[FULL])
    t7f, n_unknown = table7_fairness(runs)
    t8 = table8(runs, eff)
    diag = pad_diagnostics(runs)

    for name, df in dict(table4_per_class=t4, table5_external=t5, table5_per_class_sens=t5c, table6_ablation=t6,
                         table7_calibration=t7c, table7_selective=t7s, table7_fairness=t7f, table8_comparison=t8).items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    with open(OUT / "table4_macro.json", "w") as f:
        json.dump(macro, f, indent=1, default=lambda x: [round(float(v), 4) for v in x] if isinstance(x, tuple) else float(x))

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 40)
    with open(OUT / "step5_numbers.md", "w", encoding="utf-8") as f:
        def w(title, obj):
            f.write(f"\n## {title}\n\n```\n{obj if isinstance(obj, str) else obj.to_string()}\n```\n")
        w("Table IV per class (internal test)", t4.to_string(index=False))
        w("Table IV macro / triage", json.dumps(macro, indent=1, default=str))
        w("Table V external", t5.to_string(index=False))
        w("Table V per-class sensitivity", t5c.to_string(index=False))
        w("Table VI ablation", t6.to_string(index=False))
        w("Table VII calibration", t7c.to_string(index=False))
        w("Table VII selective prediction", t7s.to_string(index=False))
        w(f"Table VII fairness (PAD-UFES-20; {n_unknown} images without Fitzpatrick type)", t7f.to_string(index=False))
        w("Table VIII comparison", t8.to_string(index=False))
        for k, v in diag.items():
            w(f"PAD diagnostics: {k}", v if not isinstance(v, dict) else json.dumps(v, indent=1))
    print((OUT / "step5_numbers.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
