"""ISIC 2020 as a second external dermoscopic test set (evaluation only; post hoc).

Reads preds/{tag}_isic2020_fold{k}.csv written by code/eval_external.py (fold validation + all ISIC 2020 images).
Duplicate rule: the 64-bit dHash filter flags ~12k ISIC 2020 images, but pixel-level verification
(code/verify_dups.py, NCC of 64x64 grayscale, 8 flips/rotations) finds no pair with NCC >= 0.98 and visual checks
show different lesions, so an image is excluded only if its best NCC >= DUP_NCC.
ISIC 2020 labels: z = 1 means melanoma (the only malignant class); ~5.4k images also carry a fine diagnosis
(NV / BKL / MEL). Thresholds t_mal and tau_u come from each fold's validation split (as in evaluate.py).

Phase 1 (default): per-fold metrics, mean ± SD  -> results/step5/isic2020_table.csv, isic2020_per_fold.csv, isic2020.json
Phase 2 (--ci):     patient-level bootstrap of fold-averaged AUROCs, one model at a time, saved per model in
                    results/step5/isic2020_boot/{tag}.npy (resumable); then CIs, deltas vs main model, Holm
                    are added to isic2020_table.csv.
Usage: python _tools/isic2020_analysis.py            # phase 1
       python _tools/isic2020_analysis.py --ci --B 1000
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
import step5_analysis as S

ROOT = S.ROOT
DUP_NCC = 0.98
OURS = ["student", "abl_tau2", "abl_tau1", "abl_kdnone", "teacher"]
FOLDERS = {t: "results/isic2020_ours" for t in OURS}
FOLDERS.update({t: "results/isic2020_base" for t in S.BASELINES})
LABEL = {"student": "TrustDerm-Net (main, tau = 4)", "abl_tau2": "KD tau = 2", "abl_tau1": "KD tau = 1",
         "abl_kdnone": "w/o distillation", "teacher": "Teacher (ConvNeXt-B)"}
LABEL.update({t: v[1] for t, v in S.BASELINES.items()})
COLS = ["image_id", "eval_set", "y", "z", "p_mal", "u"] + S.P7
BOOT = S.OUT / "isic2020_boot"
METRICS = ("triage_auroc", "mel_auroc")


def load(tag):
    """fold -> (val, isic) with only the needed columns, float32."""
    out = {}
    for f in sorted((ROOT / FOLDERS[tag] / "preds").glob(f"{tag}_isic2020_fold*.csv")):
        k = int(f.stem.rsplit("_fold", 1)[1])
        p = pd.read_csv(f, usecols=COLS, dtype={c: np.float32 for c in COLS[4:]})
        out[k] = {n: g.drop(columns="eval_set").sort_values("image_id").reset_index(drop=True)
                  for n, g in p.groupby("eval_set", sort=False)}
    return out or None


def keep_ids(ref_ids):
    v = pd.read_csv(ROOT / "results" / "dupcheck_isic" / "dup_verify_isic2020.csv")
    excl = set(v.image_id[v.ncc >= DUP_NCC])
    return np.array(sorted(set(ref_ids) - excl)), len(excl), len(v)


def fold_metrics(d, t_mal, taus):
    z, pm, pmel = d.z.to_numpy().astype(int), d.p_mal.to_numpy(), d.p_MEL.to_numpy()
    flag = pm >= t_mal
    r = dict(triage_auroc=roc_auc_score(z, pm), mel_auroc=roc_auc_score(z, pmel),
             sens=flag[z == 1].mean(), spec=(~flag[z == 0]).mean(), flagged=flag.mean(),
             ppv=(flag & (z == 1)).sum() / max(1, flag.sum()))
    lab = d[d.y >= 0]
    y = lab.y.to_numpy().astype(int); yhat = lab[S.P7].to_numpy().argmax(1)
    r["bacc_labelled"] = np.mean([(yhat[y == c] == c).mean() for c in np.unique(y)])
    acc = d.u.to_numpy() <= taus[0.8]
    r["coverage@0.8"] = acc.mean()
    r["system_sens@0.8"] = ((flag & acc) | ~acc)[z == 1].mean()
    return r


def isic_scores(tag, ids):
    """fold -> (z, p_mal, p_MEL, t_mal, taus) restricted to ids (sorted)."""
    r = load(tag)
    out = {}
    for k, pr in r.items():
        d = pr["isic2020"]
        d = d[d.image_id.isin(ids)].sort_values("image_id").reset_index(drop=True)
        assert (d.image_id.to_numpy() == ids).all()
        out[k] = (d, *S.thresholds(pr["val"]))
    return out


def phase1():
    ref = load("student")[0]["isic2020"]
    ids, n_excl, n_flag = keep_ids(ref.image_id)
    man = pd.read_csv(ROOT / "results" / "prep_isic" / "manifest.csv", usecols=["image_id", "group"])
    r0 = ref[ref.image_id.isin(ids)]
    info = dict(n_total=len(ref), n_dhash_flagged=n_flag, n_excluded_after_ncc=n_excl, n_evaluated=len(ids),
                n_melanoma=int(r0.z.sum()), n_fine_label=int((r0.y >= 0).sum()),
                fine_label_counts={S.CLASSES[int(c)]: int(n) for c, n in r0.y[r0.y >= 0].value_counts().items()},
                n_patients=int(man[man.image_id.isin(ids)].group.nunique()), dup_rule=f"dHash<=3 AND NCC>={DUP_NCC}")
    per, rows = {}, []
    for t in FOLDERS:
        sc = isic_scores(t, ids)
        p = pd.DataFrame([dict(fold=k, t_mal=tm, **fold_metrics(d, tm, ta)) for k, (d, tm, ta) in sc.items()]).set_index("fold")
        per[t] = p
        rows.append(dict(tag=t, model=LABEL[t], **{c: S.ms(p[c]) for c in p.columns}))
        print("done", t, flush=True)
        del sc
    df = pd.DataFrame(rows)
    S.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(S.OUT / "isic2020_table.csv", index=False)
    pd.concat(per).to_csv(S.OUT / "isic2020_per_fold.csv")
    (S.OUT / "isic2020.json").write_text(json.dumps(info, indent=1))
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(json.dumps(info, indent=1))
    print(df.drop(columns=["model"]).to_string(index=False))


def fast_auc(z, s):
    """AUROC via Mann-Whitney ranks (ties averaged)."""
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
    if (cnt > 1).any():   # average ranks of ties
        csum = np.cumsum(cnt)
        avg = csum - (cnt - 1) / 2.0
        ranks[order] = avg[inv]
    n1 = z.sum(); n0 = len(z) - n1
    return (ranks[z == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def phase2(B):
    ref = load("student")[0]["isic2020"]
    ids, _, _ = keep_ids(ref.image_id)
    del ref
    man = pd.read_csv(ROOT / "results" / "prep_isic" / "manifest.csv", usecols=["image_id", "group"]).set_index("image_id")
    grp = man.loc[ids, "group"].to_numpy()
    uniq, inv = np.unique(grp, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv))]
    BOOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(S.SEED)
    picks = rng.integers(0, len(uniq), size=(B, len(uniq)), dtype=np.int32)   # same resamples for every model
    for t in FOLDERS:
        f = BOOT / f"{t}.npy"
        if f.exists() and np.load(f).shape[0] == B:
            print("skip", t); continue
        sc = isic_scores(t, ids)
        arrs = [(d.z.to_numpy().astype(np.int8), d.p_mal.to_numpy(), d.p_MEL.to_numpy()) for d, _, _ in sc.values()]
        del sc
        res = np.zeros((B, 2))
        for b in range(B):
            g = picks[b]
            idx = np.concatenate([order[starts[i]:starts[i + 1]] for i in g])
            ta = [fast_auc(z[idx], pm[idx]) for z, pm, _ in arrs]
            ma = [fast_auc(z[idx], pe[idx]) for z, _, pe in arrs]
            res[b] = np.mean(ta), np.mean(ma)
        np.save(f, res)
        print("boot done", t, flush=True)
    df = pd.read_csv(S.OUT / "isic2020_table.csv")
    per = pd.read_csv(S.OUT / "isic2020_per_fold.csv", index_col=[0, 1])
    base = np.load(BOOT / "student.npy")
    for i, t in enumerate(df.tag):
        bt = np.load(BOOT / f"{t}.npy")
        for j, m in enumerate(METRICS):
            df.loc[i, f"{m}_ci"] = f"[{np.percentile(bt[:, j], 2.5):.3f}, {np.percentile(bt[:, j], 97.5):.3f}]"
            if t != "student":
                delta = bt[:, j] - base[:, j]
                df.loc[i, f"{m}_delta_vs_main"] = (f"{per.loc[t][m].mean() - per.loc['student'][m].mean():+.3f} "
                                                   f"[{np.percentile(delta, 2.5):+.3f}, {np.percentile(delta, 97.5):+.3f}]")
                df.loc[i, f"{m}_p_boot"] = max(2 * min((delta <= 0).mean(), (delta >= 0).mean()), 1 / B)
    for m in METRICS:
        mask = df[f"{m}_p_boot"].notna()
        df.loc[mask, f"{m}_p_holm"] = S.holm(df.loc[mask, f"{m}_p_boot"])
    df.to_csv(S.OUT / "isic2020_table.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(df[["tag"] + [c for c in df.columns if any(c.startswith(m) for m in METRICS)]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    phase2(a.B) if a.ci else phase1()
