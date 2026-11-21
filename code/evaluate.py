"""Predict with every fold checkpoint of a run and compute all metrics of Section IV-E.

For each fold: thresholds t_mal (Eq. 21) and tau_u (Eq. 22) are fixed on that fold's validation split,
then applied unchanged to the internal test set, the Derm7pt test split and the external sets.
Outputs: preds/{tag}_fold{k}.csv (one row per image), results/{tag}_fold{k}.json, results/{tag}_summary.csv

Usage: python evaluate.py --tag student [--s_star 0.95]
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from config import CKPT, CLASSES, CONCEPTS, DERM7PT_TEST, EXTERNAL, MANIFEST, MEL, PREDS, RESULTS
from data import SkinDataset, eval_tf
from model import build, probs_and_uncertainty

KAPPAS = (0.7, 0.8, 0.9, 1.0)


@torch.no_grad()
def predict(model, df, dev, bs=128, workers=4):
    dl = DataLoader(SkinDataset(df, eval_tf(model.cfg["img_size"])), bs, num_workers=workers, pin_memory=True)
    P, PM, U, C = [], [], [], []
    model.eval()
    for b in dl:
        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(b["image"], b["age"], b["sex"], b["site"])
        p7, pm, u = probs_and_uncertainty(out, model.hierarchical, model.evidential)
        c = torch.sigmoid(out["concept_logits"].float()) if "concept_logits" in out \
            else torch.full((len(p7), len(CONCEPTS)), float("nan"), device=dev)
        P.append(p7.cpu()); PM.append(pm.cpu()); U.append(u.cpu()); C.append(c.cpu())
    pred = df[["image_id", "dataset", "split", "y", "z", "skin_tone"] + CONCEPTS].reset_index(drop=True).copy()
    pred[[f"p_{c}" for c in CLASSES]] = torch.cat(P).numpy()
    pred["p_mal"], pred["u"] = torch.cat(PM).numpy(), torch.cat(U).numpy()
    pred[[f"chat_{c}" for c in CONCEPTS]] = torch.cat(C).numpy()
    return pred


def decide(p7, p_mal, t_mal):
    """Fine-grained decision after the Stage-1 malignancy threshold."""
    mal = p_mal >= t_mal
    return np.where(mal, 4 + p7[:, 4:].argmax(1), p7[:, :4].argmax(1))


def ece(conf, correct, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def aurc(u, correct):
    order = np.argsort(u)
    risk = np.cumsum(1 - correct[order]) / np.arange(1, len(u) + 1)
    return risk.mean()


def safe_auc(y, s, ap=False):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, s) if ap else roc_auc_score(y, s)


def multiclass_metrics(d, t_mal, taus):
    """Fine-grained metrics use the arg-max of the 7-class distribution. The high-sensitivity threshold
    t_mal is a triage operating point (binary metrics, referral); the 7-class accuracy obtained when
    t_mal also routes the fine-grained label is reported separately as bacc_routed."""
    p7 = d[[f"p_{c}" for c in CLASSES]].to_numpy()
    y = d.y.to_numpy().astype(int)
    yhat = p7.argmax(1)
    present = sorted(np.unique(y))
    r = dict(n=len(d), bacc=balanced_accuracy_score(y, yhat),
             bacc_routed=balanced_accuracy_score(y, decide(p7, d.p_mal.to_numpy(), t_mal)),
             macro_f1=f1_score(y, yhat, labels=present, average="macro", zero_division=0),
             macro_auroc=np.nanmean([safe_auc(y == k, p7[:, k]) for k in present]))
    for k in present:
        r[f"sens_{CLASSES[k]}"] = ((yhat == k) & (y == k)).sum() / max(1, (y == k).sum())
        r[f"spec_{CLASSES[k]}"] = ((yhat != k) & (y != k)).sum() / max(1, (y != k).sum())
    if MEL in present:
        r["mel_auroc"], r["mel_auprc"] = safe_auc(y == MEL, p7[:, MEL]), safe_auc(y == MEL, p7[:, MEL], ap=True)
    correct = (yhat == y).astype(float)
    r["ece"] = ece(p7.max(1), correct)
    r["brier"] = np.mean(((p7 - np.eye(7)[y]) ** 2).sum(1))
    r["aurc"] = aurc(d.u.to_numpy(), correct)
    r.update(selective(d, yhat == y, t_mal, taus))
    return r


def fold_metrics(preds, k, s_star):
    """All metrics of one fold from its prediction tables (dict eval_set -> DataFrame)."""
    v = preds["val"]
    pos = v.p_mal[v.z == 1].to_numpy()
    t_mal = float(np.quantile(pos, 1 - s_star, method="lower"))           # Eq. 21
    taus = {kap: float(np.quantile(v.u, kap)) for kap in KAPPAS}           # Eq. 22
    res = dict(fold=k, t_mal=t_mal, tau_u=taus)
    for n, d in preds.items():
        if n == "val":
            continue
        if n in ("test", DERM7PT_TEST, "pad_ufes"):
            res[n] = multiclass_metrics(d, t_mal, taus)
            res[n].update({f"bin_{x}": y for x, y in binary_metrics(d, t_mal, taus).items()})
            if n == DERM7PT_TEST:
                res[n].update(concept_metrics(d))
        else:
            res[n] = binary_metrics(d, t_mal, taus)
    return res


def binary_metrics(d, t_mal, taus):
    z, pm = d.z.to_numpy().astype(int), d.p_mal.to_numpy()
    flag = pm >= t_mal
    r = dict(n=len(d), n_malignant=int(z.sum()), auroc=safe_auc(z, pm), auprc=safe_auc(z, pm, ap=True),
             sens=(flag & (z == 1)).sum() / max(1, z.sum()), spec=(~flag & (z == 0)).sum() / max(1, (z == 0).sum()))
    r["ece"] = ece(np.maximum(pm, 1 - pm), (flag == (z == 1)).astype(float))
    r["brier"] = np.mean((pm - z) ** 2)
    if d.dataset.iloc[0] == "isic2020":    # malignant = melanoma in ISIC 2020
        r["mel_auroc"] = safe_auc(z, d[f"p_{CLASSES[MEL]}"].to_numpy())
    r.update(selective(d, flag == (z == 1), t_mal, taus))
    if (d.skin_tone > 0).any():              # skin-tone (Fitzpatrick) analysis; not used in the paper
        for st, g in d[d.skin_tone > 0].groupby("skin_tone"):
            zz, ff = g.z.to_numpy(), g.p_mal.to_numpy() >= t_mal
            r[f"auroc_FST{st}"] = safe_auc(zz, g.p_mal.to_numpy())
            r[f"sens_FST{st}"] = (ff & (zz == 1)).sum() / max(1, zz.sum())
            r[f"spec_FST{st}"] = (~ff & (zz == 0)).sum() / max(1, (zz == 0).sum())
        groups = [r[k] for k in r if k.startswith("auroc_FST")]
        r["auroc_gap"] = np.nanmax(groups) - np.nanmin(groups)
        sens = [r[k] for k in r if k.startswith("sens_FST")]
        r["sens_gap"] = np.nanmax(sens) - np.nanmin(sens)
    return r


def selective(d, correct, t_mal, taus):
    """Coverage-controlled referral: accuracy on accepted cases and system malignancy sensitivity
    (malignant lesions either flagged by the model or referred to the dermatologist)."""
    r, u, z = {}, d.u.to_numpy(), d.z.to_numpy().astype(int)
    flag = d.p_mal.to_numpy() >= t_mal
    for kappa, tau in taus.items():
        acc = u <= tau
        r[f"coverage@{kappa}"] = acc.mean()
        r[f"acc_accepted@{kappa}"] = correct[acc].mean() if acc.any() else np.nan
        if (z == 1).any():
            caught = (flag & acc & (z == 1)) | (~acc & (z == 1))
            r[f"system_mal_sens@{kappa}"] = caught.sum() / (z == 1).sum()
    return r


def concept_metrics(d):
    r = {}
    for c in CONCEPTS:
        y, s = d[c].to_numpy(), d[f"chat_{c}"].to_numpy()
        if np.isnan(s).all() or (y < 0).all():
            continue
        r[f"auroc_{c}"] = safe_auc(y, s)
        r[f"acc_{c}"] = ((s >= 0.5) == (y == 1)).mean()
    if r:
        r["concept_mean_auroc"] = np.nanmean([v for k, v in r.items() if k.startswith("auroc_")])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--s_star", type=float, default=0.95)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--from_preds", action="store_true",
                    help="recompute metrics from saved preds/{tag}_fold*.csv (no model, no images)")
    a = ap.parse_args()
    os.makedirs(PREDS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    rows = []
    if a.from_preds:
        runs = [(int(p.rsplit("_fold", 1)[1][:-4]), p) for p in sorted(glob.glob(os.path.join(PREDS, f"{a.tag}_fold*.csv")))]
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = pd.read_csv(MANIFEST, dtype={"dhash": str})
        runs = [(None, p) for p in sorted(glob.glob(os.path.join(CKPT, f"{a.tag}_fold*.pt")))]
    for k, path in runs:
        if a.from_preds:
            all_p = pd.read_csv(path)
            preds = {n: g.drop(columns="eval_set").reset_index(drop=True) for n, g in all_p.groupby("eval_set", sort=False)}
        else:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = build(ck["model_cfg"], pretrained=False)
            model.load_state_dict(ck["state_dict"]); model.to(dev)
            k = ck["fold"]
            sets = {"val": m[m.split == f"fold{k}"], "test": m[m.split == "test"],
                    DERM7PT_TEST: m[m.split == DERM7PT_TEST]}
            sets.update({e: m[m.split == e] for e in EXTERNAL})
            preds = {n: predict(model, df, dev, workers=a.workers) for n, df in sets.items() if len(df)}
            pd.concat([d.assign(eval_set=n) for n, d in preds.items()]).to_csv(
                os.path.join(PREDS, f"{a.tag}_fold{k}.csv"), index=False)
        res = fold_metrics(preds, k, a.s_star)
        t_mal = res["t_mal"]
        with open(os.path.join(RESULTS, f"{a.tag}_fold{k}.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
        for n in res:
            if isinstance(res[n], dict) and n != "tau_u":
                rows += [dict(fold=k, set=n, metric=key, value=val) for key, val in res[n].items()]
        print(f"fold {k}: t_mal={t_mal:.3f}  test bAcc={res['test']['bacc']:.4f}")
    df = pd.DataFrame(rows)
    summ = df.groupby(["set", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
    summ.to_csv(os.path.join(RESULTS, f"{a.tag}_summary.csv"), index=False)
    df.to_csv(os.path.join(RESULTS, f"{a.tag}_per_fold.csv"), index=False)
    key = summ[summ.metric.isin(["bacc", "bacc_routed", "macro_f1", "macro_auroc", "mel_auroc", "auroc", "ece",
                                 "aurc", "bin_auroc", "bin_sens", "bin_spec", "system_mal_sens@0.8",
                                 "acc_accepted@0.8", "bin_auroc_gap", "concept_mean_auroc"])]
    print("\n" + key.to_string(index=False))


if __name__ == "__main__":
    main()
