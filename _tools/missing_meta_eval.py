"""B1 of the guide's review: does TrustDerm-Net tolerate missing metadata at TEST time?

Runs the five saved student fold models on the 1,470 HAM10000 internal-test images under five conditions:
all fields present, age missing, sex missing, site missing, all missing. Missing fields are encoded exactly
as in training data (age = NaN, sex/site = -1), so the model substitutes its learned 'missing' tokens.

Images are re-created locally with the Kaggle preprocessing (code/preprocess.py), as for Fig. 8.
The 'all present' condition is checked against the saved Kaggle predictions (sanity check).
Triage uses each fold's validation-fixed t_mal (5th percentile of validation malignant p_mal), as in Section 3.10.

Usage: python _tools/missing_meta_eval.py prepare | run | summary
Output: results/step5/missing_meta/{per_fold.csv, summary.csv}
"""
import os, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CLASSES                                   # noqa: E402
from data import SkinDataset, eval_tf                        # noqa: E402
from model import build, probs_and_uncertainty               # noqa: E402
from preprocess import process                               # noqa: E402

OUT = ROOT / "results" / "step5" / "missing_meta"
CACHE = OUT / "cache"
HAM_ZIP = Path(os.environ.get("HAM_ZIP", "archive (1).zip"))    # path to the HAM10000 zip
CKPT = ROOT / "results" / "student_v1" / "ckpt"
PREDS = ROOT / "results" / "student_eval_v2" / "preds"
P7 = [f"p_{c}" for c in CLASSES]
N_FOLDS, S_STAR = 5, 0.95
CONDITIONS = ["all present", "age missing", "sex missing", "site missing", "all missing"]
torch.set_num_threads(4)


def test_df():
    m = pd.read_csv(ROOT / "results" / "student_v1" / "manifest.csv", dtype={"dhash": str}).set_index("image_id")
    ids = pd.read_csv(PREDS / "student_fold0.csv").query("eval_set == 'test'").image_id.tolist()
    d = m.loc[ids].reset_index()
    d["cache_path"] = [str(CACHE / f"{i}.jpg") for i in d.image_id]
    return d


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    d = test_df()
    z = zipfile.ZipFile(HAM_ZIP)
    ham = {n.rsplit("/", 1)[-1]: n for n in z.namelist() if n.startswith("HAM10000_images_part")}
    raw = OUT / "raw"; raw.mkdir(exist_ok=True)
    bad = 0
    for n, r in enumerate(d.itertuples()):
        dst = Path(r.cache_path)
        if dst.exists():
            continue
        src = raw / f"{r.image_id}.jpg"
        src.write_bytes(z.read(ham[f"{r.image_id}.jpg"]))
        h = process((str(src), str(dst), 448))
        bad += str(h) != str(r.dhash)
        src.unlink()
        if n % 100 == 0:
            print(n, "images prepared", flush=True)
    print("prepared", len(d), "images; dHash mismatches:", bad)


def masked(d, cond):
    d = d.copy()
    if cond in ("age missing", "all missing"):
        d["age"] = np.nan
    if cond in ("sex missing", "all missing"):
        d["sex"] = -1
    if cond in ("site missing", "all missing"):
        d["site"] = -1
    return d


@torch.no_grad()
def image_tokens(model, d, bs=64):
    """Backbone tokens (Eq. 4) once per fold; they do not depend on the metadata."""
    ds = SkinDataset(d, eval_tf(224))
    out = []
    for s in range(0, len(ds), bs):
        x = torch.stack([ds[i]["image"] for i in range(s, min(s + bs, len(ds)))])
        out.append(model.tokens(x))
        print(f"  tokens {min(s + bs, len(ds))}/{len(ds)}", flush=True)
    return torch.cat(out)


@torch.no_grad()
def predict(model, T, d, bs=256):
    """Fusion + heads only, on cached tokens T, with the (possibly masked) metadata of d."""
    age = torch.tensor(d.age.to_numpy(dtype=float), dtype=torch.float32)
    sex = torch.tensor(d.sex.to_numpy(dtype=int)); site = torch.tensor(d.site.to_numpy(dtype=int))
    real_tokens = model.tokens
    model.tokens = lambda t: t                     # forward() receives the cached tokens instead of images
    try:
        p7s, pms, us = [], [], []
        for s in range(0, len(d), bs):
            sl = slice(s, s + bs)
            out = model(T[sl], age[sl], sex[sl], site[sl])
            p7, pm, u = probs_and_uncertainty(out, model.hierarchical, model.evidential)
            p7s.append(p7.numpy()); pms.append(pm.numpy()); us.append(u.numpy())
    finally:
        model.tokens = real_tokens
    return np.concatenate(p7s), np.concatenate(pms), np.concatenate(us)


def run():
    d = test_df()
    y = d.y.to_numpy().astype(int); z = d.z.to_numpy().astype(int)
    rows = []
    done = OUT / "per_fold.csv"
    prev = pd.read_csv(done) if done.exists() else pd.DataFrame(columns=["fold", "condition"])
    for k in range(N_FOLDS):
        ck = torch.load(CKPT / f"student_fold{k}.pt", map_location="cpu", weights_only=False)
        model = build(ck["model_cfg"], pretrained=False); model.load_state_dict(ck["state_dict"]); model.eval()
        ref = pd.read_csv(PREDS / f"student_fold{k}.csv")
        val = ref[ref.eval_set == "val"]
        t_mal = float(np.quantile(val.p_mal[val.z == 1].to_numpy(), 1 - S_STAR, method="lower"))
        ref_t = ref[ref.eval_set == "test"].set_index("image_id").loc[d.image_id]
        todo = [c for c in CONDITIONS if not ((prev.fold == k) & (prev.condition == c)).any()]
        if not todo:
            continue
        T = image_tokens(model, d)
        for cond in todo:
            p7, pm, u = predict(model, T, masked(d, cond))
            pred = p7.argmax(1)
            row = dict(fold=k, condition=cond,
                       bacc=balanced_accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
                       macro_auroc=roc_auc_score(y, p7, multi_class="ovr", average="macro"),
                       triage_auroc=roc_auc_score(z, pm), triage_sens=float((pm[z == 1] >= t_mal).mean()),
                       triage_spec=float((pm[z == 0] < t_mal).mean()), mean_u=float(u.mean()))
            if cond == "all present":
                row["max_abs_dp_vs_kaggle"] = float(np.abs(p7 - ref_t[P7].to_numpy()).max())
                row["argmax_agree_vs_kaggle"] = float((pred == ref_t[P7].to_numpy().argmax(1)).mean())
            print(row, flush=True)
            rows.append(row)
            pd.concat([prev, pd.DataFrame(rows)]).to_csv(done, index=False)
    summary()


def summary():
    df = pd.read_csv(OUT / "per_fold.csv")
    cols = ["bacc", "macro_f1", "macro_auroc", "triage_auroc", "triage_sens", "triage_spec", "mean_u"]
    g = df.groupby("condition", sort=False)[cols]
    s = g.mean().round(4).astype(str) + " ± " + g.std(ddof=1).round(4).astype(str)
    s = s.reindex([c for c in CONDITIONS if c in s.index])
    s.to_csv(OUT / "summary.csv")
    print(s.to_string())
    full = df[df.condition == "all present"].set_index("fold")
    for c in CONDITIONS[1:]:
        o = df[df.condition == c].set_index("fold")
        if len(o) == len(full):
            dd = (o.bacc - full.bacc) * 100
            print(f"{c}: delta bAcc vs all present = {dd.mean():+.1f} ± {dd.std(ddof=1):.1f} points (per fold: {', '.join(f'{v:+.1f}' for v in dd)})")


if __name__ == "__main__":
    {"prepare": prepare, "run": run, "summary": summary}[sys.argv[1]]()
