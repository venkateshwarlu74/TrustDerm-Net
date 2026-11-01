"""Fig. 8 and Section V-F: Grad-CAM, concept activations and deletion/insertion faithfulness.

Runs locally on CPU with the saved fold checkpoints. Images are re-created from the original archives with the
same preprocessing as on Kaggle (code/preprocess.py). `check` verifies that every re-created image has the same
dHash as on Kaggle and that local predictions match the Kaggle prediction files.

Cases are selected by a fixed rule (fold-0 model, the most confident case of each category); the faithfulness
sample is a stratified random sample of HAM10000 test images (15 per class, seed 0), scored with all fold models.

Usage:
  python _tools/gradcam_fig8.py prepare     # extract + preprocess the needed images
  python _tools/gradcam_fig8.py check       # dHash and local-vs-Kaggle prediction check
  python _tools/gradcam_fig8.py faith       # deletion/insertion AUC (Grad-CAM vs random order), all folds
  python _tools/gradcam_fig8.py figure      # Fig. 8
"""
import math, os, subprocess, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CLASSES, CONCEPTS, IMAGENET_MEAN, IMAGENET_STD     # noqa: E402
from data import SkinDataset, eval_tf                                  # noqa: E402
from model import build, probs_and_uncertainty                         # noqa: E402
from preprocess import process                                         # noqa: E402

WORK = ROOT / "results" / "step5" / "fig8"
RAW, CACHE = WORK / "raw", WORK / "cache"
HAM_ZIP = Path(os.environ.get("HAM_ZIP", "archive (1).zip"))    # path to the HAM10000 zip
KAGGLE = Path(os.environ["APPDATA"]) / "Python" / "Python314" / "Scripts" / "kaggle.exe"
CKPT = ROOT / "results" / "student_v1" / "ckpt"
PREDS = ROOT / "results" / "student_eval_v2" / "preds"
P7 = [f"p_{c}" for c in CLASSES]
N_FOLDS = 5
STEPS, PER_CLASS, SEED = 32, 15, 0
torch.set_num_threads(4)

CASES = [  # (label, eval_set, true class, predicted class)
    ("Correct: melanoma", "test", "MEL", "MEL"),
    ("Correct: basal cell carcinoma", "test", "BCC", "BCC"),
    ("Correct: benign keratosis", "test", "BKL", "BKL"),
    ("Failure: melanoma called nevus", "test", "MEL", "NV"),
    ("Failure: nevus called melanoma", "test", "NV", "MEL"),
    ("Failure: PAD-UFES-20 melanoma called nevus", "pad_ufes", "MEL", "NV"),
]


# ------------------------------------------------------------------------------------------------ selection
def fold_preds(k):
    p = pd.read_csv(PREDS / f"student_fold{k}.csv")
    p["pred"] = p[P7].to_numpy().argmax(1)
    return p


def select_cases():
    p = fold_preds(0)
    rows = []
    for label, s, t, pr in CASES:
        ti, pi = CLASSES.index(t), CLASSES.index(pr)
        c = p[(p.eval_set == s) & (p.y == ti) & (p.pred == pi)].sort_values(f"p_{pr}", ascending=False)
        rows.append(dict(label=label, image_id=c.image_id.iloc[0], eval_set=s, cls=pi, n_candidates=len(c)))
    return pd.DataFrame(rows)


def select_sample():
    p = fold_preds(0)
    t = p[p.eval_set == "test"]
    rng = np.random.default_rng(SEED)
    ids = []
    for k in range(7):
        pool = sorted(t[t.y == k].image_id)
        ids += list(rng.choice(pool, size=min(PER_CLASS, len(pool)), replace=False))
    return ids


def manifest():
    return pd.read_csv(ROOT / "results" / "student_v1" / "manifest.csv", dtype={"dhash": str})


# ------------------------------------------------------------------------------------------------ prepare/check
def prepare():
    m = manifest().set_index("image_id")
    ids = list(dict.fromkeys(list(select_cases().image_id) + select_sample()))
    RAW.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(HAM_ZIP)
    ham = {n.rsplit("/", 1)[-1]: n for n in z.namelist() if n.startswith("HAM10000_images_part")}
    for i in ids:
        r = m.loc[i]
        src = RAW / Path(r.path).name
        if not src.exists():
            if r.dataset == "ham10000":
                src.write_bytes(z.read(ham[f"{i}.jpg"]))
            else:   # PAD-UFES-20 image from the public Kaggle copy
                rel = r.path.split("/skin-cancer/", 1)[1]
                subprocess.run([str(KAGGLE), "datasets", "download", "mahdavi1202/skin-cancer", "-f", rel, "-p", str(RAW)],
                               check=True, capture_output=True)
                zp = RAW / (Path(rel).name + ".zip")
                if zp.exists():
                    zipfile.ZipFile(zp).extract(Path(rel).name, RAW); zp.unlink()
        dst = CACHE / r.dataset / f"{i}.jpg"
        h = process((str(src), str(dst), 448))
        print(i, r.dataset, "dhash match" if str(h) == str(r.dhash) else f"DHASH DIFFERS ({h} vs {r.dhash})")


def local_df(ids):
    m = manifest().set_index("image_id").loc[ids].reset_index()
    m["cache_path"] = [str(CACHE / d / f"{i}.jpg") for d, i in zip(m.dataset, m.image_id)]
    return m


def load_model(k):
    ck = torch.load(CKPT / f"student_fold{k}.pt", map_location="cpu", weights_only=False)
    model = build(ck["model_cfg"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def batch(df):
    ds = SkinDataset(df, eval_tf(224))
    items = [ds[i] for i in range(len(ds))]
    return {k: torch.stack([b[k] if torch.is_tensor(b[k]) else torch.tensor(b[k]) for b in items]) for k in items[0]}


@torch.no_grad()
def predict(model, b):
    out = model(b["image"], b["age"], b["sex"], b["site"])
    p7, pm, u = probs_and_uncertainty(out, model.hierarchical, model.evidential)
    return p7, u, torch.sigmoid(out["concept_logits"])


def check():
    ids = list(dict.fromkeys(list(select_cases().image_id) + select_sample()))
    df = local_df(ids)
    for k in range(N_FOLDS):
        model = load_model(k)
        p7, _, _ = predict(model, batch(df))
        ref = fold_preds(k).drop_duplicates("image_id").set_index("image_id").loc[ids, P7].to_numpy()
        diff = np.abs(p7.numpy() - ref)
        agree = (p7.numpy().argmax(1) == ref.argmax(1)).mean()
        print(f"fold {k}: max |dp| = {diff.max():.4f}, mean |dp| = {diff.mean():.5f}, arg-max agreement = {agree:.3f}")


# ------------------------------------------------------------------------------------------------ Grad-CAM
def gradcam(model, b, cls):
    """Grad-CAM on the backbone's last feature map, target log p(y = cls) of the hierarchical evidential output."""
    store = {}
    orig = model.encoder.forward_features

    def ff(x):
        f = orig(x)
        f.retain_grad()
        store["f"] = f
        return f
    model.encoder.forward_features = ff
    try:
        out = model(b["image"], b["age"], b["sex"], b["site"])
        p7, _, _ = probs_and_uncertainty(out, model.hierarchical, model.evidential)
        model.zero_grad()
        torch.log(p7[torch.arange(len(cls)), cls]).sum().backward()
    finally:
        del model.encoder.forward_features
    f, g = store["f"], store["f"].grad
    cam = F.relu((g.mean((2, 3), keepdim=True) * f).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=b["image"].shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    lo, hi = cam.flatten(1).min(1).values[:, None, None], cam.flatten(1).max(1).values[:, None, None]
    return ((cam - lo) / (hi - lo + 1e-8)).detach()


@torch.no_grad()
def curve(model, x, meta, cls, order, mode, blurred):
    """Probability of `cls` as pixels are deleted (set to the dataset mean) or inserted into a blurred image."""
    C, H, W = x.shape
    n = H * W
    per = math.ceil(n / STEPS)
    ims = []
    for s in range(STEPS + 1):
        idx = order[:min(n, s * per)]
        if mode == "del":
            im = x.clone().view(C, -1); im[:, idx] = 0.0
        else:
            im = blurred.clone().view(C, -1); im[:, idx] = x.view(C, -1)[:, idx]
        ims.append(im.view(C, H, W))
    X = torch.stack(ims)
    r = len(X)
    p7, _, _ = predict(model, dict(image=X, age=meta["age"].repeat(r), sex=meta["sex"].repeat(r), site=meta["site"].repeat(r)))
    return p7[:, cls].numpy()


def faith():
    from torchvision.transforms.functional import gaussian_blur
    import gc
    ids = select_sample()
    df = local_df(ids)
    for k in range(N_FOLDS):                      # one file per fold, so an interrupted run resumes at the next fold
        if (WORK / f"faith_fold{k}.csv").exists() and (WORK / f"faith_curves_fold{k}.npz").exists():
            print(f"fold {k}: already done, skipped", flush=True)
            continue
        rows, curves = [], {c: [] for c in ("del_cam", "ins_cam", "del_rand", "ins_rand")}
        rng = np.random.default_rng(SEED + k)   # per-fold seed: identical results whether resumed or not
        model = load_model(k)
        b = batch(df)
        p7, _, _ = predict(model, b)
        cls = p7.argmax(1)
        cams = torch.cat([gradcam(model, {kk: v[i:i + 16] for kk, v in b.items()}, cls[i:i + 16])
                          for i in range(0, len(df), 16)])
        for i in range(len(df)):
            x = b["image"][i]
            meta = {kk: b[kk][i:i + 1] for kk in ("age", "sex", "site")}
            blurred = gaussian_blur(x, kernel_size=31, sigma=10.0)
            o_cam = torch.argsort(cams[i].flatten(), descending=True)
            o_rand = torch.from_numpy(rng.permutation(x.shape[1] * x.shape[2]))
            r = dict(fold=k, image_id=df.image_id[i], y=CLASSES[int(df.y[i])], pred=CLASSES[int(cls[i])],
                     correct=bool(cls[i] == int(df.y[i])))
            for name, order, mode in (("del_cam", o_cam, "del"), ("ins_cam", o_cam, "ins"),
                                      ("del_rand", o_rand, "del"), ("ins_rand", o_rand, "ins")):
                c = curve(model, x, meta, int(cls[i]), order, mode, blurred)
                curves[name].append(c)
                r[name] = float(np.trapezoid(c, dx=1 / STEPS))
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"fold {k}: {i + 1}/{len(df)}", flush=True)
        np.savez(WORK / f"faith_curves_fold{k}.npz", **{c: np.array(v) for c, v in curves.items()})
        pd.DataFrame(rows).to_csv(WORK / f"faith_fold{k}.csv", index=False)   # written last = fold complete
        print(f"fold {k}: saved", flush=True)
        del model, b, cams, curves, rows
        gc.collect()
    # combine all folds
    res = pd.concat([pd.read_csv(WORK / f"faith_fold{k}.csv") for k in range(N_FOLDS)], ignore_index=True)
    parts = [np.load(WORK / f"faith_curves_fold{k}.npz") for k in range(N_FOLDS)]
    res.to_csv(WORK / "faithfulness_per_image.csv", index=False)
    np.savez(WORK / "faithfulness_curves.npz", **{c: np.concatenate([p[c] for p in parts])
                                                  for c in ("del_cam", "ins_cam", "del_rand", "ins_rand")},
             fold=res.fold.to_numpy())
    per_fold = res.groupby("fold")[["del_cam", "del_rand", "ins_cam", "ins_rand"]].mean()
    print(per_fold.round(3))
    print("mean ± SD over folds:\n", per_fold.agg(["mean", "std"]).round(3))
    # paired test over images (all folds pooled per image: mean over folds)
    from scipy.stats import wilcoxon
    img = res.groupby("image_id")[["del_cam", "del_rand", "ins_cam", "ins_rand"]].mean()
    for a, bb in (("del_cam", "del_rand"), ("ins_cam", "ins_rand")):
        print(a, "vs", bb, "Wilcoxon over images: p =", wilcoxon(img[a], img[bb]).pvalue,
              "| share of images where Grad-CAM is better:",
              ((img[a] < img[bb]) if a.startswith("del") else (img[a] > img[bb])).mean().round(3))


BLOCK = 32   # 224 / 32 = 7: the resolution of the MobileViT feature map behind Grad-CAM


def block_order(rng, H, W):
    """Random order of BLOCK x BLOCK squares; pixels of one block are removed/revealed together."""
    gh, gw = H // BLOCK, W // BLOCK
    idx = torch.arange(H * W).view(H, W)
    blocks = [idx[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK].flatten() for r in range(gh) for c in range(gw)]
    return torch.cat([blocks[j] for j in rng.permutation(len(blocks))])


def faith_block():
    """Block-wise random baseline for deletion/insertion (same images, models and target class as `faith`)."""
    from torchvision.transforms.functional import gaussian_blur
    import gc
    df = local_df(select_sample())
    for k in range(N_FOLDS):
        if (WORK / f"faith_block_fold{k}.csv").exists() and (WORK / f"faith_block_curves_fold{k}.npz").exists():
            print(f"fold {k}: already done, skipped", flush=True)
            continue
        rng = np.random.default_rng(SEED + 100 + k)
        model = load_model(k)
        b = batch(df)
        p7, _, _ = predict(model, b)
        cls = p7.argmax(1)
        rows, curves = [], {"del_block": [], "ins_block": []}
        for i in range(len(df)):
            x = b["image"][i]
            meta = {kk: b[kk][i:i + 1] for kk in ("age", "sex", "site")}
            blurred = gaussian_blur(x, kernel_size=31, sigma=10.0)
            order = block_order(rng, x.shape[1], x.shape[2])
            r = dict(fold=k, image_id=df.image_id[i])
            for name, mode in (("del_block", "del"), ("ins_block", "ins")):
                c = curve(model, x, meta, int(cls[i]), order, mode, blurred)
                curves[name].append(c)
                r[name] = float(np.trapezoid(c, dx=1 / STEPS))
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"fold {k}: {i + 1}/{len(df)}", flush=True)
        np.savez(WORK / f"faith_block_curves_fold{k}.npz", **{c: np.array(v) for c, v in curves.items()})
        pd.DataFrame(rows).to_csv(WORK / f"faith_block_fold{k}.csv", index=False)
        print(f"fold {k}: saved", flush=True)
        del model, b, curves, rows
        gc.collect()
    blk = pd.concat([pd.read_csv(WORK / f"faith_block_fold{k}.csv") for k in range(N_FOLDS)], ignore_index=True)
    res = pd.read_csv(WORK / "faithfulness_per_image.csv").merge(blk, on=["fold", "image_id"], validate="one_to_one")
    res.to_csv(WORK / "faithfulness_per_image.csv", index=False)
    parts = [np.load(WORK / f"faith_block_curves_fold{k}.npz") for k in range(N_FOLDS)]
    old = dict(np.load(WORK / "faithfulness_curves.npz"))
    old.update({c: np.concatenate([p[c] for p in parts]) for c in ("del_block", "ins_block")})
    np.savez(WORK / "faithfulness_curves.npz", **old)
    cols = ["del_cam", "del_block", "del_rand", "ins_cam", "ins_block", "ins_rand"]
    per_fold = res.groupby("fold")[cols].mean()
    print(per_fold.round(3))
    print("mean ± SD over folds:\n", per_fold.agg(["mean", "std"]).round(3))
    from scipy.stats import wilcoxon
    img = res.groupby("image_id")[cols].mean()
    for a, bb in (("del_cam", "del_block"), ("ins_cam", "ins_block")):
        print(a, "vs", bb, "Wilcoxon over images: p =", wilcoxon(img[a], img[bb]).pvalue,
              "| share of images where Grad-CAM is better:",
              ((img[a] < img[bb]) if a.startswith("del") else (img[a] > img[bb])).mean().round(3))


# ------------------------------------------------------------------------------------------------ figure
SHORT = ["AN", "BWV", "AV", "IS", "IDG", "IB", "RS"]   # expanded in the caption
TITLES = ["Correct: MEL", "Correct: BCC", "Correct: BKL", "Failure: MEL → NV", "Failure: NV → MEL",
          "Failure (PAD): MEL → NV"]


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "_tools"))
    from figs_results import OI, W2, save   # shared style

    cases = select_cases()
    df = local_df(list(cases.image_id))
    model = load_model(0)
    b = batch(df)
    p7, u, chat = predict(model, b)
    cams = gradcam(model, b, torch.tensor(cases.cls.to_numpy()))
    mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]

    fig = plt.figure(figsize=(W2, 8.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.22)
    top = gs[0].subgridspec(3, 2, wspace=0.18, hspace=0.45)
    for j in range(len(cases)):
        cell = top[j % 3, j // 3].subgridspec(1, 3, width_ratios=[1, 1, 0.85], wspace=0.45)
        img = (b["image"][j] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        a0, a1, a2 = (fig.add_subplot(cell[i]) for i in range(3))
        a0.imshow(img); a1.imshow(img); a1.imshow(cams[j].numpy(), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        for a in (a0, a1):
            a.set_xticks([]); a.set_yticks([])
        k = int(cases.cls[j])
        a0.text(0, 1.16, TITLES[j], transform=a0.transAxes, fontsize=7, va="bottom", fontweight="bold")
        a0.text(0, 1.03, f"p({CLASSES[k]}) = {p7[j, k]:.2f}, u = {u[j]:.2f}", transform=a0.transAxes, fontsize=6,
                va="bottom")
        c = chat[j].numpy()
        a2.barh(range(7), c, color=[OI["vermillion"] if v >= 0.5 else OI["sky"] for v in c], height=0.7)
        a2.set_xlim(0, 1); a2.axvline(0.5, color=OI["grey"], lw=0.5, ls=":")
        a2.set_yticks(range(7), SHORT, fontsize=5.5)
        a2.invert_yaxis(); a2.tick_params(axis="x", labelsize=5.5)
        if j % 3 == 2:
            a2.set_xlabel("Concept probability", fontsize=6)
        if j == 0:
            a0.text(-0.12, 1.16, "(a)", transform=a0.transAxes, fontweight="bold", fontsize=8, ha="right", va="bottom")

    curves = np.load(WORK / "faithfulness_curves.npz")
    res = pd.read_csv(WORK / "faithfulness_per_image.csv")
    xs = np.linspace(0, 1, STEPS + 1)
    bottom = gs[1].subgridspec(1, 2, wspace=0.3)
    for i, (mode, title, letter) in enumerate((("del", "Deletion (lower AUC = more faithful)", "b"),
                                               ("ins", "Insertion (higher AUC = more faithful)", "c"))):
        ax = fig.add_subplot(bottom[i])
        srcs = [("cam", "Grad-CAM order", OI["blue"]), ("block", "Random 32-px blocks", OI["orange"]),
                ("rand", "Random pixels", OI["grey"])]
        for src, lab, col in [x for x in srcs if f"{mode}_{x[0]}" in curves]:
            arr = curves[f"{mode}_{src}"]
            per_fold = np.array([arr[curves["fold"] == k].mean(0) for k in range(N_FOLDS)])
            auc = res.groupby("fold")[f"{mode}_{src}"].mean()
            ax.plot(xs, per_fold.mean(0), color=col, label=f"{lab}, AUC {auc.mean():.3f} ± {auc.std(ddof=1):.3f}")
            ax.fill_between(xs, per_fold.min(0), per_fold.max(0), color=col, alpha=0.2, lw=0)
        ax.set_xlabel("Fraction of pixels " + ("removed" if mode == "del" else "revealed"))
        ax.set_ylabel("p(predicted class)")
        ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=6, loc="upper right" if mode == "del" else "upper left")
        ax.set_title(f"({letter}) {title}", loc="left")
    save(fig, "Fig8_explanations")


if __name__ == "__main__":
    {"prepare": prepare, "check": check, "faith": faith, "faith_block": faith_block, "figure": figure}[sys.argv[1]]()
