"""Figs. 4-7 of the paper (Section V) from the saved prediction files.

Style: IEEE widths (3.5 in single column, 7.16 in double), 8 pt fonts, Okabe-Ito colour-blind-safe palette,
vector PDF/SVG plus 300 dpi PNG. Curves are averaged over all fold models (mean, with +-1 SD band where
shown); confusion matrices sum the counts of all fold models and normalise each row.

Usage: python _tools/figs_results.py [4 5 6 7]     -> figures/Fig{n}_*.{pdf,svg,png}
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step5_analysis import (BASELINES, CLASSES, FULL, P7, ROOT, RUNS, SET_NAME, apply_temperature,  # noqa: E402
                            fit_temperature, load, thresholds)

FIG = ROOT / "figures"
W1, W2 = 3.5, 7.16
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#999999")
CLASS_COL = dict(NV=OI["sky"], BKL=OI["green"], DF=OI["yellow"], VASC=OI["purple"], MEL=OI["vermillion"],
                 BCC=OI["orange"], AKIEC=OI["blue"])
SET_COL = {"test": OI["blue"], "derm7pt_test": OI["orange"], "pad_ufes": OI["vermillion"]}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0, "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


def panel(ax, letter):
    """Prefix the panel letter to the (left-aligned) title, or place it alone when the axes has no title."""
    t = ax.get_title()
    if t:
        ax.set_title(f"({letter}) {t}", loc="left"); ax.set_title("")
    else:
        ax.set_title(f"({letter})", loc="left", fontweight="bold")


# ------------------------------------------------------------------------------------------------ Fig. 4
def fig4():
    m = pd.read_csv(ROOT / "results/student_v1/manifest.csv", usecols=["image_id", "dataset", "split", "y"])
    m = m[m.split != "excluded_duplicate"]
    parts = [("HAM10000 dev. (all 5 folds)", (m.dataset == "ham10000") & m.split.str.startswith("fold")),
             ("HAM10000 internal test", (m.dataset == "ham10000") & (m.split == "test")),
             ("Derm7pt dev. (all 5 folds)", (m.dataset == "derm7pt") & m.split.str.startswith("fold")),
             ("Derm7pt test", m.split == "derm7pt_test"),
             ("PAD-UFES-20 (external)", m.dataset == "pad_ufes")]
    cols = [OI["blue"], OI["sky"], OI["orange"], OI["yellow"], OI["vermillion"]]
    fig = plt.figure(figsize=(W2, 3.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1], hspace=0.22)
    ax = fig.add_subplot(gs[0])
    x, bw = np.arange(7), 0.16
    for j, ((lab, mask), c) in enumerate(zip(parts, cols)):
        cnt = np.bincount(m[mask].y.astype(int), minlength=7)
        bars = ax.bar(x + (j - 2) * bw, np.where(cnt > 0, cnt, np.nan), bw, color=c, edgecolor="black",
                      linewidth=0.3, label=f"{lab}, n = {cnt.sum():,}")
        for b, v in zip(bars, cnt):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_yscale("log"); ax.set_ylim(1, 2e4)
    ax.set_xticks(x, CLASSES); ax.set_ylabel("Images (log scale)")
    ax.text(-0.085, 1.07, "(a)", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")
    ax.legend(ncol=3, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=1.2, columnspacing=1, fontsize=6.5)
    samples = [("figures/samples/ISIC_0024314.jpg", "NV"), ("figures/samples/ISIC_0024337.jpg", "BKL"),
               ("figures/samples/ISIC_0024318.jpg", "DF"), ("figures/samples/ISIC_0024370.jpg", "VASC"),
               ("figures/samples/ISIC_0024310.jpg", "MEL"), ("figures/samples/ISIC_0024360.jpg", "BCC"),
               ("figures/samples/ISIC_0024763.jpg", "AKIEC"),
               ("figures/samples/PAT_100_393_595.png", "NV"), ("figures/samples/PAT_109_868_113.png", "MEL"),
               ("figures/samples/PAT_101_1041_651.png", "BCC"), ("figures/samples/PAT_1006_53_385.png", "AKIEC")]
    sub = gs[1].subgridspec(1, len(samples), wspace=0.05)
    for i, (p, lab) in enumerate(samples):
        a = fig.add_subplot(sub[i])
        im = Image.open(ROOT / p).convert("RGB")
        s = min(im.size); l, t = (im.width - s) // 2, (im.height - s) // 2
        a.imshow(im.crop((l, t, l + s, t + s)).resize((256, 256)))
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(OI["blue"] if i < 7 else OI["vermillion"]); sp.set_linewidth(1.2)
        a.set_title(lab, fontsize=7, pad=2)
        if i == 0:
            a.text(-0.04, 1.2, "(b)", transform=a.transAxes, ha="right", va="bottom", fontweight="bold")
            a.text(0, -0.12, "HAM10000 (dermoscopic)", transform=a.transAxes, va="top", fontsize=7, color=OI["blue"])
        if i == 7:
            a.text(0, -0.12, "PAD-UFES-20 (smartphone clinical)", transform=a.transAxes, va="top", fontsize=7,
                   color=OI["vermillion"])
    save(fig, "Fig4_data_distribution")


# ------------------------------------------------------------------------------------------------ Fig. 5
def confusion(run, s):
    cm = np.zeros((7, 7))
    for pr in run.values():
        d = pr[s]
        y, yh = d.y.to_numpy().astype(int), d[P7].to_numpy().argmax(1)
        np.add.at(cm, (y, yh), 1)
    return cm


def fig5(run):
    cmap = LinearSegmentedColormap.from_list("w2b", ["#FFFFFF", OI["blue"]])
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.55), gridspec_kw=dict(wspace=0.45))
    for ax, s, letter in zip(axes, ["test", "derm7pt_test", "pad_ufes"], "abc"):
        cm = confusion(run, s)
        rows = [k for k in range(7) if cm[k].sum() > 0]
        cmn = cm[rows] / cm[rows].sum(1, keepdims=True)
        ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(rows)):
            for j in range(7):
                v = cmn[i, j]
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.55 else "black")
        ax.set_xticks(range(7), CLASSES, rotation=90)
        ax.set_yticks(range(len(rows)), [f"{CLASSES[k]} ({int(cm[k].sum() / len(run))})" for k in rows])
        ax.axvline(3.5, color=OI["vermillion"], lw=0.8, ls="--")
        b = sum(1 for k in rows if k < 4)
        ax.axhline(b - 0.5, color=OI["vermillion"], lw=0.8, ls="--")
        ax.set_xlabel("Predicted"); ax.set_title(SET_NAME[s])
        for sp in ax.spines.values():
            sp.set_visible(True)
        if letter == "a":
            ax.set_ylabel("True class (n)")
        panel(ax, letter)
    save(fig, "Fig5_confusion")


# ------------------------------------------------------------------------------------------------ Fig. 6
GRID = np.linspace(0, 1, 201)


def mean_curve(run, s, score, label_fn, kind="roc"):
    ys = []
    for pr in run.values():
        d = pr[s]
        y, sc = label_fn(d), score(d)
        if kind == "roc":
            f, t, _ = roc_curve(y, sc)
            ys.append(np.interp(GRID, f, t)); ys[-1][0] = 0
        else:
            p, r, _ = precision_recall_curve(y, sc)
            ys.append(np.interp(GRID, r[::-1], p[::-1]))
    ys = np.array(ys)
    return ys.mean(0), ys.std(0, ddof=1)


def fig6(run):
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.4), gridspec_kw=dict(wspace=0.35))
    ax = axes[0]
    for k, c in enumerate(CLASSES):
        mu, sd = mean_curve(run, "test", lambda d: d[f"p_{c}"], lambda d: d.y == k)
        mel = c == "MEL"
        ax.plot(GRID, mu, color=CLASS_COL[c], lw=1.8 if mel else 0.9, zorder=3 if mel else 2, label=c)
        if mel:
            ax.fill_between(GRID, mu - sd, mu + sd, color=CLASS_COL[c], alpha=0.2, lw=0)
    ax.plot([0, 1], [0, 1], color=OI["grey"], lw=0.5, ls=":")
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.set_title("One-vs-rest ROC (HAM test)"); ax.legend(frameon=False, loc="lower right", ncol=2)
    panel(ax, "a")
    ax = axes[1]
    for s in ["test", "derm7pt_test", "pad_ufes"]:
        mu, sd = mean_curve(run, s, lambda d: d.p_MEL, lambda d: d.y == 4, kind="pr")
        prev = (run[0][s].y == 4).mean()
        ax.plot(GRID, mu, color=SET_COL[s], label=SET_NAME[s].replace(" internal", ""))
        ax.fill_between(GRID, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=SET_COL[s], alpha=0.15, lw=0)
        ax.axhline(prev, color=SET_COL[s], lw=0.6, ls=":")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Melanoma precision–recall"); ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.2))
    panel(ax, "b")
    ax = axes[2]
    for s in ["test", "derm7pt_test", "pad_ufes"]:
        mu, sd = mean_curve(run, s, lambda d: d.p_mal, lambda d: d.z == 1)
        ax.plot(GRID, mu, color=SET_COL[s], label=SET_NAME[s].replace(" internal", ""))
        ax.fill_between(GRID, mu - sd, mu + sd, color=SET_COL[s], alpha=0.15, lw=0)
        ops = []
        for pr in run.values():
            t_mal, _ = thresholds(pr["val"])
            d = pr[s]
            f = (d.p_mal >= t_mal)
            ops.append(((f & (d.z == 0)).sum() / (d.z == 0).sum(), (f & (d.z == 1)).sum() / (d.z == 1).sum()))
        ops = np.array(ops)
        ax.plot(ops[:, 0].mean(), ops[:, 1].mean(), "o", ms=4, mfc="white", mec=SET_COL[s], mew=1.2, zorder=4)
    ax.plot([0, 1], [0, 1], color=OI["grey"], lw=0.5, ls=":")
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity (malignant)")
    ax.set_title("Benign/malignant triage"); ax.legend(frameon=False, loc="lower right")
    panel(ax, "c")
    save(fig, "Fig6_roc_pr")


# ------------------------------------------------------------------------------------------------ Fig. 7
def reliability(ds, bins=15):
    p = np.concatenate([d[P7].to_numpy() for d in ds]); y = np.concatenate([d.y.to_numpy() for d in ds]).astype(int)
    conf, corr = p.max(1), (p.argmax(1) == y)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (conf > lo) & (conf <= hi)
        if mk.sum() >= 10:
            xs.append(conf[mk].mean()); ys.append(corr[mk].mean())
    return np.array(xs), np.array(ys)


def risk_coverage(ds, score):
    cov = np.linspace(0.05, 1, 96)
    out = []
    for d in ds:
        y = d.y.to_numpy().astype(int)
        err = (d[P7].to_numpy().argmax(1) != y)[np.argsort(score(d))]
        risk = np.cumsum(err) / np.arange(1, len(err) + 1)
        out.append(risk[np.maximum(1, np.round(cov * len(err)).astype(int)) - 1])
    out = np.array(out)
    return cov, out.mean(0), out.std(0, ddof=1)


LABEL_POS = {  # (dx, dy, ha) in points; short names, parameter count is shown by marker size and Table VIII.
    # Each label sits off its own marker with a thin leader line, so no label lies next to another model's marker.
    "TrustDerm-Net": ("Ours", 8, 12, "left"), "MobileNetV2": ("MobileNetV2", -8, -30, "left"),
    "EfficientNet-B3": ("EffNet-B3", -3, -11, "left"), "ResNet-50": ("ResNet-50", 8, -9, "left"),
    "DeiT-S": ("DeiT-S", 8, -8, "left"), "Swin-T": ("Swin-T", -7, 9, "right"),
    "ConvNeXt-T": ("ConvNeXt-T", 7, -7, "left")}


def fig7(runs):
    full, soft = runs[FULL], runs["abl_softmax"]
    ts = []
    for pr in full.values():
        t = fit_temperature(pr["val"][P7].to_numpy(), pr["val"].y.to_numpy().astype(int))
        d = pr["test"].copy(); d[P7] = apply_temperature(d[P7].to_numpy(), t); ts.append(d)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.4), gridspec_kw=dict(wspace=0.45))
    ax = axes[0]
    for ds, lab, c, mk in [([pr["test"] for pr in full.values()], "TrustDerm-Net (evidential)", OI["blue"], "o"),
                           (ts, "+ temperature scaling", OI["sky"], "s"),
                           ([pr["test"] for pr in soft.values()], "softmax variant (v)", OI["vermillion"], "^")]:
        x, y = reliability(ds)
        ax.plot(x, y, marker=mk, ms=3, color=c, label=lab)
    ax.plot([0, 1], [0, 1], color=OI["grey"], lw=0.6, ls=":")
    ax.set_xlabel("Confidence (max class probability)"); ax.set_ylabel("Accuracy")
    ax.set_title("Reliability, HAM10000 test"); ax.legend(frameon=False, loc="lower right", fontsize=6)
    panel(ax, "a")
    ax = axes[1]
    for ds, score, lab, c, ls in [
            ([pr["test"] for pr in full.values()], lambda d: d.u, "Ours, u = K/S", OI["blue"], "-"),
            ([pr["test"] for pr in full.values()], lambda d: 1 - d[P7].max(1), "Ours, 1 − max p (exploratory)", OI["blue"], "--"),
            ([pr["test"] for pr in soft.values()], lambda d: 1 - d[P7].max(1), "Softmax variant (v)", OI["vermillion"], "-"),
            ([pr["derm7pt_test"] for pr in full.values()], lambda d: d.u, "Ours, u = K/S, Derm7pt test", OI["orange"], "-")]:
        cov, mu, sd = risk_coverage(ds, score)
        ax.plot(cov, mu, color=c, ls=ls, label=lab)
        ax.fill_between(cov, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
    for kap in (0.7, 0.8, 0.9):
        ax.axvline(kap, color=OI["grey"], lw=0.4, ls=":")
    ax.set_xlabel("Coverage (fraction not referred)"); ax.set_ylabel("Risk (error on accepted)")
    ax.set_ylim(0, 0.42); ax.set_title("Risk–coverage"); ax.legend(frameon=False, loc="upper left", fontsize=6)
    panel(ax, "b")
    ax = axes[2]
    eff = pd.read_csv(ROOT / "results/step5/efficiency.csv").set_index("model")
    pts = [("TrustDerm-Net", FULL, "TrustDerm-Net (student)", OI["blue"])]
    pts += [(v[1], t, v[2], OI["grey"]) for t, v in BASELINES.items()]
    for lab, tag, en, c in pts:
        r = runs.get(tag)
        if r is None:
            continue
        ba = [balanced_accuracy_score(pr["test"].y, pr["test"][P7].to_numpy().argmax(1)) for pr in r.values()]
        e = eff.loc[en]
        ax.errorbar(e.latency_ms_median, np.mean(ba), yerr=np.std(ba, ddof=1), fmt="o", color=c,
                    ms=3 + 1.2 * np.sqrt(e.params_M), mec="black", mew=0.4, elinewidth=0.6, capsize=1.5)
        short, dx, dy, ha = LABEL_POS.get(lab, (lab, 5, 0, "left"))
        ax.annotate(short, (e.latency_ms_median, np.mean(ba)), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6, va="center", ha=ha,
                    color=OI["blue"] if tag == FULL else "black",
                    arrowprops=dict(arrowstyle="-", color=OI["grey"], lw=0.4, shrinkA=0, shrinkB=2.5),
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=5)
    ax.set_xlim(35, 215)
    ax.set_xlabel("CPU latency, ms (median, 4 threads)"); ax.set_ylabel("Balanced accuracy")
    ax.set_title("bAcc (HAM test) vs. latency")
    panel(ax, "c")
    save(fig, "Fig7_calibration_selective_efficiency")


def main():
    which = sys.argv[1:] or ["4", "5", "6", "7"]
    FIG.mkdir(exist_ok=True)
    runs = {t: load(t, f) for t, (f, _) in RUNS.items()}
    runs.update({t: load(t, v[0]) for t, v in BASELINES.items()})
    if "4" in which: fig4()
    if "5" in which: fig5(runs[FULL])
    if "6" in which: fig6(runs[FULL])
    if "7" in which: fig7(runs)


if __name__ == "__main__":
    main()
