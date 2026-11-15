"""Decision-curve analysis (Vickers and Elkin, 2006) of the benign/malignant triage output p(z = 1).

Net benefit NB(p_t) = TP/N - FP/N * p_t / (1 - p_t), computed per fold model and averaged over the five folds,
for the HAM10000 internal test set, the Derm7pt test split and PAD-UFES-20. Reference strategies: refer all, refer none.
Output: results/step5/decision_curve.csv, figures/FigS1_decision_curve.{png,pdf}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "results" / "student_eval_v2" / "preds"
SETS = [("test", "HAM10000 internal test"), ("derm7pt_test", "Derm7pt test"), ("pad_ufes", "PAD-UFES-20 (smartphone)")]
PT = np.round(np.arange(0.005, 0.505, 0.005), 3)
OI = dict(blue="#0072B2", grey="#999999", black="#000000", orange="#E69F00")


def nb(p, z, pt):
    n = len(z)
    flag = p >= pt
    return ((flag & (z == 1)).sum() - (flag & (z == 0)).sum() * pt / (1 - pt)) / n


rows = []
for k in range(5):
    d = pd.read_csv(PREDS / f"student_fold{k}.csv")
    val = d[d.eval_set == "val"]
    t_mal = float(np.quantile(val.p_mal[val.z == 1].to_numpy(), 0.05, method="lower"))
    for s, _ in SETS:
        e = d[d.eval_set == s]
        p, z = e.p_mal.to_numpy(), e.z.to_numpy()
        prev = z.mean()
        for pt in PT:
            rows.append(dict(fold=k, set=s, pt=pt, nb_model=nb(p, z, pt), nb_all=prev - (1 - prev) * pt / (1 - pt), t_mal=t_mal))
df = pd.DataFrame(rows)
g = df.groupby(["set", "pt"]).agg(nb_model=("nb_model", "mean"), nb_model_sd=("nb_model", "std"), nb_all=("nb_all", "mean")).reset_index()
g.to_csv(ROOT / "results" / "step5" / "decision_curve.csv", index=False)
t_mal = df.groupby("fold").t_mal.first().mean()

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=False)
for ax, (s, title) in zip(axes, SETS):
    h = g[g.set == s]
    ax.plot(h.pt, h.nb_model, color=OI["blue"], lw=1.2, label="TrustDerm-Net triage")
    ax.fill_between(h.pt, h.nb_model - h.nb_model_sd, h.nb_model + h.nb_model_sd, color=OI["blue"], alpha=0.15, lw=0)
    ax.plot(h.pt, h.nb_all, color=OI["grey"], lw=1, ls="--", label="Refer all")
    ax.axhline(0, color=OI["black"], lw=0.8, label="Refer none")
    ax.axvline(t_mal, color=OI["orange"], lw=0.8, ls=":", label="Operating point t$_{mal}$")
    top = max(h.nb_all.max(), h.nb_model.max())
    ax.set_ylim(-0.05, top * 1.1 + 0.01)
    ax.set_xlim(0, 0.5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Threshold probability", fontsize=7)
    ax.tick_params(labelsize=6.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Net benefit", fontsize=7)
axes[0].legend(fontsize=5.5, frameon=False, loc="upper right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(ROOT / "figures" / f"FigS1_decision_curve.{ext}", dpi=300, bbox_inches="tight")

# numbers for the text
for s, _ in SETS:
    h = g[g.set == s].set_index("pt")
    gain = h.nb_model - np.maximum(h.nb_all, 0)
    better = h.index[gain > 0.005]
    print(s, f"prevalence-based refer-all NB at pt=0.015: {h.nb_all.loc[0.015]:.3f}; model {h.nb_model.loc[0.015]:.3f}")
    for pt in (0.05, 0.1, 0.2, 0.3):
        print(f"   pt={pt}: model {h.nb_model.loc[pt]:.3f} ± {h.nb_model_sd.loc[pt]:.3f}, refer-all {h.nb_all.loc[pt]:.3f}")
    print("   model > max(all, none) + 0.005 for pt in:", (better.min(), better.max()) if len(better) else "none")
print("mean t_mal", round(t_mal, 4))
