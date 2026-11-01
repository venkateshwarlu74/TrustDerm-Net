"""Split PAD-UFES-20's merged AKIEC group back into squamous cell carcinoma (SCC) and actinic keratosis (ACK)
and report how the TrustDerm-Net student handled each, from the saved predictions (no retraining).

Usage: python -I _tools/scc_breakdown.py <PAD metadata.csv> [run folder, default results/student_eval_v2]
Writes results/step5/scc_breakdown.csv.
"""
import csv, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["NV", "BKL", "DF", "VASC", "MEL", "BCC", "AKIEC"]
meta = {Path(r["img_id"]).stem: r["diagnostic"] for r in csv.DictReader(open(sys.argv[1], encoding="utf-8"))}
run = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results/student_eval_v2")


def auroc(pos, neg):
    """Mann-Whitney AUROC of scores pos (label 1) vs neg (label 0)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


rows = []
for k in range(5):
    th = json.load(open(run / "results" / f"student_fold{k}.json"))
    t_mal, tau = th["t_mal"], th["tau_u"]["0.8"]
    pad = [r for r in csv.DictReader(open(run / "preds" / f"student_fold{k}.csv")) if r["eval_set"] == "pad_ufes"]
    benign = [float(r["p_mal"]) for r in pad if int(r["z"]) == 0]
    for dx in ["SCC", "ACK", "BCC", "MEL"]:
        g = [r for r in pad if meta.get(r["image_id"]) == dx]
        p = [float(r["p_mal"]) for r in g]
        arg = [CLASSES[max(range(7), key=lambda c: float(r[f"p_{CLASSES[c]}"]))] for r in g]
        rows.append({"fold": k, "diagnosis": dx, "n": len(g),
                     "triage_sens": sum(x >= t_mal for x in p) / len(g),
                     "auroc_vs_benign": auroc(p, benign),
                     "argmax_AKIEC": arg.count("AKIEC") / len(g),
                     "argmax_BCC": arg.count("BCC") / len(g),
                     "argmax_MEL": arg.count("MEL") / len(g),
                     "argmax_benign": sum(a in ("NV", "BKL", "DF", "VASC") for a in arg) / len(g),
                     "referred_at_80pct": sum(float(r["u"]) > tau for r in g) / len(g)})

out = ROOT / "results" / "step5" / "scc_breakdown.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out.relative_to(ROOT)}\n")
keys = [c for c in rows[0] if c not in ("fold", "diagnosis", "n")]
print(f"{'dx':5} {'n':>4}  " + "  ".join(f"{c:>17}" for c in keys))
for dx in ["SCC", "ACK", "BCC", "MEL"]:
    g = [r for r in rows if r["diagnosis"] == dx]
    cells = []
    for c in keys:
        v = [r[c] for r in g]
        cells.append(f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}".rjust(17))
    print(f"{dx:5} {g[0]['n']:>4}  " + "  ".join(cells))

