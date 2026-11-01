"""Numbers for the faithfulness text of Section 5.6, from results/step5/fig8/faithfulness_per_image.csv.

Usage: python _tools/faith_summary.py     -> results/step5/fig8/faith_summary.json
"""
import json
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

WORK = Path(__file__).resolve().parents[1] / "results" / "step5" / "fig8"
r = pd.read_csv(WORK / "faithfulness_per_image.csv")
cols = ["del_cam", "del_block", "del_rand", "ins_cam", "ins_block", "ins_rand"]
pf = r.groupby("fold")[cols].mean()
img = r.groupby("image_id")[cols].mean()
out = dict(n_folds=int(r.fold.nunique()), n_images=int(r.image_id.nunique()),
           mean={c: round(float(pf[c].mean()), 3) for c in cols}, sd={c: round(float(pf[c].std(ddof=1)), 3) for c in cols},
           del_folds_better=int((pf.del_cam < pf.del_block).sum()), ins_folds_better=int((pf.ins_cam > pf.ins_block).sum()),
           del_img_better=round(float((img.del_cam < img.del_block).mean()), 3),
           ins_img_better=round(float((img.ins_cam > img.ins_block).mean()), 3),
           del_p=float(wilcoxon(img.del_cam, img.del_block).pvalue), ins_p=float(wilcoxon(img.ins_cam, img.ins_block).pvalue))
for name, g in (("correct", r[r.correct]), ("wrong", r[~r.correct])):
    out[name] = dict(n=len(g), **{c: round(float(g[c].mean()), 3) for c in cols})
(WORK / "faith_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
