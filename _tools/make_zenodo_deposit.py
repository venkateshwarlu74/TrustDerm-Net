"""Build the research-data deposit (Zenodo) for the TrustDerm-Net paper.

Usage: python make_zenodo_deposit.py <Paper2 folder> <output folder>
Writes <out>/TrustDerm-Net_data/ (splits, near-duplicate lists, predictions, result tables, weights, README)
and <out>/TrustDerm-Net_data_v1.zip. No images are included.
Derm7pt (CC BY-NC-ND, no redistribution): only image identifiers, split assignments and model outputs are
kept; its diagnosis, coarse label and seven-point concept labels are blanked in every file.
"""
import hashlib, json, os, shutil, sys, zipfile
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
R = os.path.join(root, "results")
D = os.path.join(out, "TrustDerm-Net_data")
if os.path.exists(D):
    shutil.rmtree(D)
for sub in ["splits", "near_duplicates", "predictions/internal_external", "predictions/isic2020", "results_tables", "weights"]:
    os.makedirs(os.path.join(D, sub), exist_ok=True)

GT = ["y", "z", "atypical_network", "blue_whitish_veil", "atypical_vascular", "irregular_streaks",
      "irregular_dots_globules", "irregular_blotches", "regression"]


def scrub(df):
    """Blank Derm7pt ground truth (licence forbids redistribution)."""
    if "dataset" in df:
        mask = df["dataset"] == "derm7pt"
        for c in GT:
            if c in df:
                df[c] = df[c].astype("object")
                df.loc[mask, c] = ""
    return df


# 1. splits
m = pd.read_csv(os.path.join(R, "prep_isic", "manifest.csv"))
s = m[["image_id", "dataset", "group", "split"]].rename(columns={"group": "lesion_or_patient_group"}).copy()
flag = (s.dataset == "isic2020") & (s.split == "excluded_duplicate")
s["note"] = ""
s.loc[flag, "note"] = "dHash candidate; NCC check found no true duplicate; kept and evaluated"
s.loc[s.dataset == "isic2020", "split"] = "isic2020_external"
s.loc[(s.split == "excluded_duplicate"), "note"] = "near-duplicate of a HAM10000 image; excluded"
s.to_csv(os.path.join(D, "splits", "lesion_level_splits.csv"), index=False)

# 2. near-duplicate lists
s[s.split == "excluded_duplicate"].to_csv(os.path.join(D, "near_duplicates", "excluded_near_duplicates_derm7pt_pad.csv"), index=False)
v = pd.read_csv(os.path.join(R, "dupcheck_isic", "dup_verify_isic2020.csv"))
v = v.rename(columns={"match": "closest_dev_image", "dhash_d": "dhash_hamming", "ncc": "max_ncc_8_orientations"})
v[["image_id", "closest_dev_image", "n_cand", "dhash_hamming", "max_ncc_8_orientations"]].to_csv(
    os.path.join(D, "near_duplicates", "isic2020_dhash_candidates_ncc.csv"), index=False)

# 3. predictions (internal test, Derm7pt test, PAD-UFES-20, validation folds)
src = {"student": "student_eval_v2", "teacher": "teacher_eval"}
for d in sorted(os.listdir(R)):
    if d.startswith(("abl_", "base_")) and os.path.isdir(os.path.join(R, d, "preds")):
        src[d] = d
n_pred = 0
for tag, d in src.items():
    for k in range(5):
        f = os.path.join(R, d, "preds", f"{tag}_fold{k}.csv")
        df = scrub(pd.read_csv(f))
        df.to_csv(os.path.join(D, "predictions/internal_external", f"{tag}_fold{k}.csv"), index=False)
        n_pred += 1
for d in ["isic2020_ours", "isic2020_base"]:
    for f in sorted(os.listdir(os.path.join(R, d, "preds"))):
        df = scrub(pd.read_csv(os.path.join(R, d, "preds", f)))
        df.to_csv(os.path.join(D, "predictions/isic2020", f), index=False)
        n_pred += 1

# 4. result tables
for f in sorted(os.listdir(os.path.join(R, "step5"))):
    if f.endswith((".csv", ".json")):
        shutil.copy(os.path.join(R, "step5", f), os.path.join(D, "results_tables", f))

# 5. weights + validation thresholds of the five student models
for k in range(5):
    shutil.copy(os.path.join(R, "student_v1", "ckpt", f"student_fold{k}.pt"), os.path.join(D, "weights", f"student_fold{k}.pt"))
    j = json.load(open(os.path.join(R, "student_eval_v2", "results", f"student_fold{k}.json")))
    json.dump({"fold": k, "t_mal": j["t_mal"], "tau_u": j["tau_u"]},
              open(os.path.join(D, "weights", f"student_fold{k}_thresholds.json"), "w"), indent=1)

shutil.copy(os.path.join(root, "_tools", "zenodo_README.md"), os.path.join(D, "README.md"))

# checksums
lines = []
for dp, _, fs in os.walk(D):
    for f in sorted(fs):
        p = os.path.join(dp, f)
        lines.append(hashlib.md5(open(p, "rb").read()).hexdigest() + "  " + os.path.relpath(p, D).replace("\\", "/"))
open(os.path.join(D, "MD5SUMS.txt"), "w").write("\n".join(sorted(lines, key=lambda l: l[34:])) + "\n")

zp = os.path.join(out, "TrustDerm-Net_data_v1.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for dp, _, fs in os.walk(D):
        for f in fs:
            p = os.path.join(dp, f)
            z.write(p, os.path.join("TrustDerm-Net_data", os.path.relpath(p, D)).replace("\\", "/"))
print(f"{n_pred} prediction files, {len(lines)} files in total; zip {os.path.getsize(zp)/1e6:.0f} MB -> {zp}")
