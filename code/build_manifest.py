"""Find every dataset under /kaggle/input and build one harmonised manifest.csv.

Each dataset is located by a characteristic file, so the Kaggle dataset names (slugs) do not matter:
  HAM10000      HAM10000_metadata.csv
  ISIC 2019     ISIC_2019_Training_GroundTruth.csv (+ ISIC_2019_Training_Metadata.csv)
  Derm7pt       meta/meta.csv (+ train_indexes.csv, valid_indexes.csv, test_indexes.csv)
  ISIC 2020     train.csv with columns image_name, target
  PAD-UFES-20   metadata.csv with column diagnostic
  DDI           ddi_metadata.csv
Missing datasets are skipped with a warning, so you can start with HAM10000 alone.

Usage: python build_manifest.py [--input /kaggle/input] [--out /kaggle/working/manifest.csv]
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from config import CLASSES, CONCEPTS, SITES, INPUT_ROOT, MANIFEST, DEV, DERM7PT_TEST

COLS = ["image_id", "path", "dataset", "role", "group", "y", "z", "age", "sex", "site", "skin_tone"] + CONCEPTS


def find(root, pattern, must_have=None):
    for p in sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True)):
        if must_have is None:
            return p
        try:
            cols = pd.read_csv(p, nrows=1).columns
        except Exception:
            continue
        if all(c in cols for c in must_have):
            return p
    return None


def index_images(folder):
    idx = {}
    for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            idx.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return idx


def map_site(s):
    if not isinstance(s, str):
        return -1
    s = s.lower()
    rules = [("palms/soles", ["palm", "sole", "acral"]), ("oral/genital", ["genital", "oral"]),
             ("head/neck", ["head", "neck", "face", "scalp", "ear", "nose", "lip"]),
             ("upper extremity", ["upper", "arm", "hand"]),
             ("lower extremity", ["lower", "leg", "thigh", "foot", "shin", "knee"]),
             ("torso", ["torso", "trunk", "back", "chest", "abdomen", "buttock", "flank"])]
    for site, keys in rules:
        if any(k in s for k in keys):
            return SITES.index(site)
    return -1


def map_sex(s):
    s = str(s).lower()
    return 0 if s in ("male", "m") else 1 if s in ("female", "f") else -1


def frame(rows):
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df:
            df[c] = -1 if c in CONCEPTS or c in ("y", "z", "sex", "site", "skin_tone") else np.nan
    return df[COLS]


def ham10000(root):
    meta = find(root, "HAM10000_metadata.csv")
    if not meta:
        return None
    imgs = index_images(os.path.dirname(meta))
    df = pd.read_csv(meta)
    dx = {"nv": "NV", "bkl": "BKL", "df": "DF", "vasc": "VASC", "mel": "MEL", "bcc": "BCC", "akiec": "AKIEC"}
    rows = [dict(image_id=r.image_id, path=imgs.get(r.image_id), dataset="ham10000", role=DEV,
                 group="HAM_" + r.lesion_id, y=CLASSES.index(dx[r.dx]), age=r.age, sex=map_sex(r.sex),
                 site=map_site(r.localization)) for r in df.itertuples()]
    return frame(rows)


def isic2019(root, exclude_ids):
    gt = find(root, "ISIC_2019_Training_GroundTruth.csv")
    if not gt:
        return None
    md = find(root, "ISIC_2019_Training_Metadata.csv")
    imgs = index_images(os.path.dirname(gt))
    df = pd.read_csv(gt)
    if md:
        df = df.merge(pd.read_csv(md), on="image", how="left")
    cls_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    to7 = {"AK": "AKIEC", "SCC": "AKIEC"}  # SCC merged into the keratinocyte group
    rows = []
    for r in df.itertuples():
        base = r.image.replace("_downsampled", "")
        if base in exclude_ids:          # HAM10000 images are already included
            continue
        lab = cls_cols[int(np.argmax([getattr(r, c) for c in cls_cols]))]
        lesion = getattr(r, "lesion_id", np.nan)
        rows.append(dict(image_id=base, path=imgs.get(r.image), dataset="isic2019", role=DEV,
                         group=f"I19_{lesion}" if isinstance(lesion, str) else f"I19_{base}",
                         y=CLASSES.index(to7.get(lab, lab)), age=getattr(r, "age_approx", np.nan),
                         sex=map_sex(getattr(r, "sex", "")), site=map_site(getattr(r, "anatom_site_general", ""))))
    return frame(rows)


def derm7pt_concepts(r):
    g = lambda c: str(r.get(c, "")).lower()
    return [int(g("pigment_network") == "atypical"),
            int(g("blue_whitish_veil") == "present"),
            int(any(k in g("vascular_structures") for k in ("dotted", "linear irregular"))),
            int(g("streaks") == "irregular"),
            int(g("dots_and_globules") == "irregular"),
            int("irregular" in g("pigmentation")),
            int(g("regression_structures") not in ("absent", "", "nan"))]


def derm7pt_dx(d):
    d = str(d).lower()
    for key, c in [("melanoma", "MEL"), ("basal cell", "BCC"), ("nevus", "NV"), ("dermatofibroma", "DF"),
                   ("seborrheic", "BKL"), ("lentigo", "BKL"), ("melanosis", "BKL"), ("vascular", "VASC")]:
        if key in d:
            return CLASSES.index(c)
    return None  # 'miscellaneous' and anything unmapped is dropped


def derm7pt(root):
    meta = find(root, os.path.join("meta", "meta.csv"))
    if not meta:
        return None
    base = os.path.dirname(os.path.dirname(meta))
    df = pd.read_csv(meta)
    test_idx = set(pd.read_csv(os.path.join(os.path.dirname(meta), "test_indexes.csv")).iloc[:, 0])
    rows = []
    for i, r in df.iterrows():
        y = derm7pt_dx(r["diagnosis"])
        if y is None:
            continue
        row = dict(image_id=f"D7_{r['case_num']}", path=os.path.join(base, "images", r["derm"]), dataset="derm7pt",
                   role=DERM7PT_TEST if i in test_idx else DEV, group=f"D7_{r['case_num']}", y=y,
                   sex=map_sex(r.get("sex", "")), site=map_site(r.get("location", "")))
        row.update(dict(zip(CONCEPTS, derm7pt_concepts(r))))
        rows.append(row)
    return frame(rows)


def isic2020(root):
    csv = find(root, "train.csv", must_have=["image_name", "target", "diagnosis"])
    if not csv:
        return None
    imgs = index_images(os.path.dirname(csv))
    dx = {"melanoma": "MEL", "nevus": "NV", "seborrheic keratosis": "BKL", "lentigo nos": "BKL",
          "solar lentigo": "BKL", "lichenoid keratosis": "BKL"}
    rows = []
    for r in pd.read_csv(csv).itertuples():
        d = dx.get(str(r.diagnosis).lower())
        rows.append(dict(image_id=r.image_name, path=imgs.get(r.image_name), dataset="isic2020", role="isic2020",
                         group=f"I20_{r.patient_id}", y=CLASSES.index(d) if d else -1, z=int(r.target),
                         age=r.age_approx, sex=map_sex(r.sex), site=map_site(r.anatom_site_general_challenge)))
    return frame(rows)


def pad_ufes(root):
    csv = find(root, "metadata.csv", must_have=["diagnostic", "img_id"])
    if not csv:
        return None
    imgs = index_images(os.path.dirname(csv))
    dx = {"NEV": "NV", "SEK": "BKL", "MEL": "MEL", "BCC": "BCC", "ACK": "AKIEC", "SCC": "AKIEC"}
    rows = []
    for r in pd.read_csv(csv).itertuples():
        stem = os.path.splitext(r.img_id)[0]
        fst = pd.to_numeric(getattr(r, "fitspatrick", np.nan), errors="coerce")   # column name as spelled in PAD
        tone = -1 if pd.isna(fst) else (12 if fst <= 2 else 34 if fst <= 4 else 56)  # DDI-style groups
        rows.append(dict(image_id=stem, path=imgs.get(stem), dataset="pad_ufes", role="pad_ufes",
                         group=f"PAD_{r.patient_id}", y=CLASSES.index(dx[r.diagnostic]), age=r.age,
                         sex=map_sex(r.gender), site=map_site(r.region), skin_tone=tone))
    return frame(rows)


def ddi(root):
    csv = find(root, "ddi_metadata.csv")
    if not csv:
        return None
    imgs = index_images(os.path.dirname(csv))
    rows = []
    for r in pd.read_csv(csv).itertuples():
        stem = os.path.splitext(r.DDI_file)[0]
        mal = str(r.malignant).lower() in ("true", "1")
        rows.append(dict(image_id=f"DDI_{stem}", path=imgs.get(stem), dataset="ddi", role="ddi",
                         group=f"DDI_{r.DDI_ID}", z=int(mal), skin_tone=int(r.skin_tone)))
    return frame(rows)  # DDI has no age/sex/site: every metadata token is 'missing'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_ROOT)
    ap.add_argument("--out", default=MANIFEST)
    a = ap.parse_args()
    parts = []
    ham = ham10000(a.input)
    ham_ids = set(ham.image_id) if ham is not None else set()
    for name, df in [("HAM10000", ham), ("ISIC 2019", isic2019(a.input, ham_ids)), ("Derm7pt", derm7pt(a.input)),
                     ("ISIC 2020", isic2020(a.input)), ("PAD-UFES-20", pad_ufes(a.input)), ("DDI", ddi(a.input))]:
        if df is None:
            print(f"[skip] {name}: not found under {a.input}")
            continue
        missing = df.path.isna().sum()
        print(f"[ok]   {name}: {len(df)} rows" + (f"  (WARNING: {missing} images not found, dropped)" if missing else ""))
        parts.append(df[df.path.notna()])
    m = pd.concat(parts, ignore_index=True)
    m.loc[m.y >= 0, "z"] = (m.loc[m.y >= 0, "y"] >= 4).astype(int)   # coarse label from fine label
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    m.to_csv(a.out, index=False)
    print(f"\nManifest: {len(m)} images -> {a.out}")
    print(m.groupby(["role", "dataset"]).size().to_string())


if __name__ == "__main__":
    main()
