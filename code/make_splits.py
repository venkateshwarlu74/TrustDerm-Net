"""Lesion-level splits and near-duplicate removal.

Development data -> fixed internal test set (~15%, 1/7 of lesion groups) + 5 cross-validation folds
on the rest. Grouping by lesion/patient guarantees that no lesion appears in two partitions.
External images whose dHash is within Hamming distance 3 of any development image are flagged and excluded.

Usage: python make_splits.py [--seed 42] [--max_hamming 3]
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from config import CLASSES, DEV, MANIFEST


def near_duplicates(dev_hashes, other_hashes, max_d):
    """Return a boolean mask over other_hashes. Pigeonhole banding: any pair within distance <= 3 shares
    at least one identical 16-bit band, so only those candidates are compared."""
    dev = np.array(dev_hashes, dtype=np.uint64)
    buckets = [dict() for _ in range(4)]
    for j, h in enumerate(dev):
        for b in range(4):
            buckets[b].setdefault(int(h >> np.uint64(16 * b)) & 0xFFFF, []).append(j)
    out = np.zeros(len(other_hashes), bool)
    for i, h in enumerate(np.array(other_hashes, dtype=np.uint64)):
        cand = set()
        for b in range(4):
            cand.update(buckets[b].get(int(h >> np.uint64(16 * b)) & 0xFFFF, []))
        if cand:
            x = np.bitwise_xor(dev[list(cand)], h)
            dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
            out[i] = dist.min() <= max_d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_hamming", type=int, default=3)
    a = ap.parse_args()
    m = pd.read_csv(MANIFEST, dtype={"dhash": str})
    m["split"] = m.role
    dev = m.role == DEV

    # Each development dataset is split on its own, so adding a dataset later never changes the
    # assignment of images that were already split (and trained-on models stay valid).
    # Datasets listed in frozen_splits.csv reuse their stored assignment exactly (fold assignments of
    # StratifiedGroupKFold differ between scikit-learn versions).
    frozen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_splits.csv")
    frozen = pd.read_csv(frozen).set_index("image_id").split if os.path.exists(frozen) else pd.Series(dtype=str)
    for name in sorted(m.loc[dev, "dataset"].unique()):
        d = m[dev & (m.dataset == name)]
        if d.image_id.isin(frozen.index).all():
            m.loc[d.index, "split"] = frozen.loc[d.image_id].to_numpy()
            print(f"{name}: {len(d)} images, splits taken from frozen_splits.csv")
            continue
        # 1) hold out ~1/7 of lesion groups as the internal test set
        #    (not for Derm7pt: its official test split is already reported separately)
        if name != "derm7pt":
            sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=a.seed)
            _, test_idx = next(sgkf.split(d, d.y, d.group))
            m.loc[d.index[test_idx], "split"] = "test"
        # 2) 5 folds on the remaining development data
        rest = m[dev & (m.dataset == name) & (m.split == DEV)]
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=a.seed)
        for k, (_, val_idx) in enumerate(sgkf.split(rest, rest.y, rest.group)):
            m.loc[rest.index[val_idx], "split"] = f"fold{k}"

    # Secondary development sets (e.g. Derm7pt) may contain copies of primary images (HAM10000 / ISIC 2019):
    # such images are excluded, otherwise a copy of a test or validation image could be trained on.
    primary = dev & m.dataset.isin(["ham10000", "isic2019"])
    secondary = dev & ~primary
    if "dhash" in m and primary.any() and secondary.any():
        dup = near_duplicates([int(h) for h in m.loc[primary, "dhash"]], [int(h) for h in m.loc[secondary, "dhash"]],
                              a.max_hamming)
        m.loc[m.index[secondary][dup], "split"] = "excluded_duplicate"
        print(f"near-duplicates of primary development images removed from secondary development sets: {dup.sum()}")
    dev = m.split.str.startswith("fold") | (m.split == "test")

    # sanity: no lesion group in more than one partition
    leak = m[dev].groupby("group").split.nunique()
    assert (leak == 1).all(), f"{(leak > 1).sum()} lesion groups leak across partitions"

    # 3) near-duplicates between development data and every other set
    other = ~dev & (m.split != "excluded_duplicate")
    if "dhash" in m and other.any():
        dup = near_duplicates([int(h) for h in m.loc[dev, "dhash"]], [int(h) for h in m.loc[other, "dhash"]],
                              a.max_hamming)
        m.loc[m.index[other][dup], "split"] = "excluded_duplicate"
        print(f"near-duplicates of development images removed from external sets: {dup.sum()}")

    m.to_csv(MANIFEST, index=False)
    tab = m[m.y >= 0].pivot_table(index="split", columns="y", values="image_id", aggfunc="count", fill_value=0)
    tab.columns = [CLASSES[c] for c in tab.columns]
    tab["total"] = tab.sum(axis=1)
    print("\nImages per split and class (use for Table II / Fig. 4):\n" + tab.to_string())
    print("\nImages per split (all):\n" + m.split.value_counts().to_string())


if __name__ == "__main__":
    main()
