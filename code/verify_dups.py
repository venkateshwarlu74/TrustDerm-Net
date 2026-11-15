"""Second-stage check of dHash near-duplicate candidates (added for the ISIC 2020 evaluation).

The 64-bit dHash (8x8 gradients) gives many false matches on ISIC 2020, where most images show one
blob on plain skin. For every external image of the chosen dataset flagged as `excluded_duplicate`,
each development image within dHash distance <= max_hamming is compared at pixel level:
zero-mean normalised cross-correlation (NCC) of 64x64 grayscale versions of the cached images,
taking the maximum over the 8 flips/rotations. Only pairs with NCC >= --thr stay excluded.

Usage:
  python verify_dups.py --dataset isic2020 --report            # write dup_verify.csv + montage, change nothing
  python verify_dups.py --dataset isic2020 --apply --thr 0.95  # re-admit images whose best NCC < thr
"""
import argparse, os
import cv2
import numpy as np
import pandas as pd
from config import MANIFEST

W = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def small(path, n=64):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    return g


def variants(g):
    out = []
    for k in range(4):
        r = np.rot90(g, k)
        out += [r, r[:, ::-1]]
    return [(v - v.mean()) / (v.std() + 1e-6) for v in out]


def ncc(a_var, b):
    b = (b - b.mean()) / (b.std() + 1e-6)
    return max(float((a * b).mean()) for a in a_var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="isic2020")
    ap.add_argument("--max_hamming", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    m = pd.read_csv(MANIFEST, dtype={"dhash": str})
    dev = m[m.split.str.startswith("fold") | (m.split == "test")]
    flagged = m[(m.dataset == a.dataset) & (m.split == "excluded_duplicate")]
    dh = np.array([int(h) for h in dev.dhash], dtype=np.uint64)
    cache, rows = {}, []

    def get(p):
        if p not in cache:
            cache[p] = small(p)
        return cache[p]

    for n, (i, r) in enumerate(flagged.iterrows()):
        x = np.bitwise_xor(dh, np.uint64(int(r.dhash)))
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        cand = np.where(d <= a.max_hamming)[0]
        av = variants(small(r.cache_path))
        best, bj = -1.0, -1
        for j in cand:
            s = ncc(av, get(dev.cache_path.iloc[j]))
            if s > best:
                best, bj = s, j
        rows.append(dict(index=i, image_id=r.image_id, y=r.y, z=r.z, n_cand=len(cand),
                         match=dev.image_id.iloc[bj] if bj >= 0 else "", match_path=dev.cache_path.iloc[bj] if bj >= 0 else "",
                         dhash_d=int(d[bj]) if bj >= 0 else -1, ncc=best))
        if n % 1000 == 0:
            print(f"{n}/{len(flagged)}", flush=True)
    v = pd.DataFrame(rows)
    v.drop(columns="match_path").to_csv(os.path.join(W, f"dup_verify_{a.dataset}.csv"), index=False)
    print("best-NCC quantiles:", np.round(np.percentile(v.ncc, [1, 5, 25, 50, 75, 95, 99, 99.9]), 3).tolist())
    for t in (0.8, 0.85, 0.9, 0.95, 0.98):
        print(f"NCC >= {t}: {(v.ncc >= t).sum()} images")
    if a.report:   # montage of the highest-NCC pairs and of pairs around the thresholds
        v = v.sort_values("ncc", ascending=False).reset_index(drop=True)
        picks = list(range(min(12, len(v))))
        for t in (0.98, 0.95, 0.9, 0.85):
            k = int((v.ncc >= t).sum())
            picks += [p for p in range(max(0, k - 2), min(len(v), k + 2))]
        tiles = []
        for p in dict.fromkeys(picks):
            r = v.iloc[p]
            A = cv2.resize(cv2.imread(m.loc[r["index"], "cache_path"]), (240, 180))
            B = cv2.resize(cv2.imread(r.match_path), (240, 180))
            row = np.hstack([A, np.full((180, 6, 3), 255, np.uint8), B])
            cv2.putText(row, f"{r.image_id} ncc={r.ncc:.3f} d={r.dhash_d}", (4, 16), 0, .45, (0, 255, 255), 1)
            cv2.putText(row, str(r.match), (250, 16), 0, .45, (0, 255, 255), 1)
            tiles.append(row)
        for s in range(0, len(tiles), 12):
            cv2.imwrite(os.path.join(W, f"dup_montage_{s // 12}.jpg"), np.vstack(tiles[s:s + 12]))
    if a.apply:
        keep = v.loc[v.ncc < a.thr, "index"].to_numpy()
        m.loc[keep, "split"] = a.dataset
        m.to_csv(MANIFEST, index=False)
        print(f"re-admitted {len(keep)} {a.dataset} images (best NCC < {a.thr}); "
              f"{(v.ncc >= a.thr).sum()} stay excluded")
        print(m[m.dataset == a.dataset].split.value_counts().to_string())
        print("melanoma (z=1):", m[(m.dataset == a.dataset) & (m.split == a.dataset)].z.sum())


if __name__ == "__main__":
    main()
