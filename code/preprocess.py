"""Hair removal (Eq. 1), Shades-of-Gray colour constancy (Eqs. 2-3) and resizing, cached once to disk.
Also computes a 64-bit difference hash (dHash) per image for near-duplicate detection.

Usage: python preprocess.py [--short 448] [--workers 4]
"""
import argparse, os
from multiprocessing import Pool
import cv2
import numpy as np
import pandas as pd
from config import CACHE, MANIFEST

SHORT = 448


def hair_removal(img, ksize=17, thresh=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    response = np.zeros_like(gray)
    for angle in (0, 45, 90, 135):          # linear structuring elements at four orientations
        k = np.zeros((ksize, ksize), np.uint8)
        k[ksize // 2, :] = 1
        rot = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
        k = (cv2.warpAffine(k, rot, (ksize, ksize)) > 0).astype(np.uint8)
        response = np.maximum(response, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k))
    mask = (response > thresh).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def shades_of_gray(img, p=6):
    f = img.astype(np.float32) + 1e-6
    e = np.power(np.mean(np.power(f, p), axis=(0, 1)), 1.0 / p)   # per-channel illuminant
    out = f * (e.mean() / e)
    return np.clip(out, 0, 255).astype(np.uint8)


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def read_reduced(src, short):
    """Decode large JPEGs at 1/2, 1/4 or 1/8 scale when that still leaves >= `short` pixels (much faster)."""
    try:
        from PIL import Image
        w, h = Image.open(src).size
    except Exception:
        return cv2.imread(src)
    for f, flag in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2)):
        if min(w, h) / f >= short:
            return cv2.imread(src, flag)
    return cv2.imread(src)


def process(args):
    src, dst, short = args
    if os.path.exists(dst):
        img = cv2.imread(dst)
        return dhash(img) if img is not None else -1
    img = read_reduced(src, short)
    if img is None:
        return -1
    h, w = img.shape[:2]
    s = short / min(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    img = shades_of_gray(hair_removal(img))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dhash(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", type=int, default=SHORT)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    m = pd.read_csv(MANIFEST)
    m["cache_path"] = [os.path.join(CACHE, d, f"{i}.jpg") for d, i in zip(m.dataset, m.image_id)]
    jobs = [(s, c, a.short) for s, c in zip(m.path, m.cache_path)]
    with Pool(a.workers) as pool:
        hashes = []
        for k, h in enumerate(pool.imap(process, jobs, chunksize=32)):
            hashes.append(h)
            if (k + 1) % 2000 == 0:
                print(f"{k + 1}/{len(m)} images", flush=True)
    m["dhash"] = [str(h) for h in hashes]
    bad = (m.dhash == "-1").sum()
    if bad:
        print(f"WARNING: {bad} unreadable images dropped")
    m = m[m.dhash != "-1"]
    m.to_csv(MANIFEST, index=False)
    print(f"done: {len(m)} images cached in {CACHE}")


if __name__ == "__main__":
    main()
