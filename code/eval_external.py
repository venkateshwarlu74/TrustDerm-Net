"""Predict one external dataset (default ISIC 2020) with every fold checkpoint of the given runs.

Unlike evaluate.py, only the fold's validation split (needed for t_mal and tau_u) and the external set
are predicted. ALL images of the external set are predicted, including those flagged `excluded_duplicate`
by the dHash filter, and the split column is kept, so duplicate handling can be decided afterwards.
Metrics are computed offline from preds/{tag}_{dataset}_fold{k}.csv.

Usage: python eval_external.py --dataset isic2020 --tags student teacher base_swint ...
"""
import argparse, glob, os
import pandas as pd
import torch
from config import CKPT, MANIFEST, PREDS
from evaluate import predict
from model import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="isic2020")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(PREDS, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = pd.read_csv(MANIFEST, dtype={"dhash": str})
    ext = m[m.dataset == a.dataset]
    print(f"{a.dataset}: {len(ext)} images; split counts: {ext.split.value_counts().to_dict()}", flush=True)
    for tag in a.tags:
        paths = sorted(glob.glob(os.path.join(CKPT, f"{tag}_fold*.pt")))
        print(f"{tag}: {len(paths)} checkpoints", flush=True)
        for path in paths:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = build(ck["model_cfg"], pretrained=False)
            model.load_state_dict(ck["state_dict"]); model.to(dev)
            k = ck["fold"]
            out = []
            for name, df in (("val", m[m.split == f"fold{k}"]), (a.dataset, ext)):
                p = predict(model, df, dev, workers=a.workers)
                p["orig_split"] = df.split.to_numpy()
                out.append(p.assign(eval_set=name))
            pd.concat(out).to_csv(os.path.join(PREDS, f"{tag}_{a.dataset}_fold{k}.csv"), index=False)
            print(f"  fold {k} done", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
