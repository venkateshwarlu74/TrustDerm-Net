"""Efficiency of every compared architecture (Table VIII) without trained checkpoints.

Parameters, MACs and latency do not depend on the weights, so models are built with random weights from
the same configurations used in training. Latency: batch 1, CPU, N threads, 10 warm-up runs then R timed runs.

Usage: python efficiency_table.py [--threads 4] [--runs 50] [--out efficiency.csv]
"""
import argparse, io, platform, statistics, time
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode
from model import build

BASE = dict(img_size=224, use_meta=False, fusion="gated", hierarchical=False, evidential=False,
            use_concepts=False, p_md=0.0)
MODELS = {
    "TrustDerm-Net (student)": dict(backbone="mobilevit_s", img_size=224, use_meta=True, fusion="gated",
                                    hierarchical=True, evidential=True, use_concepts=True, p_md=0.3),
    "Teacher (ConvNeXt-B)": dict(backbone="convnext_base", img_size=320, use_meta=True, fusion="gated",
                                 hierarchical=True, evidential=False, use_concepts=False, p_md=0.3),
    "ResNet-50": dict(BASE, backbone="resnet50"),
    "EfficientNet-B3": dict(BASE, backbone="efficientnet_b3"),
    "MobileNetV2": dict(BASE, backbone="mobilenetv2_100"),
    "DeiT-S": dict(BASE, backbone="deit_small_patch16_224"),
    "Swin-T": dict(BASE, backbone="swin_tiny_patch4_window7_224"),
    "ConvNeXt-T": dict(BASE, backbone="convnext_tiny"),
}


def inputs(size):
    return (torch.randn(1, 3, size, size), torch.full((1,), 50.0),
            torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))


@torch.no_grad()
def measure(cfg, threads, runs):
    model = build(cfg, pretrained=False).eval()
    size = cfg["img_size"]
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    fc = FlopCounterMode(display=False)
    with fc:
        model(*inputs(size))
    torch.set_num_threads(threads)
    x = inputs(size)
    for _ in range(10):
        model(*x)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); model(*x); t.append((time.perf_counter() - t0) * 1000)
    return dict(params_M=sum(p.numel() for p in model.parameters()) / 1e6, GMACs=fc.get_total_flops() / 2 / 1e9,
                size_MB_fp32=buf.tell() / 2 ** 20, input=size, latency_ms_mean=statistics.mean(t),
                latency_ms_median=statistics.median(t), latency_ms_sd=statistics.stdev(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", default="efficiency.csv")
    a = ap.parse_args()
    rows = []
    for name, cfg in MODELS.items():
        r = dict(model=name, **measure(cfg, a.threads, a.runs))
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["cpu"], df["threads"], df["torch"] = platform.processor(), a.threads, torch.__version__
    df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
