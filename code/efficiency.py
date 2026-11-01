"""Parameters, FLOPs, model size, CPU latency and GPU throughput for a trained checkpoint (Table VIII).

Usage: python efficiency.py --ckpt /kaggle/working/ckpt/student_fold0.pt [--threads 4]
"""
import argparse, io, json, time
import torch
from torch.utils.flop_counter import FlopCounterMode
from model import build


def inputs(n, size, device):
    return (torch.randn(n, 3, size, size, device=device), torch.full((n,), 50.0, device=device),
            torch.zeros(n, dtype=torch.long, device=device), torch.zeros(n, dtype=torch.long, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=50)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build(ck["model_cfg"], pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.eval()
    size = ck["model_cfg"]["img_size"]
    r = dict(ckpt=a.ckpt, backbone=ck["model_cfg"]["backbone"], img_size=size,
             params_M=sum(p.numel() for p in model.parameters()) / 1e6)
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    r["size_MB_fp32"] = buf.tell() / 2 ** 20
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            model(*inputs(1, size, "cpu"))
        r["GFLOPs"] = fc.get_total_flops() / 2 / 1e9     # multiply-accumulates, the usual "FLOPs" convention
        torch.set_num_threads(a.threads)
        x = inputs(1, size, "cpu")
        for _ in range(10):
            model(*x)
        t = time.perf_counter()
        for _ in range(a.runs):
            model(*x)
        r[f"cpu_latency_ms_{a.threads}threads"] = (time.perf_counter() - t) / a.runs * 1000
        if torch.cuda.is_available():
            model.cuda(); x = inputs(64, size, "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                for _ in range(5):
                    model(*x)
                torch.cuda.synchronize(); t = time.perf_counter()
                for _ in range(20):
                    model(*x)
                torch.cuda.synchronize()
            r["gpu_images_per_s"] = 64 * 20 / (time.perf_counter() - t)
            r["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
