"""Train the teacher or the student for one cross-validation fold (Algorithm 1).

Teacher:  python train.py --role teacher --fold 0
Student:  python train.py --role student --fold 0 --teacher_ckpt /kaggle/working/ckpt/teacher_fold0.pt
Ablation: add e.g. --no_meta / --fusion concat / --flat / --softmax / --no_concepts / --kd none|plain / --p_md 0
"""
import argparse, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from config import CKPT, CLASSES, DEV, MANIFEST
from data import SkinDataset, eval_tf, train_tf
from losses import class_balanced_weights, concept_loss, hierarchical_loss, kd_loss
from model import build, probs_and_uncertainty


def args_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["teacher", "student"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default=None, help="run name (default: role)")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--img", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    # method switches (defaults = full TrustDerm-Net)
    ap.add_argument("--no_meta", action="store_true")
    ap.add_argument("--fusion", choices=["gated", "concat"], default="gated")
    ap.add_argument("--flat", action="store_true", help="single 7-class head instead of the hierarchy")
    ap.add_argument("--softmax", action="store_true", help="softmax heads instead of evidential heads")
    ap.add_argument("--no_concepts", action="store_true")
    ap.add_argument("--p_md", type=float, default=0.3)
    ap.add_argument("--kd", choices=["none", "plain", "weighted"], default="weighted")
    ap.add_argument("--teacher_ckpt", default=None)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--lambda_kd", type=float, default=1.0)
    ap.add_argument("--lambda_c", type=float, default=0.5)
    ap.add_argument("--concept_repeat", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.999)
    ap.add_argument("--max_train", type=int, default=0, help="debug: limit training images")
    a = ap.parse_args()
    teacher = a.role == "teacher"
    a.backbone = a.backbone or ("convnext_base.fb_in22k_ft_in1k" if teacher else "mobilevit_s.cvnets_in1k")
    a.img = a.img or (320 if teacher else 224)
    a.epochs = a.epochs or (20 if teacher else 40)
    a.bs = a.bs or (24 if teacher else 64)
    a.lr = a.lr or (1e-4 if teacher else 3e-4)
    if teacher:  # the teacher is a hierarchical softmax model without concepts or distillation
        a.softmax, a.flat, a.kd, a.no_concepts = True, False, "none", True
    a.tag = a.tag or a.role
    return a


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}


@torch.no_grad()
def predict(model, loader, dev, size=None):
    model.eval()
    P, Y = [], []
    for b in loader:
        b = to_dev(b, dev)
        x = b["image"] if size is None else F.interpolate(b["image"], size=size, mode="bilinear", antialias=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(x, b["age"], b["sex"], b["site"])
        p7, _, _ = probs_and_uncertainty(out, model.hierarchical, model.evidential)
        P.append(p7.cpu()); Y.append(b["y"].cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def main():
    a = args_()
    seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = pd.read_csv(MANIFEST, dtype={"dhash": str})
    tr = m[(m.split.str.startswith("fold")) & (m.split != f"fold{a.fold}")]
    va = m[m.split == f"fold{a.fold}"]
    if a.max_train:
        tr = tr.sample(min(a.max_train, len(tr)), random_state=a.seed)
    print(f"fold {a.fold}: train {len(tr)}  val {len(va)}  device {dev}")

    teacher, t_img = None, None
    if a.role == "student" and a.kd != "none":
        assert a.teacher_ckpt, "--teacher_ckpt is required unless --kd none"
        ck = torch.load(a.teacher_ckpt, map_location="cpu", weights_only=False)
        teacher = build(ck["model_cfg"], pretrained=False)
        teacher.load_state_dict(ck["state_dict"])
        teacher.to(dev).eval().requires_grad_(False)
        t_img = ck["model_cfg"]["img_size"]
    load_img = max(a.img, t_img or 0)   # one augmented view, downsampled for the student

    cfg = dict(backbone=a.backbone, img_size=a.img, use_meta=not a.no_meta, fusion=a.fusion,
               hierarchical=not a.flat, evidential=not a.softmax, use_concepts=not a.no_concepts,
               p_md=a.p_md if not a.no_meta else 0.0)
    model = build(cfg).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    ds_tr = SkinDataset(tr, train_tf(load_img), concept_repeat=a.concept_repeat if not a.no_concepts else 1)
    dl_tr = DataLoader(ds_tr, a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(SkinDataset(va, eval_tf(a.img)), a.bs * 2, num_workers=a.workers, pin_memory=True)

    counts = np.bincount(tr.y.astype(int), minlength=7)
    w7 = class_balanced_weights(counts, a.beta).to(dev)
    w2 = class_balanced_weights([counts[:4].sum(), counts[4:].sum()], a.beta).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps, warm = a.epochs * len(dl_tr), a.warmup * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    os.makedirs(CKPT, exist_ok=True)
    ck_path = os.path.join(CKPT, f"{a.tag}_fold{a.fold}.pt")
    log_path = os.path.join(CKPT, f"{a.tag}_fold{a.fold}_log.csv")
    best, bad, log = -1.0, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        lam = min(1.0, ep / 10)                                  # KL annealing, Eq. 15
        t0, tot, n = time.time(), 0.0, 0
        for b in dl_tr:
            b = to_dev(b, dev)
            x = b["image"]
            xs = x if x.shape[-1] == a.img else F.interpolate(x, size=a.img, mode="bilinear", antialias=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                out = model(xs, b["age"], b["sex"], b["site"])
                t_out = None
                if teacher is not None:
                    with torch.no_grad():
                        xt = x if x.shape[-1] == t_img else F.interpolate(x, size=t_img, mode="bilinear", antialias=True)
                        t_out = teacher(xt, b["age"], b["sex"], b["site"])
            loss = hierarchical_loss(out, b["y"], b["z"], w7, w2, lam, model.hierarchical, model.evidential)
            if t_out is not None:
                loss = loss + a.lambda_kd * kd_loss(out, t_out, a.tau, a.kd == "weighted",
                                                    model.hierarchical, model.evidential)
            lc = concept_loss(out, b["concepts"])
            if lc is not None:
                loss = loss + a.lambda_c * lc
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x); n += len(x)
        P, Y = predict(model, dl_va, dev)
        bacc = balanced_accuracy_score(Y, P.argmax(1))
        log.append(dict(epoch=ep, train_loss=tot / n, val_bacc=bacc, minutes=(time.time() - t0) / 60))
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(f"ep {ep:3d}  loss {tot / n:.4f}  val bAcc {bacc:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if bacc > best:
            best, bad = bacc, 0
            torch.save(dict(state_dict=model.state_dict(), model_cfg=model.cfg, args=vars(a), fold=a.fold,
                            val_bacc=bacc, epoch=ep), ck_path)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break
    print(json.dumps(dict(tag=a.tag, fold=a.fold, best_val_bacc=best, ckpt=ck_path)))


if __name__ == "__main__":
    main()
