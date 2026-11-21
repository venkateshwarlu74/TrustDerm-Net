"""TrustDerm-Net desktop demo: load a dermoscopic image, enter optional metadata, and see the triage
decision, the 7-class diagnosis, the evidential uncertainty with referral, the seven-point checklist
concepts and a Grad-CAM map.

Research prototype only. It is not a medical device and must not be used for diagnosis.

Usage:
  python app/trustderm_gui.py [--open IMAGE]       # open the window (optionally with an image)
  python app/trustderm_gui.py --cli IMAGE [--age 60 --sex male --site torso] [--model ens|0|1|2|3|4]
"""
import argparse, json, sys, threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CLASSES, CONCEPTS, SITES          # noqa: E402
from data import eval_tf                             # noqa: E402
from model import build, probs_and_uncertainty       # noqa: E402
from preprocess import SHORT, hair_removal, shades_of_gray   # noqa: E402

CKPT = ROOT / "results" / "student_v1" / "ckpt"
THRESH = ROOT / "results" / "student_eval_v2" / "results"
KAPPA = "0.8"        # referral threshold: the model keeps the 80% most certain validation cases
NAMES = {"NV": "Melanocytic nevus", "BKL": "Benign keratosis", "DF": "Dermatofibroma", "VASC": "Vascular lesion",
         "MEL": "Melanoma", "BCC": "Basal cell carcinoma", "AKIEC": "Actinic keratosis / SCC"}
CONCEPT_NAMES = {"atypical_network": "Atypical pigment network", "blue_whitish_veil": "Blue-whitish veil",
                 "atypical_vascular": "Atypical vascular pattern", "irregular_streaks": "Irregular streaks",
                 "irregular_dots_globules": "Irregular dots/globules", "irregular_blotches": "Irregular blotches",
                 "regression": "Regression structures"}
DISCLAIMER = ("Research prototype — not a medical device. Not for diagnosis. Trained on dermoscopic images "
              "(HAM10000). Melanoma AUROC was 0.76 on external dermoscopic images (ISIC 2020) but only 0.43 "
              "on ordinary phone photos (PAD-UFES-20), so it is unreliable on phone photos.")
torch.set_num_threads(4)


# ------------------------------------------------------------------------------------------------ model
class Predictor:
    """The five fold students of the paper; 'ens' averages their outputs and thresholds."""

    def __init__(self):
        self.models, self.t_mal, self.tau = [], [], []
        for k in range(5):
            ck = torch.load(CKPT / f"student_fold{k}.pt", map_location="cpu", weights_only=False)
            m = build(ck["model_cfg"], pretrained=False)
            m.load_state_dict(ck["state_dict"])
            self.models.append(m.eval())
            th = json.loads((THRESH / f"student_fold{k}.json").read_text())
            self.t_mal.append(float(th["t_mal"]))
            self.tau.append(float(th["tau_u"][KAPPA]))
        self.tf = eval_tf(self.models[0].cfg["img_size"])

    @staticmethod
    def preprocess(path):
        """Same steps as the training cache: short side 448, hair removal, Shades of Gray, JPEG q95."""
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot read image: {path}")
        h, w = img.shape[:2]
        s = SHORT / min(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        img = shades_of_gray(hair_removal(img))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def run(self, path, age=None, sex=-1, site=-1, which="ens"):
        pil = self.preprocess(path)
        x = self.tf(pil).unsqueeze(0)
        meta = (torch.tensor([float("nan") if age is None else float(age)]),
                torch.tensor([sex]), torch.tensor([site]))
        idx = list(range(5)) if which == "ens" else [int(which)]
        P7, PM, U, C, cams = [], [], [], [], []
        for k in idx:
            p7, pm, u, c, cam = self._one(self.models[k], x, meta)
            P7.append(p7); PM.append(pm); U.append(u); C.append(c); cams.append(cam)
        p7, pm, u, c = (np.mean(v, 0) for v in (P7, PM, U, C))
        t_mal = float(np.mean([self.t_mal[k] for k in idx]))
        tau = float(np.mean([self.tau[k] for k in idx]))
        crop = np.asarray(self.tf.transforms[1](self.tf.transforms[0](pil)))      # the 224 x 224 view the model saw
        return dict(p7=p7, p_mal=float(pm), u=float(u), concepts=c, t_mal=t_mal, tau=tau,
                    malignant_flag=bool(pm >= t_mal), refer=bool(u > tau), pred=int(p7.argmax()),
                    image=crop, cam=np.mean(cams, 0), folds=idx)

    @staticmethod
    def _one(model, x, meta):
        """Forward pass plus Grad-CAM for the predicted class (same method as Fig. 7 of the paper)."""
        store, orig = {}, model.encoder.forward_features

        def ff(inp):
            f = orig(inp)
            f.retain_grad()
            store["f"] = f
            return f
        model.encoder.forward_features = ff
        try:
            out = model(x, *meta)
            p7, pm, u = probs_and_uncertainty(out, model.hierarchical, model.evidential)
            cls = int(p7.argmax())
            model.zero_grad()
            torch.log(p7[0, cls]).backward()
        finally:
            del model.encoder.forward_features
        f, g = store["f"], store["f"].grad
        cam = F.relu((g.mean((2, 3), keepdim=True) * f).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        c = torch.sigmoid(out["concept_logits"].float())[0]
        return (p7[0].detach().numpy(), pm[0].item(), u[0].item(), c.detach().numpy(), cam.detach().numpy())


def overlay(img, cam, alpha=0.45):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]
    return np.uint8((1 - alpha) * img + alpha * heat)


def report(r, path):
    lines = [f"Image: {path}", f"Model: {'ensemble of folds 0-4' if len(r['folds']) == 5 else 'fold ' + str(r['folds'][0])}",
             "",
             f"Triage: {'SCREEN-POSITIVE (review)' if r['malignant_flag'] else 'screen-negative'}"
             f"   P(malignant) = {r['p_mal']:.3f}  (threshold {r['t_mal']:.3f}, set for 95% sensitivity;\n"
             f"        on the HAM10000 test set it caught 95.5% of malignant lesions but also flagged ~40% of benign ones)",
             f"Uncertainty u = {r['u']:.3f}  (referral threshold {r['tau']:.3f})  ->  "
             f"{'REFER to a dermatologist' if r['refer'] else 'confident'}",
             f"Most likely class: {NAMES[CLASSES[r['pred']]]} ({r['p7'][r['pred']]:.1%})", "", "7-class probabilities:"]
    lines += [f"  {NAMES[c]:<28} {p:6.1%}" for c, p in sorted(zip(CLASSES, r["p7"]), key=lambda t: -t[1])]
    lines += ["", "Seven-point checklist concepts (probability):"]
    lines += [f"  {CONCEPT_NAMES[c]:<28} {p:5.2f}  {'present' if p >= 0.5 else ''}" for c, p in zip(CONCEPTS, r["concepts"])]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ GUI
def gui(start_image=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("TrustDerm-Net — research demo")
    root.geometry("1180x760")
    root.minsize(1000, 680)
    BG, CARD, ACC = "#f4f6f8", "#ffffff", "#1f4e79"
    root.configure(bg=BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=CARD)
    style.configure("TLabel", background=BG, font=("Segoe UI", 10))
    style.configure("Card.TLabel", background=CARD, font=("Segoe UI", 10))
    style.configure("H.TLabel", background=CARD, font=("Segoe UI", 11, "bold"), foreground=ACC)
    style.configure("Big.TLabel", background=CARD, font=("Segoe UI", 15, "bold"))
    style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    state = dict(pred=None, path=None, result=None, photos=[])

    banner = tk.Label(root, text=DISCLAIMER, bg="#fff4ce", fg="#5c4400", font=("Segoe UI", 9), wraplength=1140,
                      justify="left", anchor="w", padx=10, pady=6)
    banner.pack(fill="x")

    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)
    main.columnconfigure(2, weight=1)
    main.rowconfigure(0, weight=1)

    # ---- left: inputs
    left = ttk.Frame(main, style="Card.TFrame", padding=12)
    left.grid(row=0, column=0, sticky="ns", padx=(0, 10))
    ttk.Label(left, text="1. Image", style="H.TLabel").pack(anchor="w")
    ttk.Button(left, text="Open dermoscopic image…", command=lambda: choose()).pack(fill="x", pady=(4, 2))
    file_lbl = ttk.Label(left, text="No image loaded", style="Card.TLabel", wraplength=220, foreground="#666")
    file_lbl.pack(anchor="w", pady=(0, 12))

    ttk.Label(left, text="2. Patient data (optional)", style="H.TLabel").pack(anchor="w")
    ttk.Label(left, text="Leave blank / Unknown if not known.", style="Card.TLabel", foreground="#666").pack(anchor="w")
    ttk.Label(left, text="Age (years)", style="Card.TLabel").pack(anchor="w", pady=(8, 0))
    age_var = tk.StringVar()
    ttk.Entry(left, textvariable=age_var, width=12).pack(anchor="w")
    ttk.Label(left, text="Sex", style="Card.TLabel").pack(anchor="w", pady=(8, 0))
    sex_var = tk.StringVar(value="Unknown")
    ttk.Combobox(left, textvariable=sex_var, values=["Unknown", "Male", "Female"], state="readonly", width=20).pack(anchor="w")
    ttk.Label(left, text="Lesion site", style="Card.TLabel").pack(anchor="w", pady=(8, 0))
    site_var = tk.StringVar(value="Unknown")
    ttk.Combobox(left, textvariable=site_var, values=["Unknown"] + SITES, state="readonly", width=20).pack(anchor="w")

    ttk.Label(left, text="3. Model", style="H.TLabel").pack(anchor="w", pady=(14, 0))
    model_var = tk.StringVar(value="Ensemble of 5 folds")
    ttk.Combobox(left, textvariable=model_var, state="readonly", width=20,
                 values=["Ensemble of 5 folds", "Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"]).pack(anchor="w")

    run_btn = ttk.Button(left, text="Analyse", style="Accent.TButton", command=lambda: analyse(), state="disabled")
    run_btn.pack(fill="x", pady=(18, 4))
    save_btn = ttk.Button(left, text="Save report…", command=lambda: save(), state="disabled")
    save_btn.pack(fill="x")
    status = ttk.Label(left, text="Loading models…", style="Card.TLabel", foreground="#666", wraplength=220)
    status.pack(anchor="w", pady=(12, 0))

    # ---- middle: images
    mid = ttk.Frame(main, style="Card.TFrame", padding=12)
    mid.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
    ttk.Label(mid, text="Model input (after preprocessing)", style="H.TLabel").pack(anchor="w")
    img_lbl = tk.Label(mid, bg="#e9ecef", width=34, height=15)
    img_lbl.pack(pady=(4, 10))
    ttk.Label(mid, text="Grad-CAM: where the model looked", style="H.TLabel").pack(anchor="w")
    cam_lbl = tk.Label(mid, bg="#e9ecef", width=34, height=15)
    cam_lbl.pack(pady=(4, 0))

    # ---- right: results
    right = ttk.Frame(main, style="Card.TFrame", padding=12)
    right.grid(row=0, column=2, sticky="nsew")
    triage_lbl = ttk.Label(right, text="—", style="Big.TLabel")
    triage_lbl.pack(anchor="w")
    triage_sub = ttk.Label(right, text="", style="Card.TLabel", foreground="#444", wraplength=380)
    triage_sub.pack(anchor="w")
    refer_lbl = tk.Label(right, text="", font=("Segoe UI", 11, "bold"), bg=CARD, anchor="w", justify="left", wraplength=380)
    refer_lbl.pack(anchor="w", fill="x", pady=(8, 8))
    ttk.Label(right, text="Diagnosis probabilities", style="H.TLabel").pack(anchor="w")
    bars = tk.Canvas(right, width=380, height=190, bg=CARD, highlightthickness=0)
    bars.pack(anchor="w", pady=(2, 8))
    ttk.Label(right, text="Seven-point checklist concepts", style="H.TLabel").pack(anchor="w")
    cbars = tk.Canvas(right, width=380, height=190, bg=CARD, highlightthickness=0)
    cbars.pack(anchor="w", pady=(2, 0))

    def draw_bars(canvas, items, mark=None, threshold=None):
        canvas.delete("all")
        y = 6
        for name, p, color in items:
            canvas.create_text(4, y + 8, text=name, anchor="w", font=("Segoe UI", 9))
            x0, x1 = 185, 330
            canvas.create_rectangle(x0, y + 2, x1, y + 15, fill="#e9ecef", width=0)
            canvas.create_rectangle(x0, y + 2, x0 + (x1 - x0) * float(p), y + 15, fill=color, width=0)
            canvas.create_text(x1 + 6, y + 8, text=f"{p:.0%}" if mark != "raw" else f"{p:.2f}", anchor="w",
                               font=("Segoe UI", 9, "bold" if p >= 0.5 else "normal"))
            y += 26
        if threshold is not None:
            xt = 185 + 145 * threshold
            canvas.create_line(xt, 2, xt, y - 6, dash=(3, 2), fill="#555")

    def show(arr, lbl, size=300):
        ph = ImageTk.PhotoImage(Image.fromarray(arr).resize((size, size)))
        state["photos"].append(ph)
        state["photos"] = state["photos"][-4:]
        lbl.configure(image=ph, width=size, height=size)

    def choose(p=None):
        p = p or filedialog.askopenfilename(title="Choose a dermoscopic image",
                                            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not p:
            return
        state["path"] = p
        file_lbl.configure(text=Path(p).name)
        img = Image.open(p).convert("RGB")
        img.thumbnail((300, 300))
        side = Image.new("RGB", (300, 300), (233, 236, 239))
        side.paste(img, ((300 - img.width) // 2, (300 - img.height) // 2))
        show(np.asarray(side), img_lbl)
        cam_lbl.configure(image="")
        if state["pred"]:
            run_btn.configure(state="normal")

    def inputs():
        a = age_var.get().strip()
        age = None
        if a:
            age = float(a)
            if not 0 <= age <= 110:
                raise ValueError("Age must be between 0 and 110, or blank.")
        sex = {"Male": 0, "Female": 1}.get(sex_var.get(), -1)
        site = SITES.index(site_var.get()) if site_var.get() in SITES else -1
        which = "ens" if model_var.get().startswith("Ens") else model_var.get()[-1]
        return age, sex, site, which

    def analyse():
        try:
            age, sex, site, which = inputs()
        except ValueError as e:
            messagebox.showerror("Input", str(e) if "Age" in str(e) else "Age must be a number, or blank.")
            return
        run_btn.configure(state="disabled")
        status.configure(text="Analysing… (a few seconds on CPU)")

        def work():
            try:
                r = state["pred"].run(state["path"], age, sex, site, which)
                root.after(0, lambda: done(r))
            except Exception as e:      # shown to the user, not swallowed
                root.after(0, lambda: (messagebox.showerror("Error", str(e)), run_btn.configure(state="normal"),
                                       status.configure(text="Error.")))
        threading.Thread(target=work, daemon=True).start()

    def done(r):
        state["result"] = r
        show(r["image"], img_lbl)
        show(overlay(r["image"], r["cam"]), cam_lbl)
        if r["malignant_flag"]:
            triage_lbl.configure(text="Triage: screen-positive", foreground="#b42318")
        else:
            triage_lbl.configure(text="Triage: screen-negative", foreground="#1a7f37")
        triage_sub.configure(text=f"P(malignant) = {r['p_mal']:.3f}; triage threshold {r['t_mal']:.3f} "
                                  f"(high-sensitivity setting: catches ~95% of malignant lesions but also flags ~40% of benign ones). "
                                  f"Most likely: {NAMES[CLASSES[r['pred']]]} ({r['p7'][r['pred']]:.0%}).")
        if r["refer"]:
            refer_lbl.configure(text=f"⚠ Uncertain (u = {r['u']:.2f} > {r['tau']:.2f}): refer to a dermatologist",
                                fg="#b54708")
        else:
            refer_lbl.configure(text=f"✓ Model is confident (u = {r['u']:.2f} ≤ {r['tau']:.2f})", fg="#1a7f37")
        order = np.argsort(-r["p7"])
        draw_bars(bars, [(NAMES[CLASSES[i]], r["p7"][i], "#b42318" if i >= 4 else "#2f6fb0") for i in order])
        draw_bars(cbars, [(CONCEPT_NAMES[c], p, "#7a4cc2") for c, p in zip(CONCEPTS, r["concepts"])],
                  mark="raw", threshold=0.5)
        status.configure(text="Done. Red bars = malignant classes.")
        run_btn.configure(state="normal")
        save_btn.configure(state="normal")

    def save():
        p = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=Path(state["path"]).stem + "_report.txt",
                                         filetypes=[("Text", "*.txt")])
        if p:
            Path(p).write_text(report(state["result"], state["path"]), encoding="utf-8")
            Image.fromarray(overlay(state["result"]["image"], state["result"]["cam"])).save(Path(p).with_suffix(".cam.png"))
            status.configure(text=f"Saved {Path(p).name} and the Grad-CAM image.")

    def load():
        try:
            pr = Predictor()
            root.after(0, lambda: (state.update(pred=pr), status.configure(text="Ready. Open an image."),
                                   run_btn.configure(state="normal" if state["path"] else "disabled"),
                                   analyse() if start_image else None))
        except Exception as e:
            root.after(0, lambda: status.configure(text=f"Could not load models: {e}"))
    if start_image:
        choose(start_image)
    threading.Thread(target=load, daemon=True).start()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", metavar="IMAGE")
    ap.add_argument("--age", type=float)
    ap.add_argument("--sex", choices=["male", "female"])
    ap.add_argument("--site", choices=SITES)
    ap.add_argument("--model", default="ens", choices=["ens", "0", "1", "2", "3", "4"])
    ap.add_argument("--open", metavar="IMAGE", help="GUI: load this image and analyse it at start-up (for demos)")
    a = ap.parse_args()
    if not a.cli:
        return gui(a.open)
    r = Predictor().run(a.cli, a.age, {"male": 0, "female": 1}.get(a.sex, -1),
                        SITES.index(a.site) if a.site else -1, a.model)
    print(report(r, a.cli))


if __name__ == "__main__":
    main()
