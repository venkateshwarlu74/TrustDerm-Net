"""TrustDerm-Net benign/malignant demo: a simplified app that shows only the Stage-1 triage decision
(benign vs. malignant), P(malignant), the evidential referral flag and a Grad-CAM map for that decision.

It uses the same five fold students, thresholds and preprocessing as trustderm_gui.py and the paper:
  - triage:   P(malignant) >= t_mal, set on validation data for 95% sensitivity (Eq. 21)
  - referral: uncertainty u (Eq. 14) > tau_u, the threshold that keeps 80% of validation cases (Eq. 22)
"Malignant" here means MEL, BCC or AKIEC (actinic keratosis / intraepithelial carcinoma, incl. SCC).

Research prototype only. It is not a medical device and must not be used for diagnosis.

Usage:
  python app/trustderm_binary_gui.py [--open IMAGE]
  python app/trustderm_binary_gui.py --cli IMAGE [--age 60 --sex male --site torso] [--model ens|0|1|2|3|4]
"""
import argparse, sys, threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trustderm_gui import DISCLAIMER, SITES, Predictor, overlay, probs_and_uncertainty   # noqa: E402

POS_TEXT = "Possibly malignant: needs review"
NEG_TEXT = "Likely benign"
OPERATING_POINT = ("The threshold is set low on purpose, for 95% sensitivity. On the HAM10000 test set it caught "
                   "95.5% of malignant lesions but also flagged about 40% of benign ones (specificity 59.8%). "
                   "\"Possibly malignant\" therefore means \"needs review\", not \"cancer\".")


class BinaryPredictor(Predictor):
    """Same models as the full demo; Grad-CAM explains the benign/malignant decision, not the 7-class one."""

    def run(self, path, age=None, sex=-1, site=-1, which="ens"):
        pil = self.preprocess(path)
        x = self.tf(pil).unsqueeze(0)
        meta = (torch.tensor([float("nan") if age is None else float(age)]),
                torch.tensor([sex]), torch.tensor([site]))
        idx = list(range(5)) if which == "ens" else [int(which)]
        t_mal = float(np.mean([self.t_mal[k] for k in idx]))
        tau = float(np.mean([self.tau[k] for k in idx]))
        PM, U = [], []
        for k in idx:                      # first pass: probabilities, to know which decision to explain
            with torch.no_grad():
                _, pm, u = probs_and_uncertainty(self.models[k](x, *meta), self.models[k].hierarchical,
                                                 self.models[k].evidential)
            PM.append(pm[0].item()); U.append(u[0].item())
        p_mal, u = float(np.mean(PM)), float(np.mean(U))
        positive = p_mal >= t_mal
        cams = [self._cam(self.models[k], x, meta, positive) for k in idx]
        crop = np.asarray(self.tf.transforms[1](self.tf.transforms[0](pil)))
        return dict(p_mal=p_mal, u=u, t_mal=t_mal, tau=tau, malignant_flag=positive, refer=u > tau,
                    image=crop, cam=np.mean(cams, 0), folds=idx)

    @staticmethod
    def _cam(model, x, meta, positive):
        """Grad-CAM of log P(malignant) if screen-positive, else of log P(benign)."""
        store, orig = {}, model.encoder.forward_features

        def ff(inp):
            f = orig(inp)
            f.retain_grad()
            store["f"] = f
            return f
        model.encoder.forward_features = ff
        try:
            _, pm, _ = probs_and_uncertainty(model(x, *meta), model.hierarchical, model.evidential)
            target = pm[0] if positive else 1.0 - pm[0]
            model.zero_grad()
            torch.log(target.clamp_min(1e-8)).backward()
        finally:
            del model.encoder.forward_features
        f, g = store["f"], store["f"].grad
        cam = F.relu((g.mean((2, 3), keepdim=True) * f).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().numpy()


def report(r, path):
    model = "ensemble of folds 0-4" if len(r["folds"]) == 5 else "fold " + str(r["folds"][0])
    return "\n".join([
        f"Image: {path}", f"Model: {model}", "",
        f"Result: {(POS_TEXT if r['malignant_flag'] else NEG_TEXT).upper()}",
        f"  P(malignant) = {r['p_mal']:.3f}   (threshold t_mal = {r['t_mal']:.3f})",
        f"  {OPERATING_POINT}", "",
        f"Uncertainty u = {r['u']:.3f}   (referral threshold {r['tau']:.3f})  ->  "
        f"{'REFER to a dermatologist' if r['refer'] else 'model is confident'}",
        "", "Malignant = melanoma, basal cell carcinoma or actinic keratosis / SCC.", "", DISCLAIMER])


# ------------------------------------------------------------------------------------------------ GUI
def gui(start_image=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("TrustDerm-Net — benign / malignant (research demo)")
    root.geometry("1060x700")
    root.minsize(940, 640)
    BG, CARD, ACC = "#f4f6f8", "#ffffff", "#1f4e79"
    RED, GREEN, AMBER = "#b42318", "#1a7f37", "#b54708"
    root.configure(bg=BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=CARD)
    style.configure("Card.TLabel", background=CARD, font=("Segoe UI", 10))
    style.configure("H.TLabel", background=CARD, font=("Segoe UI", 11, "bold"), foreground=ACC)
    style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    state = dict(pred=None, path=None, result=None, photos=[])

    tk.Label(root, text=DISCLAIMER, bg="#fff4ce", fg="#5c4400", font=("Segoe UI", 9), wraplength=1020,
             justify="left", anchor="w", padx=10, pady=6).pack(fill="x")
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
    img_lbl = tk.Label(mid, bg="#e9ecef", width=34, height=13)
    img_lbl.pack(pady=(4, 10))
    ttk.Label(mid, text="Grad-CAM: what drove the decision", style="H.TLabel").pack(anchor="w")
    cam_lbl = tk.Label(mid, bg="#e9ecef", width=34, height=13)
    cam_lbl.pack(pady=(4, 0))

    # ---- right: the result
    right = ttk.Frame(main, style="Card.TFrame", padding=14)
    right.grid(row=0, column=2, sticky="nsew")
    ttk.Label(right, text="Result", style="H.TLabel").pack(anchor="w")
    result_box = tk.Label(right, text="—", font=("Segoe UI", 18, "bold"), bg="#e9ecef", fg="#333",
                          wraplength=340, justify="center", padx=10, pady=18)
    result_box.pack(fill="x", pady=(4, 12))
    ttk.Label(right, text="Probability of malignancy", style="H.TLabel").pack(anchor="w")
    gauge = tk.Canvas(right, width=340, height=62, bg=CARD, highlightthickness=0)
    gauge.pack(anchor="w", pady=(2, 2))
    pm_lbl = ttk.Label(right, text="", style="Card.TLabel", wraplength=340)
    pm_lbl.pack(anchor="w")
    refer_lbl = tk.Label(right, text="", font=("Segoe UI", 11, "bold"), bg=CARD, anchor="w", justify="left",
                         wraplength=340)
    refer_lbl.pack(anchor="w", fill="x", pady=(12, 8))
    ttk.Label(right, text=OPERATING_POINT + "\n\nMalignant = melanoma, basal cell carcinoma or actinic keratosis / SCC.",
              style="Card.TLabel", foreground="#555", wraplength=320, justify="left").pack(anchor="w", pady=(6, 0))

    def draw_gauge(p, t):
        """Bar for P(malignant) on a square-root scale, so the low threshold is visible; dashed line = t_mal."""
        gauge.delete("all")
        x0, x1, y0, y1 = 10, 318, 8, 30
        pos = lambda v: x0 + (x1 - x0) * np.sqrt(max(0.0, min(1.0, v)))
        gauge.create_rectangle(x0, y0, x1, y1, fill="#e9ecef", width=0)
        gauge.create_rectangle(x0, y0, pos(p), y1, fill=RED if p >= t else GREEN, width=0)
        xt = pos(t)
        gauge.create_line(xt, y0 - 5, xt, y1 + 5, dash=(3, 2), fill="#333", width=2)
        gauge.create_text(xt + 3, y1 + 13, text=f"threshold {t:.3f}", anchor="w", font=("Segoe UI", 8))
        for v in (0, 0.25, 1.0):
            gauge.create_text(pos(v), y1 + 26, text=f"{v:g}", font=("Segoe UI", 8), fill="#777")

    def show(arr, lbl, size=270):
        ph = ImageTk.PhotoImage(Image.fromarray(arr).resize((size, size)))
        state["photos"] = (state["photos"] + [ph])[-4:]
        lbl.configure(image=ph, width=size, height=size)

    def choose(p=None):
        p = p or filedialog.askopenfilename(title="Choose a dermoscopic image",
                                            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not p:
            return
        state["path"] = p
        file_lbl.configure(text=Path(p).name)
        img = Image.open(p).convert("RGB")
        img.thumbnail((270, 270))
        side = Image.new("RGB", (270, 270), (233, 236, 239))
        side.paste(img, ((270 - img.width) // 2, (270 - img.height) // 2))
        show(np.asarray(side), img_lbl)
        cam_lbl.configure(image="")
        result_box.configure(text="—", bg="#e9ecef", fg="#333")
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
            result_box.configure(text=POS_TEXT, bg="#fde8e7", fg=RED)
        else:
            result_box.configure(text=NEG_TEXT, bg="#e6f4ea", fg=GREEN)
        draw_gauge(r["p_mal"], r["t_mal"])
        pm_lbl.configure(text=f"P(malignant) = {r['p_mal']:.3f}   (threshold {r['t_mal']:.3f})")
        if r["refer"]:
            refer_lbl.configure(text=f"⚠ Uncertain (u = {r['u']:.2f} > {r['tau']:.2f}): refer to a dermatologist",
                                fg=AMBER)
        else:
            refer_lbl.configure(text=f"✓ Model is confident (u = {r['u']:.2f} ≤ {r['tau']:.2f})", fg=GREEN)
        status.configure(text="Done.")
        run_btn.configure(state="normal")
        save_btn.configure(state="normal")

    def save():
        p = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=Path(state["path"]).stem + "_binary_report.txt",
                                         filetypes=[("Text", "*.txt")])
        if p:
            Path(p).write_text(report(state["result"], state["path"]), encoding="utf-8")
            Image.fromarray(overlay(state["result"]["image"], state["result"]["cam"])).save(Path(p).with_suffix(".cam.png"))
            status.configure(text=f"Saved {Path(p).name} and the Grad-CAM image.")

    def load():
        try:
            pr = BinaryPredictor()
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
    r = BinaryPredictor().run(a.cli, a.age, {"male": 0, "female": 1}.get(a.sex, -1),
                              SITES.index(a.site) if a.site else -1, a.model)
    print(report(r, a.cli))


if __name__ == "__main__":
    main()
