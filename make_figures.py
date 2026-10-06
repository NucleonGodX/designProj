"""Paper figures from runs/main -> runs/main/figures/*.png|pdf  (300 dpi, colour-blind-safe palette)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

RUN = sys.argv[1] if len(sys.argv) > 1 else r"C:\designProj\runs\main"
OUT = f"{RUN}/figures"
os.makedirs(OUT, exist_ok=True)
C = dict(blue="#1f77b4", orange="#e07b00", green="#2a9d5c", grey="#6b7280", red="#c0392b")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)


h = json.load(open(f"{RUN}/history.json"))
res = json.load(open(f"{RUN}/results.json"))
rows = json.load(open(f"{RUN}/test_per_patient.json"))

# Fig 1: loss curves
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(h["epoch"], h["train_loss"], color=C["blue"], label="Train")
ax[0].plot(h["epoch"], h["val_loss"], color=C["orange"], label="Validation")
ax[0].set(xlabel="Epoch", ylabel="Loss", title="(a) Total loss (main + vessel aux; train also deep supervision)")
ax[0].legend(frameon=False)
ax[1].plot(h["epoch"], h["train_main"], color=C["blue"], label="Train")
ax[1].plot(h["epoch"], h["val_main"], color=C["orange"], label="Validation")
ax[1].set(xlabel="Epoch", ylabel="Dice + BCE loss", title="(b) Main segmentation head loss")
ax[1].legend(frameon=False)
for a in ax:
    a.title.set_fontsize(9)
save(fig, "fig1_loss_curves")

# Fig 2: validation Dice vs epoch
fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.plot(h["val_dice_epoch"], h["val_dice"], "o-", color=C["green"], ms=4)
b = int(np.argmax(h["val_dice"]))
ax.scatter([h["val_dice_epoch"][b]], [h["val_dice"][b]], s=70, facecolor="none", edgecolor=C["red"], zorder=3,
           label=f"Best: {h['val_dice'][b]:.3f} (epoch {h['val_dice_epoch'][b]})")
ax.set(xlabel="Epoch", ylabel="Validation Dice (full volume)", title="Validation Dice during training")
ax.legend(frameon=False)
save(fig, "fig2_val_dice")

# Fig 3: ROC + PR
d = np.load(f"{RUN}/roc_pr.npz")
fpr, tpr, _ = roc_curve(d["y"], d["s"], sample_weight=d["w"])
pr, rc, _ = precision_recall_curve(d["y"], d["s"], sample_weight=d["w"])
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.8))
ax[0].plot(fpr, tpr, color=C["blue"], lw=2, label=f"AUC = {res['roc_auc']:.3f}")
ax[0].plot([0, 1], [0, 1], "--", color=C["grey"], lw=1)
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) Voxel-wise ROC", xlim=(0, 1), ylim=(0, 1.01))
ax[0].legend(frameon=False, loc="lower right")
ax[1].plot(rc, pr, color=C["orange"], lw=2, label=f"AP = {res['avg_precision']:.3f}")
ax[1].set(xlabel="Recall", ylabel="Precision", title="(b) Voxel-wise precision-recall", xlim=(0, 1), ylim=(0, 1.01))
ax[1].legend(frameon=False, loc="upper right")
for a in ax:
    a.title.set_fontsize(9)
save(fig, "fig3_roc_pr")

# Fig 4: per-patient test Dice vs lesion volume + distribution
r = [x for x in rows if x["gt_vox"] > 0]
vol = np.array([x["gt_vox"] for x in r]) / 1000.0
dice = np.array([x["dice"] for x in r])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
ax[0].scatter(vol, dice, s=34, color=C["blue"], alpha=0.85)
ax[0].set_xscale("log")
ax[0].axhline(np.median(dice), color=C["red"], ls="--", lw=1, label=f"Median Dice {np.median(dice):.3f}")
ax[0].set(xlabel="Ground-truth lesion volume (mL, log scale)", ylabel="Dice", ylim=(-0.02, 1.02),
          title="(a) Test Dice vs total lesion volume")
ax[0].legend(frameon=False, loc="lower right")
ax[1].boxplot(dice, widths=0.5, patch_artist=True, boxprops=dict(facecolor="#cfe3f5", color=C["blue"]),
              medianprops=dict(color=C["red"]))
ax[1].scatter(np.random.RandomState(0).normal(1, 0.04, len(dice)), dice, s=14, color=C["blue"], alpha=0.7)
ax[1].set(xticks=[1], xticklabels=[f"n = {len(dice)}"], ylabel="Dice", ylim=(-0.02, 1.02), title="(b) Distribution")
for a in ax:
    a.title.set_fontsize(9)
save(fig, "fig4_test_dice")

# Fig 5 (bonus): qualitative overlays
q = np.load(f"{RUN}/qual.npz")
cases = sorted({k.split("__")[0] for k in q.files})
if cases:
    fig, ax = plt.subplots(len(cases), 3, figsize=(8, 2.7 * len(cases)))
    ax = np.atleast_2d(ax)
    for i, c in enumerate(cases):
        t1c, gt, pr_ = q[f"{c}__t1c"].astype(np.float32), q[f"{c}__gt"], q[f"{c}__prob"].astype(np.float32) > 0.5
        ys, xs = np.nonzero(gt)
        cy, cx = int(ys.mean()), int(xs.mean())
        hw = 40
        y0, x0 = max(0, cy - hw), max(0, cx - hw)
        sl = (slice(y0, y0 + 2 * hw), slice(x0, x0 + 2 * hw))
        for j, (title, m, col) in enumerate([("T1c", None, None), ("Ground truth", gt, C["green"]), ("Prediction", pr_, C["red"])]):
            ax[i, j].imshow(t1c[sl], cmap="gray", vmin=-1, vmax=3)
            if m is not None:
                ax[i, j].contour(m[sl].astype(float), levels=[0.5], colors=[col], linewidths=1.3)
            ax[i, j].axis("off")
            ax[i, j].set_title(title if i == 0 else "", fontsize=9)
        ax[i, 0].text(2, 8, f"{c.split('-')[-2]}  Dice {float(q[f'{c}__dice']):.2f}", color="white", fontsize=8)
    save(fig, "fig5_qualitative")
print("saved to", OUT)
