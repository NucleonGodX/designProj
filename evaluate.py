"""Evaluate best checkpoint on the held-out 15% test patients. Writes runs/main/{results.json,test_per_patient.json,roc_pr.npz,qual.npz}."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "arch"))
sys.path.insert(0, os.path.dirname(__file__))
import train as T  # noqa: E402  (also applies fast_mta patch)


def lesion_stats(pred, gt):
    st = np.ones((3, 3, 3))
    gl, ng = ndimage.label(gt, st)
    pl, npred = ndimage.label(pred, st)
    det = sum(1 for i in range(1, ng + 1) if pred[gl == i].any()) if ng else 0
    fp = sum(1 for j in range(1, npred + 1) if not gt[pl == j].any()) if npred else 0
    return ng, det, npred, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"C:\designProj\runs\main")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--feature_size", type=int, default=24)
    ap.add_argument("--heads", type=int, nargs=4, default=[2, 4, 8, 16])
    ap.add_argument("--window", type=int, default=4)
    a = ap.parse_args()

    split = json.load(open(f"{a.run}/split.json"))
    model = T.build_model(a)
    model.load_state_dict(torch.load(f"{a.run}/{a.ckpt}", map_location="cuda"))
    model.eval()
    rng = np.random.RandomState(0)

    rows, ys, ss, ws = [], [], [], []
    qual = {}
    tot = dict(correct=0, vox=0, tp=0, fp=0, fn=0, lg=0, ld=0, lp=0, lfp=0)
    for cid in split["test"]:
        v = T.load_case(cid)
        prob = T.sliding_window(model, v["img"])
        gt = v["tc"] > 0
        pred = prob > a.thr
        brain = v["img"][1] != 0
        tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
        ng, det, npred, lfp = lesion_stats(pred, gt)
        dice = T.dice_np(pred, gt)
        rows.append(dict(case=cid, dice=dice, gt_vox=int(gt.sum()), pred_vox=int(pred.sum()), tp=tp, fp=fp, fn=fn,
                         gt_lesions=ng, detected=det, pred_lesions=npred, fp_lesions=lfp,
                         voxel_acc=float(((pred == gt) & brain).sum() / brain.sum())))
        for k, val in dict(correct=int(((pred == gt) & brain).sum()), vox=int(brain.sum()), tp=tp, fp=fp, fn=fn,
                           lg=ng, ld=det, lp=npred, lfp=lfp).items():
            tot[k] += val
        # ROC/PR samples: all positives + random brain negatives, negatives re-weighted
        pos = np.flatnonzero(gt.ravel())
        neg_all = np.flatnonzero((brain & ~gt).ravel())
        neg = rng.choice(neg_all, size=min(100000, len(neg_all)), replace=False)
        idx = np.concatenate([pos, neg])
        ys.append(gt.ravel()[idx].astype(np.uint8))
        ss.append(prob.ravel()[idx].astype(np.float32))
        ws.append(np.concatenate([np.ones(len(pos)), np.full(len(neg), len(neg_all) / max(1, len(neg)))]).astype(np.float32))
        qual[cid] = dict(dice=dice, prob=prob.astype(np.float16), gt=gt, t1c=v["img"][1], flair=v["img"][3])
        print(f"{cid}: Dice {dice:.3f}  gt_vox {int(gt.sum()):6d}  lesions {det}/{ng}  FP lesions {lfp}", flush=True)

    y, s, w = np.concatenate(ys), np.concatenate(ss), np.concatenate(ws)
    np.savez_compressed(f"{a.run}/roc_pr.npz", y=y, s=s, w=w)
    # qualitative: 4 cases spanning the Dice range (with lesions)
    ordered = sorted([r for r in rows if r["gt_vox"] > 0], key=lambda r: r["dice"])
    pick = [ordered[int(i)] ["case"] for i in np.linspace(len(ordered) - 1, 0, 4)] if ordered else []
    out = {}
    for cid in pick:
        q = qual[cid]
        z = int(np.argmax(q["gt"].sum((1, 2))))
        out[cid] = dict(z=z, dice=q["dice"], t1c=q["t1c"][z], gt=q["gt"][z], prob=q["prob"][z])
    np.savez_compressed(f"{a.run}/qual.npz", **{f"{c}__{k}": v for c, d in out.items() for k, v in d.items()})

    from sklearn.metrics import average_precision_score, roc_auc_score
    d = np.array([r["dice"] for r in rows if r["gt_vox"] > 0])
    has = [r for r in rows if r["gt_vox"] > 0]
    res = dict(
        n_test=len(rows), threshold=a.thr,
        dice_mean=float(d.mean()), dice_median=float(np.median(d)), dice_std=float(d.std()),
        dice_pooled=float(2 * tot["tp"] / (2 * tot["tp"] + tot["fp"] + tot["fn"])),
        voxel_accuracy=tot["correct"] / tot["vox"],
        voxel_sensitivity=tot["tp"] / max(1, tot["tp"] + tot["fn"]),
        voxel_precision=tot["tp"] / max(1, tot["tp"] + tot["fp"]),
        lesion_sensitivity=tot["ld"] / max(1, tot["lg"]),
        lesion_precision=(tot["lp"] - tot["lfp"]) / max(1, tot["lp"]),
        fp_lesions_per_patient=tot["lfp"] / len(rows),
        patient_detection_rate=float(np.mean([r["detected"] > 0 for r in has])),
        roc_auc=float(roc_auc_score(y, s, sample_weight=w)),
        avg_precision=float(average_precision_score(y, s, sample_weight=w)),
        n_gt_lesions=tot["lg"], n_detected=tot["ld"],
    )
    json.dump(res, open(f"{a.run}/results.json", "w"), indent=1)
    json.dump(rows, open(f"{a.run}/test_per_patient.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
