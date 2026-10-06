"""Train Spatial-MTA Swin UNETR (arch/hybrid_swin_architecture.py, unmodified) on Pretreat-MetsToBrain-Masks.

Target : lesion = enhancing tumour + necrosis (BraTS labels 3 and 1).
Split  : 85% train+val / 15% test, patient-level, seed fixed. 10% of the 85% is held out as an internal
         validation set for checkpoint selection (the 15% test set is never used for selection).
Aux    : vessel pseudo-labels (Frangi, see prep.py) for the model's vessel head; deep supervision on.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "arch"))
from hybrid_swin_architecture import SpatialMTASwinUNETR, dice_bce_loss  # noqa: E402
import fast_mta  # noqa: E402  (exact, faster SpatialKQConv3D.forward; verified identical in fp64)

fast_mta.patch()

PROC = r"C:\designProj\data\proc"
dev = "cuda"


# ----------------------------------------------------------------------------- data
def make_split(seed=42, test_frac=0.15, val_frac=0.10):
    cases = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{PROC}/*.npz"))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(cases))
    n_test = int(round(test_frac * len(cases)))
    test = [cases[i] for i in perm[:n_test]]
    rest = [cases[i] for i in perm[n_test:]]
    n_val = int(round(val_frac * len(rest)))
    return {"train": rest[n_val:], "val": rest[:n_val], "test": test}


def load_case(cid):
    d = np.load(f"{PROC}/{cid}.npz")
    return {"img": d["img"], "tc": d["tc"], "vessel": d["vessel"]}


def pad_to(a, size):
    pads = [(0, 0)] * (a.ndim - 3) + [(0, max(0, size - s)) for s in a.shape[-3:]]
    return np.pad(a, pads) if any(p[1] for p in pads) else a


class PatchSampler:
    def __init__(self, cases, patch=96, p_fg=0.7, seed=0):
        self.c, self.patch, self.p_fg = cases, patch, p_fg
        self.rng = np.random.RandomState(seed)
        self.fg = {k: np.argwhere(v["tc"] > 0) for k, v in cases.items()}
        self.keys = [k for k in cases]

    def sample(self, augment=True):
        k = self.keys[self.rng.randint(len(self.keys))]
        v, P = self.c[k], self.patch
        D, H, W = v["tc"].shape
        if len(self.fg[k]) and self.rng.rand() < self.p_fg:
            ctr = self.fg[k][self.rng.randint(len(self.fg[k]))] + self.rng.randint(-P // 3, P // 3 + 1, 3)
        else:
            ctr = np.array([self.rng.randint(D), self.rng.randint(H), self.rng.randint(W)])
        lo = [int(np.clip(c - P // 2, 0, max(0, s - P))) for c, s in zip(ctr, (D, H, W))]
        sl = tuple(slice(l, l + P) for l in lo)
        img, tc, ves = v["img"][(slice(None),) + sl], v["tc"][sl], v["vessel"][sl]
        img, tc, ves = pad_to(img, P), pad_to(tc, P), pad_to(ves, P)
        img = img.astype(np.float32)
        if augment:
            for ax in range(3):
                if self.rng.rand() < 0.5:
                    img, tc, ves = (np.flip(a, ax + (1 if a.ndim == 4 else 0)) for a in (img, tc, ves))
            img = img * self.rng.uniform(0.9, 1.1, (4, 1, 1, 1)) + self.rng.uniform(-0.1, 0.1, (4, 1, 1, 1))
        return (np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(tc, dtype=np.float32),
                np.ascontiguousarray(ves, dtype=np.float32))

    def batch(self, n, augment=True):
        s = [self.sample(augment) for _ in range(n)]
        x, y, v = (torch.from_numpy(np.stack(a)) for a in zip(*s))
        return x.to(dev), y[:, None].to(dev), v[:, None].to(dev)


# ----------------------------------------------------------------------------- model / loss
def build_model(a):
    return SpatialMTASwinUNETR(in_channels=4, out_channels=1, feature_size=a.feature_size,
                               num_heads=tuple(a.heads), window_size=a.window, mta=True,
                               deep_supervision=True, use_vessel_aux=True, drop_path_rate=0.1).to(dev)


def total_loss(out, y, v, w_ds=0.5, w_ves=0.3):
    main = dice_bce_loss(out["pred"].float(), y)
    loss = main + w_ds * sum(dice_bce_loss(o.float(), y) for o in out["ds"]) + w_ves * dice_bce_loss(out["vessel"].float(), v)
    return loss, main


@torch.no_grad()
def sliding_window(model, img, patch=96, overlap=0.5, bs=2):
    """img (4,D,H,W) float32 numpy -> prob (D,H,W) numpy."""
    model.eval()
    x = torch.from_numpy(img.astype(np.float32))
    _, D, H, W = x.shape
    P = patch
    xp = F.pad(x, (0, max(0, P - W), 0, max(0, P - H), 0, max(0, P - D)))
    _, Dp, Hp, Wp = xp.shape
    step = max(1, int(P * (1 - overlap)))
    starts = lambda n: sorted(set(list(range(0, n - P + 1, step)) + [n - P]))
    coords = [(a, b, c) for a in starts(Dp) for b in starts(Hp) for c in starts(Wp)]
    acc = torch.zeros(Dp, Hp, Wp, device=dev)
    cnt = torch.zeros_like(acc)
    for i in range(0, len(coords), bs):
        cs = coords[i:i + bs]
        xb = torch.stack([xp[:, a:a + P, b:b + P, c:c + P] for a, b, c in cs]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(xb)["pred"]
        o = torch.sigmoid(o.float())[:, 0]
        for (a, b, c), p in zip(cs, o):
            acc[a:a + P, b:b + P, c:c + P] += p
            cnt[a:a + P, b:b + P, c:c + P] += 1
    return (acc / cnt.clamp_min(1))[:D, :H, :W].cpu().numpy()


def dice_np(p, g, eps=1e-6):
    return float((2 * (p & g).sum() + eps) / (p.sum() + g.sum() + eps))


@torch.no_grad()
def evaluate_dice(model, cases, thr=0.5):
    ds = []
    for k, v in cases.items():
        pr = sliding_window(model, v["img"]) > thr
        gt = v["tc"] > 0
        if gt.sum() == 0:
            continue
        ds.append(dice_np(pr, gt))
    return float(np.mean(ds)) if ds else 0.0


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iters", type=int, default=100, help="iterations per epoch")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--feature_size", type=int, default=24)
    ap.add_argument("--heads", type=int, nargs=4, default=[2, 4, 8, 16])
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--out", default=r"C:\designProj\runs\main")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    split = make_split()
    json.dump(split, open(f"{a.out}/split.json", "w"), indent=1)
    print({k: len(v) for k, v in split.items()}, flush=True)
    if a.smoke:
        split = {k: v[:3] for k, v in split.items()}
        a.epochs, a.iters, a.val_every = 2, 5, 1

    t0 = time.time()
    tr = {k: load_case(k) for k in split["train"]}
    va = {k: load_case(k) for k in split["val"]}
    print(f"loaded train/val in {time.time()-t0:.0f}s", flush=True)
    sampler, vsampler = PatchSampler(tr, a.patch, seed=1), PatchSampler(va, a.patch, seed=2)
    val_fixed = [vsampler.batch(a.bs, augment=False) for _ in range(20)]   # fixed val patches for val loss

    model = build_model(a)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params: {n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    total_steps = a.epochs * a.iters
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / total_steps)) + 1e-3)

    hist = {"epoch": [], "train_loss": [], "train_main": [], "val_loss": [], "val_main": [], "lr": [],
            "val_dice_epoch": [], "val_dice": [], "time": []}
    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        tl, tm = [], []
        for _ in range(a.iters):
            x, y, v = sampler.batch(a.bs)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            loss, main = total_loss(out, y, v)
            if not torch.isfinite(loss):
                print("non-finite loss, skipping step", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
            tm.append(main.item())
        model.eval()
        vl, vm = [], []
        with torch.no_grad():
            for x, y, v in val_fixed:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                # eval-mode output has no ds; recompute main+vessel only
                l_main = dice_bce_loss(out["pred"].float(), y)
                l_v = dice_bce_loss(out["vessel"].float(), v)
                vm.append(l_main.item())
                vl.append((l_main + 0.3 * l_v).item())
        hist["epoch"].append(ep)
        hist["train_loss"].append(float(np.mean(tl)))
        hist["train_main"].append(float(np.mean(tm)))
        hist["val_loss"].append(float(np.mean(vl)))
        hist["val_main"].append(float(np.mean(vm)))
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["time"].append(time.time() - t0)
        msg = f"ep {ep:3d}/{a.epochs} train {hist['train_loss'][-1]:.4f} val {hist['val_loss'][-1]:.4f}"
        if ep % a.val_every == 0 or ep == a.epochs:
            d = evaluate_dice(model, va)
            hist["val_dice_epoch"].append(ep)
            hist["val_dice"].append(d)
            msg += f" | val Dice {d:.4f}"
            if d > best:
                best = d
                torch.save(model.state_dict(), f"{a.out}/best.pt")
                msg += " *"
        print(msg + f" | {hist['time'][-1]/60:.1f} min", flush=True)
        json.dump(hist, open(f"{a.out}/history.json", "w"))
    torch.save(model.state_dict(), f"{a.out}/last.pt")
    print(f"done. best val Dice {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
