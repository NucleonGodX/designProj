"""Preprocess Pretreat-MetsToBrain-Masks -> data/proc/<case>.npz

img    (4, D, H, W) float16 : T1n, T1c, T2w, FLAIR; z-scored inside brain, 0 outside; cropped to brain bbox
tc     (D, H, W) uint8      : lesion target = enhancing tumour (3) + necrosis (1)   [labels 1 or 3]
edema  (D, H, W) uint8      : peritumoral edema (label 2)   (kept for reference, not trained on)
vessel (D, H, W) uint8      : Frangi-vesselness pseudo-label from T1c (bright tubes, outside lesions)
"""
import glob
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

RAW = r"C:\designProj\data\raw\PKG - Pretreat-MetsToBrain-Masks\Pretreat-MetsToBrain-Masks"
OUT = r"C:\designProj\data\proc"
MODS = ["t1n", "t1c", "t2w", "t2f"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def gauss_kernel(sigma):
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def smooth(v, sigma):
    k = gauss_kernel(sigma)
    r = len(k) // 2
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - ax):2 * (2 - ax) + 2] = [r, r]
        v = F.conv3d(F.pad(v, pad, mode="replicate"), k.view(shape))
    return v


def vesselness(vol, brain, sigmas=(1.0, 1.5, 2.0)):
    """Frangi vesselness for bright tubular structures. vol (D,H,W) tensor on dev."""
    v = vol[None, None]
    best = torch.zeros_like(vol)
    for s in sigmas:
        g = smooth(v, s)[0, 0]
        d = lambda t, ax: (torch.roll(t, -1, ax) - torch.roll(t, 1, ax)) / 2
        gx, gy, gz = d(g, 0), d(g, 1), d(g, 2)
        H = torch.stack([d(gx, 0), d(gx, 1), d(gx, 2), d(gy, 1), d(gy, 2), d(gz, 2)], -1) * s ** 2
        Hm = torch.zeros(*vol.shape, 3, 3, device=dev)
        Hm[..., 0, 0], Hm[..., 0, 1], Hm[..., 0, 2] = H[..., 0], H[..., 1], H[..., 2]
        Hm[..., 1, 0], Hm[..., 1, 1], Hm[..., 1, 2] = H[..., 1], H[..., 3], H[..., 4]
        Hm[..., 2, 0], Hm[..., 2, 1], Hm[..., 2, 2] = H[..., 2], H[..., 4], H[..., 5]
        flat = Hm.reshape(-1, 3, 3)
        ev = torch.cat([torch.linalg.eigvalsh(flat[i:i + 200000]) for i in range(0, flat.shape[0], 200000)])
        idx = ev.abs().argsort(-1)
        ev = torch.gather(ev, 1, idx)                                          # |l1|<=|l2|<=|l3|
        l1, l2, l3 = ev[:, 0], ev[:, 1], ev[:, 2]
        eps = 1e-6
        Ra = l2.abs() / (l3.abs() + eps)
        Rb = l1.abs() / (torch.sqrt(l2.abs() * l3.abs()) + eps)
        S = torch.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)
        c = 0.5 * S.max().clamp_min(eps)
        out = (1 - torch.exp(-Ra ** 2 / 0.5)) * torch.exp(-Rb ** 2 / 0.5) * (1 - torch.exp(-S ** 2 / (2 * c ** 2)))
        out = torch.where((l2 < 0) & (l3 < 0), out, torch.zeros_like(out)).reshape(vol.shape)
        best = torch.maximum(best, out)
    return best * brain


def process(case_dir):
    cid = os.path.basename(case_dir)
    vols = [np.asarray(nib.load(f"{case_dir}/{cid}-{m}.nii.gz").dataobj, dtype=np.float32) for m in MODS]
    seg = np.asarray(nib.load(f"{case_dir}/{cid}-seg.nii.gz").dataobj).round().astype(np.uint8)
    brain = np.any([v > 0 for v in vols], axis=0)
    nz = np.argwhere(brain)
    lo, hi = nz.min(0), nz.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    brain_c = brain[sl]
    chans = []
    for v in vols:
        v = v[sl]
        vals = v[brain_c]
        z = (v - vals.mean()) / (vals.std() + 1e-6)
        z = np.clip(z, -5, 5) * brain_c
        chans.append(z.astype(np.float32))
    img = np.stack(chans)
    seg = seg[sl]
    tc = ((seg == 1) | (seg == 3)).astype(np.uint8)

    t1c = torch.from_numpy(img[1]).to(dev)
    bt = torch.from_numpy(brain_c.astype(np.float32)).to(dev)
    vs = vesselness(t1c, bt)
    lesion_d = F.max_pool3d(torch.from_numpy((seg > 0).astype(np.float32))[None, None].to(dev), 7, 1, 3)[0, 0] > 0
    vs = vs * (~lesion_d)
    thr = torch.quantile(vs[bt > 0][:: 7], 0.985)
    vessel = ((vs > thr) & (bt > 0) & (t1c > 0.5)).cpu().numpy().astype(np.uint8)

    np.savez_compressed(f"{OUT}/{cid}.npz", img=img.astype(np.float16), tc=tc,
                        edema=(seg == 2).astype(np.uint8), vessel=vessel)
    return cid, img.shape[1:], int(tc.sum()), int(vessel.sum())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cases = sorted(glob.glob(f"{RAW}/BraTS-MET-*"))
    if len(sys.argv) > 1:
        cases = cases[: int(sys.argv[1])]
    print(f"{len(cases)} cases, device={dev}", flush=True)
    for i, c in enumerate(cases):
        if os.path.exists(f"{OUT}/{os.path.basename(c)}.npz"):
            continue
        cid, shp, ntc, nv = process(c)
        print(f"[{i+1}/{len(cases)}] {cid} crop={shp} lesion_vox={ntc} vessel_vox={nv}", flush=True)
