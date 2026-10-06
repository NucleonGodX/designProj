"""
Spatial-MTA Swin UNETR (3D)  --  architecture + stability test-suite
=====================================================================

A pure-PyTorch (no MONAI dependency) 3D Swin UNETR with Spatial Multi-Token Attention (MTA)
and Deep Supervision / Auxiliary Vessel Pseudo-Label Heads.

Proposed Architecture Implementations Included:
  1. Spatial-MTA Swin UNETR: 
     - Stages 1-2: Spatial key-query convolution over window attention logits.
     - Stages 3-4: Head-mixing to combine specialized heads.
  2. Head-group specialization (Vessel Aux Head): 
     - An auxiliary vessel segmentation head branching from high-resolution encoder 
       features (Stage 1) to act as an anti-false-positive mechanism against vessels.
  3. Deep Supervision:
     - Multi-scale deep supervision in the decoder for improved gradient flow, 
       essential for small lesion detection.

Stability features (enabled by default): identity-initialized gates, shifted-window 
masking inside MTA convs, logit clamping to protect fp16/bf16, and depth-scaled GroupNorm.

Run:   python architecture.py                # full stability suite (tiny CPU config)
       python architecture.py --quick        # skips the training-loop test
       python architecture.py --feature_size 24 --window 5 --steps 30 --seeds 3
Needs: torch >= 2.0
"""

import argparse
import math
import sys
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Small helpers
# =============================================================================
def _triple(v):
    if isinstance(v, int):
        return (v, v, v)
    v = tuple(v)
    assert len(v) == 3, "expected an int or a 3-tuple"
    return v


class DropPath(nn.Module):
    """Stochastic depth."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


def window_partition_3d(x, ws):
    """(B, D, H, W, C) -> (B * nW, N, C); D/H/W must be multiples of ws."""
    B, D, H, W, C = x.shape
    wd, wh, ww = ws
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, wd * wh * ww, C)
    return x


def window_reverse_3d(windows, ws, B, D, H, W):
    wd, wh, ww = ws
    C = windows.shape[-1]
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, D, H, W, C)
    return x


def _axis_slices(ws, s):
    if s == 0:
        return [slice(None)]
    return [slice(0, -ws), slice(-ws, -s), slice(-s, None)]


def compute_shift_mask(dims, ws, shift, device):
    """Bool mask (nW, N, N); True = pair is NOT allowed to attend (different shifted region)."""
    D, H, W = dims
    img = torch.zeros(1, D, H, W, 1, device=device)
    cnt = 0
    for sd, sh, sw in product(_axis_slices(ws[0], shift[0]),
                              _axis_slices(ws[1], shift[1]),
                              _axis_slices(ws[2], shift[2])):
        img[:, sd, sh, sw, :] = cnt
        cnt += 1
    mw = window_partition_3d(img, ws).squeeze(-1)          # (nW, N)
    return mw.unsqueeze(1) != mw.unsqueeze(2)               # (nW, N, N) bool


# =============================================================================
# MTA components (the novel part)
# =============================================================================
class SpatialKQConv3D(nn.Module):
    """
    Spatial key-query convolution on window attention logits.

    logits: (B_, heads, N, N) with N = wd*wh*ww.  Both the query axis and the key axis are
    reshaped to (wd, wh, ww) and convolved with a depth-wise (one kernel per head) 3D
    conv:   key-neighbourhood conv  ->  query-neighbourhood conv   (factorised, so cost is
    cq^3 + ck^3 per head rather than cq^3 * ck^3).

        out = logits + gate * (conv(logits) - logits)          gate init 0  => identity
    """
    is_mta = True

    def __init__(self, heads, window, kq=(3, 3), gate_init=0.0, identity_noise=1e-2, clamp=50.0):
        super().__init__()
        self.heads, self.window = heads, _triple(window)
        cq, ck = kq
        assert cq % 2 == 1 and ck % 2 == 1, "kernel sizes must be odd to preserve window size"
        self.q_conv = nn.Conv3d(heads, heads, cq, padding=cq // 2, groups=heads, bias=False)
        self.k_conv = nn.Conv3d(heads, heads, ck, padding=ck // 2, groups=heads, bias=False)
        self.gate = nn.Parameter(torch.full((heads,), float(gate_init)))
        self.clamp = clamp
        self.enabled = True
        self.reset_identity(identity_noise)

    @torch.no_grad()
    def reset_identity(self, noise=1e-2):
        for conv in (self.q_conv, self.k_conv):
            conv.weight.zero_()
            c = conv.kernel_size[0] // 2
            conv.weight[:, :, c, c, c] = 1.0
            if noise > 0:
                conv.weight.add_(noise * torch.randn_like(conv.weight))

    def forward(self, logits):
        if not self.enabled:
            return logits
        B, h, N, _ = logits.shape
        wd, wh, ww = self.window
        # key-neighbourhood conv (conv over the key axes, batch over queries)
        x = logits.reshape(B, h, N, wd, wh, ww).permute(0, 2, 1, 3, 4, 5).reshape(B * N, h, wd, wh, ww)
        x = self.k_conv(x)
        x = x.reshape(B, N, h, wd, wh, ww).permute(0, 2, 1, 3, 4, 5).reshape(B, h, N, N)
        # query-neighbourhood conv (conv over the query axes, batch over keys)
        y = x.reshape(B, h, wd, wh, ww, N).permute(0, 5, 1, 2, 3, 4).reshape(B * N, h, wd, wh, ww)
        y = self.q_conv(y)
        y = y.reshape(B, N, h, wd, wh, ww).permute(0, 2, 3, 4, 5, 1).reshape(B, h, N, N)
        g = self.gate.view(1, h, 1, 1)
        out = logits + g * (y - logits)
        if self.clamp is not None:
            out = out.clamp(-self.clamp, self.clamp)
        return out


class HeadMixing(nn.Module):
    """
    Head-mixing over groups of `group` heads, applied AFTER softmax.
    convex=True  -> weights are relu'd + row-normalised, gate clamped to [0,1], so every
                    mixed attention row is still a probability distribution.
        out = attn + gate * (mix(attn) - attn)                 gate init 0 => identity
    """
    is_mta = True

    def __init__(self, heads, group=2, gate_init=0.0, identity_noise=1e-2, convex=True):
        super().__init__()
        assert heads % group == 0, f"heads ({heads}) must be divisible by head-group size ({group})"
        self.heads, self.group, self.convex = heads, group, convex
        G = heads // group
        w = torch.eye(group).unsqueeze(0).repeat(G, 1, 1) + identity_noise * torch.randn(G, group, group)
        self.weight = nn.Parameter(w)
        self.gate = nn.Parameter(torch.full((heads,), float(gate_init)))
        self.enabled = True

    def _matrix(self):
        if not self.convex:
            return self.weight
        w = F.relu(self.weight)
        return w / w.sum(-1, keepdim=True).clamp_min(1e-6)

    def forward(self, attn):
        if not self.enabled:
            return attn
        B, h, N, M = attn.shape
        G, c = h // self.group, self.group
        a = attn.reshape(B, G, c, N, M)
        mixed = torch.einsum("goi,bginm->bgonm", self._matrix().to(attn.dtype), a).reshape(B, h, N, M)
        if self.convex:
            # straight-through clamp: forward value in [0,1] (rows stay valid distributions),
            # gradient passes unchanged. A plain clamp(0,1) has ZERO gradient at gate==0 in
            # current PyTorch, which would leave a gate initialised at 0 permanently closed.
            g = self.gate + (self.gate.clamp(0.0, 1.0) - self.gate).detach()
        else:
            g = self.gate
        g = g.view(1, h, 1, 1).to(attn.dtype)
        return attn + g * (mixed - attn)


class HeadGroupNorm(nn.Module):
    """Per-head, per-token norm of attention outputs with MTA-style depth scaling (gated)."""
    is_mta = True

    def __init__(self, heads, dim, layer_depth, max_depth, gate_init=0.0):
        super().__init__()
        self.gn = nn.GroupNorm(heads, dim)
        self.scale = nn.Parameter(torch.tensor(float(layer_depth) / float(max_depth)))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.enabled = True

    def forward(self, x):                       # x: (B_, N, C)
        if not self.enabled:
            return x
        B_, N, C = x.shape
        y = self.gn(x.reshape(B_ * N, C)).reshape(B_, N, C) * self.scale
        return x + self.gate * (y - x)


# =============================================================================
# Swin 3D backbone
# =============================================================================
class WindowAttention3D(nn.Module):
    def __init__(self, dim, heads, window, mta_mode="none", kq=(3, 3), head_group=2,
                 layer_depth=1, max_depth=1, gate_init=0.0, logit_clamp=50.0, use_gn=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.window = dim, heads, _triple(window)
        self.scale = (dim // heads) ** -0.5
        wd, wh, ww = self.window

        self.rel_bias_table = nn.Parameter(torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), heads))
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij"))
        cf = coords.flatten(1)
        rel = (cf[:, :, None] - cf[:, None, :]).permute(1, 2, 0).contiguous()
        rel[..., 0] += wd - 1
        rel[..., 1] += wh - 1
        rel[..., 2] += ww - 1
        rel[..., 0] *= (2 * wh - 1) * (2 * ww - 1)
        rel[..., 1] *= (2 * ww - 1)
        self.register_buffer("rel_index", rel.sum(-1), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mta_mode = mta_mode
        self.kq = self.hm = self.gn = None
        if mta_mode == "kq":
            self.kq = SpatialKQConv3D(heads, self.window, kq, gate_init, clamp=logit_clamp)
            if use_gn:
                self.gn = HeadGroupNorm(heads, dim, layer_depth, max_depth, gate_init)
        elif mta_mode == "hm":
            self.hm = HeadMixing(heads, head_group, gate_init)
        elif mta_mode != "none":
            raise ValueError(f"unknown mta_mode {mta_mode}")

        self.store_attn = False
        self.last_attn = None

    @staticmethod
    def _apply_mask(attn, mask, fill):
        nW = mask.shape[0]
        B_, h, N, M = attn.shape
        a = attn.reshape(B_ // nW, nW, h, N, M).masked_fill(mask[None, :, None], fill)
        return a.reshape(B_, h, N, M)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)                       # (B_, h, N, N)
        bias = self.rel_bias_table[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if self.kq is not None:
            if mask is not None:
                attn = self._apply_mask(attn, mask, 0.0)     # nothing leaks INTO the conv
            attn = self.kq(attn)
        if mask is not None:
            attn = self._apply_mask(attn, mask, -100.0)       # nothing leaks OUT of the conv
        attn = attn.softmax(dim=-1)
        if self.hm is not None:
            attn = self.hm(attn)
        if self.store_attn:
            self.last_attn = attn.detach()
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.gn is not None:
            out = self.gn(out)
        return self.proj_drop(self.proj(out))


class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, window, shift, mlp_ratio=4.0, drop_path=0.0, **attn_kwargs):
        super().__init__()
        self.window, self.shift = _triple(window), _triple(shift)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, heads, self.window, **attn_kwargs)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self._mask_cache = {}

    def _mask(self, dims, shift, device):
        key = (dims, shift, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = compute_shift_mask(dims, self.window, shift, device)
        return self._mask_cache[key]

    def forward(self, x):                      # x: (B, D, H, W, C) channel-last
        B, D, H, W, C = x.shape
        ws = self.window
        shortcut = x
        x = self.norm1(x)
        pd, ph, pw = (-D) % ws[0], (-H) % ws[1], (-W) % ws[2]
        x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))
        Dp, Hp, Wp = D + pd, H + ph, W + pw
        # no point shifting along an axis that has a single window
        shift = tuple(s if dim > w else 0 for s, dim, w in zip(self.shift, (Dp, Hp, Wp), ws))
        if any(shift):
            x = torch.roll(x, shifts=tuple(-s for s in shift), dims=(1, 2, 3))
            mask = self._mask((Dp, Hp, Wp), shift, x.device)
        else:
            mask = None
        xw = window_partition_3d(x, ws)
        aw = self.attn(xw, mask)
        x = window_reverse_3d(aw, ws, B, Dp, Hp, Wp)
        if any(shift):
            x = torch.roll(x, shifts=shift, dims=(1, 2, 3))
        x = x[:, :D, :H, :W, :].contiguous()
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x):                      # (B, D, H, W, C)
        _, D, H, W, _ = x.shape
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))
        parts = [x[:, i::2, j::2, k::2, :] for i, j, k in product((0, 1), repeat=3)]
        return self.reduction(self.norm(torch.cat(parts, dim=-1)))


class SwinEncoder3D(nn.Module):
    def __init__(self, in_channels, feature_size, depths, num_heads, window, mta, kq_stages, hm_stages,
                 kq=(3, 3), head_group=2, drop_path_rate=0.0, gate_init=0.0, logit_clamp=50.0, use_gn=True):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_channels, feature_size, kernel_size=2, stride=2)
        total = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total).tolist()
        self.stages, self.merges = nn.ModuleList(), nn.ModuleList()
        layer = 0
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            dim = feature_size * 2 ** i
            mode = "kq" if (mta and i in kq_stages) else "hm" if (mta and i in hm_stages) else "none"
            blocks = []
            for j in range(depth):
                layer += 1
                blocks.append(SwinBlock3D(
                    dim, heads, window, shift=(0, 0, 0) if j % 2 == 0 else tuple(w // 2 for w in _triple(window)),
                    drop_path=dpr[layer - 1], mta_mode=mode, kq=kq, head_group=head_group,
                    layer_depth=layer, max_depth=total, gate_init=gate_init,
                    logit_clamp=logit_clamp, use_gn=use_gn))
            self.stages.append(nn.Sequential(*blocks))
            self.merges.append(PatchMerging3D(dim))

    def forward(self, x):
        x = self.patch_embed(x)
        feats = [x]                                         # /2 , C = fs
        x = x.permute(0, 2, 3, 4, 1)
        for stage, merge in zip(self.stages, self.merges):
            x = merge(stage(x))
            feats.append(x.permute(0, 4, 1, 2, 3).contiguous())
        return feats                                        # /2, /4, /8, /16, /32


# =============================================================================
# UNETR-style CNN decoder
# =============================================================================
class ResConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_c, affine=True)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_c, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip = None
        if in_c != out_c:
            self.skip = nn.Sequential(nn.Conv3d(in_c, out_c, 1, bias=False), nn.InstanceNorm3d(out_c, affine=True))

    def forward(self, x):
        r = x if self.skip is None else self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, skip_c):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2)
        self.block = ResConvBlock(out_c + skip_c, out_c)

    def forward(self, x, skip):
        return self.block(torch.cat([self.up(x), skip], dim=1))


# =============================================================================
# Full model with Deep Supervision and Vessel Aux Head
# =============================================================================
class SpatialMTASwinUNETR(nn.Module):
    """
    in_channels=4 (T1, T1c, T2, FLAIR) by default; use in_channels=1 for T1c-only BM pipelines.
    Spatial size is padded internally to a multiple of 32 (min 64) and cropped back.
    mta=False  => plain Swin UNETR with identical parameter names (MTA params simply absent).
    deep_supervision=True => Returns a list of predictions [final, ds_dec1, ds_dec2, ds_dec3] during training.
    use_vessel_aux=True => Adds an auxiliary output for vessel prediction from Stage 1. 
                           Returned as a dict: {"pred": logits, "vessel": aux_logits, "ds": [deep_sup_logits]}
    """

    def __init__(self, in_channels=4, out_channels=1, feature_size=48, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mta=True, kq_stages=(0, 1), hm_stages=(2, 3),
                 kq_kernel=(3, 3), head_group=2, drop_path_rate=0.0, gate_init=0.0,
                 logit_clamp=50.0, use_gn=True, deep_supervision=True, use_vessel_aux=True):
        super().__init__()
        fs = feature_size
        self.mta = mta
        self.deep_supervision = deep_supervision
        self.use_vessel_aux = use_vessel_aux

        self.encoder = SwinEncoder3D(in_channels, fs, depths, num_heads, window_size, mta, kq_stages,
                                     hm_stages, kq_kernel, head_group, drop_path_rate, gate_init,
                                     logit_clamp, use_gn)
        self.enc1 = ResConvBlock(in_channels, fs)
        self.enc2 = ResConvBlock(fs, fs)
        self.enc3 = ResConvBlock(2 * fs, 2 * fs)
        self.enc4 = ResConvBlock(4 * fs, 4 * fs)
        self.enc10 = ResConvBlock(16 * fs, 16 * fs)
        
        self.dec4 = UpBlock(16 * fs, 8 * fs, 8 * fs)
        self.dec3 = UpBlock(8 * fs, 4 * fs, 4 * fs)
        self.dec2 = UpBlock(4 * fs, 2 * fs, 2 * fs)
        self.dec1 = UpBlock(2 * fs, fs, fs)
        self.dec0 = UpBlock(fs, fs, fs)
        self.head = nn.Conv3d(fs, out_channels, kernel_size=1)

        # Proposed Architecture 2: Head-group specialization with vessel pseudo-labels.
        # We attach this auxiliary head to f1 (Stage 1 output, high resolution) 
        # to classify vessels early in the network and separate them from true lesions.
        if self.use_vessel_aux:
            self.vessel_head = nn.Sequential(
                nn.Conv3d(2 * fs, fs, kernel_size=3, padding=1),
                nn.InstanceNorm3d(fs),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(fs, 1, kernel_size=1)
            )

        # Deep Supervision Heads
        if self.deep_supervision:
            self.ds3 = nn.Conv3d(4 * fs, out_channels, kernel_size=1)
            self.ds2 = nn.Conv3d(2 * fs, out_channels, kernel_size=1)
            self.ds1 = nn.Conv3d(fs, out_channels, kernel_size=1)

    @staticmethod
    def _pad_target(n):
        return max(64, int(math.ceil(n / 32) * 32))

    def forward(self, x):
        D, H, W = x.shape[2:]
        tgt = [self._pad_target(n) for n in (D, H, W)]
        pads = (0, tgt[2] - W, 0, tgt[1] - H, 0, tgt[0] - D)
        if any(pads):
            x = F.pad(x, pads)
            
        f0, f1, f2, f3, f4 = self.encoder(x)
        
        # Auxiliary Vessel Prediction (from stage 1 output f1)
        vessel_logits = None
        if self.use_vessel_aux:
            # f1 is at /2 resolution, upsample to original resolution for loss calculation
            v_log = self.vessel_head(f1)
            vessel_logits = F.interpolate(v_log, size=(D, H, W), mode="trilinear", align_corners=False)

        e1 = self.enc1(x)
        e2 = self.enc2(f0)
        e3 = self.enc3(f1)
        e4 = self.enc4(f2)
        b = self.enc10(f4)
        
        d4 = self.dec4(b, f3)
        d3 = self.dec3(d4, e4)
        d2 = self.dec2(d3, e3)
        d1 = self.dec1(d2, e2)
        d0 = self.dec0(d1, e1)
        
        final_logits = self.head(d0)[:, :, :D, :H, :W]

        if not self.training:
            if self.use_vessel_aux:
                return {"pred": final_logits, "vessel": vessel_logits}
            return final_logits

        # In training, return deep supervision outputs if enabled
        out_dict = {"pred": final_logits}
        if self.use_vessel_aux:
            out_dict["vessel"] = vessel_logits
            
        if self.deep_supervision:
            ds3_out = F.interpolate(self.ds3(d3), size=(D, H, W), mode="trilinear", align_corners=False)
            ds2_out = F.interpolate(self.ds2(d2), size=(D, H, W), mode="trilinear", align_corners=False)
            ds1_out = F.interpolate(self.ds1(d1), size=(D, H, W), mode="trilinear", align_corners=False)
            out_dict["ds"] = [ds1_out, ds2_out, ds3_out]

        return out_dict

    # ---- MTA utilities ------------------------------------------------------------
    def mta_modules(self):
        return [(n, m) for n, m in self.named_modules() if getattr(m, "is_mta", False)]

    def set_mta_enabled(self, flag):
        for _, m in self.mta_modules():
            m.enabled = bool(flag)

    @torch.no_grad()
    def set_mta_gates(self, value):
        for _, m in self.mta_modules():
            m.gate.fill_(value)

    @torch.no_grad()
    def mta_gate_summary(self):
        out = {}
        for name, m in self.mta_modules():
            out.setdefault(type(m).__name__, []).append(m.gate.abs().mean().item())
        return {k: sum(v) / len(v) for k, v in out.items()}

    def attention_modules(self):
        return [m for m in self.modules() if isinstance(m, WindowAttention3D)]


# =============================================================================
# Diagnostics usable on real data too
# =============================================================================
@torch.no_grad()
def attention_diagnostics(attn):
    """attn (B_, heads, N, N) -> (mean entropy, mean pairwise cosine similarity between heads)."""
    ent = -(attn.clamp_min(1e-12).log() * attn).sum(-1).mean().item()
    B_, h, N, M = attn.shape
    if h < 2:
        return ent, float("nan")
    flat = F.normalize(attn.reshape(B_, h, N * M), dim=-1)
    sim = flat @ flat.transpose(1, 2)
    off = (sim.sum() - sim.diagonal(dim1=1, dim2=2).sum()) / (B_ * h * (h - 1))
    return ent, off.item()


def estimate_kq_logit_memory_gb(volume, window, heads_stage1, patch=2, bytes_per=4, copies=1):
    """Size of ONE window-logit tensor of stage 1 (the dominant cost of spatial KQ conv)."""
    tokens = [math.ceil(v / patch) for v in volume]
    ws = _triple(window)
    padded = [math.ceil(t / w) * w for t, w in zip(tokens, ws)]
    n_win = math.prod(p // w for p, w in zip(padded, ws))
    N = math.prod(ws)
    return n_win * heads_stage1 * N * N * bytes_per * copies / 1024 ** 3


def dice_bce_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    dice = 1 - (2 * inter + eps) / (prob.sum() + target.sum() + eps)
    return dice + F.binary_cross_entropy_with_logits(logits, target)


# =============================================================================
# Test harness
# =============================================================================
RESULTS = []

def banner(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")

def check(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    sym = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[status]
    RESULTS.append((status, name))
    print(f"  {sym} [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    return ok

def build(args, mta=True, **over):
    cfg = dict(in_channels=4, out_channels=1, feature_size=args.feature_size,
               num_heads=tuple(args.heads), window_size=args.window, mta=mta)
    cfg.update(over)
    return SpatialMTASwinUNETR(**cfg)

def synthetic_bm_batch(B=1, size=64, in_ch=4, gen=None):
    """Small bright 'metastases' (labelled) + thin bright 'vessels' (NOT labelled) on noise."""
    g = gen or torch.Generator().manual_seed(0)
    vol = 0.3 * torch.randn(B, in_ch, size, size, size, generator=g)
    lab = torch.zeros(B, 1, size, size, size)
    vessel_lab = torch.zeros(B, 1, size, size, size)
    
    ax = torch.arange(size).float()
    zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")
    
    for b in range(B):
        # Add metastases
        for _ in range(int(torch.randint(1, 5, (1,), generator=g))):
            c = torch.randint(6, size - 6, (3,), generator=g).float()
            r = float(torch.randint(1, 4, (1,), generator=g))
            m = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r
            lab[b, 0][m] = 1.0
            vol[b, 1][m] += 2.0
            vol[b, 0][m] += 0.3
            
        # Add vessels
        a = torch.randint(4, size - 4, (3,), generator=g).float()
        e = torch.randint(4, size - 4, (3,), generator=g).float()
        for t in torch.linspace(0, 1, 2 * size):
            p = (a + t * (e - a)).round().long().clamp(0, size - 1)
            vol[b, 1, p[0], p[1], p[2]] += 1.5
            vessel_lab[b, 0, p[0], p[1], p[2]] = 1.0
            
    return vol, lab, vessel_lab

# ---------------------------------------------------------------------------------
def test_components(args):
    banner("1. Component correctness (windows, masks, spatial key-query conv)")
    x = torch.randn(2, 8, 12, 8, 5)
    ws = (4, 4, 4)
    xr = window_reverse_3d(window_partition_3d(x, ws), ws, 2, 8, 12, 8)
    check("window partition/reverse round-trip", torch.equal(x, xr))
    
    # spatial KQ support
    kq = SpatialKQConv3D(heads=1, window=(5, 5, 5), kq=(3, 3), gate_init=1.0, clamp=None)
    with torch.no_grad():
        kq.q_conv.weight.fill_(1.0)
        kq.k_conv.weight.fill_(1.0)
    logits = torch.zeros(1, 1, 125, 125)
    c = 2 * 25 + 2 * 5 + 2
    logits[0, 0, c, c] = 1.0
    out = kq(logits)[0, 0]
    nz = out.nonzero()
    nq = out.abs().sum(1).nonzero().numel()
    nk = out.abs().sum(0).nonzero().numel()
    check("spatial KQ support is 27 query-neighbours x 27 key-neighbours",
          nz.shape[0] == 729 and nq == 27 and nk == 27, f"nonzeros={nz.shape[0]}, q={nq}, k={nk}")

def test_shapes_and_outputs(args):
    banner("2. Shape tests & Deep Supervision / Aux outputs")
    model = build(args).train()
    tot = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {tot:,}  ({tot * 4 / 1024 ** 2:.2f} MB fp32)")
    
    shape = (64, 64, 64)
    x = torch.randn(1, 4, *shape)
    with torch.no_grad():
        out = model(x)
        
    check("Outputs dictionary in training mode", isinstance(out, dict))
    check("Main prediction shape correct", out["pred"].shape == (1, 1, *shape))
    check("Deep supervision outputs present", "ds" in out and len(out["ds"]) == 3)
    check("Vessel aux prediction present", "vessel" in out and out["vessel"].shape == (1, 1, *shape))
    
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    check("Outputs dict with pred/vessel in eval mode", isinstance(out_eval, dict) and "vessel" in out_eval)

def test_training_stability(args):
    banner(f"3. Short training run on synthetic metastases + vessel distractor ({args.steps} steps x {args.seeds} seeds)")
    for mta in (False, True):
        tag = "MTA " if mta else "base"
        finals, peaks, firsts = [], [], []
        ok = True
        for s in range(args.seeds):
            torch.manual_seed(s)
            model = build(args, mta=mta).train()
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
            gen = torch.Generator().manual_seed(1000 + s)
            losses, gns = [], []
            for _ in range(args.steps):
                x, y, vessel_y = synthetic_bm_batch(1, 64, gen=gen)
                opt.zero_grad(set_to_none=True)
                out = model(x)
                
                # Main + Deep Supervision + Vessel Loss
                loss = dice_bce_loss(out["pred"], y)
                for ds_out in out["ds"]:
                    loss += 0.5 * dice_bce_loss(ds_out, y)
                loss += 0.5 * dice_bce_loss(out["vessel"], vessel_y)
                
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                opt.step()
                losses.append(loss.item())
                gns.append(gn)
            ok &= all(math.isfinite(v) for v in losses + gns)
            head = sum(losses[:3]) / 3
            tail = sum(losses[-3:]) / 3
            firsts.append(head)
            finals.append(tail)
            med = sorted(gns)[len(gns) // 2]
            peaks.append(max(gns) / max(med, 1e-12))
            
        mean = sum(finals) / len(finals)
        std = (sum((f - mean) ** 2 for f in finals) / len(finals)) ** 0.5
        check(f"[{tag}] all losses / grad-norms finite", ok)
        check(f"[{tag}] loss decreases", all(f < h for f, h in zip(finals, firsts)), f"final {mean:.3f}")

def run_all(args):
    banner("Spatial-MTA Swin UNETR -- Hybrid Architecture Validation")
    torch.set_grad_enabled(True)
    test_components(args)
    test_shapes_and_outputs(args)
    if not args.quick:
        test_training_stability(args)
    
    banner("SUMMARY")
    n_pass = sum(s == "PASS" for s, _ in RESULTS)
    print(f"  passed {n_pass}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_size", type=int, default=12)
    ap.add_argument("--heads", type=int, nargs=4, default=[1, 2, 4, 8])
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--quick", action="store_true", help="skip the training-loop test")
    sys.exit(run_all(ap.parse_args()))
