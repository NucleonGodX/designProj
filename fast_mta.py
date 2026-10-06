"""Drop-in, mathematically identical replacement for SpatialKQConv3D.forward.

The original runs two depth-wise 3D convs on tiny (wd,wh,ww) windows, which is pathologically slow in cuDNN
(especially backward). Since the window is small, each conv is just a fixed (N x N) linear map per head
(N = wd*wh*ww) whose entries are kernel weights -> two batched matmuls. Same parameters, same outputs.
Call `patch()` before building/running the model; the architecture file itself is left untouched.
"""
import itertools

import torch

import hybrid_swin_architecture as arch


def _index_tables(window, k, device):
    wd, wh, ww = window
    pos = torch.tensor(list(itertools.product(range(wd), range(wh), range(ww))), device=device)   # (N,3)
    c = k // 2
    off = pos[None, :, :] - pos[:, None, :]            # off[p, q] = q - p
    valid = (off.abs() <= c).all(-1)
    ki = ((off + c).clamp(0, k - 1))
    flat = (ki[..., 0] * k + ki[..., 1]) * k + ki[..., 2]   # (N,N) index into flattened k^3 kernel
    return flat, valid


def _matrix(conv, window):
    """conv: depth-wise Conv3d(h,h,k,groups=h) -> (h, N, N) with M[h,p,q] = w[h, q-p+c]."""
    key = (tuple(window), conv.weight.device)
    cache = getattr(conv, "_mat_idx", None)
    if cache is None or cache[0] != key:
        k = conv.kernel_size[0]
        flat, valid = _index_tables(window, k, conv.weight.device)
        conv._mat_idx = (key, flat, valid)
    _, flat, valid = conv._mat_idx
    w = conv.weight.reshape(conv.weight.shape[0], -1)       # (h, k^3)
    return w[:, flat] * valid                                # (h, N, N)


def _forward(self, logits):
    if not self.enabled:
        return logits
    Mk = _matrix(self.k_conv, self.window).to(logits.dtype)
    Mq = _matrix(self.q_conv, self.window).to(logits.dtype)
    x = torch.einsum("hjk,bhnk->bhnj", Mk, logits)           # key-neighbourhood conv
    y = torch.einsum("hij,bhjm->bhim", Mq, x)                # query-neighbourhood conv
    g = self.gate.view(1, -1, 1, 1).to(logits.dtype)
    out = logits + g * (y - logits)
    if self.clamp is not None:
        out = out.clamp(-self.clamp, self.clamp)
    return out


def patch():
    arch.SpatialKQConv3D._orig_forward = arch.SpatialKQConv3D.forward
    arch.SpatialKQConv3D.forward = _forward
