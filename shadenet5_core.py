"""Core helpers for the ShadeNet-5 ComfyUI node.

Adapted from the model repository's inference modules (ensemble.py, framing.py,
rotation.py of singam96/ShadeNet-5-35M) so the node renders exactly what the
model card reports.

The model takes a [1, 3, 448, 448] photo in [-1, 1] and returns the 7 maps
[1, 7, 448, 448] in [-1, 1]: albedo 0:3, depth 3, normal 4:7 (normal x right,
y UP, z toward the viewer).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

NORMAL_X = 4  # first channel of the normal
SIZE = 448    # the ONNX files are fixed at 448x448
ROT_DEG = 4.0


# ---------------------------------------------------------------- framing ---

def letterbox_tensor(x: torch.Tensor, size: int = SIZE):
    """Scale the photo so its LONG side is `size` and fill the rest of the
    square by mirroring the photo's own edge (numpy 'symmetric' padding, which
    is safe even when the padding is longer than the photo itself).

    x: [1, 3, H, W] in [0, 1]. Returns (square [1, 3, S, S], box) where box is
    (left, top, w, h) of the real photo inside the square.
    """
    _, _, h, w = x.shape
    scale = size / float(max(w, h))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = F.interpolate(x.float(), size=(nh, nw), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0.0, 1.0)
    arr = small[0].permute(1, 2, 0).numpy()  # HWC in [0, 1]
    left, top = (size - nw) // 2, (size - nh) // 2
    pad = ((top, size - nh - top), (left, size - nw - left), (0, 0))
    out = np.pad(arr, pad, mode="symmetric")
    return torch.from_numpy(out).permute(2, 0, 1)[None].to(x.dtype), (left, top, nw, nh)


def crop_to_box(maps: torch.Tensor, box) -> torch.Tensor:
    """Cut [B, C, S, S] maps down to the photo's own pixels."""
    left, top, w, h = box
    return maps[:, :, top:top + h, left:left + w]


# -------------------------------------------------------------- rotation ---

def _theta(angle_deg: float, device, dtype) -> torch.Tensor:
    """affine_grid matrix [1, 2, 3]: output pixel -> input sample position, so
    the picture content turns counter-clockwise by `angle_deg` (image
    coordinates: x right, y down)."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[[c, -s, 0.0], [s, c, 0.0]]], device=device,
                        dtype=dtype)


def rotate_image(x: torch.Tensor, angle_deg: float, mode: str = "bicubic",
                 padding: str = "reflection") -> torch.Tensor:
    """Rotate [B, C, H, W] counter-clockwise about the centre."""
    b = x.shape[0]
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = x.float()
        theta = _theta(angle_deg, x.device, torch.float32).expand(b, -1, -1)
        grid = F.affine_grid(theta, list(xf.shape), align_corners=False)
        out = F.grid_sample(xf, grid, mode=mode, padding_mode=padding,
                            align_corners=False)
    return out.to(x.dtype)


def _reorient_normals(maps: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Apply R(angle) to the normal's (nx, ny); nz is unchanged."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    out = maps.clone()
    nx, ny = maps[:, 4], maps[:, 5]
    out[:, 4] = c * nx - s * ny
    out[:, 5] = s * nx + c * ny
    return out


def unrotate_maps(maps: torch.Tensor, angle_deg: float):
    """Maps predicted on a photo rotated by `angle_deg`, brought back to the
    original frame (rotate by -angle, normals reoriented by R(-angle)).

    Returns (maps, valid): `valid` [B, 1, H, W] marks original-frame pixels
    the rotated canvas actually covered."""
    ones = torch.ones_like(maps[:, :1])
    r = rotate_image(maps, -angle_deg, "bilinear", "zeros")
    valid = rotate_image(ones, -angle_deg, "bilinear", "zeros") > 0.999
    return _reorient_normals(r, -angle_deg), valid


# -------------------------------------------------------------- ensemble ---

def unit_normals(maps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out = maps.clone()
    n = out[:, NORMAL_X:NORMAL_X + 3]
    out[:, NORMAL_X:NORMAL_X + 3] = n / n.norm(dim=1, keepdim=True).clamp_min(eps)
    return out


@torch.no_grad()
def tta_average(refine_fn, x, flip=False, rot_deg=0.0, progress=None):
    """Per-pixel mean over the mirror (optional) x rotations (0, -rot, +rot).

    Each pass: mirror -> rotate the photo (reflected border) -> predict ->
    rotate the maps back, reorienting the normals -> un-mirror. A pass only
    counts at the pixels its rotated canvas covers.
    """
    import itertools
    angles = (0.0, -float(rot_deg), float(rot_deg)) if rot_deg else (0.0,)
    flips = (False, True) if flip else (False,)
    passes = list(itertools.product(flips, angles))
    num = den = None
    for f, a in passes:
        c = x.flip(-1) if f else x
        if a:
            c = rotate_image(c, a, "bicubic", "reflection").clamp(-1.0, 1.0)
        p = refine_fn(c).float()
        valid = torch.ones_like(p[:, :1])
        if a:
            p, v = unrotate_maps(p, a)
            valid = v.to(p.dtype)
        if f:
            p = p.flip(-1).clone()
            p[:, NORMAL_X] = -p[:, NORMAL_X]
            valid = valid.flip(-1)
        num = p * valid if num is None else num + p * valid
        den = valid if den is None else den + valid
        if progress is not None:
            progress(1)
    return num / den.clamp_min(1.0)
