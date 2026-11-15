"""ComfyUI nodes for ShadeNet-5 (singam96/ShadeNet-5-35M).

Single-image inverse rendering: one RGB photo in, albedo + relative depth +
surface normal out, via the quantised ONNX export (no training code needed).
"""
import os

import numpy as np
import torch

import folder_paths
from comfy.utils import ProgressBar

from .shadenet5_core import (NORMAL_X, ROT_DEG, SIZE, crop_to_box,
                             letterbox_tensor, tta_average, unit_normals)

REPO = "singam96/ShadeNet-5-35M"
FILES = {
    "merged (refined, recommended)": "onnx/shadenet5_mix_refined.onnx",
    "mix (single pass, no refiner)": "onnx/shadenet5_mix_int4_int8.onnx",
    "mix2 (smaller, less accurate)": "onnx/shadenet5_mix2_int4_int8.onnx",
}

_MODEL_CACHE = {}


def _model_dir():
    d = os.path.join(folder_paths.models_dir, "shadenet5")
    os.makedirs(d, exist_ok=True)
    return d


def _download(filename: str) -> str:
    # Prefer an already-present local file (offline / manual install).
    local = os.path.join(_model_dir(), filename)
    if os.path.isfile(local):
        return local
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, filename, local_dir=_model_dir())
    return path


class ShadeNet5Loader:
    """Load a ShadeNet-5 ONNX model (downloads from HuggingFace on first use)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(FILES.keys()), {"default": "merged (refined, recommended)"}),
                "provider": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "threads": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                    "tooltip": "ONNX Runtime intra-op threads. 0 = default."}),
            }
        }

    RETURN_TYPES = ("SHADENET5_MODEL",)
    FUNCTION = "load"
    CATEGORY = "ShadeNet5"

    def load(self, model, provider, threads):
        key = (model, provider, threads)
        if key in _MODEL_CACHE:
            return (_MODEL_CACHE[key],)

        import onnxruntime as ort
        path = _download(FILES[model])

        so = ort.SessionOptions()
        if threads > 0:
            so.intra_op_num_threads = threads
        providers = {
            "auto": None,  # let ORT pick (CUDA if onnxruntime-gpu is installed)
            "cpu": ["CPUExecutionProvider"],
            "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        }[provider]
        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in available]
        sess = ort.InferenceSession(path, so, providers=providers)

        wrapper = {"session": sess, "name": model, "path": path}
        _MODEL_CACHE[key] = wrapper
        return (wrapper,)


class ShadeNet5Predict:
    """Run ShadeNet-5 on an IMAGE batch and output albedo / depth / normal."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SHADENET5_MODEL",),
                "images": ("IMAGE",),
                "mode": (["refined", "enhanced"], {"default": "refined",
                        "tooltip": "refined: one forward pass + refiner (~5x faster). "
                                   "enhanced: six passes (mirror + +-4deg rotations) averaged; "
                                   "better normals, ~6x slower."}),
                "flip_normal_y": ("BOOLEAN", {"default": False,
                        "tooltip": "Flip the normal map's vertical axis (the model "
                                   "outputs camera-frame normals with y UP; some "
                                   "tools expect y down)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "depth", "normal")
    FUNCTION = "predict"
    CATEGORY = "ShadeNet5"

    def predict(self, model, images, mode, flip_normal_y):
        sess = model["session"]
        imgs = images.detach().cpu().float()
        n = imgs.shape[0]
        passes_per_image = 6 if mode == "enhanced" else 1
        pb = ProgressBar(n * passes_per_image)

        def run_pass(x):
            out = sess.run(None, {"input_rgb": np.ascontiguousarray(x.numpy())})[0]
            pb.update(1)
            return torch.from_numpy(out)

        albedo, depth, normal = [], [], []
        for i in range(n):
            x01 = imgs[i].permute(2, 0, 1)[None]  # [1, 3, H, W], [0, 1]
            square, box = letterbox_tensor(x01, SIZE)
            x = square * 2.0 - 1.0  # [-1, 1]
            with torch.no_grad():
                if mode == "enhanced":
                    maps = unit_normals(
                        tta_average(run_pass, x, flip=True, rot_deg=ROT_DEG,
                                    progress=pb.update))
                else:
                    maps = run_pass(x).float()
                maps = crop_to_box(maps, box)[0]  # [7, h, w] in [-1, 1]

            alb = (maps[0:3] + 1.0) * 0.5
            dep = (maps[3:4].expand(3, -1, -1) + 1.0) * 0.5
            nor = (maps[4:7] + 1.0) * 0.5
            if flip_normal_y:
                nor[1] = 1.0 - nor[1]
            albedo.append(alb.permute(1, 2, 0))
            depth.append(dep.permute(1, 2, 0))
            normal.append(nor.permute(1, 2, 0))

        return (torch.stack(albedo), torch.stack(depth), torch.stack(normal))


NODE_CLASS_MAPPINGS = {
    "ShadeNet5Loader": ShadeNet5Loader,
    "ShadeNet5Predict": ShadeNet5Predict,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShadeNet5Loader": "ShadeNet-5 Loader",
    "ShadeNet5Predict": "ShadeNet-5 Inverse Render",
}
