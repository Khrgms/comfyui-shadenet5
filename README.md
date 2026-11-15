# comfyui-shadenet5

ComfyUI custom nodes for [ShadeNet-5](https://huggingface.co/singam96/ShadeNet-5-35M)
(single-image inverse rendering / intrinsic decomposition, Apache-2.0).

One RGB photo in → **albedo + relative depth + surface normal** out, via the
model's quantised ONNX export (`onnx/shadenet5_mix_refined.onnx`, 35 MB). The
preprocessing, output convention and the optional six-pass test-time averaging
are ported verbatim from the repository's `inference/` scripts.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and install deps into ComfyUI's
Python environment:

```
python -m pip install -r requirements.txt
```

For GPU inference install `onnxruntime-gpu` instead of `onnxruntime`. `torch`
and `numpy` are already provided by ComfyUI.

The model file downloads automatically from HuggingFace into
`ComfyUI/models/shadenet5/` on first use (or drop the `.onnx` there manually
for offline machines).

## Usage

1. **ShadeNet-5 Loader** — picks the ONNX file and execution provider.
2. **ShadeNet-5 Inverse Render** — takes a normal ComfyUI `IMAGE` and outputs
   three `IMAGE`s:
   - `albedo` — base colour (lighting removed)
   - `depth` — relative depth, grayscale (per-image scale, not metric)
   - `normal` — surface normals in camera frame (x right, y **up**, z toward
     the viewer); enable `flip_normal_y` if your downstream tool expects
     OpenGL/DirectX-style y-down green channels.

   `mode`:
   - `refined` — one forward pass + the small refiner (recommended, ~5x faster)
   - `enhanced` — the model card's six-pass recipe (mirror + ±4° rotations,
     per-pixel average with unit-length normals). Better normals, ~6x slower.

Inputs are processed whole: the long side is scaled to 448 and the remainder
of the square is edge-mirrored; the padding is cropped off the maps, so
outputs match the input's aspect ratio.

## Notes

- The ONNX files are fixed at 448×448; larger detail is limited by that.
- Depth is scale-and-shift invariant (relative), so it is not comparable
  across images — handy for relighting, ControlNet-style conditioning,
  geometry hints, etc.
- Cached sessions live in memory per (model, provider, threads) combination.

## Files

- `nodes.py` — ComfyUI node definitions
- `shadenet5_core.py` — letterbox / rotation / TTA helpers ported from the
  model repo (`framing.py`, `rotation.py`, `ensemble.py`)
