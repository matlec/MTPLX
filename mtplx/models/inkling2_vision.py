# SPDX-License-Identifier: Apache-2.0
#
# Vendored from jundot/omlx v0.8.2 (5eeec7757f83bfc99eeb87abe49686d00a0bba1f),
# omlx/patches/inkling2/vision.py and the image processor of processing.py.
# Copyright the oMLX authors, Apache-2.0. Adaptations: one module, no
# tokenizer-side processor (the server expands the image placeholders).
"""Inkling 2 vision tower (checkpoint ``model.visual.*``).

A ViT with axial 2D RoPE on interleaved channel pairs, a 32 x 32 position table
resampled bilinearly to each image, and one class token per image. Each 2 x 2
patch block becomes one image token; the processor sends one 40 px pixel block
per image token in raster order.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionArgs:
    hidden_size: int = 1024
    num_hidden_layers: int = 23
    num_attention_heads: int = 16
    patch_size: int = 20
    spatial_merge_size: int = 2
    posemb_grid_size: int = 32
    projection_hidden_dim: int = 4096
    projector_bias: bool = False
    layer_norm_eps: float = 1e-5
    mlp_ratio: float = 4.0
    num_channels: int = 3
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, params: dict) -> VisionArgs:
        names = cls.__dataclass_fields__
        args = cls(**{k: v for k, v in params.items() if k in names})
        if "native_patch_size" in params:
            args.patch_size = int(params["native_patch_size"])
        if "adapter_merge_size" in params:
            args.spatial_merge_size = int(params["adapter_merge_size"])
        rope = params.get("rope_parameters") or {}
        if "rope_theta" in rope:
            args.rope_theta = float(rope["rope_theta"])
        return args

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def _grid_coords(rows: int, cols: int, merge: int) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) of every patch of a ``rows x cols`` patch grid, listed in
    merge-block order (block row, block col, row in block, col in block)."""
    within = np.arange(rows * cols)
    blocks_w = cols // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    return block_row * merge + in_row, block_col * merge + in_col


def _bilinear_taps(index: np.ndarray, size: int, side: int):
    """Taps and weights of ``F.interpolate(mode="bilinear", align_corners=False)``
    from a ``side``-long table to ``size`` positions (border clamped)."""
    src = (index.astype(np.float32) + 0.5) * side / size - 0.5
    floor = np.floor(src)
    offsets = np.arange(2)
    taps = np.clip(floor[:, None].astype(np.int64) + offsets, 0, side - 1)
    weights = np.clip(1.0 - np.abs(src[:, None] - floor[:, None] - offsets), 0.0, None)
    return taps, weights.astype(np.float32)


class Embeddings(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.args = args
        dims = args.num_channels * args.patch_size**2
        self.patch_embedding = nn.Linear(dims, args.hidden_size, bias=False)
        self.position_embedding = mx.zeros((args.posemb_grid_size**2, args.hidden_size))
        self.class_embedding = mx.zeros((args.hidden_size,))
        self.cls_position_embedding = mx.zeros((args.hidden_size,))

    def __call__(self, blocks: mx.array, rows: int, cols: int) -> mx.array:
        """``blocks``: one image's [rows * cols, P*m, P*m, C] pixel blocks.

        Returns the class token followed by the patch embeddings, patches in
        merge-block order.
        """
        args = self.args
        merge, patch, channels = (
            args.spatial_merge_size,
            args.patch_size,
            args.num_channels,
        )
        x = blocks.reshape(-1, merge, patch, merge, patch, channels)
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, patch * patch * channels)
        h = self.patch_embedding(x.astype(self.patch_embedding.weight.dtype))

        grid_h, grid_w = rows * merge, cols * merge
        row, col = _grid_coords(grid_h, grid_w, merge)
        side = args.posemb_grid_size
        h_taps, h_w = _bilinear_taps(row, grid_h, side)
        w_taps, w_w = _bilinear_taps(col, grid_w, side)
        index = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
        weight = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
        table = self.position_embedding[mx.array(index)].astype(mx.float32)
        pos = mx.sum(table * mx.array(weight)[..., None], axis=1)
        h = h + pos.astype(h.dtype)

        cls = (self.class_embedding + self.cls_position_embedding).astype(h.dtype)
        return mx.concatenate([cls[None], h], axis=0)


def _axial_rope(rows: int, cols: int, args: VisionArgs) -> tuple[mx.array, mx.array]:
    """cos/sin [1 + patches, head_dim] for the class token at (0, 0) and the
    patches at their 1-indexed (row, col)."""
    merge = args.spatial_merge_size
    row, col = _grid_coords(rows * merge, cols * merge, merge)
    pos_h = np.concatenate([[0], row + 1]).astype(np.float32)
    pos_w = np.concatenate([[0], col + 1]).astype(np.float32)
    half = args.head_dim // 2
    inv_freq = 1.0 / (
        args.rope_theta ** (np.arange(0, half, 2, dtype=np.float32) / half)
    )
    # Width frequencies first, then height; each angle covers one channel pair.
    angles = np.concatenate(
        [pos_w[:, None] * inv_freq, pos_h[:, None] * inv_freq], axis=-1
    )
    angles = np.repeat(angles, 2, axis=-1)
    return mx.array(np.cos(angles)), mx.array(np.sin(angles))


def _rotate_pairs(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate interleaved channel pairs (x1, x2) -> (x1 c - x2 s, x2 c + x1 s)."""
    xf = x.astype(mx.float32)
    pairs = xf.reshape(*xf.shape[:-1], -1, 2)
    rotated = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(xf.shape)
    return (xf * cos + rotated * sin).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dims = args.hidden_size
        self.heads = args.num_attention_heads
        self.scale = args.head_dim**-0.5
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        length = x.shape[0]
        shape = (length, self.heads, -1)
        cos, sin = cos[:, None], sin[:, None]
        q = _rotate_pairs(self.q_proj(x).reshape(shape), cos, sin)
        k = _rotate_pairs(self.k_proj(x).reshape(shape), cos, sin)
        v = self.v_proj(x).reshape(shape)
        out = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None],
            v.transpose(1, 0, 2)[None],
            scale=self.scale,
        )
        return self.out_proj(out[0].transpose(1, 0, 2).reshape(length, -1))


class MLP(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        hidden = int(args.hidden_size * args.mlp_ratio)
        self.c_fc = nn.Linear(args.hidden_size, hidden)
        self.c_proj = nn.Linear(hidden, args.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class LayerScale(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.gamma = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class EncoderLayer(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dims, eps = args.hidden_size, args.layer_norm_eps
        self.ln_1 = nn.LayerNorm(dims, eps=eps)
        self.ln_2 = nn.LayerNorm(dims, eps=eps)
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.ls_1 = LayerScale(dims)
        self.ls_2 = LayerScale(dims)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.ls_1(self.self_attn(self.ln_1(x), cos, sin))
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class Backbone(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.embeddings = Embeddings(args)
        self.ln_pre = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.layers = [EncoderLayer(args) for _ in range(args.num_hidden_layers)]


class VisionTower(nn.Module):
    def __init__(self, args: VisionArgs, text_hidden_size: int):
        super().__init__()
        self.args = args
        merged = args.spatial_merge_size**2 * args.hidden_size
        bias = args.projector_bias
        self.backbone = Backbone(args)
        self.proj_in = nn.Linear(merged, args.projection_hidden_dim, bias=bias)
        self.proj_out = nn.Linear(
            args.projection_hidden_dim, text_hidden_size, bias=bias
        )

    def __call__(self, pixel_values: mx.array, image_grid_thw) -> mx.array:
        """Image-token embeddings [sum(rows * cols), text hidden] for the
        processor's pixel blocks, images in order."""
        merge = self.args.spatial_merge_size
        outputs, start = [], 0
        for _, rows, cols in np.asarray(image_grid_thw).reshape(-1, 3).tolist():
            count = rows * cols
            blocks = pixel_values[start : start + count]
            start += count
            h = self.backbone.embeddings(blocks, rows, cols)
            h = self.backbone.ln_pre(h)
            cos, sin = _axial_rope(rows, cols, self.args)
            for layer in self.backbone.layers:
                h = layer(h, cos, sin)
            patches = h[1:].reshape(count, merge * merge * h.shape[-1])
            outputs.append(self.proj_out(nn.gelu(self.proj_in(patches))))
        return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)


# --------------------------------------------------------------------------
# Image processor (torch-free port of transformers' Inkling 2 processor)
# --------------------------------------------------------------------------

from PIL import Image  # noqa: E402


class Inkling2ImageProcessor:
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        size=None,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        rescale_factor=1 / 255,
        **kwargs,
    ):
        size = size or {"height": 40, "width": 40}
        self.block_h = int(size["height"])
        self.block_w = int(size["width"])
        self.mean = np.asarray(image_mean, dtype=np.float32)
        self.std = np.asarray(image_std, dtype=np.float32)
        self.rescale_factor = float(rescale_factor)

    @staticmethod
    def _rgb(image) -> np.ndarray:
        """HWC float32 RGB on the 0-255 scale."""
        if isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=-1)
            return array[..., :3].astype(np.float32)
        if image.has_transparency_data:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.int32)
            rows = np.arange(rgba.shape[0])[:, None] // 8
            cols = np.arange(rgba.shape[1])[None, :] // 8
            background = np.where((rows + cols) % 2 == 0, 204, 153)[..., None]
            alpha = rgba[..., 3:4]
            rgb = (rgba[..., :3] * alpha + background * (255 - alpha) + 127) // 255
            return rgb.astype(np.float32)
        return np.asarray(image.convert("RGB"), dtype=np.float32)

    def preprocess(self, images):
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]
        blocks, grids = [], []
        for image in images:
            x = self._rgb(image)
            height, width, channels = x.shape
            rows = math.ceil(height / self.block_h)
            cols = math.ceil(width / self.block_w)
            padded = np.full(
                (rows * self.block_h, cols * self.block_w, channels), -1.0, np.float32
            )
            padded[:height, :width] = x
            x = (padded * self.rescale_factor - self.mean) / self.std
            x = x.reshape(rows, self.block_h, cols, self.block_w, channels)
            blocks.append(
                x.transpose(0, 2, 1, 3, 4).reshape(
                    -1, self.block_h, self.block_w, channels
                )
            )
            grids.append([1, rows, cols])
        return {
            "pixel_values": np.concatenate(blocks, axis=0),
            "image_grid_thw": np.asarray(grids, dtype=np.int64),
        }

    def __call__(self, images, **kwargs):
        return self.preprocess(images)


def load_image_processor(model_dir) -> Inkling2ImageProcessor:
    """The checkpoint's image processor (``processor_config.json``)."""
    import json
    from pathlib import Path

    path = Path(model_dir) / "processor_config.json"
    config = json.loads(path.read_text()) if path.is_file() else {}
    return Inkling2ImageProcessor(**config.get("image_processor", {}))


def encode_images(model, processor: Inkling2ImageProcessor, images: list[bytes]) -> list[mx.array]:
    """Image-token rows [rows * cols, text hidden] for each encoded image, in order."""
    import io

    out = []
    for raw in images:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            processed = processor.preprocess([image])
        rows = model.encode_image(mx.array(processed["pixel_values"]), processed["image_grid_thw"])
        mx.eval(rows)
        out.append(rows)
    return out
