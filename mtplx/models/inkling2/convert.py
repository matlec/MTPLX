# SPDX-License-Identifier: Apache-2.0
"""Build an MTPLX Inkling 2 pack from the published (ModelOpt) checkpoint.

The pack stores the runtime layout (``loader.sanitize_weights``), so loading
does no remapping. Routed experts keep their source NVFP4 codes; the MXFP8
coda experts, the attention projections and the LM head are re-quantized to
the requested carriers. Every decode step reads every attention projection
once per layer visit (48 per token) and the 201K-row LM head once, so those
two dominate decode bandwidth after the experts.

  python -m mtplx.models.inkling2.convert SRC DST [--attn-bits 8] [--head-bits 6]
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .load import _model_args, read_config

PACK_FORMAT = "mtplx-inkling2-pack"
PACK_VERSION = 1
_COPY = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "mtp.safetensors",
    "README.md",
)


def _affine(bits: int, group: int) -> dict:
    return {"group_size": group, "bits": bits, "mode": "affine"}


def _spec(value: str) -> dict | None:
    """``"8"`` / ``"4:64"`` -> affine; ``"nvfp4"`` / ``"mxfp4"`` / ``"mxfp8"``;
    ``"bf16"`` keeps the tensor unquantized."""
    if value in ("bf16", "none"):
        return None
    if value in ("nvfp4",):
        return {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    if value in ("mxfp4",):
        return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    if value in ("mxfp8",):
        return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    bits, _, group = value.partition(":")
    return _affine(int(bits), int(group or 64))


def _quantize_linear(weight: mx.array, spec: dict) -> dict[str, mx.array]:
    out = mx.quantize(
        weight.astype(mx.bfloat16),
        group_size=spec["group_size"],
        bits=spec["bits"],
        mode=spec["mode"],
    )
    names = ("weight", "scales", "biases")
    return {n: a for n, a in zip(names, out)}


def _requantize_experts(prefix: str, weights: dict, spec: dict) -> dict:
    """MXFP8 expert codes -> another carrier (dequantize, then quantize)."""
    w = mx.dequantize(
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.scales"],
        None,
        group_size=32,
        bits=8,
        mode="mxfp8",
        dtype=mx.bfloat16,
    )
    out = _quantize_linear(w, spec)
    return {f"{prefix}.{k}": v for k, v in out.items()}


def convert(
    src: Path,
    dst: Path,
    *,
    attn: str = "8",
    head: str = "6",
    embed: str = "bf16",
    coda_experts: str = "keep",
    shard_layers: int = 4,
) -> None:
    from .model import Model

    config = read_config(src)
    model = Model(_model_args(config))
    raw: dict[str, mx.array] = {}
    for shard in sorted(glob.glob(str(src / "model*.safetensors"))):
        raw.update(mx.load(shard))
    weights = model.sanitize(raw)
    del raw
    text = config["text_config"]
    n_layers = int(text["num_hidden_layers"])
    attn_spec, head_spec, embed_spec = _spec(attn), _spec(head), _spec(embed)
    coda_spec = None if coda_experts == "keep" else _spec(coda_experts)

    quant: dict[str, dict] = {}
    # Every routed-expert projection records its carrier, so the pack loads
    # without inspecting tensors.
    for i in range(n_layers):
        experts = model.model.layers[i].mlp.experts
        for which in ("w13", "w2"):
            proj = getattr(experts, which)
            quant[f"model.layers.{i}.mlp.experts.{which}"] = {
                "mode": proj.mode,
                "bits": proj.bits,
                "group_size": proj.group_size,
            }

    def emit_layer(i: int) -> dict[str, mx.array]:
        pre = f"model.layers.{i}."
        out = {k: v for k, v in weights.items() if k.startswith(pre)}
        for name in ("attn.in_proj", "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"):
            key = pre + name + ".weight"
            if key not in out or attn_spec is None:
                continue
            q = _quantize_linear(out.pop(key), attn_spec)
            for k, v in q.items():
                out[pre + name + "." + k] = v
            quant[pre + name] = dict(attn_spec)
        if coda_spec is not None and quant[pre + "mlp.experts.w13"]["mode"] == "mxfp8":
            for which in ("w13", "w2"):
                base = pre + f"mlp.experts.{which}"
                out.update(_requantize_experts(base, out, coda_spec))
                if coda_spec["mode"] != "affine":
                    out.pop(base + ".biases", None)
                if coda_spec["mode"] == "nvfp4":
                    # Unit tensor scale in MLX's convention (see ExpertProj).
                    experts = out[base + ".weight"].shape[0]
                    out[base + ".global_scale"] = mx.full((experts,), 448.0 * 6.0, mx.float32)
                quant[base] = dict(coda_spec)
        return out

    dst.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    shards: list[dict[str, mx.array]] = []
    for start in range(0, n_layers, shard_layers):
        shard: dict[str, mx.array] = {}
        for i in range(start, min(n_layers, start + shard_layers)):
            shard.update(emit_layer(i))
        shards.append(shard)
    rest = {k: v for k, v in weights.items() if not k.startswith("model.layers.")}
    if head_spec is not None:
        for k, v in _quantize_linear(rest.pop("lm_head.weight"), head_spec).items():
            rest["lm_head." + k] = v
        quant["lm_head"] = dict(head_spec)
    if embed_spec is not None:
        for k, v in _quantize_linear(rest.pop("model.embed.weight"), embed_spec).items():
            rest["model.embed." + k] = v
        quant["model.embed"] = dict(embed_spec)
    shards.append(rest)
    predicate = model.cast_predicate
    total = len(shards)
    for n, shard in enumerate(shards, 1):
        name = f"model-{n:05d}-of-{total:05d}.safetensors"
        shard = {
            k: (
                v.astype(mx.bfloat16)
                if predicate(k) and mx.issubdtype(v.dtype, mx.floating) and v.dtype == mx.float32
                else v
            )
            for k, v in shard.items()
        }
        mx.eval(shard)
        mx.save_safetensors(str(dst / name), shard, metadata={"format": "mlx"})
        for k in shard:
            index[k] = name
        print(f"wrote {name} ({len(shard)} tensors)", flush=True)
        shard.clear()
        shards[n - 1] = {}
    out_config = dict(config)
    out_config.pop("quantization_config", None)
    out_config.pop("vision_config", None)
    out_config["quantization"] = quant
    out_config["mtplx_pack"] = {
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "source": str(src),
        "attn": attn,
        "head": head,
        "embed": embed,
        "coda_experts": coda_experts,
    }
    (dst / "config.json").write_text(json.dumps(out_config, indent=2))
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index}, indent=2)
    )
    for name in _COPY:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    if (src / "tiktoken").is_dir():
        shutil.copytree(src / "tiktoken", dst / "tiktoken", dirs_exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--attn", default="8", help="attention projections: bf16 | BITS[:GROUP] | nvfp4 | mxfp4 | mxfp8")
    p.add_argument("--head", default="6", help="LM head carrier")
    p.add_argument("--embed", default="bf16", help="embedding carrier")
    p.add_argument("--coda-experts", default="keep", help="MXFP8 coda experts: keep | BITS[:GROUP] | nvfp4")
    args = p.parse_args()
    convert(args.src, args.dst, attn=args.attn, head=args.head, embed=args.embed, coda_experts=args.coda_experts)


if __name__ == "__main__":
    main()
