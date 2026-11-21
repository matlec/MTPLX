# SPDX-License-Identifier: Apache-2.0
"""Load an Inkling 2 checkpoint (source ModelOpt layout or an MTPLX pack)."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import INKLING2_MODEL_TYPES


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") not in INKLING2_MODEL_TYPES:
        raise ValueError(f"{path} is not an Inkling 2 checkpoint")
    return config


def _model_args(config: dict[str, Any]):
    from .model import ModelArgs

    quant = config.get("quantization")
    # A source checkpoint carries ModelOpt's quantization_config; the expert
    # carriers are then detected from the tensors in sanitize().
    return ModelArgs(
        model_type=config["model_type"],
        text_config=config["text_config"],
        vision_config=None,
        mtp_config=config.get("mtp_config"),
        eos_token_id=config.get("eos_token_id", 200006),
        image_token_id=config.get("image_token_id", 200054),
        quantization=quant if isinstance(quant, dict) else None,
    )


def _quantize_from_config(model: nn.Module, config: dict[str, Any]) -> None:
    """Re-create the quantized modules a converted pack declares."""
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        return

    def predicate(path: str, module: nn.Module):
        spec = quant.get(path)
        if isinstance(spec, dict):
            return spec
        if spec is False:
            return False
        return False

    nn.quantize(model, class_predicate=predicate)


def load_model(path: str | Path, *, lazy: bool = False):
    """Returns ``(model, config)``."""
    from .model import Model

    path = Path(path)
    config = read_config(path)
    model = Model(_model_args(config))
    weights: dict[str, mx.array] = {}
    for shard in sorted(glob.glob(str(path / "model*.safetensors"))):
        weights.update(mx.load(shard))
    weights = model.sanitize(weights)
    _quantize_from_config(model, config)
    predicate = model.cast_predicate
    dtype = mx.bfloat16
    weights = {
        k: (v.astype(dtype) if predicate(k) and mx.issubdtype(v.dtype, mx.floating) and v.dtype != dtype else v)
        for k, v in weights.items()
    }
    model.load_weights(list(weights.items()), strict=True)
    if not lazy:
        mx.eval(model.parameters())
        model.pack_fused()
    model.eval()
    return model, config
