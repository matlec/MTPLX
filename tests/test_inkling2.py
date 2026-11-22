"""Inkling 2 step engine and backend: a small random model with the real
head geometry (head_dim 128, relative bias band, sliding window, looped core)
checked against the reference graph in ``mtplx.models.inkling2.model``."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="Metal kernels")

_TEXT = {
    "hidden_size": 256,
    "num_hidden_layers": 12,
    "vocab_size": 512,
    "unpadded_vocab_size": 500,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "swa_num_attention_heads": 2,
    "swa_num_key_value_heads": 1,
    "swa_head_dim": 128,
    "sliding_window_size": 64,
    "d_rel": 16,
    "rel_extent": 64,
    "log_scaling_n_floor": 32,
    "n_layers_per_attn_res_block": 2,
    "n_routed_experts": 32,
    "num_experts_per_tok": 4,
    "intermediate_size": 128,
    "expert_formats": ["nvfp4"] * 10 + ["mxfp8"] * 2,
    "loop_config": {"core_start": 4, "core_end": 8, "num_passes": 3, "gate_rank": 16},
}


@pytest.fixture(scope="module")
def model():
    from mtplx.models.inkling2.model import Model, ModelArgs

    mx.random.seed(7)
    m = Model(ModelArgs(model_type="inkling2_loop", text_config=dict(_TEXT)))
    params = []
    for key, value in tree_flatten(m.parameters()):
        if value.dtype in (mx.uint32, mx.uint8) or key.endswith("global_scale"):
            continue
        if key.endswith("norm.weight"):
            new = mx.ones(value.shape)
        elif key.endswith("router_bias"):
            # Well-separated expert preferences: random routing logits have
            # near-ties that rounding noise flips, which says nothing about
            # the kernels under test.
            new = mx.random.permutation(value.shape[0]).astype(mx.float32) * 0.5
        elif key.endswith("router_weight"):
            new = mx.random.normal(value.shape) * 0.002
        elif key.endswith("q_scale") or key.endswith("r_scale"):
            new = 1.0 + 0.1 * mx.random.normal(value.shape)
        else:
            new = mx.random.normal(value.shape) * 0.05
        # The loader's casting: BF16 except the tensors kept in FP32.
        dtype = mx.bfloat16 if m.cast_predicate(key) else mx.float32
        params.append((key, new.astype(dtype)))
    m.load_weights(params, strict=False)
    for layer in m.model.layers:
        for name in ("w13", "w2"):
            proj = getattr(layer.mlp.experts, name)
            e, n, packed = proj.weight.shape
            k = packed * 32 // proj.bits
            w = (mx.random.normal((e, n, k)) * 0.05).astype(mx.bfloat16)
            proj.weight, proj.scales = mx.quantize(
                w, group_size=proj.group_size, bits=proj.bits, mode=proj.mode
            )[:2]
    import mlx.nn as nn

    nn.quantize(
        m,
        group_size=64,
        bits=8,
        class_predicate=lambda p, mod: isinstance(mod, nn.Linear)
        and (p == "lm_head" or ".attn." in p),
    )
    mx.eval(m.parameters())
    m.pack_fused()
    return m


def _kl(a: mx.array, b: mx.array) -> tuple[float, float]:
    la = a - mx.logsumexp(a, -1, keepdims=True)
    lb = b - mx.logsumexp(b, -1, keepdims=True)
    k = (mx.exp(la) * (la - lb)).sum(-1)
    return float(k.mean()), float(k.max())


def _ids(n: int, seed: int = 0) -> list[int]:
    mx.random.seed(seed)
    return mx.random.randint(0, 500, (n,)).tolist()


def _reference_decode(model, ids, start):
    cache = model.make_cache()
    model(mx.array([ids[:start]]), cache=cache)
    rows = [model(mx.array([[t]]), cache=cache)[0, -1] for t in ids[start:]]
    return mx.stack(rows)[:, :500]


def _engine(model, ids, start):
    from mtplx.models.inkling2.decode import StepCache, StepEngine

    cache = model.make_cache()
    model(mx.array([ids[:start]]), cache=cache)
    return StepCache.from_reference(model, cache), StepEngine(model)


def test_engine_decode_matches_reference(model):
    # 100 tokens: the 64-token sliding window wraps the ring (cap 80).
    ids = _ids(100)
    ref = _reference_decode(model, ids, 30)
    cache, engine = _engine(model, ids, 30)
    out = mx.concatenate([engine.step(cache, [t])[0][:, :500] for t in ids[30:]])
    mean, worst = _kl(ref, out)
    assert mean < 2e-3 and worst < 2e-2, (mean, worst)
    assert cache.offset == len(ids)


def test_engine_block_step_and_rollback(model):
    ids = _ids(60, seed=1)
    cache, engine = _engine(model, ids, 40)
    single = mx.concatenate([engine.step(cache, [t])[0][:, :500] for t in ids[40:48]])
    cache, engine = _engine(model, ids, 40)
    block = mx.concatenate(
        [engine.step(cache, ids[i : i + 4])[0][:, :500] for i in (40, 44)]
    )
    assert _kl(single, block)[0] < 2e-3
    # Roll back two of the last block's four tokens and redo them one by one.
    cache.rollback(2)
    assert cache.offset == 46
    redo = mx.concatenate([engine.step(cache, [t])[0][:, :500] for t in ids[46:48]])
    assert _kl(single[6:8], redo)[0] < 2e-3


def test_reference_round_trip_and_snapshot(model):
    from mtplx.models.inkling2.decode import StepCache

    ids = _ids(140, seed=2)
    cache, engine = _engine(model, ids, 90)
    for t in ids[90:94]:
        engine.step(cache, [t])
    snap = cache.snapshot()
    ref = cache.to_reference(model)
    model(mx.array([ids[94:130]]), cache=ref)
    resumed = StepCache.from_reference(model, ref, position=130)
    a = engine.step(resumed, [ids[130]])[0][:, :500]
    direct, engine2 = _engine(model, ids, 130)
    b = engine2.step(direct, [ids[130]])[0][:, :500]
    assert _kl(b, a)[0] < 5e-3
    # Advancing the original must not disturb the snapshot.
    for t in ids[94:100]:
        engine.step(cache, [t])
    c = engine.step(snap, [ids[94]])[0][:, :500]
    fresh, engine3 = _engine(model, ids, 94)
    d = engine3.step(fresh, [ids[94]])[0][:, :500]
    assert _kl(d, c)[0] < 2e-3


class _Tok:
    """The control tokens of the Inkling chat format, text as code points."""

    special = {
        "<|message_model|>": 1000,
        "<|content_thinking|>": 1001,
        "<|content_text|>": 1002,
        "<|content_invoke_tool_json|>": 1003,
        "<|content_invoke_tool_text|>": 1004,
        "<|content_tool_error|>": 1005,
        "<|end_message|>": 1006,
    }

    def convert_tokens_to_ids(self, name):
        return self.special.get(name)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_translator_maps_messages_to_qwen_markup():
    from mtplx.backends.inkling2_ar import _Translator

    tok = _Tok()
    sp = tok.special
    t = _Translator(tok)
    stream = (
        [sp["<|message_model|>"], sp["<|content_thinking|>"], ord("r"), sp["<|end_message|>"]]
        + [sp["<|message_model|>"], sp["<|content_text|>"], ord("a"), sp["<|end_message|>"]]
        + [sp["<|message_model|>"], ord("f"), ord("n"), sp["<|content_invoke_tool_json|>"]]
        + [ord("{"), ord("}"), sp["<|end_message|>"]]
    )
    out = "".join(chr(c) for token in stream for c in t(token))
    assert out == "<think>r</think>\n\na<tool_call>\n{}\n</tool_call>"


def test_cache_keys_separate_images():
    from types import SimpleNamespace

    from mtplx.backends.inkling2_ar import cache_key_ids

    ids = [5, 9, 9, 9, 7, 9, 9, 3]
    a = SimpleNamespace(image_pad_token_id=9, image_digests=(11, 22), pad_counts=(3, 2))
    b = SimpleNamespace(image_pad_token_id=9, image_digests=(11, 23), pad_counts=(3, 2))
    ka, kb = cache_key_ids(ids, a), cache_key_ids(ids, b)
    # Same first image: same key up to the second image, then they differ.
    assert ka[:5] == kb[:5] and ka[5] != kb[5]
    assert len(set(ka[1:4])) == 3 and ka[0] == 5 and ka[4] == 7
    assert cache_key_ids(ids, None) == ids
    assert cache_key_ids(ids, SimpleNamespace(image_pad_token_id=9, image_digests=None, pad_counts=None)) is None
