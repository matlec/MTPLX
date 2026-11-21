# SPDX-License-Identifier: Apache-2.0
"""Single-stream DSpark speculative decoding for Inkling 2.

One cycle:

1. The drafter turns the anchor (the newest sampled token, not yet run
   through the target) plus ``block_size - 1`` mask positions into
   ``block_size - 1`` proposals. It attends the projected context taps of
   the newest ``sliding_window_size - 1`` committed positions.
2. The target runs ``[anchor, d1 .. d_k]`` in one forward and captures the
   draft taps for those positions.
3. The longest prefix the target agrees with is kept; the target's own token
   after it becomes the next anchor. Rejected positions are trimmed from the
   KV caches and the short-conv states are re-cut from the verify windows.

Acceptance is exact for the target distribution: greedy compares argmaxes;
sampling treats the drafter as a point mass q = onehot(d) and accepts with
probability p(d), resampling a rejection from p with d removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from .dspark import Inkling2DSparkModel, load_inkling2_dspark_model, mix_rmsnorm


@dataclass
class DraftState:
    """Projected context K/V per draft layer, newest ``slots`` positions."""

    keys: list[mx.array] = field(default_factory=list)
    values: list[mx.array] = field(default_factory=list)
    fed: int = 0  # committed positions fed so far (absolute)
    last_row: mx.array | None = None  # newest raw context row [1, 1, T*H]


def markov_latent(markov, prev_tokens: mx.array, step: int) -> mx.array:
    """One RNN step of the Markov head: advances its state and returns the
    [1, rank] latent (``Inkling2MarkovHead.apply_step_logits`` without the
    full-vocabulary projection)."""
    state = markov._state
    dtype = state.dtype
    inputs = mx.concatenate(
        [
            state,
            markov.markov_w1(prev_tokens.reshape(-1)).astype(dtype),
            markov._hidden[:, step].astype(dtype),
            markov._rows_f.astype(dtype),
        ],
        axis=-1,
    )
    proj = markov.rnn_proj(inputs).astype(mx.float32)
    r = markov.rank
    gate = mx.sigmoid(proj[:, :r]).astype(dtype)
    kept = (gate * state).astype(dtype)
    blend = ((1.0 - gate) * mx.tanh(proj[:, r : 2 * r]).astype(dtype)).astype(dtype)
    markov._state = kept + blend
    return mx.tanh(proj[:, 2 * r :]).astype(dtype)


class DSparkDrafter:
    # Candidates the Markov head re-ranks per position (0: whole vocabulary).
    candidates = 0

    def __init__(self, model: Inkling2DSparkModel):
        self.model = model
        cfg = model.config
        self.block_size = int(cfg.block_size)
        self.slots = int(cfg.sliding_window) - 1
        self.target_layer_ids = [int(i) for i in cfg.target_layer_ids]
        self.mask_token_id = int(cfg.mask_token_id)

    @classmethod
    def load(cls, model_path: str | Path, target) -> "DSparkDrafter":
        return cls(load_inkling2_dspark_model(model_path).bind(target))

    # -- context -------------------------------------------------------------
    def new_state(self) -> DraftState:
        return DraftState()

    def feed(self, state: DraftState, captures: list[mx.array]) -> None:
        """Append committed positions: ``captures`` are the target taps
        ``[1, n, H]`` in ``target_layer_ids`` order."""
        context = mx.concatenate(captures, axis=-1)
        n = int(context.shape[1])
        if n == 0:
            return
        if n > self.slots:
            state.fed += n - self.slots
            context = context[:, -self.slots :]
            n = self.slots
        model = self.model
        base = state.fed
        for index, layer in enumerate(model.layers):
            attn = layer.self_attn
            k, v = attn.project_context_kv(context)
            k = attn.k_norm_ctx(k.reshape(1, n, attn.n_kv_heads, attn.head_dim))
            k = model.rope(k.transpose(0, 2, 1, 3), offset=base)
            v = v.reshape(1, n, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            if index < len(state.keys):
                k = mx.concatenate([state.keys[index], k], axis=2)
                v = mx.concatenate([state.values[index], v], axis=2)
                if k.shape[2] > self.slots:
                    k = k[:, :, -self.slots :]
                    v = v[:, :, -self.slots :]
                state.keys[index] = k
                state.values[index] = v
            else:
                state.keys.append(k)
                state.values.append(v)
        state.fed = base + n
        state.last_row = context[:, -1:]

    # -- draft ---------------------------------------------------------------
    def draft(self, state: DraftState, anchor: int, count: int | None = None) -> list[int]:
        """Greedy proposals after ``anchor`` (at most ``block_size - 1``)."""
        model = self.model
        block = self.block_size
        count = block - 1 if count is None else min(count, block - 1)
        h = model._embed_input_tokens(mx.array([[anchor]], dtype=mx.int32))
        fill = model.mask_embed.astype(h.dtype)
        h = mx.concatenate([h, mx.broadcast_to(fill, (1, block - 1, h.shape[-1]))], axis=1)
        offset = state.fed
        for index, layer in enumerate(model.layers):
            attn = layer.self_attn
            residual = h
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).reshape(1, block, attn.n_heads, -1)).transpose(0, 2, 1, 3)
            k = attn.k_norm(attn.k_proj(x).reshape(1, block, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = attn.v_proj(x).reshape(1, block, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q = model.rope(q, offset=offset)
            k = model.rope(k, offset=offset)
            if state.keys:
                k = mx.concatenate([state.keys[index], k], axis=2)
                v = mx.concatenate([state.values[index], v], axis=2)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale)
            out = out.transpose(0, 2, 1, 3).reshape(1, block, -1)
            h = residual + attn.o_proj(out)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        hidden = model.norm(h)[:, 1:]
        logits = model._logits(hidden).astype(mx.float32)
        markov = model.markov_head
        if markov._vocab_bias is not None:
            logits = logits + markov._vocab_bias
        if state.last_row is not None:
            markov._rows_f = mix_rmsnorm(
                state.last_row, markov._alpha, markov.dflare_norm.weight, markov._eps
            )[0]
        else:
            markov._rows_f = mx.zeros((1, hidden.shape[-1]), dtype=hidden.dtype)
        markov._hidden = hidden
        markov._state = mx.zeros((1, markov.rank), dtype=hidden.dtype)
        markov._step = 0
        prev = mx.array([anchor], dtype=mx.int32)
        tokens = []
        if self.candidates:
            # The Markov bias only re-ranks: score the base logits' top
            # candidates instead of the whole vocabulary (the fold matrix is
            # V x rank, ~100 MB per step otherwise). Proposals only steer the
            # verify; acceptance stays exact.
            kc = self.candidates
            cand = mx.argpartition(-logits[0, :count], kc - 1, axis=-1)[:, :kc]
            base = mx.take_along_axis(logits[0, :count], cand, axis=-1)
            fold_w = markov._fold_weight[cand]  # [count, kc, rank]
            fold_b = markov._fold_bias[cand].astype(mx.float32)
        for step in range(count):
            if not self.candidates:
                step_logits = markov.apply_step_logits(logits[:, step], prev)
                prev = mx.argmax(step_logits, axis=-1)
            else:
                latent = markov_latent(markov, prev, step)
                bias = (fold_w[step] @ latent[0]).astype(mx.float32) + fold_b[step]
                pick = mx.argmax(base[step] + bias)
                prev = cand[step, pick].reshape(1)
            tokens.append(prev)
        out = mx.concatenate(tokens) if tokens else mx.zeros((0,), dtype=mx.int32)
        return [int(t) for t in np.array(out).tolist()]


# ---------------------------------------------------------------------------
# Target side
# ---------------------------------------------------------------------------


def _softcapped_logits(target, hidden):
    return target.logits_from_hidden(hidden)


def target_forward(target, tokens: list[int], cache, capture_ids: list[int]):
    """Run ``tokens`` through the target; returns (logits [L, V], captures)."""
    out = target.forward(
        mx.array([tokens], dtype=mx.int32), cache, capture_layer_ids=capture_ids
    )
    return out.logits[0], out.hidden_states


def _sample(logits: mx.array, temperature: float, top_p: float, top_k: int, rng) -> tuple[int, np.ndarray | None]:
    """Sample one token; returns it and the filtered probability row (None
    for greedy)."""
    if temperature <= 0:
        return int(mx.argmax(logits).item()), None
    lp = logits.astype(mx.float32) / temperature
    probs = np.array(mx.softmax(lp, axis=-1)).astype(np.float64)
    if top_k and top_k < probs.shape[-1]:
        kth = np.partition(probs, -top_k)[-top_k]
        probs = np.where(probs >= kth, probs, 0.0)
    if 0.0 < top_p < 1.0:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        keep = cum - probs[order] < top_p * cum[-1]
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[keep]] = True
        probs = np.where(mask, probs, 0.0)
    probs /= probs.sum()
    return int(rng.choice(probs.shape[-1], p=probs)), probs


@dataclass
class SpecStats:
    cycles: int = 0
    drafted: int = 0
    accepted: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    draft_s: float = 0.0
    verify_s: float = 0.0


def generate_engine(
    target,
    drafter: DSparkDrafter | None,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
    stop_ids: set[int] | None = None,
    draft_tokens: int | None = None,
    prefill_chunk: int = 2048,
    on_token: Callable[[int], None] | None = None,
    stats: SpecStats | None = None,
    engine=None,
) -> list[int]:
    """``generate`` on the step engine: reference prefill, engine decode
    and (with ``drafter``) engine verify."""
    import time

    from .decode import StepCache, StepEngine

    engine = engine or StepEngine(target)
    stats = stats if stats is not None else SpecStats()
    rng = np.random.default_rng(seed)
    stop_ids = stop_ids or set()
    ref_cache = target.make_cache()
    capture_ids = drafter.target_layer_ids if drafter is not None else None
    dstate = drafter.new_state() if drafter is not None else None
    t0 = time.perf_counter()
    logits = None
    for start in range(0, len(prompt_ids), prefill_chunk):
        chunk = prompt_ids[start : start + prefill_chunk]
        if drafter is not None and len(prompt_ids) - start - len(chunk) < drafter.slots:
            lg, caps = target_forward(target, chunk, ref_cache, capture_ids)
            drafter.feed(dstate, caps)
            logits = lg[-1]
        else:
            if drafter is not None:
                dstate.fed += len(chunk)
            logits = target(mx.array([chunk], dtype=mx.int32), cache=ref_cache)[0, -1]
        mx.eval(logits)
    cache = StepCache.from_reference(target, ref_cache)
    del ref_cache
    stats.prefill_s += time.perf_counter() - t0

    out: list[int] = []

    def emit(token: int) -> bool:
        out.append(token)
        if on_token is not None:
            on_token(token)
        return token in stop_ids or len(out) >= max_tokens

    t0 = time.perf_counter()
    anchor, _ = _sample(logits, temperature, top_p, top_k, rng)
    if emit(anchor):
        stats.decode_s += time.perf_counter() - t0
        return out
    if drafter is None:
        if temperature <= 0:
            # Greedy: chain on-device so the next step is built while the
            # GPU runs this one.
            tok = mx.array([anchor], dtype=mx.int32)
            pending = []
            while True:
                lg, _ = engine.step(cache, tok)
                tok = mx.argmax(lg[-1:], axis=-1).astype(mx.int32)
                mx.async_eval(tok)
                pending.append(tok)
                if len(pending) >= 2 or len(out) + len(pending) >= max_tokens:
                    t = int(pending.pop(0).item())
                    if emit(t):
                        break
            stats.decode_s += time.perf_counter() - t0
            return out
        while True:
            lg, _ = engine.step(cache, [anchor])
            anchor, _ = _sample(lg[-1], temperature, top_p, top_k, rng)
            if emit(anchor):
                break
        stats.decode_s += time.perf_counter() - t0
        return out
    while True:
        budget = max_tokens - len(out)
        td = time.perf_counter()
        drafts = drafter.draft(dstate, anchor, min(draft_tokens or drafter.block_size - 1, budget))
        stats.draft_s += time.perf_counter() - td
        tv = time.perf_counter()
        logits, caps = engine.step(cache, [anchor] + drafts, capture_ids)
        mx.eval(logits, *caps)
        stats.verify_s += time.perf_counter() - tv
        stats.cycles += 1
        stats.drafted += len(drafts)
        accepted, next_token, done = _accept(logits, drafts, temperature, top_p, top_k, rng, emit)
        stats.accepted += accepted
        if not done and next_token is None:
            next_token, _ = _sample(logits[len(drafts)], temperature, top_p, top_k, rng)
        cache.rollback(len(drafts) - accepted)
        drafter.feed(dstate, [c[:, : accepted + 1] for c in caps])
        if done:
            break
        anchor = next_token
        if emit(anchor):
            break
    stats.decode_s += time.perf_counter() - t0
    return out


def _accept(logits, drafts, temperature, top_p, top_k, rng, emit):
    """Longest accepted prefix. Returns (accepted, correction or None, done)."""
    if temperature <= 0:
        targets = np.array(mx.argmax(logits, axis=-1)).tolist()
    accepted = 0
    for i, d in enumerate(drafts):
        if temperature <= 0:
            if targets[i] == d:
                accepted += 1
                if emit(d):
                    return accepted, None, True
                continue
            return accepted, targets[i], False
        _, probs = _sample(logits[i], temperature, top_p, top_k, rng)
        if rng.random() < probs[d]:
            accepted += 1
            if emit(d):
                return accepted, None, True
            continue
        probs = probs.copy()
        probs[d] = 0.0
        probs /= probs.sum()
        return accepted, int(rng.choice(probs.shape[-1], p=probs)), False
    return accepted, None, False


def generate(
    target,
    drafter: DSparkDrafter | None,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
    stop_ids: set[int] | None = None,
    draft_tokens: int | None = None,
    prefill_chunk: int = 2048,
    on_token: Callable[[int], None] | None = None,
    stats: SpecStats | None = None,
) -> list[int]:
    """Generate up to ``max_tokens`` tokens; DSpark when ``drafter`` is given."""
    import time

    stats = stats if stats is not None else SpecStats()
    rng = np.random.default_rng(seed)
    stop_ids = stop_ids or set()
    cache = target.make_cache()
    capture_ids = drafter.target_layer_ids if drafter is not None else None
    dstate = drafter.new_state() if drafter is not None else None

    t0 = time.perf_counter()
    logits = None
    for start in range(0, len(prompt_ids), prefill_chunk):
        chunk = prompt_ids[start : start + prefill_chunk]
        if drafter is not None:
            lg, caps = target_forward(target, chunk, cache, capture_ids)
            # Only the newest window positions are ever attended.
            if len(prompt_ids) - start - len(chunk) < drafter.slots:
                drafter.feed(dstate, caps)
            else:
                dstate.fed += len(chunk)
            logits = lg[-1]
        else:
            logits = target(mx.array([chunk], dtype=mx.int32), cache=cache)[0, -1]
        mx.eval(logits)
    stats.prefill_s += time.perf_counter() - t0

    out: list[int] = []

    def emit(token: int) -> bool:
        out.append(token)
        if on_token is not None:
            on_token(token)
        return token in stop_ids or len(out) >= max_tokens

    t0 = time.perf_counter()
    anchor, _ = _sample(logits, temperature, top_p, top_k, rng)
    if emit(anchor):
        stats.decode_s += time.perf_counter() - t0
        return out
    while True:
        if drafter is None:
            logits = target(mx.array([[anchor]], dtype=mx.int32), cache=cache)[0, -1]
            anchor, _ = _sample(logits, temperature, top_p, top_k, rng)
            if emit(anchor):
                break
            continue
        budget = max_tokens - len(out)
        td = time.perf_counter()
        drafts = drafter.draft(dstate, anchor, min(draft_tokens or drafter.block_size - 1, budget))
        stats.draft_s += time.perf_counter() - td
        tv = time.perf_counter()
        logits, caps = target_forward(target, [anchor] + drafts, cache, capture_ids)
        mx.eval(logits, *caps)
        stats.verify_s += time.perf_counter() - tv
        stats.cycles += 1
        stats.drafted += len(drafts)
        accepted = 0
        next_token = None
        done = False
        for i, d in enumerate(drafts):
            if temperature <= 0:
                t = int(mx.argmax(logits[i]).item())
                if t == d:
                    accepted += 1
                    if emit(d):
                        done = True
                        break
                    continue
                next_token = t
                break
            _, probs = _sample(logits[i], temperature, top_p, top_k, rng)
            if rng.random() < probs[d]:
                accepted += 1
                if emit(d):
                    done = True
                    break
                continue
            probs = probs.copy()
            probs[d] = 0.0
            probs /= probs.sum()
            next_token = int(rng.choice(probs.shape[-1], p=probs))
            break
        stats.accepted += accepted
        if not done and next_token is None:
            # Every draft accepted: the bonus token comes from the last row.
            next_token, _ = _sample(logits[len(drafts)], temperature, top_p, top_k, rng)
        keep = accepted + 1  # anchor + accepted drafts are committed
        if keep < len(drafts) + 1:
            if not target.mtp_partial_rollback(cache, accepted, len(drafts)):
                raise RuntimeError("Inkling 2 verify rollback failed")
        else:
            for entry in cache:
                entry[1]._omlx_verify_xp = None
        drafter.feed(dstate, [c[:, :keep] for c in caps])
        if done:
            break
        anchor = next_token
        if emit(anchor):
            break
    stats.decode_s += time.perf_counter() - t0
    return out
