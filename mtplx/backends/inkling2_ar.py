"""Inkling 2 (``inkling2`` / ``inkling2_loop``) target-only backend.

The model and its step engine live in ``mtplx.models.inkling2``. This module
adapts them to the server: a runtime object, and a generate loop that owns
prefill (reference graph) and decode (compiled step engine, sampling on the
GPU so the next step is built while the current one runs).

Inkling marks reasoning, answers and tool calls with control tokens
(``<|content_thinking|>`` ... ``<|end_message|>``, ``<|content_text|>``,
``<|content_invoke_tool_json|>``). The generate loop re-expresses them as the
``<think>`` / ``<tool_call>`` markup the server's Qwen-style reasoning and
tool parsers already split, so the wire protocol needs no new parser.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mtplx.runtime import MTPLXRuntime

BACKEND_ID = "inkling2_ar"
ARCH_ID = "inkling2-ar"
MODEL_TYPES = frozenset({"inkling2", "inkling2_loop"})

# Load-time re-quantization of the BF16 attention projections and LM head
# (8-bit both: KL 0.0034 vs the source on the bench text).
_DEFAULT_ATTN = "8:64"
_DEFAULT_HEAD = "8:64"


def is_inkling2_config(config: dict[str, Any] | None) -> bool:
    return str((config or {}).get("model_type") or "").lower() in MODEL_TYPES


class Inkling2ARRuntime(MTPLXRuntime):
    """Target-only runtime; generation goes through ``generate_inkling2_ar``."""

    backend_id = BACKEND_ID

    def forward_ar(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        input_embeddings=None,
    ):
        del return_hidden, hidden_variant, emit_logits, logits_keep
        return self.model(input_ids, cache=cache, input_embeddings=input_embeddings)

    def make_cache(self):
        return self.model.make_cache()

    def repage_target_prefill_cache(self, cache: Any) -> bool:
        del cache
        return False


def load_inkling2_runtime(path: Path, config: dict[str, Any]) -> Inkling2ARRuntime:
    from mtplx.models.inkling2.decode import StepEngine
    from mtplx.models.inkling2.load import load_model
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import _load_tokenizer_resilient

    attn = os.environ.get("MTPLX_INKLING2_ATTN", _DEFAULT_ATTN) or None
    head = os.environ.get("MTPLX_INKLING2_HEAD", _DEFAULT_HEAD) or None
    if attn in {"bf16", "none"}:
        attn = None
    if head in {"bf16", "none"}:
        head = None
    model, _config = load_model(path, attn=attn, head=head)
    tokenizer = _load_tokenizer_resilient(path, config)
    template = path / "chat_template.jinja"
    if not getattr(tokenizer, "chat_template", None) and template.is_file():
        tokenizer.chat_template = template.read_text()
    runtime = Inkling2ARRuntime(model, tokenizer, path, False, MTPContract())
    runtime.engine = StepEngine(model)
    runtime.drafter = None
    if os.environ.get("MTPLX_INKLING2_DSPARK", "0") not in {"", "0", "false"}:
        from mtplx.models.inkling2.spec import DSparkDrafter

        runtime.drafter = DSparkDrafter.load(path, model)
    runtime.laguna_fused_report = []
    return runtime


# ---------------------------------------------------------------------------
# Control-token translation
# ---------------------------------------------------------------------------


class _Translator:
    """Stateful map from Inkling's message tokens to Qwen-style markup.

    ``<|message_model|>`` opens a header (an optional tool name follows) that
    ends at a content marker; ``<|end_message|>`` closes the current section.
    """

    def __init__(self, tokenizer: Any):
        def tid(name: str) -> int:
            value = tokenizer.convert_tokens_to_ids(name)
            return int(value) if value is not None else -1

        def ids(text: str) -> list[int]:
            return list(tokenizer.encode(text, add_special_tokens=False))

        self.message = tid("<|message_model|>")
        self.thinking = tid("<|content_thinking|>")
        self.text = tid("<|content_text|>")
        self.tool_json = tid("<|content_invoke_tool_json|>")
        self.tool_text = tid("<|content_invoke_tool_text|>")
        self.tool_error = tid("<|content_tool_error|>")
        self.end = tid("<|end_message|>")
        self.think_open = ids("<think>")
        self.think_close = ids("</think>\n\n")
        self.tool_open = ids("<tool_call>\n")
        self.tool_close = ids("\n</tool_call>")
        self.section: str | None = None
        self.in_header = False

    def __call__(self, token: int) -> list[int]:
        if token == self.message:
            self.in_header = True
            return []
        if token == self.thinking:
            self.in_header = False
            self.section = "think"
            return list(self.think_open)
        if token in (self.text, self.tool_text, self.tool_error):
            self.in_header = False
            self.section = "text"
            return []
        if token == self.tool_json:
            self.in_header = False
            self.section = "tool"
            return list(self.tool_open)
        if token == self.end:
            section, self.section = self.section, None
            if section == "think":
                return list(self.think_close)
            if section == "tool":
                return list(self.tool_close)
            return []
        if self.in_header:
            # The tool name between the header and its content marker; the
            # JSON payload repeats it.
            return []
        return [token]


# ---------------------------------------------------------------------------
# Sampling on the GPU
# ---------------------------------------------------------------------------


def _sample_lazy(mx, logits, sampler, counts):
    """One token from a [1, V] FP32 row, as a lazy [1] int32 array."""
    temperature = float(getattr(sampler, "temperature", 0.0) or 0.0)
    top_p = float(getattr(sampler, "top_p", 1.0) or 1.0)
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    presence = float(getattr(sampler, "presence_penalty", 0.0) or 0.0)
    frequency = float(getattr(sampler, "frequency_penalty", 0.0) or 0.0)
    row = logits.reshape(-1)
    if counts is not None and (presence or frequency):
        row = row - presence * (counts > 0) - frequency * counts
    if temperature <= 0:
        return mx.argmax(row, keepdims=True).astype(mx.int32)
    row = row / temperature
    if 0 < top_k < row.shape[-1]:
        cand = mx.argpartition(-row, top_k - 1)[:top_k]
        vals = row[cand]
    else:
        cand = None
        vals = row
    if 0.0 < top_p < 1.0:
        order = mx.argsort(-vals)
        sorted_vals = vals[order]
        probs = mx.softmax(sorted_vals)
        keep = (mx.cumsum(probs) - probs) < top_p
        masked = mx.where(keep, sorted_vals, -mx.inf)
        pick = order[mx.random.categorical(masked)]
    else:
        pick = mx.random.categorical(vals)
    token = cand[pick] if cand is not None else pick
    return token.reshape(1).astype(mx.int32)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@dataclass
class _Timing:
    prefill_s: float = 0.0
    decode_s: float = 0.0
    steps: int = 0


def _prefill(runtime, prompt_ids, prefill_callback, abort_check, chunk: int = 2048):
    import mlx.core as mx

    from mtplx.models.inkling2.decode import StepCache

    model = runtime.model
    drafter = getattr(runtime, "drafter", None)
    capture_ids = drafter.target_layer_ids if drafter is not None else None
    dstate = drafter.new_state() if drafter is not None else None
    cache = model.make_cache()
    started = time.perf_counter()
    logits = None
    total = len(prompt_ids)
    for start in range(0, total, chunk):
        piece = prompt_ids[start : start + chunk]
        ids = mx.array([piece], dtype=mx.int32)
        if drafter is not None and total - start - len(piece) < drafter.slots:
            out = model.forward(ids, cache, capture_layer_ids=capture_ids)
            drafter.feed(dstate, out.hidden_states)
            logits = out.logits[0, -1:]
        else:
            if drafter is not None:
                dstate.fed += len(piece)
            logits = model(ids, cache=cache)[0, -1:]
        mx.eval(logits)
        if prefill_callback is not None:
            try:
                prefill_callback(
                    {
                        "phase": "progress",
                        "tokens_done": min(total, start + len(piece)),
                        "tokens_total": total,
                        "cached_tokens": 0,
                        "new_prefill_tokens": total,
                        "elapsed_s": time.perf_counter() - started,
                        "started_s": started,
                    }
                )
            except Exception:
                pass
        if abort_check is not None and abort_check():
            break
    step_cache = StepCache.from_reference(model, cache)
    return step_cache, logits, dstate, time.perf_counter() - started


def generate_inkling2_ar(
    runtime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    token_callback: Callable[[list[int]], None] | None = None,
    trace_label: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    prefill_callback: Callable[[dict[str, Any]], None] | None = None,
    repetition_stop: bool = False,
    first_token_logprobs_top_k: int | None = None,
    abort_check: Callable[[], bool] | None = None,
    **_unused: Any,
):
    import mlx.core as mx

    from mtplx.generation import (
        GenerationOutput,
        GenerationStats,
        _default_stop_tokens,
        _generation_rate_fields,
        _long_cycle_stop,
        _repetition_stop_config,
        _trim_repeated_suffix,
        first_token_logprobs,
    )

    del trace_label, trace_metadata
    tokenizer = runtime.tokenizer
    stop_ids = set(stop_token_ids) if stop_token_ids is not None else _default_stop_tokens(tokenizer)
    end_sampling = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
    if isinstance(end_sampling, int) and end_sampling >= 0:
        stop_ids.add(end_sampling)
    mx.random.seed(int(seed))
    started_all = time.perf_counter()
    if prefill_callback is not None:
        try:
            prefill_callback(
                {
                    "phase": "started",
                    "tokens_done": 0,
                    "tokens_total": len(prompt_ids),
                    "cached_tokens": 0,
                    "new_prefill_tokens": len(prompt_ids),
                    "elapsed_s": 0.0,
                    "started_s": started_all,
                }
            )
        except Exception:
            pass
    cache, logits, dstate, prefill_s = _prefill(runtime, prompt_ids, prefill_callback, abort_check)
    if prefill_callback is not None:
        try:
            prefill_callback(
                {
                    "phase": "completed",
                    "tokens_total": len(prompt_ids),
                    "tokens_done": len(prompt_ids),
                    "cached_tokens": 0,
                    "new_prefill_tokens": len(prompt_ids),
                    "elapsed_s": prefill_s,
                    "prompt_eval_time_s": prefill_s,
                    "prefill_tok_s": len(prompt_ids) / prefill_s if prefill_s > 0 else None,
                    "cache_hit": False,
                    "cache_miss_reason": "inkling2_no_session_bank",
                }
            )
        except Exception:
            pass

    engine = runtime.engine
    translate = _Translator(tokenizer)
    generated: list[int] = []  # raw model tokens
    wire: list[int] = []  # translated tokens (text and callbacks)
    repetition_config = _repetition_stop_config(bool(repetition_stop))
    repetition_long_cycle = _long_cycle_stop(repetition_config)
    repetition_result = None
    first_logprobs = None
    finish_reason = "length"
    penalties = bool(
        getattr(sampler, "presence_penalty", 0.0) or getattr(sampler, "frequency_penalty", 0.0)
    )
    vocab = int(logits.shape[-1])
    counts = mx.zeros((vocab,), dtype=mx.float32) if penalties else None

    def commit(token: int) -> bool:
        """Record one model token; True when generation must stop."""
        nonlocal repetition_result, finish_reason
        generated.append(token)
        if token in stop_ids:
            finish_reason = "stop"
            return True
        out = translate(token)
        if out:
            wire.extend(out)
            if token_callback is not None:
                token_callback(list(out))
        repetition_result = _trim_repeated_suffix(generated, repetition_config, repetition_long_cycle)
        if repetition_result is not None:
            finish_reason = "stop"
            return True
        if len(generated) >= max_tokens:
            return True
        return bool(abort_check is not None and abort_check())

    decode_started = time.perf_counter()
    token = _sample_lazy(mx, logits, sampler, counts)
    if first_token_logprobs_top_k is not None:
        first_id = int(token.item())
        first_logprobs = first_token_logprobs(
            logits[0], token_id=first_id, top_k=first_token_logprobs_top_k
        )
    # Keep one step in flight: step N + 1 is built and queued before step
    # N's token is read back for stop checks and streaming.
    pending = token
    mx.async_eval(pending)
    steps = 0
    while True:
        if counts is not None:
            counts = counts.at[pending].add(1.0)
        lg, _ = engine.step(cache, pending)
        steps += 1
        nxt = _sample_lazy(mx, lg[-1:], sampler, counts)
        mx.async_eval(nxt)
        if commit(int(pending.item())):
            break
        pending = nxt
    decode_s = time.perf_counter() - decode_started
    elapsed = time.perf_counter() - started_all
    text_tokens = list(wire)
    stats = GenerationStats(
        mode="ar",
        generated_tokens=len(generated),
        elapsed_s=elapsed,
        **_generation_rate_fields(
            generated_tokens=len(generated),
            elapsed_s=elapsed,
            prompt_eval_time_s=prefill_s,
        ),
        runtime_mtp_enabled=False,
        prompt_eval_time_s=prefill_s,
        cached_tokens=0,
        new_prefill_tokens=len(prompt_ids),
        session_cache_hit=False,
        cache_miss_reason="inkling2_no_session_bank",
        target_forward_time_s=prefill_s + decode_s,
        verify_time_s=decode_s,
        verify_calls=steps,
        peak_memory_bytes=int(mx.get_peak_memory()),
        repetition_stop_triggered=repetition_result is not None,
        repetition_stop_reason=(
            repetition_result.reason if repetition_result is not None else None
        ),
    )
    try:
        text = tokenizer.decode(text_tokens, clean_up_tokenization_spaces=False)
    except TypeError:
        text = tokenizer.decode(text_tokens)
    return GenerationOutput(
        tokens=text_tokens,
        text=text,
        stats=stats,
        final_state=None,
        finish_reason=finish_reason,
        first_token_logprobs=first_logprobs,
    )
