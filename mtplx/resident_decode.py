"""Device-resident greedy depth-1 decode rounds (``MTPLX_M1_RESIDENT_DECODE=1``).

The stock round reads the verify's argmax on the host, then commits, drafts and
builds the next verify. On M1 Max that host chain leaves the GPU idle for about
3.7 ms of a 19.5 ms A3B round at 2K. Here everything a round decides stays on
the device:

* ``accepted = argmax(row 0) == draft``
* the next primary is ``where(accepted, argmax(row 1), argmax(row 0))``
* the target cache commit picks the captured GDN row ``accepted`` and moves the
  full-attention offsets back by ``1 - accepted``
* the draft history always appends ``(hidden row 0, draft)`` and then moves its
  offset back by ``1 - accepted`` (a rejected row is overwritten by the next
  draft), exactly what the stock rollback + append leaves

so round k+1 is queued before the host reads round k. The host reads each
round one round late, emits its tokens and stops the pipeline when a round
holds a stop token, would pass ``max_tokens`` or the caller asks to stop. On
exit every round the host has not accepted is discarded and the caches are
rewound to the start of the oldest such round (offsets and GDN leaves are kept
per round; KV rows past an offset are dead), so the stock loop resumes exactly
there and handles the stop with its own rules.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import mlx.core as mx


class ResidentUnsupported(RuntimeError):
    pass


def enabled() -> bool:
    return (os.environ.get("MTPLX_M1_RESIDENT_DECODE") or "").strip() == "1"


def max_context() -> int:
    """Context (prompt + completion) up to which rounds stay resident.

    A queued verify cannot donate the KV buffers the previous, still running
    round holds, so its slice_update copies them; the copy grows with the
    context. M1 Max A3B, greedy: +11% at 2K, +13% at 8K, +4.5% at 16K, +1% at
    30K (moe-s3/resident, 2026-10-14)."""
    try:
        return int(os.environ.get("MTPLX_M1_RESIDENT_MAX_CONTEXT") or 20480)
    except ValueError:
        return 20480


def _select_row(value: mx.array, index: mx.array) -> mx.array:
    """``value[:, index]`` for a device scalar ``index`` (no host read)."""

    shape = (int(value.shape[0]), 1) + tuple(int(s) for s in value.shape[2:])
    sel = mx.broadcast_to(index.reshape((1, 1) + (1,) * (value.ndim - 2)), shape)
    return mx.take_along_axis(value, sel, axis=1)[:, 0]


def device_commit(cache: list[Any], captures: dict, accepted: mx.array) -> None:
    """Commit a 2-row verify window keeping ``1 + accepted`` rows, on device."""

    from .cache_state import replace_recurrent_cache_state
    from .gdn_capture import (
        _maybe_contiguous_authoritative_gdn_leaf,
        linear_gated_delta_from_conv_tape_replay_steps,
    )

    if captures.get("__final_only__"):
        raise ResidentUnsupported("final_only_captures")
    keep = accepted.astype(mx.int32)
    trim = (1 - keep).astype(mx.int32)
    for layer_idx, entry in enumerate(cache):
        capture = captures.get(layer_idx)
        if capture is not None and hasattr(entry, "state"):
            if "conv_states" not in capture or int(capture.get("capture_start", 0)) != 0:
                raise ResidentUnsupported("capture_layout")
            conv_state = mx.contiguous(_select_row(capture["conv_states"], keep))
            if "tape" in capture:
                replayed = linear_gated_delta_from_conv_tape_replay_steps(
                    capture["tape"],
                    capture["conv_out"],
                    capture["g"],
                    capture["state_in"],
                    capture.get("gdn_meta", capture.get("gdn")),
                    keep + 1,
                )
                if replayed is None:
                    raise ResidentUnsupported("tape_replay")
                gdn_state = _maybe_contiguous_authoritative_gdn_leaf(replayed)
            else:
                gdn_state = mx.contiguous(_select_row(capture["states"], keep))
            replace_recurrent_cache_state(entry, [conv_state, gdn_state])
        elif capture is None and entry is not None and getattr(entry, "is_trimmable", lambda: False)():
            leaves = getattr(entry, "cache", None)
            if not (isinstance(leaves, list) and len(leaves) >= 3 and isinstance(leaves[2], mx.array)):
                raise ResidentUnsupported(f"attention_entry:{type(entry).__name__}")
            rollback = getattr(entry, "rollback_state", None)
            if rollback is not None:
                for slot in range(len(rollback)):
                    rollback[slot] = None
            leaves[2] = mx.maximum(leaves[2] - trim, mx.array(0, dtype=leaves[2].dtype))


def _snapshot(cache: list[Any], mtp_cache: list[Any]) -> list[tuple[Any, int, Any]]:
    """References that rewind both caches to this point (offsets, GDN leaves)."""

    snap: list[tuple[Any, int, Any]] = []
    for entry in list(cache) + list(mtp_cache):
        if entry is None:
            continue
        if hasattr(entry, "state") and not isinstance(getattr(entry, "cache", None), list):
            snap.append((entry, -1, list(entry.state)))
        elif hasattr(entry, "state") and isinstance(getattr(entry, "cache", None), list) and not (
            len(entry.cache) >= 3 and isinstance(entry.cache[2], mx.array) and entry.cache[2].ndim == 0
        ):
            snap.append((entry, -1, list(entry.cache)))
        elif isinstance(getattr(entry, "cache", None), list) and len(entry.cache) >= 3:
            snap.append((entry, 2, entry.cache[2]))
    return snap


def _restore(snap: list[tuple[Any, int, Any]]) -> None:
    from .cache_state import replace_recurrent_cache_state

    for entry, slot, value in snap:
        if slot == 2:
            entry.cache[2] = value
            rollback = getattr(entry, "rollback_state", None)
            if rollback is not None:
                for i in range(len(rollback)):
                    rollback[i] = None
        else:
            replace_recurrent_cache_state(entry, list(value))


@dataclass
class _Round:
    out: mx.array  # int32 [4]: draft, t0, t1, accepted
    snap: list  # rewind point = this round's start
    primary: mx.array
    hidden: mx.array
    logits: mx.array


@dataclass
class ResidentResult:
    tokens: list[int] = field(default_factory=list)  # newly committed, in order
    rounds: int = 0
    accepted: int = 0
    primary: int | None = None  # pending primary at the rewind point
    hidden: Any = None  # draft hidden at the rewind point
    logits: Any = None
    exit_reason: str = ""
    verify_time_s: float = 0.0
    draft_time_s: float = 0.0
    wait_s: float = 0.0
    wall_s: float = 0.0


class ResidentMTP:
    """Compiled draft / history steps over a tensor-offset MTP cache."""

    def __init__(self, rt: Any, mtp_cache: list[Any], *, mtp_hidden_variant: str, reserve: int, hidden: mx.array, primary: mx.array):
        from .attention_context import attention_phase
        from .generation import _device_core_state_tree
        from .graphbank import TensorOffsetKVCache, promote_kv_cache_offsets

        promote_kv_cache_offsets(mtp_cache, reserve_tokens=reserve, initial_reserve_tokens=reserve)
        if not mtp_cache or not all(isinstance(e, TensorOffsetKVCache) for e in mtp_cache):
            raise ResidentUnsupported("mtp_cache_not_tensor_offset")
        self.mtp_cache = mtp_cache
        variant = mtp_hidden_variant

        def draft(h, p):
            logits, _h = rt.draft_mtp(
                h, p, mtp_cache=mtp_cache, return_hidden=True,
                mtp_hidden_variant=variant, mtp_depth=1,
            )
            return mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1).astype(mx.int32)

        def commit_draft(h0, d, accepted, h, p):
            with attention_phase("ar_decode"):
                rt.update_mtp_cache(h0, d, mtp_cache=mtp_cache, mtp_hidden_variant=variant)
            trim = (1 - accepted).astype(mx.int32)
            offsets = []
            for entry in mtp_cache:
                entry.cache[2] = entry.cache[2] - trim
                for i in range(len(entry.rollback_state)):
                    entry.rollback_state[i] = None
                offsets.append(entry.cache[2])
            # The committed history offsets (before the next draft's row) are
            # the rewind point of the next round.
            return (draft(h, p), *offsets)

        tree = _device_core_state_tree(mtp_cache)
        self._draft = mx.compile(draft, inputs=tree, outputs=tree)
        self._commit_draft = mx.compile(commit_draft, inputs=tree, outputs=tree)

    def draft(self, h, p):
        return self._draft(h, p)

    def commit_draft(self, h0, d, accepted, h, p):
        """Returns the next draft and the committed history offsets."""
        out = self._commit_draft(h0, d, accepted, h, p)
        return out[0], list(out[1:])


def run_resident_rounds(
    *,
    cache: list[Any],
    mtp: ResidentMTP,
    verify: Callable[[mx.array, int], tuple[Any, Any, Any]],
    reserve_round: Callable[[int], bool],
    hidden: mx.array,
    primary: int,
    committed: int,
    max_tokens: int,
    stop_token_ids: set[int] | None,
    should_stop: Callable[[], bool] | None = None,
    on_tokens: Callable[[list[int]], None] | None = None,
    max_rounds: int | None = None,
    context_limit: int | None = None,
) -> ResidentResult:
    """Run pipelined rounds from ``primary`` (committed, not yet forwarded).

    ``committed`` is the host count of committed completion tokens (including
    ``primary``). ``verify(ids, committed_upper)`` runs the target verify on the
    two-token window; ``reserve_round(committed_upper)`` grows the paged KV for
    one round and returns False when it cannot.
    """

    stops = set(int(t) for t in (stop_token_ids or ()))
    res = ResidentResult()
    started = time.perf_counter()
    p_dev = mx.array([[int(primary)]], dtype=mx.int32)
    h_dev = hidden
    logits_dev = None
    snap = _snapshot(cache, mtp.mtp_cache)
    t0 = time.perf_counter()
    d_dev = mtp.draft(h_dev, p_dev)
    res.draft_time_s += time.perf_counter() - t0
    inflight: list[_Round] = []
    host_committed = int(committed)
    queued_upper = host_committed
    exit_reason = ""
    while True:
        # Queue one more round when it cannot pass max_tokens even if every
        # in-flight round accepts.
        can_queue = (
            not exit_reason
            and queued_upper + 2 < max_tokens
            and (max_rounds is None or res.rounds + len(inflight) < max_rounds)
        )
        if can_queue and context_limit is not None and queued_upper + 2 > context_limit:
            # Past this context a queued verify cannot donate the KV it
            # extends (the previous round still holds it), and the copy eats
            # the gain; the stock loop takes over.
            can_queue = False
            exit_reason = "context"
        if can_queue and not reserve_round(queued_upper + 2):
            can_queue = False
            exit_reason = "reservation"
        if can_queue:
            tv = time.perf_counter()
            ids = mx.concatenate([p_dev, d_dev], axis=1)
            logits, vhidden, captures = verify(ids, queued_upper)
            targets = mx.argmax(logits[0, :2, :], axis=-1).astype(mx.int32)
            accepted = (targets[0] == d_dev.reshape(())).astype(mx.int32)
            device_commit(cache, captures, accepted)
            keep = accepted.astype(mx.bool_)
            p_next = mx.where(keep, targets[1], targets[0]).reshape(1, 1)
            h_next = mx.where(keep, vhidden[:, 1:2, :], vhidden[:, 0:1, :])
            l_next = mx.where(keep, logits[:, 1, :], logits[:, 0, :])
            out = mx.concatenate([d_dev.reshape(1), targets, accepted.reshape(1)])
            res.verify_time_s += time.perf_counter() - tv
            td = time.perf_counter()
            next_snap = _snapshot(cache, [])
            d_next, mtp_offsets = mtp.commit_draft(vhidden[:, 0:1, :], d_dev, accepted, h_next, p_next)
            next_snap += [(e, 2, o) for e, o in zip(mtp.mtp_cache, mtp_offsets)]
            res.draft_time_s += time.perf_counter() - td
            mx.async_eval(out, p_next, h_next, d_next)
            inflight.append(_Round(out=out, snap=snap, primary=p_dev, hidden=h_dev, logits=logits_dev))
            snap, p_dev, h_dev, d_dev, logits_dev = next_snap, p_next, h_next, d_next, l_next
            queued_upper += 2
            if len(inflight) < 2:
                continue
        if not inflight:
            break
        # Read the oldest round (the GPU finished it or is on its last kernels).
        rnd = inflight[0]
        tw = time.perf_counter()
        d, a0, a1, acc = (int(v) for v in rnd.out.tolist())
        res.wait_s += time.perf_counter() - tw
        emitted = [d, a1] if acc else [a0]
        if any(t in stops for t in emitted) or host_committed + len(emitted) >= max_tokens:
            exit_reason = exit_reason or (
                "stop" if any(t in stops for t in emitted) else "max_tokens"
            )
            break
        inflight.pop(0)
        res.tokens.extend(emitted)
        if on_tokens is not None:
            on_tokens(emitted)
        res.rounds += 1
        res.accepted += acc
        host_committed += len(emitted)
        queued_upper = host_committed + 2 * len(inflight)
        if should_stop is not None and should_stop():
            exit_reason = exit_reason or "caller"
        if exit_reason and not can_queue:
            if not inflight:
                break
    # Rewind to the start of the oldest round the host did not accept.
    if inflight:
        first = inflight[0]
        _restore(first.snap)
        res.primary = int(first.primary.reshape(-1)[0].item())
        res.hidden = first.hidden
        res.logits = first.logits
    else:
        _restore(snap)
        res.primary = int(p_dev.reshape(-1)[0].item())
        res.hidden = h_dev
        res.logits = logits_dev
    # Hand the stock loop (and the session bank) python-int MTP containers.
    for i, entry in enumerate(mtp.mtp_cache):
        if hasattr(entry, "demote"):
            mtp.mtp_cache[i] = entry.demote()
    res.exit_reason = exit_reason or "drained"
    res.wall_s = time.perf_counter() - started
    if os.environ.get("MTPLX_RESIDENT_DEBUG"):
        n = max(1, res.rounds)
        print(
            f"[resident] rounds={res.rounds} acc={res.accepted} exit={res.exit_reason} "
            f"ms/round wall={1e3 * res.wall_s / n:.2f} verify_host={1e3 * res.verify_time_s / n:.2f} "
            f"draft_host={1e3 * res.draft_time_s / n:.2f} wait={1e3 * res.wait_s / n:.2f}",
            flush=True,
        )
    return res
