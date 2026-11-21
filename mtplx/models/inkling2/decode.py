# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 step engine: decode and speculative-verify steps of <= 16 tokens.

Runs the same math as ``Inkling2TextModel`` on its own cache layout with the
kernels of ``decode_kernels`` (about 13 graph nodes per layer visit instead
of ~90; a step on M1 is bound by the host's per-node cost). Prefill keeps
the reference graph and hands its cache over with ``StepCache.from_reference``.

Cache: one ``[2, Hkv, cap, D]`` K/V buffer per layer visit. Sliding visits
keep a ring of ``window + 16`` positions (a verify block may overwrite at
most 16 slots that no surviving query attends); full-attention visits grow.
Each visit also keeps its four short-conv states and, for rollback, the
windows of the last step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from . import decode_kernels as DK
from . import fused, kernels, moe_kernels

MAX_STEP_TOKENS = 16
_RING_SLACK = MAX_STEP_TOKENS
_GROW = 4096


@dataclass
class VisitCache:
    kv: mx.array
    cap: int
    ring: bool
    conv: list  # k, v, attn-out, mlp-out states
    windows: list | None = None


@dataclass
class StepCache:
    visits: list = field(default_factory=list)
    offset: int = 0
    last_tokens: int = 0  # tokens of the last step (rollback bound)

    @classmethod
    def from_reference(cls, model, ref_cache, position: int | None = None) -> "StepCache":
        """Adopt a reference (prefill) cache: one CacheList(KVCache,
        ArraysCache(4)) per visit.

        ``position``: tokens the sequence holds. A sliding visit's reference
        cache may keep only its newest keys (``to_reference``); they are the
        positions just before ``position``. Defaults to the full-attention
        visits' length.
        """
        text = model.model
        cache = cls()
        if position is None:
            lengths = {
                int(entry[0].offset)
                for (idx, _p), entry in zip(text._schedule, ref_cache)
                if not text.layers[idx].attn.window
            }
            if len(lengths) != 1:
                raise ValueError(f"reference cache visits disagree on length: {lengths}")
            position = lengths.pop()
        for (idx, _p), entry in zip(text._schedule, ref_cache):
            attn = text.layers[idx].attn
            kvc, conv = entry[0], entry[1]
            n = int(kvc.offset)
            hkv, d = attn.n_kv, attn.head_dim
            k = kvc.keys[..., :n, :][0] if kvc.keys is not None else mx.zeros((hkv, 0, d), mx.bfloat16)
            v = kvc.values[..., :n, :][0] if kvc.values is not None else mx.zeros((hkv, 0, d), mx.bfloat16)
            if attn.window:
                cap = attn.window + _RING_SLACK
                keep = min(n, cap)
                buf = mx.zeros((2, hkv, cap, d), dtype=mx.bfloat16)
                if keep:
                    # Position p lives in slot p % cap: the kept positions
                    # fill a run of slots modulo cap, written in <= 2 pieces.
                    start = int((position - keep) % cap)
                    first = min(keep, cap - start)
                    kk = k[:, n - keep :]
                    vv = v[:, n - keep :]
                    buf[0, :, start : start + first] = kk[:, :first]
                    buf[1, :, start : start + first] = vv[:, :first]
                    if keep > first:
                        buf[0, :, : keep - first] = kk[:, first:]
                        buf[1, :, : keep - first] = vv[:, first:]
                vc = VisitCache(buf, cap, True, [])
            else:
                if n != position:
                    raise ValueError(f"full-attention visit holds {n} of {position} tokens")
                cap = max(_GROW, ((n + _GROW) // _GROW) * _GROW)
                buf = mx.zeros((2, hkv, cap, d), dtype=mx.bfloat16)
                if n:
                    buf[0, :, :n] = k
                    buf[1, :, :n] = v
                vc = VisitCache(buf, cap, False, [])
            widths = (attn.dkv, attn.dkv, text.args.hidden_size, text.args.hidden_size)
            kc = text.args.sconv_kernel_size
            for slot, width in enumerate(widths):
                s = conv[slot] if conv is not None else None
                if s is None:
                    s = mx.zeros((kc - 1, width), dtype=mx.float32)
                vc.conv.append(s.reshape(kc - 1, width).astype(mx.float32))
            cache.visits.append(vc)
        cache.offset = int(position)
        mx.eval([v.kv for v in cache.visits] + [s for v in cache.visits for s in v.conv])
        return cache

    def to_reference(self, model) -> list:
        """A reference cache holding this state, to continue with the
        reference graph (long prefill chunks). Sliding visits get their ring
        in position order; their attention only reads the newest window."""
        from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

        out = []
        for vc in self.visits:
            if vc.ring:
                m = min(self.offset, vc.cap)
                slots = mx.arange(self.offset - m, self.offset) % vc.cap
                kv = mx.take(vc.kv, slots, axis=2)
            else:
                m = self.offset
                kv = vc.kv[:, :, :m]
            kvc = KVCache()
            if m:
                kvc.keys = mx.contiguous(kv[0][None])
                kvc.values = mx.contiguous(kv[1][None])
            kvc.offset = m
            conv = ArraysCache(4)
            for slot, state in enumerate(vc.conv):
                conv[slot] = state.reshape(1, *state.shape[-2:])
            out.append(CacheList(kvc, conv))
        return out

    def snapshot(self) -> "StepCache":
        """A copy that shares the buffers: a later in-place step on either
        copy makes MLX copy the shared buffer first, so both stay valid."""
        return StepCache(
            [VisitCache(v.kv, v.cap, v.ring, list(v.conv)) for v in self.visits],
            self.offset,
            0,
        )

    def nbytes(self) -> int:
        return sum(int(v.kv.nbytes) + sum(int(c.nbytes) for c in v.conv) for v in self.visits)

    def rollback(self, rejected: int) -> None:
        """Drop the last ``rejected`` tokens of the last step."""
        if rejected <= 0:
            return
        if rejected > self.last_tokens:
            raise ValueError("cannot roll back past the last step")
        keep = self.last_tokens - rejected
        self.offset -= rejected
        for vc in self.visits:
            new = []
            for w in vc.windows:
                width = w.shape[-1]
                flat = w.reshape(-1, width)
                kc1 = flat.shape[0] - self.last_tokens
                new.append(flat[keep : keep + kc1] * 1.0)
            vc.conv = new
        self.last_tokens = keep


def _packed(key, make):
    return DK.const(key, make)


def _arange(n: int) -> mx.array:
    return DK.const(("arange", n), lambda: mx.arange(n, dtype=mx.int32))


class StepEngine:
    """Decode / verify steps for one sequence on a ``StepCache``."""

    def __init__(self, model):
        self.model = model
        self.text = model.model
        args = self.text.args
        self.hidden = args.hidden_size
        self.rscale = kernels._scalar(float(args.hidden_size**-0.5))
        self.cscale = kernels._scalar(1.0 / (args.sconv_kernel_size + 1) ** 0.5)
        self._compiled: dict = {}
        self.compile = True

    # -- one layer visit -----------------------------------------------------
    def _visit(self, layer, h, blocks, bank, kv, conv, cap: int, ring: bool, loop_pass: bool, tokens: int, info):
        """One layer visit; returns (h, kv, conv states, conv windows)."""
        attn = layer.attn
        lid = id(layer)
        mixed = msg = None
        if layer.layer_idx == 0:
            r = DK.resmix(
                mode=0, hidden=self.hidden, tokens=tokens, blocks=[], query=None,
                rscale=self.rscale, h=h, norm_w=layer.attn_norm.weight, first=True,
            )
            hn, xq = r["hn"], r["out0"]
            xk = xv = xq
        else:
            query = layer.attn_input.query
            if attn.sliding:
                norm_w = layer.attn_norm.weight
            else:
                norm_w = _packed(
                    ("qkvnorm", lid),
                    lambda: mx.stack(
                        [layer.q_input_norm.weight, layer.k_input_norm.weight, layer.v_input_norm.weight]
                    ),
                )
            if not loop_pass:
                r = DK.resmix(
                    mode=0, hidden=self.hidden, tokens=tokens, blocks=blocks, query=query,
                    rscale=self.rscale, h=h, norm_w=norm_w,
                )
                hn = r["hn"]
                xq = r["out0"]
                xk = r.get("out1", xq)
                xv = r.get("out2", xq)
            else:
                r = DK.resmix(
                    mode=0, hidden=self.hidden, tokens=tokens, blocks=blocks, query=query,
                    rscale=self.rscale, h=h, norm_w=None, bank=bank,
                    mquery=layer.loop_message.query,
                )
                hn, msg = r["hn"], r["msg"]
                down_t, gf_t = layer._fused_mix
                mixed = fused.mix_rows(
                    r["outraw"], msg, down_t, gf_t, n=3, x_shared=attn.sliding,
                    msg_offset=0, w_offset=0,
                )
                if attn.sliding:
                    norm_w = _packed(("attnnorm", lid), lambda: layer.attn_norm.weight[None])
                xq, xk, xv = DK.norm_rows(mixed, norm_w, shared=attn.sliding)
        # Projections.
        if attn.sliding:
            pq = attn.in_proj(xq)
            pk = attn.in_proj(xk) if loop_pass else pq
            pv = attn.in_proj(xv) if loop_pass else pq
            dq, dkv, dr = attn.n_heads * attn.head_dim, attn.dkv, attn.n_heads * attn.d_rel
            layout = {"q": 0, "k": dq, "v": dq + dkv, "r": dq + 2 * dkv}
            gate_col = dq + 2 * dkv + dr
        else:
            pq = attn.q_proj(xq)
            pk = attn.k_proj(xk)
            pv = attn.v_proj(xv)
            dq, dr = attn.n_heads * attn.head_dim, attn.n_heads * attn.d_rel
            layout = {"q": 0, "r": dq, "k": 0, "v": 0, "dyn": attn.dkv}
            gate_col = dq + dr
        sink_col = gate_col + attn.n_heads * attn.head_dim
        q, kvn, rr, kwin, vwin, knext, vnext = DK.qkv_epilogue(
            attn, pq, pk, pv, conv[0], conv[1], info, tokens=tokens, layout=layout
        )
        # Cache write: slots from the step's position, so the traced step is
        # the same at every offset (ring visits wrap inside the scatter).
        slots = info + _arange(tokens)
        if ring:
            slots = slots % cap
        kv[:, :, slots] = kvn
        out = DK.attention(
            attn, q, kv, rr, pq, info, tokens=tokens,
            keys=attn.window if ring else cap,
            cap=cap, ring=ring, gate_col=gate_col, sink_col=sink_col,
        )
        a = attn.o_proj(out)
        if layer.is_boundary:
            blocks.append(hn)
        mlp_w = _packed(("mlpnorm", lid), lambda: layer.mlp_norm.weight[None])
        if not loop_pass:
            r2 = DK.resmix(
                mode=1, hidden=self.hidden, tokens=tokens, blocks=blocks,
                query=layer.mlp_input.query, rscale=self.rscale, h=h, norm_w=mlp_w,
                a_in=a, state=conv[2], conv_w=layer.attn_conv, cscale=self.cscale,
                boundary=layer.is_boundary,
            )
            x = r2["out0"]
        else:
            r2 = DK.resmix(
                mode=1, hidden=self.hidden, tokens=tokens, blocks=blocks,
                query=layer.mlp_input.query, rscale=self.rscale, h=h, norm_w=None,
                a_in=a, state=conv[2], conv_w=layer.attn_conv, cscale=self.cscale,
                boundary=layer.is_boundary,
            )
            down_t, gf_t = layer._fused_mix
            y = fused.mix_rows(
                r2["outraw"], msg, down_t, gf_t, n=1, x_shared=False, msg_offset=3, w_offset=3
            )
            (x,) = DK.norm_rows(y, mlp_w, shared=True)
        inds, weights = DK.route(layer.mlp, x)
        h_new, s3win, s3next = moe_kernels.moe_block(
            layer.mlp.experts, x, inds, weights, r2["resid"], layer.mlp_conv,
            conv[3], layer._conv_scale,
        )
        return (
            h_new,
            kv,
            [knext, vnext, r2["next_state"], s3next],
            [kwin, vwin, r2["win"], s3win],
        )

    # -- one step ------------------------------------------------------------
    def _step_fn(self, tokens: int, caps: tuple, rings: tuple, capture_ids: tuple):
        """The pure step for one shape: (ids, info, kvs, convs) ->
        (logits, kvs, convs, windows, captures)."""
        text = self.text

        def fn(ids, info, kvs, convs):
            h = text.embed_norm(text.embed(ids))
            blocks: list = []
            prelude = bank = None
            capture = {int(i): None for i in capture_ids}
            new_kvs, new_convs, windows = [], [], []
            core_start = text._core_start
            for visit, (idx, loop_pass) in enumerate(text._schedule):
                if idx == core_start and prelude is None:
                    prelude = (h, list(blocks))
                elif idx == core_start:
                    bank = (
                        blocks[len(prelude[1]) + 1 :],
                        mx.fast.rms_norm(h.astype(mx.float32), None, 1e-6),
                    )
                    h, blocks = prelude[0], list(prelude[1])
                if visit in capture:
                    capture[visit] = h
                h, kv, conv, win = self._visit(
                    text.layers[idx], h, blocks, bank if loop_pass else None,
                    kvs[visit], convs[visit], caps[visit], rings[visit],
                    bool(loop_pass), tokens, info,
                )
                new_kvs.append(kv)
                new_convs.append(conv)
                windows.append(win)
            r = DK.resmix(
                mode=0, hidden=self.hidden, tokens=tokens, blocks=blocks,
                query=text.final_input.query, rscale=self.rscale, h=h, norm_w=text.norm.weight,
            )
            out = r["out0"]
            if text.num_visits in capture:
                capture[text.num_visits] = out
            logits = self.model.logits_from_hidden(out)
            caps_out = [capture[int(i)][None] for i in capture_ids]
            return logits, new_kvs, new_convs, windows, caps_out

        return fn

    def step(self, cache: StepCache, tokens, capture_ids=None):
        """Run ``tokens`` (list or [T] array) at ``cache.offset``. Returns
        (logits [T, V] FP32, captures [1, T, H] per ``capture_ids``)."""
        ids = tokens if isinstance(tokens, mx.array) else mx.array(tokens, dtype=mx.int32)
        t = int(ids.shape[0])
        if t > MAX_STEP_TOKENS:
            raise ValueError(f"a step takes at most {MAX_STEP_TOKENS} tokens")
        for vc in cache.visits:
            if not vc.ring and cache.offset + t > vc.cap:
                grow = mx.zeros((2, vc.kv.shape[1], _GROW, vc.kv.shape[3]), dtype=vc.kv.dtype)
                vc.kv = mx.concatenate([vc.kv, grow], axis=2)
                vc.cap += _GROW
        capture_ids = tuple(int(i) for i in capture_ids or ())
        caps = tuple(vc.cap for vc in cache.visits)
        rings = tuple(vc.ring for vc in cache.visits)
        key = (t, caps, capture_ids)
        fn = self._compiled.get(key)
        if fn is None:
            raw = self._step_fn(t, caps, rings, capture_ids)
            fn = mx.compile(raw) if self.compile else raw
            self._compiled[key] = fn
        info = mx.array([cache.offset], dtype=mx.int32)
        kvs = [vc.kv for vc in cache.visits]
        convs = [vc.conv for vc in cache.visits]
        # Drop the cache's own references so the K/V buffers can be updated
        # in place (a still-referenced input would be copied).
        for vc in cache.visits:
            vc.kv = None
        logits, new_kvs, new_convs, windows, caps_out = fn(ids, info, kvs, convs)
        del kvs
        for vc, kv, conv, win in zip(cache.visits, new_kvs, new_convs, windows):
            vc.kv, vc.conv, vc.windows = kv, conv, win
        cache.offset += t
        cache.last_tokens = t
        return logits, caps_out
