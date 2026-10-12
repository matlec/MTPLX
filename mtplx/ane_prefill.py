"""Prefill FFN split between the GPU and the Neural Engine (M1 family), ``MTPLX_ANE_PREFILL=1``.

Why (2026-10-12, attn-s4): on an M1 Max the ANE runs int8-activation matmuls at ~9.7 TFLOPS, as fast
as the GPU's MMA peak, and the two barely slow each other (GPU 9.30 -> 9.15, ANE 9.84 -> 9.69 TFLOPS
side by side). Splash 1.3 splits each dense layer's FFN by intermediate channel between them and its
prefill runs 1.35-1.44x faster with the split (measured here, 7.7K-60K prompts).

What: for prefill chunks of at least ``MIN_ROWS`` rows, the trailing ``MTPLX_ANE_PREFILL_CHANNELS``
(default 7680) intermediate channels of every Qwen3Next MLP run on the ANE as one W16A8 program
(native_extensions/ane_ffn): the input rotated by a 512-block Hadamard and quantized to int8 per token,
the weights dequantized to fp16 and rotated alike, the intermediate rotated and quantized per token
inside the program (Splash's numerics). The GPU runs the leading channels meanwhile; their down
projections add up. Decode, verify and short chunks stay on the GPU.

Numerics: not exact. ANE part 1.57% relative RMS against fp32 on synthetic layers; on Youssofal
Speed-FP16 the simulated split changed 1/6 greedy 160-token answers (the KV 8-bit -> fp16 switch
changes 2/6).
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

MIN_ROWS = 512
_STATE: dict[str, Any] = {"installed": False, "ffn": None, "calls": 0, "fallbacks": 0, "error": None}
_LOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get("MTPLX_ANE_PREFILL", "0") == "1"


def _ext():
    path = Path(__file__).resolve().parents[1] / "native_extensions" / "ane_ffn"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import mtplx_ane_ffn

    return mtplx_ane_ffn


def _had(a: mx.array) -> mx.array:
    s = a.shape
    return mx.hadamard_transform(a.reshape(*s[:-1], s[-1] // 512, 512), scale=1 / math.sqrt(512)).reshape(s)


def _qparams(lin: Any) -> tuple[mx.array, mx.array, mx.array | None, int, int]:
    return lin.weight, lin.scales, getattr(lin, "biases", None), int(lin.group_size), int(lin.bits)


def _layer(self: Any, cut: int) -> dict[str, Any]:
    cached = getattr(self, "_mtplx_ane_split", None)
    if cached is not None and cached["cut"] == cut:
        return cached
    g, u, d = _qparams(self.gate_proj), _qparams(self.up_proj), _qparams(self.down_proj)
    pd = 32 // d[4]
    entry = {
        "cut": cut,
        "g": (g[0][:cut], g[1][:cut], None if g[2] is None else g[2][:cut], g[3], g[4]),
        "u": (u[0][:cut], u[1][:cut], None if u[2] is None else u[2][:cut], u[3], u[4]),
        "d": (d[0][:, : cut // pd], d[1][:, : cut // d[3]], None if d[2] is None else d[2][:, : cut // d[3]], d[3], d[4]),
        "ga": (g[0][cut:], g[1][cut:], None if g[2] is None else g[2][cut:], g[3], g[4]),
        "ua": (u[0][cut:], u[1][cut:], None if u[2] is None else u[2][cut:], u[3], u[4]),
        "da": (d[0][:, cut // pd :], d[1][:, cut // d[3] :], None if d[2] is None else d[2][:, cut // d[3] :], d[3], d[4]),
    }
    self._mtplx_ane_split = entry
    return entry


def _qmm(x: mx.array, p: tuple) -> mx.array:
    w, s, b, gs, bits = p
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)


def _deq(p: tuple) -> mx.array:
    w, s, b, gs, bits = p
    return mx.dequantize(w, s, b, group_size=gs, bits=bits).astype(mx.float16)


def _prep_x(x2: mx.array) -> tuple[mx.array, mx.array]:
    xr = _had(x2.astype(mx.float32))
    s = mx.maximum(mx.max(mx.abs(xr), axis=-1), 1e-6) / 127.0
    xq = mx.clip(mx.round(xr / s[:, None]), -127, 127).astype(mx.int8)
    return xq, (s * 128.0).astype(mx.float16)


def install_gdn(model: Any) -> dict[str, Any]:
    """GDN in-projections split: the GPU computes in_proj_qkv's leading rows while the ANE computes its
    trailing ``MTPLX_ANE_PREFILL_GDN_QKV_TAIL`` rows (default 2048) and all of in_proj_z, in one program
    (``MTPLX_ANE_PREFILL_GDN=1``). in_proj_z's call then returns the ANE's rows for the same input."""

    ext = _ext()
    if ext.unavailable():
        return {"gdn": "unavailable"}
    tail = int(os.environ.get("MTPLX_ANE_PREFILL_GDN_QKV_TAIL", "2048"))
    layers = getattr(getattr(model, "language_model", model), "model", None)
    layers = getattr(layers, "layers", None) or getattr(getattr(model, "model", None), "layers", [])
    tagged = 0
    for layer in layers:
        gdn = getattr(layer, "linear_attn", None)
        qkv, z = getattr(gdn, "in_proj_qkv", None), getattr(gdn, "in_proj_z", None)
        if qkv is None or z is None or not hasattr(qkv, "scales") or not hasattr(z, "scales"):
            continue
        if (qkv.bits, qkv.group_size) != (z.bits, z.group_size) or getattr(qkv, "biases", None) is None:
            continue
        n_qkv = qkv.weight.shape[0]
        qkv._mtplx_ane_gdn = {"z": z, "cut": n_qkv - tail}
        z._mtplx_ane_gdn_z = True
        tagged += 1
    if not tagged:
        return {"gdn": "no layers"}
    cls = type(next(l.linear_attn.in_proj_qkv for l in layers if hasattr(getattr(l, "linear_attn", None), "in_proj_qkv")))
    orig = cls.__call__
    stash: dict[int, mx.array] = {}

    def call(self: Any, x: mx.array) -> mx.array:
        if getattr(self, "_mtplx_ane_gdn_z", False):
            hit = stash.pop(id(x), None)
            if hit is not None:
                return hit
            return orig(self, x)
        role = getattr(self, "_mtplx_ane_gdn", None)
        rows = math.prod(x.shape[:-1])
        if role is None or rows < MIN_ROWS:
            return orig(self, x)
        z = role["z"]
        cut = role["cut"]
        H = x.shape[-1]
        n_qkv, n_z = self.weight.shape[0], z.weight.shape[0]
        n_ane = (n_qkv - cut) + n_z
        lin = _STATE.get("gdn_lin")
        if lin is None:
            lin = ext.AneLinear(H, n_ane, [512, 1024, 1536, 2048])
            _STATE["gdn_lin"] = lin
            logger.info("[ane-prefill] GDN program ready (%.1f s)", lin.compile_seconds)
        if rows > lin.max_rows:
            return orig(self, x)
        trace = os.environ.get("MTPLX_ANE_PREFILL_TRACE") == "1"
        if trace:
            import time as _time

            t0 = _time.perf_counter()
            mx.eval(x)
            t1 = _time.perf_counter()
        x2 = x.reshape(rows, H)
        xq, tx = _prep_x(x2)
        ws = [self.weight[cut:], self.scales[cut:], self.biases[cut:], z.weight, z.scales, z.biases]
        st = lin.stage(xq, tx, ws, [0, n_qkv - cut], self.bits, self.group_size)
        mx.eval(st)
        gpu = mx.quantized_matmul(x2, self.weight[:cut], self.scales[:cut], self.biases[:cut], transpose=True,
                                  group_size=self.group_size, bits=self.bits)
        if trace:
            t2 = _time.perf_counter()
        mx.async_eval(gpu)
        lin.run(rows)
        if trace:
            t3 = _time.perf_counter()
            mx.eval(gpu)
            t4 = _time.perf_counter()
            acc = _STATE.setdefault("gtrace", [0.0, 0.0, 0.0, 0.0, 0])
            acc[0] += t1 - t0; acc[1] += t2 - t1; acc[2] += t3 - t2; acc[3] += t4 - t3; acc[4] += 1
            if acc[4] % 48 == 0:
                n = acc[4]
                print(f"[ane-prefill] GDN in_proj per layer: before {acc[0]/n*1e3:.1f} ms  prep+stage {acc[1]/n*1e3:.1f} ms  ane {acc[2]/n*1e3:.1f} ms  gpu-after {acc[3]/n*1e3:.1f} ms (rows {rows})", flush=True)
                _STATE["gtrace"] = [0.0, 0.0, 0.0, 0.0, 0]
        o = lin.output()[:, :rows].T.astype(x.dtype)
        _STATE["gdn_calls"] = _STATE.get("gdn_calls", 0) + 1
        stash[id(x)] = o[:, n_qkv - cut :].reshape(*x.shape[:-1], n_z)
        return mx.concatenate([gpu, o[:, : n_qkv - cut]], axis=-1).reshape(*x.shape[:-1], n_qkv)

    cls.__call__ = call
    return {"gdn": tagged, "qkv_tail": tail}


def install(model: Any = None) -> dict[str, Any]:
    if _STATE["installed"]:
        return dict(_STATE)
    ext = _ext()
    why = ext.unavailable()
    if why:
        _STATE["error"] = why
        return dict(_STATE)
    from mlx_lm.models import qwen3_next as qn

    orig = qn.Qwen3NextMLP.__call__
    channels = int(os.environ.get("MTPLX_ANE_PREFILL_CHANNELS", "7680"))

    def call(self: Any, x: mx.array) -> mx.array:
        rows = math.prod(x.shape[:-1])
        if rows < MIN_ROWS or not hasattr(self.gate_proj, "scales"):
            return orig(self, x)
        H = x.shape[-1]
        inter = self.gate_proj.weight.shape[0]
        ffn = _STATE["ffn"]
        if ffn is None:
            with _LOCK:
                if _STATE["ffn"] is None:
                    _STATE["ffn"] = ext.AneFfn(H, channels, [512, 1024, 1536, 2048])
                    logger.info("[ane-prefill] program ready (%.1f s)", _STATE["ffn"].compile_seconds)
                ffn = _STATE["ffn"]
        if rows > ffn.max_rows:
            _STATE["fallbacks"] += 1
            return orig(self, x)
        cut = inter - channels
        L = _layer(self, cut)
        trace = os.environ.get("MTPLX_ANE_PREFILL_TRACE") == "1"
        if trace:
            import time as _time

            t0 = _time.perf_counter()
            mx.eval(x)
            t1 = _time.perf_counter()
        x2 = x.reshape(rows, H)
        xq, tx = _prep_x(x2)
        ga, ua, da = L["ga"], L["ua"], L["da"]
        if ga[2] is None or ga[4] not in (4, 8) or not (ga[4] == ua[4] == da[4] and ga[3] == ua[3] == da[3]):
            stage = [ffn.stage(xq, tx, _had(_deq(ga)), _had(_deq(ua)), _had(_deq(da)))]
        else:
            stage = [ffn.stage_input(xq, tx), ffn.stage_weights_q(list(ga[:3]), list(ua[:3]), list(da[:3]), ga[4], ga[3])]
        mx.eval(*stage)
        if trace:
            t2 = _time.perf_counter()
        h = nn.silu(_qmm(x2, L["g"])) * _qmm(x2, L["u"])
        y_gpu = _qmm(h, L["d"])
        mx.async_eval(y_gpu)
        ffn.run(rows)
        if trace:
            t3 = _time.perf_counter()
            mx.eval(y_gpu)
            t4 = _time.perf_counter()
            acc = _STATE.setdefault("trace", [0.0, 0.0, 0.0, 0.0, 0])
            acc[0] += t1 - t0; acc[1] += t2 - t1; acc[2] += t3 - t2; acc[3] += t4 - t3; acc[4] += 1
            if acc[4] % 64 == 0:
                n = acc[4]
                print(f"[ane-prefill] per layer: before-mlp {acc[0]/n*1e3:.1f} ms  prep+stage {acc[1]/n*1e3:.1f} ms  ane {acc[2]/n*1e3:.1f} ms  gpu-part-after-ane {acc[3]/n*1e3:.1f} ms  (rows {rows})", flush=True)
                _STATE["trace"] = [0.0, 0.0, 0.0, 0.0, 0]
        o = ffn.output()
        y_ane = (o[:H, :rows] * o[H : H + 1, :rows]).T
        _STATE["calls"] += 1
        return (y_gpu + y_ane.astype(y_gpu.dtype)).reshape(x.shape)

    qn.Qwen3NextMLP.__call__ = call
    _STATE["installed"] = True
    _STATE["channels"] = channels
    if model is not None and os.environ.get("MTPLX_ANE_PREFILL_GDN", "0") == "1":
        _STATE["gdn"] = install_gdn(model)
    return {k: v for k, v in _STATE.items() if k not in ("ffn", "gdn_lin")}


def stats() -> dict[str, Any]:
    return {k: v for k, v in _STATE.items() if k != "ffn"}
