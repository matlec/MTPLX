"""M1 family: the expert-sorted quantized gather with tiles cut per expert.

``mx.gather_qmm(..., sorted_indices=True)`` (MLX 0.32.2) tiles the sorted rows
32 at a time across expert boundaries; a tile whose rows belong to several
experts reruns the whole tile matmul once per expert and keeps a slice of
each. On a prefill chunk of the 35B-A3B (2,048 tokens x top-8 = 16,384 rows,
~64 per expert) that and its tile shape hold the routed gate/up projection to
4.3 TFLOPS on an M1 Max, against 7.7 for a plain quantized matmul of the same
size.

Here each threadgroup owns up to 32 sorted rows of ONE expert (each expert's
run is cut into its own tiles, and a threadgroup finds its expert by binary
search over a prefix table of tiles per expert) and runs MLX's own
``qmm_t_impl`` tile body on them (the installed MLX's templates, read from its
headers; only the address space of the scalar arguments differs). The
reduction order of every output element is MLX's, so the result is
bit-identical to the stock call; a first-use canary per shape compares the two
outputs bit for bit and keeps the stock op for that shape if they ever differ.

Measured on M1 Max (MLX 0.32.2, 2026-10-14, random routing, 2,048 tokens,
top-8, 256 experts, 4-bit g64): gate/up [1024, 2048] 16.1 -> 11.4 ms
(6.1 TFLOPS), down [2048, 512] 6.9 -> 5.9 ms. Tile 32x64x32 was the best of
16/32/64 rows, 64/128 columns and 32/64 depth on both shapes.

``MTPLX_M1_MOE_SEG_GATHER`` = 1/0 forces it; unset, it is on for the M1 GPU
family (``cache_state.m1_long_context_defaults``).
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx

BM = 32
BN = 64
BK = 32
THREADS = 128
#: Fewer routed rows keep MLX's kernel (decode and verify widths).
MIN_ROWS = 1024
_ENV = "MTPLX_M1_MOE_SEG_GATHER"

_HEADERS = (
    "mlx/backend/metal/kernels/steel/gemm/gemm.h",
    "mlx/backend/metal/kernels/quantized_utils.h",
    "mlx/backend/metal/kernels/quantized.h",
)
_PREAMBLE = "mlx/backend/metal/kernels/utils.h"
_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
_PRAGMA_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$")

_STATS: dict[str, int] = {"calls": 0, "canaries": 0, "canary_failures": 0, "header_failures": 0}
_CANARY: dict[tuple, bool] = {}


def stats() -> dict[str, int]:
    return dict(_STATS)


def enabled() -> bool:
    try:
        from mtplx import nax_detect

        from mtplx.moe_sorted_gather import guard_active

        if nax_detect.nax_hardware_available() or guard_active():
            return False
    except Exception:
        pass
    raw = (os.environ.get(_ENV) or "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    try:
        from mtplx.cache_state import m1_long_context_defaults

        return bool(m1_long_context_defaults())
    except Exception:
        return False


def _closure(root: Path, rel: str, seen: set[str]) -> None:
    if rel in seen:
        return
    seen.add(rel)
    for line in (root / rel).read_text().splitlines():
        match = _INCLUDE.match(line)
        if match:
            _closure(root, match.group(1), seen)


def _inline(root: Path, rel: str, seen: set[str], out: list[str]) -> None:
    if rel in seen:
        return
    seen.add(rel)
    for line in (root / rel).read_text().splitlines():
        match = _INCLUDE.match(line)
        if match:
            _inline(root, match.group(1), seen, out)
        elif not _PRAGMA_ONCE.match(line):
            out.append(line)


@lru_cache(maxsize=1)
def _mlx_headers() -> str | None:
    """MLX's steel GEMM and quantized templates plus a copy of ``qmm_t_impl``
    taking its scalars by value (the original binds ``constant`` references,
    which a per-threadgroup row count cannot be)."""

    try:
        root = Path(mx.__file__).resolve().parent / "include"
        for rel in _HEADERS + (_PREAMBLE,):
            if not (root / rel).is_file():
                raise FileNotFoundError(rel)
        seen: set[str] = set()
        _closure(root, _PREAMBLE, seen)
        out: list[str] = []
        for rel in _HEADERS:
            _inline(root, rel, seen, out)
        text = "\n".join(out)
        start = text.index("METAL_FUNC void qmm_t_impl(")
        template_start = text.rindex("template <", 0, start)
        end = text.index("\n}\n", start) + 3
        local = (
            text[template_start:end]
            .replace("void qmm_t_impl(", "void mtplx_qmm_t_rows(")
            .replace("const constant int&", "const int")
        )
    except Exception as exc:
        _STATS["header_failures"] += 1
        print(
            f"[moe-seg-gather] MLX kernel headers unusable ({type(exc).__name__}: {exc}); "
            "the stock sorted gather runs instead",
            file=sys.stderr,
            flush=True,
        )
        return None
    return text + "\n" + local


_SOURCE = r"""
    threadgroup T Xs[BM * (BK + 16 / sizeof(T))];
    threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
    const int tile = int(threadgroup_position_in_grid.y);
    if (tile >= tile_start[ED]) {
        return;
    }
    int lo = 0;
    int hi = ED - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (tile_start[mid] <= tile) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int expert = lo;
    const int row0 = row_start[expert] + (tile - tile_start[expert]) * BM;
    const int tile_rows = min(BM, row_start[expert + 1] - row0);
    constexpr int pack_factor = get_pack_factor<BITS, 8>();
    constexpr int bytes_per_pack = get_bytes_per_pack<BITS>();
    const size_t row_bytes = size_t(KD) * bytes_per_pack / pack_factor;
    const size_t row_groups = KD / GS;
    const device uint32_t* expert_w = (const device uint32_t*)(
        (const device uint8_t*)w + size_t(expert) * ND * row_bytes);
    const device T* expert_scales = scales + size_t(expert) * ND * row_groups;
    const device T* expert_biases = biases + size_t(expert) * ND * row_groups;
    const uint3 column_tile = uint3(threadgroup_position_in_grid.x, 0, 0);
    mtplx_qmm_t_rows<T, GS, BITS, true, BM, BK, BN>(
        expert_w, expert_scales, expert_biases,
        x + size_t(row0) * KD, y + size_t(row0) * ND, Xs, Ws,
        KD, ND, tile_rows, KD, column_tile,
        thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""


@lru_cache(maxsize=1)
def _kernel():
    headers = _mlx_headers()
    if headers is None:
        return None
    return mx.fast.metal_kernel(
        name="mtplx_m1_moe_seg_gather_qmm",
        input_names=["x", "w", "scales", "biases", "tile_start", "row_start"],
        output_names=["y"],
        source=_SOURCE,
        header=headers,
        ensure_row_contiguous=True,
    )


def applies(
    x: Any,
    w: Any,
    scales: Any,
    biases: Any,
    rhs_indices: Any,
    *,
    lhs_indices: Any = None,
    transpose: bool = True,
    group_size: int | None,
    bits: int | None,
    mode: str = "affine",
    sorted_indices: bool,
) -> bool:
    if not sorted_indices or not transpose or lhs_indices is not None:
        return False
    if mode != "affine" or biases is None or rhs_indices is None:
        return False
    if group_size not in (32, 64, 128) or bits not in (4, 8):
        return False
    if x.dtype not in (mx.float16, mx.bfloat16) or scales.dtype != x.dtype:
        return False
    if x.ndim != 3 or int(x.shape[1]) != 1 or w.ndim != 3:
        return False
    rows = int(x.shape[0])
    k = int(x.shape[2])
    n = int(w.shape[1])
    if rows < MIN_ROWS or int(rhs_indices.size) != rows:
        return False
    if k % BK or n % BN or k % int(group_size):
        return False
    return enabled() and _kernel() is not None


def _segmented(x, w, scales, biases, rhs_indices, *, group_size: int, bits: int):
    experts = int(w.shape[0])
    n = int(w.shape[1])
    rows = int(x.shape[0])
    k = int(x.shape[2])
    counts = mx.zeros((experts,), dtype=mx.int32).at[rhs_indices.astype(mx.int32)].add(1)
    zero = mx.zeros((1,), dtype=mx.int32)
    row_start = mx.concatenate([zero, mx.cumsum(counts)])
    tile_start = mx.concatenate([zero, mx.cumsum((counts + BM - 1) // BM)])
    (y,) = _kernel()(
        inputs=[x.reshape(rows, k), w, scales, biases, tile_start, row_start],
        template=[
            ("T", x.dtype), ("GS", int(group_size)), ("BITS", int(bits)),
            ("KD", k), ("ND", n), ("ED", experts), ("BM", BM), ("BN", BN), ("BK", BK),
        ],
        grid=((n // BN) * THREADS, rows // BM + experts, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(rows, 1, n)


def gather_qmm(x, w, scales, biases, rhs_indices, *, group_size: int, bits: int):
    """``mx.gather_qmm(x, ..., transpose=True, sorted_indices=True)`` for
    ``x`` [rows, 1, K]; the first call of each shape is checked bit for bit
    against the stock op (and the stock op serves that shape on a mismatch)."""

    key = (tuple(int(d) for d in w.shape), str(x.dtype), int(group_size), int(bits))
    verdict = _CANARY.get(key)
    if verdict is False:
        return None
    y = _segmented(x, w, scales, biases, rhs_indices, group_size=group_size, bits=bits)
    if verdict is None:
        _STATS["canaries"] += 1
        from mtplx.moe_sorted_gather import guarded_gather_qmm

        ref = guarded_gather_qmm(
            x, w, scales, biases, rhs_indices=rhs_indices, transpose=True,
            group_size=group_size, bits=bits, sorted_indices=True,
        )
        same = bool(mx.array_equal(y, ref).item())
        _CANARY[key] = same
        if not same:
            _STATS["canary_failures"] += 1
            print(
                f"[moe-seg-gather] first call of {key} differs from mx.gather_qmm; "
                "the stock op serves that shape",
                file=sys.stderr,
                flush=True,
            )
            return ref
    _STATS["calls"] += 1
    return y


_SWITCH_MARK = "_mtplx_m1_seg_gather"


def install_switch_linear() -> bool:
    """Route mlx-lm's quantized switch linear (the A3B's routed down
    projection) through :func:`gather_qmm` for sorted prefill-width calls."""

    if not enabled():
        return False
    from mlx_lm.models import switch_layers

    cls = switch_layers.QuantizedSwitchLinear
    current = cls.__call__
    if getattr(current, _SWITCH_MARK, False):
        return True

    def call(self, x, indices, sorted_indices=False):
        params = self
        biases = params.get("biases") if hasattr(params, "get") else getattr(self, "biases", None)
        if applies(
            x, self["weight"], self["scales"], biases, indices,
            group_size=self.group_size, bits=self.bits,
            mode=getattr(self, "mode", "affine"), sorted_indices=sorted_indices,
        ):
            y = gather_qmm(
                x, self["weight"], self["scales"], biases, indices,
                group_size=self.group_size, bits=self.bits,
            )
            if y is not None:
                if "bias" in self:
                    y = y + mx.expand_dims(self["bias"][indices], -2)
                return y
        return current(self, x, indices, sorted_indices=sorted_indices)

    setattr(call, _SWITCH_MARK, True)
    cls.__call__ = call
    return True


def _reset_for_tests() -> None:
    _CANARY.clear()
    for key in _STATS:
        _STATS[key] = 0
