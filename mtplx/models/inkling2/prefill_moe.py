# SPDX-License-Identifier: Apache-2.0
"""Expert-segmented NVFP4 / MXFP8 gather matmul for Inkling 2 prefill.

The FP-carrier counterpart of ``mtplx.kernels.moe_seg_gather_m1``: MLX's
sorted ``gather_qmm`` tiles the expert-sorted rows 32 at a time across expert
boundaries, so a tile spanning several experts reruns its matmul per expert.
Here each threadgroup owns up to BM sorted rows of ONE expert (tiles are cut
per expert; a threadgroup finds its expert by binary search over the tile
prefix table) and runs MLX's own ``fp_qmm_t_impl`` tile body on them. The
first call of each shape is compared bit for bit with the stock op, and a
mismatch keeps the stock op for that shape.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from mtplx.kernels.moe_seg_gather_m1 import _closure, _inline

BN = 64
BK = 32
THREADS = 128
# Rows per tile: short experts waste less with 16-row tiles.
_NARROW_BM = 16
_BM = 32
_NARROW_MAX_ROWS = 4096
# Routed rows below this keep the stock op (decode / verify / short chunks).
MIN_ROWS = 1024

_HEADERS = (
    "mlx/backend/metal/kernels/steel/gemm/gemm.h",
    "mlx/backend/metal/kernels/quantized_utils.h",
    "mlx/backend/metal/kernels/fp_quantized.h",
)
_PREAMBLE = "mlx/backend/metal/kernels/utils.h"

_CANARY: dict[tuple, bool] = {}
STATS = {"calls": 0, "canaries": 0, "canary_failures": 0, "header_failures": 0}


@lru_cache(maxsize=1)
def _headers() -> str | None:
    """MLX's steel GEMM and FP-quantized templates plus a copy of
    ``fp_qmm_t_impl`` taking its scalars by value (a per-threadgroup row
    count cannot bind the original's ``constant`` references)."""
    try:
        root = Path(mx.__file__).resolve().parent / "include"
        seen: set[str] = set()
        _closure(root, _PREAMBLE, seen)
        out: list[str] = []
        for rel in _HEADERS:
            if not (root / rel).is_file():
                raise FileNotFoundError(rel)
            _inline(root, rel, seen, out)
        text = "\n".join(out)
        start = text.index("METAL_FUNC void fp_qmm_t_impl(")
        template_start = text.rindex("template <", 0, start)
        end = text.index("\n}\n", start) + 3
        body = text[template_start:end].replace("const constant int&", "const int")
        local = body.replace("void fp_qmm_t_impl(", "void mtplx_fp_qmm_t_rows(")
    except Exception as exc:  # noqa: BLE001 - fall back to the stock op
        STATS["header_failures"] += 1
        print(
            f"[inkling2-prefill-moe] MLX headers unusable ({type(exc).__name__}: {exc}); "
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
    constexpr int pack_factor = get_pack_factor<8, BITS>();
    constexpr int bytes_per_pack = get_bytes_per_pack();
    const size_t row_bytes = size_t(KD) * bytes_per_pack / pack_factor;
    const size_t row_groups = KD / GS;
    const device uint32_t* expert_w = (const device uint32_t*)(
        (const device uint8_t*)w + size_t(expert) * ND * row_bytes);
    const device uint8_t* expert_scales = scales + size_t(expert) * ND * row_groups;
    const uint3 column_tile = uint3(threadgroup_position_in_grid.x, 0, 0);
    mtplx_fp_qmm_t_rows<T, GS, BITS, true, BM, BK, BN>(
        expert_w, expert_scales,
        x + size_t(row0) * KD, y + size_t(row0) * ND, Xs, Ws,
        KD, ND, tile_rows, KD, column_tile,
        thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""


@lru_cache(maxsize=1)
def _kernel():
    headers = _headers()
    if headers is None:
        return None
    return mx.fast.metal_kernel(
        name="mtplx_inkling2_fp_seg_gather_qmm",
        input_names=["x", "w", "scales", "tile_start", "row_start"],
        output_names=["y"],
        source=_SOURCE,
        header=headers,
        ensure_row_contiguous=True,
    )


def applies(proj, routed_rows: int) -> bool:
    return (
        routed_rows >= MIN_ROWS
        and getattr(proj, "mode", None) in ("nvfp4", "mxfp8")
        and mx.default_device() == mx.gpu
        and _kernel() is not None
    )


def _segmented(x, proj, sorted_experts):
    experts, n = int(proj.weight.shape[0]), int(proj.weight.shape[1])
    rows, k = int(x.shape[0]), int(x.shape[-1])
    bm = _NARROW_BM if rows <= _NARROW_MAX_ROWS else _BM
    counts = mx.zeros((experts,), dtype=mx.int32).at[sorted_experts.astype(mx.int32)].add(1)
    zero = mx.zeros((1,), dtype=mx.int32)
    row_start = mx.concatenate([zero, mx.cumsum(counts)])
    tile_start = mx.concatenate([zero, mx.cumsum((counts + bm - 1) // bm)])
    (y,) = _kernel()(
        inputs=[x.reshape(rows, k), proj.weight, proj.scales, tile_start, row_start],
        template=[
            ("T", x.dtype), ("GS", int(proj.group_size)), ("BITS", int(proj.bits)),
            ("KD", k), ("ND", n), ("ED", experts), ("BM", bm), ("BN", BN), ("BK", BK),
        ],
        grid=((n // BN) * THREADS, rows // bm + experts, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )
    return y


def gather_qmm(x, proj, sorted_experts):
    """``proj(x, sorted_experts, sorted_indices=True)`` for expert-sorted rows
    ``x`` [rows, ..., K] (``model._gather_sort``'s layout). Returns
    [rows, ..., N], or None when the stock op must serve."""
    rows = int(x.size // x.shape[-1])
    key = (tuple(int(d) for d in proj.weight.shape), proj.mode, str(x.dtype))
    verdict = _CANARY.get(key)
    if verdict is False:
        return None
    y = _segmented(x.reshape(rows, x.shape[-1]), proj, sorted_experts)
    y = y.reshape(*x.shape[:-1], y.shape[-1])
    if verdict is None:
        STATS["canaries"] += 1
        ref = proj(x, sorted_experts, True)
        verdict = bool(mx.array_equal(y, ref).item())
        _CANARY[key] = verdict
        if not verdict:
            STATS["canary_failures"] += 1
            diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item())
            print(
                f"[inkling2-prefill-moe] segmented kernel differs for {key} (max {diff:.3g}); "
                "the stock op serves it",
                file=sys.stderr,
                flush=True,
            )
            return None
    STATS["calls"] += 1
    return y
