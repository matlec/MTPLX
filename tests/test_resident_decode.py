"""MTPLX_M1_RESIDENT_DECODE building blocks: device row selection, the
device-step tape replay and the rewind snapshot."""

import mlx.core as mx
import pytest

from mtplx import resident_decode as rd
from mtplx.gdn_capture import (
    _linear_gated_delta_from_conv_tape_replay,
    linear_gated_delta_from_conv_tape_replay_steps,
)


def test_select_row_matches_host_index():
    value = mx.random.normal((1, 2, 3, 5))
    for index in (0, 1):
        got = rd._select_row(value, mx.array(index, dtype=mx.int32))
        assert mx.array_equal(got, value[:, index]).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="metal kernel")
def test_device_step_tape_replay_is_bit_identical():
    B, T, Hk, Hv, Dk, Dv = 1, 2, 2, 4, 64, 64
    key_dim = Hk * Dk
    conv_dim = 2 * key_dim + Hv * Dv
    meta = {
        "conv_dim": conv_dim,
        "head_k_dim": Dk,
        "head_v_dim": Dv,
        "num_k_heads": Hk,
        "num_v_heads": Hv,
        "key_dim": key_dim,
    }
    conv_out = mx.random.normal((B, T, conv_dim)).astype(mx.bfloat16)
    tape = mx.random.normal((B, T, Hv, Dv)).astype(mx.float32)
    g = mx.random.uniform(0.5, 1.0, (B, T, Hv)).astype(mx.float32)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32)
    for steps in (1, 2):
        ref = _linear_gated_delta_from_conv_tape_replay(
            tape, conv_out, g, state, meta, steps=steps
        )
        got = linear_gated_delta_from_conv_tape_replay_steps(
            tape, conv_out, g, state, meta, mx.array(steps, dtype=mx.int32)
        )
        assert ref is not None and got is not None
        assert mx.array_equal(ref, got).item()


class _Attn:
    def __init__(self, offset):
        self.cache = [mx.zeros((1, 2, 8, 4)), mx.zeros((1, 2, 8, 4)), mx.array(offset, dtype=mx.int32)]
        self.rollback_state = [None, None, None]

    def is_trimmable(self):
        return True


def test_device_commit_trims_attention_offset_by_rejection():
    for accepted, expected in ((1, 10), (0, 9)):
        entry = _Attn(10)
        rd.device_commit([entry], {}, mx.array(accepted, dtype=mx.int32))
        assert int(entry.cache[2].item()) == expected


def test_snapshot_restores_attention_offset():
    entry = _Attn(7)
    snap = rd._snapshot([entry], [])
    entry.cache[2] = mx.array(11, dtype=mx.int32)
    rd._restore(snap)
    assert int(entry.cache[2].item()) == 7


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_M1_RESIDENT_DECODE", raising=False)
    assert rd.enabled() is False
    monkeypatch.setenv("MTPLX_M1_RESIDENT_MAX_CONTEXT", "1234")
    assert rd.max_context() == 1234
