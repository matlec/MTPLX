"""M1 family: the 35B-A3B MoE defaults to capture_commit; dense models are untouched."""

import os
from types import SimpleNamespace

import pytest

from mtplx.commands import public as p


@pytest.fixture(autouse=True)
def _private_environ(monkeypatch):
    env = dict(os.environ)
    for key in p._M1_MOE_LANE_ENV:
        env.pop(key, None)
    env.pop("MTPLX_M1_MOE_LANE", None)
    monkeypatch.setattr(os, "environ", env)
    return env


def _args(**kw):
    return SimpleNamespace(_cli_flags=set(kw.pop("cli", ())), _injected_default_flags=set(), depth=3,
                           verify_strategy=None, draft_temperature=None, draft_top_p=None, draft_top_k=None, **kw)


def _moe_id():
    return p.QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID


def test_m1_moe_uses_capture_commit(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    monkeypatch.delenv("MTPLX_M1_MOE_VERIFY_STRATEGY", raising=False)
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    assert a.verify_strategy == "capture_commit" and a.depth == 1


def test_off_m1_and_override_keep_target_prefix(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    assert a.verify_strategy == "target_prefix"
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    monkeypatch.setenv("MTPLX_M1_MOE_VERIFY_STRATEGY", "target_prefix")
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    assert a.verify_strategy == "target_prefix"


def test_explicit_flag_and_dense_models_are_untouched(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    a = _args(cli={"verify-strategy"}); a.verify_strategy = "trim_commit"
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    assert a.verify_strategy == "trim_commit"
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, "mtplx-qwen38-27b-optimized-speed-fp16")
    assert a.verify_strategy is None and a.depth == 3


def test_m1_moe_lane_env_only_for_m1_capture_commit(_private_environ):
    env = _private_environ
    env["MTPLX_M1_LONG_CONTEXT"] = "1"
    env["MTPLX_QWEN_MOE_PACK_GATE_UP"] = "0"
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    child = a._child_env_defaults
    assert child["MTPLX_A3B_WHOLE_MOE_FUSION"] == "1"
    assert child["MTPLX_M1_MOE_DEFERRED_DRAFT"] == "1"
    assert "MTPLX_QWEN_MOE_PACK_GATE_UP" not in child  # an explicit env wins
    assert "MTPLX_LINEAR_GDN_FROM_CONV_TGY" not in child
    assert "MTPLX_A3B_WHOLE_MOE_FUSION" not in env  # never this process's env


@pytest.mark.parametrize(
    "setup",
    [
        {"MTPLX_M1_LONG_CONTEXT": "0"},
        {"MTPLX_M1_LONG_CONTEXT": "1", "MTPLX_M1_MOE_LANE": "0"},
        {"MTPLX_M1_LONG_CONTEXT": "1", "MTPLX_M1_MOE_VERIFY_STRATEGY": "target_prefix"},
    ],
)
def test_m1_moe_lane_env_stays_off(_private_environ, setup):
    env = _private_environ
    env.update(setup)
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    assert not a._child_env_defaults
    dense = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(dense, "mtplx-qwen38-27b-optimized-speed-fp16")
    assert not getattr(dense, "_child_env_defaults", None)


def test_ane_prefill_skips_moe_models():
    from types import SimpleNamespace as NS

    from mtplx import ane_prefill as ap

    class Qwen3NextSparseMoeBlock:  # noqa: N801 - mirrors the class name it detects
        pass

    moe = NS(named_modules=lambda: [("layers.0.mlp", Qwen3NextSparseMoeBlock())])
    dense = NS(named_modules=lambda: [("layers.0.mlp", object())])
    assert ap._is_moe(moe) and not ap._is_moe(dense) and not ap._is_moe(None)
