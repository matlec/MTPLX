"""M1 family: the 35B-A3B MoE defaults to capture_commit; dense models are untouched."""

from types import SimpleNamespace

from mtplx.commands import public as p


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
