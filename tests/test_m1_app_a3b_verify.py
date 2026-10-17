from types import SimpleNamespace

import mtplx.commands.public as public


def _args(**kw):
    base = dict(_cli_flags={"verify-strategy", "depth"}, depth=1, verify_strategy="target_prefix",
                app_launch_id="ABC", scheduler_mode="serial", chat_template_profile=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _fp16_id():
    return public.QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID


def test_app_launch_on_m1_takes_the_capture_commit_lane(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    monkeypatch.delenv("MTPLX_M1_MOE_VERIFY_STRATEGY", raising=False)
    args = _args()
    public._apply_qwen36_35b_optimized_speed_defaults(args, _fp16_id())
    assert args.verify_strategy == "capture_commit"


def test_cli_typed_target_prefix_is_kept(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    monkeypatch.delenv("MTPLX_M1_MOE_VERIFY_STRATEGY", raising=False)
    args = _args(app_launch_id=None)
    public._apply_qwen36_35b_optimized_speed_defaults(args, _fp16_id())
    assert args.verify_strategy == "target_prefix"


def test_batched_scheduler_and_overrides_keep_target_prefix(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    monkeypatch.delenv("MTPLX_M1_MOE_VERIFY_STRATEGY", raising=False)
    args = _args(scheduler_mode="ar_batch")
    public._apply_qwen36_35b_optimized_speed_defaults(args, _fp16_id())
    assert args.verify_strategy == "target_prefix"
    monkeypatch.setenv("MTPLX_M1_MOE_VERIFY_STRATEGY", "target_prefix")
    args = _args()
    public._apply_qwen36_35b_optimized_speed_defaults(args, _fp16_id())
    assert args.verify_strategy == "target_prefix"


def test_off_the_m1_family_nothing_changes(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    monkeypatch.delenv("MTPLX_M1_MOE_VERIFY_STRATEGY", raising=False)
    args = _args()
    public._apply_qwen36_35b_optimized_speed_defaults(args, _fp16_id())
    assert args.verify_strategy == "target_prefix"
