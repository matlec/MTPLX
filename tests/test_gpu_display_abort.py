"""Classification of macOS GPU display aborts (retried before any token)."""

from mtplx.errors import is_allocation_failure, is_gpu_display_abort


def test_display_abort_and_hang_are_recognized():
    for text in (
        "[METAL] Command buffer execution failed: Impacting Interactivity "
        "(0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)",
        "[METAL] Command buffer execution failed: Caused GPU Hang Error "
        "(00000003:kIOGPUCommandBufferCallbackErrorHang).",
        "Discarded (victim of GPU error/recovery) "
        "(00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)",
    ):
        assert is_gpu_display_abort(RuntimeError(text))


def test_memory_and_other_errors_are_not_display_aborts():
    oom = RuntimeError(
        "[METAL] Command buffer execution failed: Insufficient Memory "
        "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
    )
    assert is_allocation_failure(oom)
    assert not is_gpu_display_abort(oom)
    assert not is_gpu_display_abort(RuntimeError("shape mismatch"))
    assert not is_gpu_display_abort(ValueError("Impacting Interactivity"))
