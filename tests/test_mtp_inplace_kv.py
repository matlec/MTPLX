"""In-place MTP KV append used by the sampled draft chain (MTPLX_MTP_INPLACE_KV)."""

import pytest

mx = pytest.importorskip("mlx.core")


def _writer():
    from mtplx.kernels.mfa_prefill import _ext

    ext = _ext()
    fn = getattr(ext, "inplace_row_write", None) if ext is not None else None
    if fn is None:
        pytest.skip("mfa_attn native extension not built")
    return fn


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_row_write_matches_slice_update(dtype):
    write = _writer()
    dt = getattr(mx, dtype)
    base = mx.random.normal((1, 4, 64, 32)).astype(dt)
    mx.eval(base)
    ref = base * 1
    src = mx.random.normal((1, 2, 4, 32)).astype(dt).transpose(0, 2, 1, 3)
    out = write(base, src, 17)
    ref[..., 17:19, :] = src
    mx.eval(out, ref)
    assert mx.array_equal(out, ref).item()


def test_chain_append_matches_stock_kv_cache():
    _writer()
    from mlx_lm.models.cache import KVCache

    from mtplx import qwen4_draft_device_chain as chain

    def run(inplace):
        cache = KVCache()
        keys = mx.random.normal((1, 2, 5, 16), key=mx.random.key(0)).astype(mx.float16)
        cache.update_and_fetch(keys, keys)
        mx.eval(cache.keys, cache.values)
        outs = []
        ctx = chain.inplace_mtp_kv() if inplace else None
        if ctx is not None:
            ctx.__enter__()
        try:
            for depth in range(3):
                k = mx.full((1, 2, 1, 16), depth + 1, mx.float16)
                kk, vv = cache.update_and_fetch(k, k * 2)
                outs.append(kk.sum() + vv.sum())
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        mx.eval(*outs)
        return [o.item() for o in outs], cache.offset

    assert run(True) == run(False)
    assert not chain._INPLACE["active"]
