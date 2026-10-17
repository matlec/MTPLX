"""A stream's final chunk keeps its mtplx_stats under the byte cap (OpenCode 2 SseEventTooLarge)."""

import json

from mtplx.server import openai as srv


def _big_public(n_extra=450):
    public = {k: 123456.789 for k in list(srv.PUBLIC_MTPLX_STATS_KEYS)[:n_extra]}
    public["completion_tokens"] = 77
    public["decode_tok_s"] = 91.5
    public["accepted_by_depth"] = [5, 4, 3]
    public["raw_decode_tok_s"] = 90.0            # not in the public list: stamped by hand
    public["finish_stop_origin"] = "context_copy"
    return public


def test_small_envelope_is_untouched():
    small = {"completion_tokens": 3, "decode_tok_s": 50.0}
    assert srv._cap_stream_stats(small, {}, cap=12000) is small


def test_large_envelope_fits_and_keeps_the_core_keys():
    public = _big_public()
    assert len(json.dumps(public)) > 12000
    out = srv._cap_stream_stats(public, {"user-agent": "ai-sdk"}, cap=12000)
    assert len(json.dumps(out)) <= 12000 + 100       # the cap plus the trimmed marker
    for key in ("completion_tokens", "decode_tok_s", "accepted_by_depth", "raw_decode_tok_s", "finish_stop_origin"):
        assert out[key] == public[key]
    assert out["stats_trimmed"]["of"] == len(public)
    assert out["stats_trimmed"]["kept"] < len(public)


def test_the_mac_app_keeps_everything():
    public = _big_public()
    assert srv._cap_stream_stats(public, {"X-MTPLX-Client": "mtplx_app"}, cap=12000) is public


def test_cap_zero_turns_it_off():
    public = _big_public()
    assert srv._cap_stream_stats(public, {}, cap=0) is public


def test_env_default_and_override(monkeypatch):
    monkeypatch.delenv("MTPLX_STREAM_STATS_MAX_BYTES", raising=False)
    assert srv._stream_stats_cap_bytes() == 12000
    monkeypatch.setenv("MTPLX_STREAM_STATS_MAX_BYTES", "0")
    assert srv._stream_stats_cap_bytes() == 0
    monkeypatch.setenv("MTPLX_STREAM_STATS_MAX_BYTES", "junk")
    assert srv._stream_stats_cap_bytes() == 12000
