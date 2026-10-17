# MTPLX for M1 Macs

[![Release](https://img.shields.io/github/v/release/shunya1810/MTPLX)](https://github.com/shunya1810/MTPLX/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

English | [日本語](README.ja.md)

An unofficial build of [MTPLX](https://github.com/youssofal/MTPLX) for Macs with an **M1-family chip**, tuned for
long prompts and coding-agent sessions with **Qwen3.8-27B** and **Qwen3.6-35B-A3B**. Like upstream, it decodes with
the model's own multi-token prediction (MTP) heads, keeping the model's output distribution, and serves an OpenAI
and Anthropic compatible API. This build is MTPLX 2.12.2 plus changes that only switch on for M1-family GPUs.

- **On an M3 or newer, use [upstream MTPLX](https://github.com/youssofal/MTPLX).** Nothing here was measured there.
- Not affiliated with or endorsed by the MTPLX project. The changes were developed with AI assistance (Claude) and
  measured on one MacBook Pro M1 Max with 64 GB. Upstream's README is in [README.upstream.md](README.upstream.md).

## Key numbers

M1 Max 64 GB, 8-bit KV, three-turn conversations over a long prompt
([apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench); method in
[Performance](#performance)).

| Model | This build | Compared with |
|---|---|---|
| Qwen3.6-35B-A3B (MoE) | decode 111.8 tok/s at 2K, 86.4 at 32K, 61.2 at 128K | Splash 1.3.0-m1.1: 96.1, 63.8, 43.9 |
| Qwen3.8-27B (dense) | decode 29.4 tok/s at 2K, 28.1 at 32K, 19.8 at 128K | upstream MTPLX 2.12.0: 23.6, 14.0, 5.0 |
| A3B in an agent session | a file rewrite 219 tok/s (111 without context-copy); a short tool result after 160K tokens of history read in 0.42 s | v2.12.2-m1.5: 111 tok/s, 0.75 s |

## Install and run

You need an M1-family Mac (measured: M1 Max 64 GB, macOS 26), Xcode with the Metal toolchain (`xcrun metal` must
work), and the [MTPLX app](https://github.com/youssofal/MTPLX) opened once: this build uses the app's Python runtime,
and the models are downloaded from the app's model screen.

```bash
git clone --branch v2.12.2-m1.7 https://github.com/shunya1810/MTPLX.git MTPLX-m1
cd MTPLX-m1
scripts/build-m1-native-extensions.sh   # about a minute; ends with "mfa_attn: ok" and "ane_ffn: ok"
```

Quit the MTPLX app (two engines do not fit in memory side by side), then start the server with one of the models:

```bash
# Qwen3.6-35B-A3B (MoE): the fastest choice for agents on M1
PYTHONPATH="$PWD" "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" -m mtplx.cli serve \
  --model "$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16" \
  --context-window 262144 --paged-kv-quantization q8

# Qwen3.8-27B (dense): replace the --model path with
#   "$HOME/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"
```

Check that it answers (the server listens on `http://127.0.0.1:8000`):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "mtplx", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 64}'
```

To run this build behind the MTPLX app's own UI instead, use `scripts/open-mtplx-app-with-checkout.command`.
More: [docs/m1max-longctx/USAGE.md](docs/m1max-longctx/USAGE.md). Without the native extensions the server still runs,
with a slower GPU-only prefill.

## Use it with a coding agent

Any client that speaks the OpenAI or Anthropic API works. Two examples:

- **OpenCode**: add a provider to `~/.config/opencode/opencode.json`:

  ```json
  {
    "provider": {
      "mtplx": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "MTPLX (M1)",
        "options": { "baseURL": "http://127.0.0.1:8000/v1", "apiKey": "local", "timeout": false },
        "models": { "mtplx": { "name": "MTPLX", "reasoning": true, "tool_call": true,
                               "limit": { "context": 262144, "output": 32768 } } }
      }
    }
  }
  ```

- **Claude Code**: `ANTHROPIC_BASE_URL=http://127.0.0.1:8000 claude`

Qwen3.6 has no reasoning-effort levels: thinking is on or off. Qwen3.8 keeps `low` to `xhigh`.

## Models

| Model | Memory at peak | Notes |
|---|---|---|
| `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16` (MoE, 3B active) | 27–31 GB | tuned for M1 since v2.12.2-m1.4; GPU-resident decode rounds, Neural Engine prefill, context-copy |
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` (dense) | 32–45 GB at 2K–128K, 50 GB at 256K | the original target of this build |
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16` (dense, 8-bit) | about 9 GB more | measured; decode about 10–15% slower than Speed |

## Performance

All numbers: one MacBook Pro M1 Max 64 GB, macOS 26, 8-bit KV, temperature 0, the
[apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench) three-turn conversation (turn 1 reads
the long prompt cold, turns 2–3 add about 300 tokens, up to 256 generated tokens per turn), engines run alternately in
one session.

**Qwen3.6-35B-A3B**, v2.12.2-m1.5 against Splash 1.3.0-m1.1 (2026-10-14; on this benchmark m1.6 decodes 0.6% below m1.5 at 2K and 3.8% below at 32K, where
context-copy sometimes copies the wrong continuation)

| | 2K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| decode, tok/s (this build) | **111.8** | **110.8** | **86.4** | **76.5** | **61.2** |
| decode, tok/s (Splash) | 96.1 | 80.8 | 63.8 | 57.7 | 43.9 |
| whole conversation (this build) | **9.3 s** | **14.4 s** | **41.3 s** | **1.5 min** | **4.3 min** |
| whole conversation (Splash) | 9.5 s | 18.1 s | 60.7 s | 2.5 min | 7.7 min |

With thinking on and short prompts (greedy, 1,024 tokens), Splash decodes 9–15% faster: its DFlash 2 block drafts fit
reasoning text and boilerplate code best.

**Qwen3.8-27B**, v2.12.2-m1.1 (256K: m1.2) against upstream MTPLX 2.12.0

| | 2K | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|
| decode, tok/s (this build) | **29.4** | **29.2** | **28.1** | **25.5** | **19.8** | **14.2** |
| decode, tok/s (upstream) | 23.6 | 22.5 | 14.0 | 8.7 | 5.0 | — |
| whole conversation (this build) | **33 s** | **61 s** | **3.3 min** | **7.3 min** | **18.4 min** | **59.4 min** |
| whole conversation (upstream) | 42 s | 86 s | 5.0 min | 11.3 min | 30.6 min | — |

**Agent sessions (Qwen3.6-35B-A3B, v2.12.2-m1.6)**: rewriting an 11K-token file with thinking on (temperature 0.6)
decodes at 219 tok/s with context-copy against 111 without; writing new text, 111 against 105. A tool result of 42 tokens
appended to 160K tokens of cached history is read in 0.42 s (0.75 s in m1.5). The server's own work before each
request dropped from 80–100 ms to about 2 ms on long agent transcripts.

## What this build changes

- **Prefill**: part of the work runs on the Neural Engine while the GPU computes the rest (the approach is adapted from
  [Splash](https://github.com/incoai/splash)); attention over the cached prefix uses
  [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) blocks tuned for M1; short additions to
  a long history read the 8-bit KV in place.
- **Decode**: fewer host round trips per round; on the A3B, rounds are queued on the GPU before the host reads the
  previous one (greedy text identical, sampled output follows the same distribution), and file rewrites copy their
  unchanged lines from the prompt (context-copy, exact matches only).
- **Memory**: 8-bit paged KV, long prompts prefilled straight into the 8-bit pages, memory estimates measured on this
  build; a 256K-token conversation runs at the defaults in 64 GB.
- **Agent sessions** (any Mac): sessions found again across token seams and history rewrites, tool-argument whitespace
  kept, a request retried when macOS stops a long GPU command, restores that resume from the deepest saved state.

Every change with its measurements: [CHANGES.ja.md](docs/m1max-longctx/CHANGES.ja.md) (Japanese) and the
[release notes](https://github.com/shunya1810/MTPLX/releases).

## Switches

Each M1 default can be turned off on its own; `MTPLX_M1_LONG_CONTEXT=0` turns all of them off.

| Variable | Default on M1 | Turns off |
|---|---|---|
| `MTPLX_ANE_PREFILL=0` / `MTPLX_ANE_PREFILL_MOE=0` | on | the Neural Engine prefill (dense / MoE) |
| `MTPLX_MFA_PREFILL=0` | on | the flash-attention prefill |
| `MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN=0` | on | the chained draft |
| `MTPLX_M1_PAGED_PREFILL=0` | on from 32K | prefill straight into 8-bit pages |
| `MTPLX_M1_ENGINE_RAM_FRACTION=0.75` | 13/16 | the larger engine memory budget |
| `MTPLX_M1_TURBO_DEFAULT=0` | turbo for dense 4/8-bit | the turbo profile default |
| `MTPLX_M1_MOE_LANE=0` | on for MoE models | every MoE launch default (the MoE and A3B rows here included) |
| `MTPLX_M1_RESIDENT_DECODE=0` | on for the A3B | GPU-resident decode rounds |
| `MTPLX_CONTEXT_COPY=0` | on for the A3B | context-copy |

## Known limitations

- Measured on one M1 Max 64 GB. M1 Pro, M1 Ultra and other memory sizes run the same code but are untested.
- The Neural Engine path uses Apple's private `AppleNeuralEngine.framework` and may stop working after a macOS update;
  prefill then falls back to the GPU.
- The Neural Engine prefill computes with int8 activations, so it is a close approximation of the GPU prefill, not
  bit-identical (on the A3B, 98.25% of next tokens agree over 24,576 positions). `MTPLX_ANE_PREFILL=0` /
  `MTPLX_ANE_PREFILL_MOE=0` keep prefill on the GPU.
- Thinking models sometimes reason for a very long time on hard agent tasks (we saw 131,072-token answers on the A3B in
  long sessions). Reasoning is not capped by default; `MTPLX_THINKING_BUDGET=<tokens>` caps it on requests that carry
  tools.
- Some upstream tests assume upstream's defaults and fail with the M1 defaults on; most pass with
  `MTPLX_M1_LONG_CONTEXT=0`.

## License and credits

Apache-2.0, as upstream ([LICENSE](LICENSE), [NOTICE](NOTICE)). The Neural Engine client and FFN program layout are
adapted from Splash (`native_extensions/ane_ffn/NOTICE`); the flash-attention kernel follows metal-flash-attention
(`native_extensions/mfa_attn/MFA_LICENSE.txt`).
