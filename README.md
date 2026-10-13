# MTPLX for M1 Macs (long context)

[![Release](https://img.shields.io/github/v/release/shunya1810/MTPLX)](https://github.com/shunya1810/MTPLX/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

English | [日本語](README.ja.md)

An unofficial build of [MTPLX](https://github.com/youssofal/MTPLX) for Macs with an
M1-family chip, tuned for long prompts and multi-turn agent sessions with
Qwen3.8-27B. MTPLX runs the model's own multi-token prediction (MTP) heads for
exact speculative decoding behind an OpenAI and Anthropic compatible server and
a native Mac app. This build is based on MTPLX 2.12.2 and changes only what
runs on M1-family GPUs; on other Macs it behaves like upstream.

**On an M3 or newer, use [upstream MTPLX](https://github.com/youssofal/MTPLX).**
Nothing here was measured or tuned there.

Not affiliated with or endorsed by the MTPLX project. The changes were developed
with AI assistance (Claude) and measured on one MacBook Pro M1 Max with 64 GB.
The original README is in [README.upstream.md](README.upstream.md).

## What this build adds

- **Prompt reading on the Neural Engine.** Part of every layer's MLP runs on the
  M1's Neural Engine while the GPU computes the rest, and the two halves of a
  chunk are pipelined. Cold prompts read 1.4–1.5× faster. Uses Apple's private
  `AppleNeuralEngine.framework` (the approach is adapted from
  [Splash](https://github.com/incoai/splash), Apache-2.0); if it is unavailable,
  prefill falls back to the GPU.
- **M1 attention kernels.** Prefill attention over the cached prefix on
  [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
  blocks tuned for M1 (head size 256), and verify attention and projections
  sized for the M1 GPU.
- **Fewer host round trips in decode.** Batched reads of the verify rows and
  paged-KV offsets, and three draft depths chained on the GPU with the MTP cache
  appended in place: about 5% per decode round at 32K.
- **Long contexts in 64 GB.** 8-bit paged KV, prompts of 32K+ prefilled straight
  into the 8-bit pages, memory-guard bills measured on this build, and an engine
  budget of 13/16 of RAM: a 256K-token conversation runs at the defaults.
- **Agent-session fixes** (any Mac): session identity across token seams, tool
  argument whitespace, KV growth refusals ending an answer instead of failing it.
  See [CHANGES.ja.md](docs/m1max-longctx/CHANGES.ja.md).

## Measured

Three-turn conversations over a long prompt (turn 1 reads it cold, turns 2–3 add
~300 tokens, up to 256 generated tokens per turn), Qwen3.8-27B, 8-bit KV,
temperature 0, M1 Max 64 GB. From the
[apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench)
(this build v2.12.2-m1.1, 256K v2.12.2-m1.2; MTPLX upstream 2.12.0 run the same
way). The benchmark also compares other engines.

**Whole conversation**

| | 2K | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|
| **this build** | **33 s** | **61 s** | **3.3 min** | **7.3 min** | **18.4 min** | **59.4 min** |
| MTPLX upstream | 42 s | 86 s | 5.0 min | 11.3 min | 30.6 min | — |

**Decode**, tok/s, mean of the three turns

| | 2K | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|
| **this build** | **29.4** | **29.2** | **28.1** | **25.5** | **19.8** | **14.2** |
| MTPLX upstream | 23.6 | 22.5 | 14.0 | 8.7 | 5.0 | — |

Follow-up turns start in 0.5–2.9 s (upstream: 0.6–3.5 s at 2K–128K). Peak wired
memory is 32–45 GB at 2K–128K and 50 GB at 256K. With sampling (temperature 1.0)
decode is 31.6 / 28.3 / 23.9 tok/s at 2K / 32K / 64K (upstream 23.2 / 14.2 / 8.9).

## Install

Needs an M1-family Mac (measured on an M1 Max 64 GB, macOS 26), the
[MTPLX app](https://github.com/youssofal/MTPLX) opened once (its runtime and the
model are used), and Xcode with the Metal toolchain for the native extensions.

```bash
git clone --branch v2.12.2-m1.2 https://github.com/shunya1810/MTPLX.git MTPLX-m1
cd MTPLX-m1
scripts/build-m1-native-extensions.sh   # about a minute; prints mfa_attn: ok / ane_ffn: ok
```

Then either run it inside the app (`scripts/open-mtplx-app-with-checkout.command`)
or start the server alone:

```bash
PYTHONPATH="$PWD" "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" -m mtplx.cli serve \
  --model "$HOME/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16" \
  --context-window 262144 --paged-kv-quantization q8
```

Details: [docs/m1max-longctx/USAGE.md](docs/m1max-longctx/USAGE.md)
([日本語](docs/m1max-longctx/USAGE.ja.md)). Without the native extensions the
server still runs, without the Neural Engine prefill, the flash-attention
prefill and the chained draft.

## Models

| Model | Status |
|---|---|
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` | measured, recommended (turbo profile by default on M1) |
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16` | measured (8-bit; 9 GB more memory, ~10–15% slower decode) |
| Qwen3.6-35B-A3B (MoE) | runs, not tuned yet |

## Switches

Every M1 default can be turned off on its own; `MTPLX_M1_LONG_CONTEXT=0` turns all
of them off.

| Variable | Default on M1 | Turns off |
|---|---|---|
| `MTPLX_ANE_PREFILL=0` | on | the Neural Engine prefill |
| `MTPLX_MFA_PREFILL=0` | on | flash-attention prefill |
| `MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN=0` | on | the chained draft |
| `MTPLX_M1_PAGED_PREFILL=0` | on (32K+) | prefill straight into 8-bit pages |
| `MTPLX_M1_ENGINE_RAM_FRACTION=0.75` | 13/16 | the larger engine budget |
| `MTPLX_M1_TURBO_DEFAULT=0` | turbo for dense 4/8-bit | the turbo profile default |

The full list, with measurements: [CHANGES.ja.md](docs/m1max-longctx/CHANGES.ja.md).

## Known limitations

- Measured on one M1 Max 64 GB. M1 Pro, M1 Ultra and other memory sizes run the
  same code but are untested.
- The Neural Engine path uses a private Apple framework and may stop working
  after a macOS update (prefill then falls back to the GPU).
- Memory: at 256K the build peaks around 50 GB. Part of it is a second copy of
  the conversation the session bank keeps for fast follow-up turns.
- Some upstream tests assume upstream's defaults and fail with the M1 defaults on;
  most pass with `MTPLX_M1_LONG_CONTEXT=0`.

## License

Apache-2.0, as upstream ([LICENSE](LICENSE), [NOTICE](NOTICE)). The Neural Engine
client and FFN program layout are adapted from Splash
(`native_extensions/ane_ffn/NOTICE`); the flash-attention kernel follows
metal-flash-attention (`native_extensions/mfa_attn/MFA_LICENSE.txt`).
