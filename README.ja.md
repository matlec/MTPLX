# M1 Mac 向け MTPLX

[![Release](https://img.shields.io/github/v/release/shunya1810/MTPLX)](https://github.com/shunya1810/MTPLX/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

[English](README.md) | 日本語

[MTPLX](https://github.com/youssofal/MTPLX) を M1 系の Mac 向けに調整した非公式のフォークです。Qwen3.8-27B と Qwen3.6-35B-A3B で、長いプロンプトとコーディングエージェントの会話を速く処理することを目的にしています。本家と同じく、モデル自身の MTP（multi-token prediction）ヘッドで投機的デコードを行うので、出力の分布は変わりません。OpenAI 互換と Anthropic 互換の API で動きます。中身は MTPLX 2.12.2 に、M1 系の GPU でだけ有効になる変更を加えたものです。

- **M3 以降の Mac では [本家の MTPLX](https://github.com/youssofal/MTPLX) を使ってください**。このフォークは M3 以降では計測していません。
- MTPLX のプロジェクトとは関係のない非公式のフォークです。変更は AI（Claude）の支援を受けて作り、MacBook Pro M1 Max（64 GB）1 台で計測しました。本家の README は [README.upstream.md](README.upstream.md) にあります。

## 主な計測値

M1 Max 64 GB、8bit の KV、長いプロンプトでの 3 ターンの会話です（[apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench)。条件は「[計測結果](#計測結果)」を見てください）。

| モデル | このフォーク | 比較の相手 |
|---|---|---|
| Qwen3.6-35B-A3B（MoE） | decode 2K で 111.8 tok/s、32K で 86.4、128K で 61.2 | Splash 1.3.0-m1.1: 96.1、63.8、43.9 |
| Qwen3.8-27B（dense） | decode 2K で 29.4 tok/s、32K で 28.1、128K で 19.8 | 本家 MTPLX 2.12.0: 23.6、14.0、5.0 |
| A3B のエージェントの会話 | ファイルの書き直しが 219 tok/s（context-copy なしでは 111）、16 万トークンの履歴の後ろに足した短いツールの結果を 0.42 秒で読み込み | v2.12.2-m1.5: 111 tok/s、0.75 秒 |

## インストールと起動

次のものが必要です。

- M1 系の Mac（計測は M1 Max 64 GB、macOS 26）
- Metal のツールチェーンが入った Xcode（`xcrun metal` が動くこと）
- 一度起動した [MTPLX アプリ](https://github.com/youssofal/MTPLX)（このフォークはアプリの Python 実行環境を使います。モデルはアプリのモデル画面からダウンロードします）

```bash
git clone --branch v2.12.2-m1.7 https://github.com/shunya1810/MTPLX.git MTPLX-m1
cd MTPLX-m1
scripts/build-m1-native-extensions.sh   # 1 分ほど。「mfa_attn: ok」と「ane_ffn: ok」が出れば完了
```

MTPLX アプリを終了してから（エンジンを 2 つ同時に動かすとメモリに収まりません）、どちらかのモデルでサーバーを起動します。

```bash
# Qwen3.6-35B-A3B（MoE）: M1 でエージェントに使うなら最も速い選択
PYTHONPATH="$PWD" "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" -m mtplx.cli serve \
  --model "$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16" \
  --context-window 262144 --paged-kv-quantization q8

# Qwen3.8-27B（dense）: --model を次のパスに替える
#   "$HOME/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"
```

応答を確かめます（サーバーは `http://127.0.0.1:8000` で待ち受けます）。

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "mtplx", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 64}'
```

MTPLX アプリの画面からこのフォークを動かすときは、`scripts/open-mtplx-app-with-checkout.command` を使います。詳しくは [docs/m1max-longctx/USAGE.ja.md](docs/m1max-longctx/USAGE.ja.md) を見てください。ネイティブ拡張をビルドしなくてもサーバーは動きますが、prefill が GPU だけの遅い経路になります。

## コーディングエージェントで使う

OpenAI か Anthropic の API を使うクライアントなら動きます。OpenCode と Claude Code での設定は次のとおりです。

- OpenCode: `~/.config/opencode/opencode.json` にプロバイダーを足します。

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

- Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8000 claude` で起動します。

Qwen3.6 には reasoning effort の段階がなく、thinking のオンとオフだけです。Qwen3.8 は `low` から `xhigh` まで使えます。

## モデル

| モデル | メモリのピーク | 補足 |
|---|---|---|
| `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16`（MoE、3B が有効） | 27〜31 GB | v2.12.2-m1.4 から M1 向けに調整。decode の round を GPU の上で続けて回し、prefill で Neural Engine を使い、context-copy を使います |
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16`（dense） | 2K〜128K で 32〜45 GB、256K で 50 GB | このフォークの最初の対象 |
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16`（dense、8bit） | 約 9 GB 多い | 計測済み。decode は Speed より約 10〜15% 遅い |

## 計測結果

数値はすべて MacBook Pro M1 Max 64 GB、macOS 26、8bit の KV、temperature 0 での、[apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench) の 3 ターンの会話です。1 ターン目は長いプロンプトを最初から読み、2〜3 ターン目は約 300 トークンずつ足し、各ターンで最大 256 トークンを生成します。比べるエンジンは同じセッションで交互に動かしました。

**Qwen3.6-35B-A3B**: v2.12.2-m1.5 と Splash 1.3.0-m1.1（2026-10-14。このベンチでの m1.6 の decode は、m1.5 より 2K で 0.6%、32K で 3.8% 遅くなります。32K では context-copy が別の続きを写して外れることがあるためです）

| | 2K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| decode（tok/s、このフォーク） | 111.8 | 110.8 | 86.4 | 76.5 | 61.2 |
| decode（tok/s、Splash） | 96.1 | 80.8 | 63.8 | 57.7 | 43.9 |
| 会話全体（このフォーク） | 9.3 秒 | 14.4 秒 | 41.3 秒 | 1.5 分 | 4.3 分 |
| 会話全体（Splash） | 9.5 秒 | 18.1 秒 | 60.7 秒 | 2.5 分 | 7.7 分 |

thinking ありの短いプロンプト（greedy、1,024 トークン）では、Splash の decode が 9〜15% 速くなります。Splash の DFlash 2 のブロック単位の draft は、思考の文章や定型のコードで最もよく当たるからです。

**Qwen3.8-27B**: v2.12.2-m1.1（256K だけ m1.2）と本家 MTPLX 2.12.0

| | 2K | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|
| decode（tok/s、このフォーク） | 29.4 | 29.2 | 28.1 | 25.5 | 19.8 | 14.2 |
| decode（tok/s、本家） | 23.6 | 22.5 | 14.0 | 8.7 | 5.0 | — |
| 会話全体（このフォーク） | 33 秒 | 61 秒 | 3.3 分 | 7.3 分 | 18.4 分 | 59.4 分 |
| 会話全体（本家） | 42 秒 | 86 秒 | 5.0 分 | 11.3 分 | 30.6 分 | — |

**エージェントの会話（Qwen3.6-35B-A3B、v2.12.2-m1.6）**: 1.1 万トークンのファイルを thinking あり（temperature 0.6）で書き直すと、decode は context-copy ありで 219 tok/s、なしで 111 tok/s です。新しく文章を書くときは、context-copy ありで 111 tok/s、なしで 105 tok/s です。16 万トークンのキャッシュ済みの履歴の後ろに 42 トークンのツールの結果を足すと、0.42 秒で読み込みます（m1.5 は 0.75 秒）。長いエージェントの会話で、サーバーがリクエストごとにする前処理は 80〜100 ms から約 2 ms になりました。

## 本家からの変更点

- prefill の計算の一部を Neural Engine で行い、そのあいだに GPU が残りを計算します（仕組みは [Splash](https://github.com/incoai/splash) から移植しました）。キャッシュ済みの部分に対する attention は、M1 向けに調整した [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) のブロックで計算します。長い履歴の後ろに短い入力を足すときは、8bit の KV をそのまま読みます。
- decode では、1 ラウンドあたりのホストとの往復を減らしました。A3B では、ホストが前のラウンドの結果を読む前に、次のラウンドを GPU に積みます（greedy の出力は同じで、sampling の出力の分布も同じです）。ファイルを書き直す応答では、プロンプトと同じ行をそのまま写します（context-copy、完全に一致したところだけ）。
- メモリについては、KV を 8bit のページで持ち、長いプロンプトは最初から 8bit のページに読み込みます。メモリの見積もりはこのフォークでの実測に合わせました。64 GB で、256K トークンの会話が既定の設定のまま動きます。
- エージェントの会話（どの Mac でも有効）では、トークンの区切り方が違っても同じ会話として扱えるようにし、ツールの引数の空白が消えないようにしました。macOS が長い GPU の処理を止めたときはリクエストをやり直し、履歴が変わったときは保存した状態のうち最も深い位置から再開します。

変更点とその計測結果は、すべて [CHANGES.ja.md](docs/m1max-longctx/CHANGES.ja.md) と [リリースノート](https://github.com/shunya1810/MTPLX/releases) にあります。

## M1 向けの既定を止める

M1 向けの既定は、それぞれ個別に止められます。`MTPLX_M1_LONG_CONTEXT=0` で、すべてまとめて止まります。

| 変数 | M1 での既定 | 止まるもの |
|---|---|---|
| `MTPLX_ANE_PREFILL=0` / `MTPLX_ANE_PREFILL_MOE=0` | オン | Neural Engine の prefill（dense / MoE） |
| `MTPLX_MFA_PREFILL=0` | オン | flash-attention の prefill |
| `MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN=0` | オン | draft を続けて作る処理 |
| `MTPLX_M1_PAGED_PREFILL=0` | 32K 以上でオン | 8bit のページへの直接の読み込み |
| `MTPLX_M1_ENGINE_RAM_FRACTION=0.75` | 13/16 | エンジンのメモリの上限の引き上げ |
| `MTPLX_M1_TURBO_DEFAULT=0` | 4bit / 8bit の dense で turbo | turbo のプロファイルを既定にすること |
| `MTPLX_M1_MOE_LANE=0` | MoE のモデルでオン | MoE 向けの起動時の既定（この表の MoE と A3B の項目を含む） |
| `MTPLX_M1_RESIDENT_DECODE=0` | A3B でオン | GPU の上で decode のラウンドを続けて回すこと |
| `MTPLX_CONTEXT_COPY=0` | A3B でオン | context-copy |

## 制限事項

- 計測は M1 Max 64 GB の 1 台だけです。M1 Pro、M1 Ultra、ほかのメモリ容量でも同じコードが動きますが、試していません。
- Neural Engine の経路は、Apple の非公開の `AppleNeuralEngine.framework` を使います。macOS の更新で動かなくなる可能性があり、その場合は prefill が GPU だけに戻ります。
- Neural Engine の prefill は int8 の活性で計算するので、GPU の prefill と bit 単位では一致しません（A3B では、24,576 か所の次のトークンの 98.25% が一致しました）。`MTPLX_ANE_PREFILL=0` / `MTPLX_ANE_PREFILL_MOE=0` で、prefill を GPU だけにできます。
- thinking のあるモデルは、難しいエージェントの課題でとても長く考え込むことがあります（A3B の長い会話で、131,072 トークンの応答がありました）。既定では思考の長さに上限はありません。`MTPLX_THINKING_BUDGET=<トークン数>` で、ツールを使うリクエストの思考に上限を設定できます。
- 本家のテストの一部は本家の既定を前提にしているので、M1 向けの既定がオンだと失敗します。`MTPLX_M1_LONG_CONTEXT=0` にすると、ほとんどが通ります。

## ライセンスと謝辞

本家と同じ Apache-2.0 です（[LICENSE](LICENSE)、[NOTICE](NOTICE)）。Neural Engine のクライアントと FFN のプログラムの配置は Splash から移植しました（`native_extensions/ane_ffn/NOTICE`）。flash-attention の kernel は metal-flash-attention に従っています（`native_extensions/mfa_attn/MFA_LICENSE.txt`）。
