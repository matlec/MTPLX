# M1 Mac 向け MTPLX（長い文脈）

[![Release](https://img.shields.io/github/v/release/shunya1810/MTPLX)](https://github.com/shunya1810/MTPLX/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

[English](README.md) | 日本語

[MTPLX](https://github.com/youssofal/MTPLX) を M1 系の Mac 向けに調整した非公式版です。
Qwen3.8-27B で、長いプロンプトとエージェントの複数ターンの会話を速く回すことを目的にしています。
MTPLX は、モデル自身の MTP（multi-token prediction）ヘッドで出力を変えない投機的デコードを行い、OpenAI と Anthropic 互換のサーバーと Mac アプリとして動きます。
この版は MTPLX 2.12.2 を土台に、M1 系の GPU で動く部分だけを変えています。ほかの Mac では本家と同じ動きです。

**M3 以降の Mac では [本家の MTPLX](https://github.com/youssofal/MTPLX) を使ってください。** この版はそこでは計測も調整もしていません。

MTPLX のプロジェクトとは関係のない非公式版です。変更は AI（Claude）の支援を受けて作り、MacBook Pro M1 Max（64 GB）1 台で計測しました。
本家の README は [README.upstream.md](README.upstream.md) にあります。

## この版で足したもの

- **Neural Engine を使う prompt の読み込み**：各層の MLP の一部を M1 の Neural Engine で計算し、残りを GPU が同時に計算します。prefill の塊は半分ずつ交互に重ねます。最初の prompt の読み込みは 1.4〜1.5 倍速くなります。Apple の非公開の `AppleNeuralEngine.framework` を使います（やり方は [Splash](https://github.com/incoai/splash)（Apache-2.0）から移植しました）。使えないときは GPU だけの計算に戻ります。
- **M1 向けの attention の kernel**：キャッシュ済みの部分に対する prefill の attention を、M1 向けに調整した [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) で計算します（head size 256）。verify の attention と射影も M1 の GPU に合わせています。
- **decode のホストとの往復を減らす**：verify の行と KV の位置をまとめて読み、draft の 3 段を GPU 上で続けて作ります（MTP のキャッシュはその場で書き込みます）。32K で decode の 1 ラウンドが約 5% 短くなります。
- **64 GB で長い文脈**：KV を 8bit のページで持ち、32K 以上の prompt は最初から 8bit のページに読み込みます。メモリガードの見積もりをこの版の実測に合わせ、エンジンが RAM の 13/16 まで使えるようにしました。256K トークンの会話が既定の設定で動きます。
- **エージェントでの会話の修正**（どの Mac でも）：トークンの区切りの違いで会話が別扱いになる問題、ツールの引数の空白、KV を広げられないときに回答ごと失敗する問題などを直しました。[CHANGES.ja.md](docs/m1max-longctx/CHANGES.ja.md) を見てください。

## 計測結果

長いプロンプトでの 3 ターンの会話です（1 ターン目は最初から読み、2〜3 ターン目で約 300 トークンずつ足し、各ターン最大 256 トークンを生成）。Qwen3.8-27B、8bit の KV、temperature 0、M1 Max 64 GB。
[apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench) から抜粋しました（この版は v2.12.2-m1.1、256K は v2.12.2-m1.2。本家 MTPLX 2.12.0 も同じ方法で計測）。ほかのエンジンとの比較はベンチにあります。

**会話全体の時間**

| | 2K | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|
| **この版** | **33 秒** | **61 秒** | **3.3 分** | **7.3 分** | **18.4 分** | **59.4 分** |
| 本家 MTPLX | 42 秒 | 86 秒 | 5.0 分 | 11.3 分 | 30.6 分 | — |

**decode**（tok/s、3 ターンの平均）

| | 2K | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|
| **この版** | **29.4** | **29.2** | **28.1** | **25.5** | **19.8** | **14.2** |
| 本家 MTPLX | 23.6 | 22.5 | 14.0 | 8.7 | 5.0 | — |

2 ターン目以降は 0.5〜2.9 秒で応答を始めます（本家は 2K〜128K で 0.6〜3.5 秒）。
メモリのピーク（固定分）は、2K〜128K で 32〜45 GB、256K で 50 GB です。
サンプリング（temperature 1.0）では、decode は 2K / 32K / 64K で 31.6 / 28.3 / 23.9 tok/s でした（本家は 23.2 / 14.2 / 8.9）。

## インストール

M1 系の Mac（計測は M1 Max 64 GB、macOS 26）と、一度起動して実行環境とモデルを入れた [MTPLX アプリ](https://github.com/youssofal/MTPLX)、ネイティブ拡張のための Metal のツールチェーン入りの Xcode が必要です。

```bash
git clone --branch v2.12.2-m1.2 https://github.com/shunya1810/MTPLX.git MTPLX-m1
cd MTPLX-m1
scripts/build-m1-native-extensions.sh   # 1 分ほど。mfa_attn: ok / ane_ffn: ok と表示されれば完了
```

アプリの中で動かす（`scripts/open-mtplx-app-with-checkout.command`）か、サーバーだけを起動します。

```bash
PYTHONPATH="$PWD" "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" -m mtplx.cli serve \
  --model "$HOME/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16" \
  --context-window 262144 --paged-kv-quantization q8
```

詳しくは [docs/m1max-longctx/USAGE.ja.md](docs/m1max-longctx/USAGE.ja.md) を見てください。
ネイティブ拡張がなくてもサーバーは動きますが、Neural Engine の prefill、flash-attention の prefill、draft の chain は使われません。

## モデル

| モデル | 状態 |
|---|---|
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` | 計測済み、おすすめ（M1 では既定で turbo の profile） |
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16` | 計測済み（8bit。メモリが 9 GB 多く、decode は 1〜1.5 割遅い） |
| Qwen3.6-35B-A3B（MoE） | 動くが、まだ調整していない |

## 切り替え

M1 向けの既定値は、1 つずつ止められます。`MTPLX_M1_LONG_CONTEXT=0` で全部止まります。

| 環境変数 | M1 での既定 | 止めるもの |
|---|---|---|
| `MTPLX_ANE_PREFILL=0` | on | Neural Engine の prefill |
| `MTPLX_MFA_PREFILL=0` | on | flash-attention の prefill |
| `MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN=0` | on | draft の chain |
| `MTPLX_M1_PAGED_PREFILL=0` | on（32K 以上） | 8bit のページへの直接の prefill |
| `MTPLX_M1_ENGINE_RAM_FRACTION=0.75` | 13/16 | 広げたエンジンのメモリ上限 |
| `MTPLX_M1_TURBO_DEFAULT=0` | dense の 4/8bit は turbo | turbo の profile の既定 |

すべての変更と計測値は [CHANGES.ja.md](docs/m1max-longctx/CHANGES.ja.md) にあります。

## 既知の制約

- 計測は M1 Max 64 GB の 1 台だけです。M1 Pro、M1 Ultra、ほかのメモリ量でも同じコードが動きますが、確かめていません。
- Neural Engine の経路は Apple の非公開のフレームワークを使うので、macOS の更新で使えなくなることがあります（そのときは GPU だけの prefill に戻ります）。
- メモリ：256K ではピークが約 50 GB になります。一部は、2 ターン目以降を速く始めるためにセッションバンクが持つ、会話のもう 1 組のコピーです。
- 本家のテストのうち、本家の既定値を前提にしたものは、M1 の既定値では落ちます。ほとんどは `MTPLX_M1_LONG_CONTEXT=0` で通ります。

## ライセンス

本家と同じ Apache-2.0 です（[LICENSE](LICENSE)、[NOTICE](NOTICE)）。Neural Engine のクライアントと FFN のプログラムの構成は Splash から移植しました（`native_extensions/ane_ffn/NOTICE`）。flash-attention の kernel は metal-flash-attention にならっています（`native_extensions/mfa_attn/MFA_LICENSE.txt`）。
