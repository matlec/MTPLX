# M1 Mac 向け MTPLX（長い文脈）

[![Release](https://img.shields.io/github/v/release/shunya1810/MTPLX)](https://github.com/shunya1810/MTPLX/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

[English](README.md) | 日本語

[MTPLX](https://github.com/youssofal/MTPLX) を M1 系の Mac 向けに調整した非公式のフォークです。
Qwen3.8-27B で、長いプロンプトとエージェントの複数ターンの会話を速く処理することを目的にしています。
MTPLX は、モデル自身の MTP（multi-token prediction）ヘッドを使って、出力を変えずに投機的デコードを行います。OpenAI 互換・Anthropic 互換のサーバーと、Mac アプリとして動きます。
このフォークは MTPLX 2.12.2 をもとに、M1 系の GPU で動く部分だけを変えています。ほかの Mac では本家と同じように動きます。

**M3 以降の Mac では [本家の MTPLX](https://github.com/youssofal/MTPLX) を使ってください。** このフォークは M3 以降の Mac では計測も調整もしていません。

MTPLX のプロジェクトとは関係のない非公式のフォークです。変更は AI（Claude）の支援を受けて作り、MacBook Pro M1 Max（64 GB）1 台で計測しました。
本家の README は [README.upstream.md](README.upstream.md) にあります。

## このフォークで足したもの

- **Neural Engine を使ったプロンプトの読み込み**：各層の MLP の一部を M1 の Neural Engine で計算し、残りを GPU で同時に計算します。prefill の塊は半分に分け、GPU と Neural Engine が交互に休まず処理するように重ねます。最初のプロンプトの読み込みは 1.4〜1.5 倍速くなります。Apple の非公開の `AppleNeuralEngine.framework` を使い、仕組みは [Splash](https://github.com/incoai/splash)（Apache-2.0）から移植しました。使えないときは GPU だけの計算に戻ります。
- **M1 向けの attention の kernel**：キャッシュ済みの部分に対する prefill の attention を、M1 向けに調整した [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)（head size 256）で計算します。verify の attention と射影も M1 の GPU に合わせています。
- **decode でホストとの往復を減らす**：verify の各行と KV の位置を 1 回でまとめて読み、draft の 3 段を GPU 上で続けて作ります。MTP のキャッシュはその場で書き込みます。32K の文脈では、decode の 1 ラウンドが約 5% 短くなります。
- **64 GB で長い文脈を扱う**：KV を 8bit のページで持ち、32K トークン以上のプロンプトは最初から 8bit のページに読み込みます。メモリガードの見積もりをこのフォークでの実測に合わせ、エンジンが RAM の 13/16 まで使えるようにしました。256K トークンの会話が既定の設定で動きます。
- **エージェントでの会話の修正**（どの Mac でも有効）：トークンの区切り方の違いで同じ会話が別の会話として扱われる問題、ツールの引数の空白が消える問題、KV を広げられないときに回答ごと失敗する問題などを直しました。詳しくは [CHANGES.ja.md](docs/m1max-longctx/CHANGES.ja.md) を見てください。

## 計測結果

長いプロンプトで 3 ターンの会話をした結果です。1 ターン目はプロンプトを最初から読み、2〜3 ターン目では約 300 トークンずつ足し、各ターンで最大 256 トークンを生成します。条件は、Qwen3.8-27B、8bit の KV、temperature 0、M1 Max 64 GB です。
数値は [apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench) から抜粋しました。このフォークは v2.12.2-m1.1（256K だけ v2.12.2-m1.2）で、本家 MTPLX 2.12.0 も同じ方法で計測しています。ほかのエンジンとの比較はベンチにあります。

**会話全体の時間**

| | 2K | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|
| **このフォーク** | **33 秒** | **61 秒** | **3.3 分** | **7.3 分** | **18.4 分** | **59.4 分** |
| 本家 MTPLX | 42 秒 | 86 秒 | 5.0 分 | 11.3 分 | 30.6 分 | — |

**decode**（tok/s、3 ターンの平均）

| | 2K | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|
| **このフォーク** | **29.4** | **29.2** | **28.1** | **25.5** | **19.8** | **14.2** |
| 本家 MTPLX | 23.6 | 22.5 | 14.0 | 8.7 | 5.0 | — |

2 ターン目以降は 0.5〜2.9 秒で応答を始めます（本家は 2K〜128K で 0.6〜3.5 秒）。
使用中に固定されたメモリ（wired）のピークは、2K〜128K で 32〜45 GB、256K で 50 GB です。
サンプリング（temperature 1.0）での decode は、2K / 32K / 64K で 31.6 / 28.3 / 23.9 tok/s でした（本家は 23.2 / 14.2 / 8.9）。

## インストール

次のものが必要です。

- M1 系の Mac（計測は M1 Max 64 GB、macOS 26）
- 一度起動して、実行環境とモデルを入れた [MTPLX アプリ](https://github.com/youssofal/MTPLX)
- ネイティブ拡張をビルドするための、Metal のツールチェーンが入った Xcode

```bash
git clone --branch v2.12.2-m1.3 https://github.com/shunya1810/MTPLX.git MTPLX-m1
cd MTPLX-m1
scripts/build-m1-native-extensions.sh   # 1 分ほど。mfa_attn: ok / ane_ffn: ok と表示されれば完了
```

アプリの中で動かす場合は `scripts/open-mtplx-app-with-checkout.command` を実行します。サーバーだけを起動する場合は、次のようにします。

```bash
PYTHONPATH="$PWD" "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" -m mtplx.cli serve \
  --model "$HOME/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16" \
  --context-window 262144 --paged-kv-quantization q8
```

詳しくは [docs/m1max-longctx/USAGE.ja.md](docs/m1max-longctx/USAGE.ja.md) を見てください。
ネイティブ拡張がなくてもサーバーは動きますが、その場合は Neural Engine の prefill、flash-attention の prefill、draft の chain が使われません。

## モデル

| モデル | 状態 |
|---|---|
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` | 計測済み、おすすめ（M1 では既定で turbo の profile） |
| `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16` | 計測済み（8bit。メモリが 9 GB 多く、decode は 10〜15% 遅い） |
| Qwen3.6-35B-A3B（MoE） | 動くが、まだ調整していない |

## 切り替え

M1 向けの既定値は、1 つずつ止められます。`MTPLX_M1_LONG_CONTEXT=0` を設定すると、すべて止まります。

| 環境変数 | M1 での既定 | 止めるもの |
|---|---|---|
| `MTPLX_ANE_PREFILL=0` | on | Neural Engine の prefill |
| `MTPLX_MFA_PREFILL=0` | on | flash-attention の prefill |
| `MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN=0` | on | draft の chain |
| `MTPLX_M1_PAGED_PREFILL=0` | on（32K 以上） | 8bit のページへの直接の prefill |
| `MTPLX_M1_ENGINE_RAM_FRACTION=0.75` | 13/16 | 広げたエンジンのメモリ上限 |
| `MTPLX_M1_TURBO_DEFAULT=0` | dense の 4/8bit は turbo | turbo の profile の既定 |

すべての変更点と計測値は [CHANGES.ja.md](docs/m1max-longctx/CHANGES.ja.md) にあります。

## 既知の制約

- 計測は M1 Max 64 GB の 1 台だけです。M1 Pro、M1 Ultra、ほかのメモリ量でも同じコードが動きますが、確かめていません。
- Neural Engine の経路は Apple の非公開のフレームワークを使うので、macOS の更新で使えなくなることがあります。そのときは GPU だけの prefill に戻ります。
- 256K では、メモリのピークは 1 ターン目のプロンプトの読み込み中に約 48 GB、2 ターン目以降は 41〜42 GB です。
- 本家のテストのうち、本家の既定値を前提にしたものは、M1 の既定値のままでは失敗します。ほとんどは `MTPLX_M1_LONG_CONTEXT=0` で通ります。

## ライセンス

本家と同じ Apache-2.0 です（[LICENSE](LICENSE)、[NOTICE](NOTICE)）。Neural Engine のクライアントと FFN のプログラムの構成は Splash から移植しました（`native_extensions/ane_ffn/NOTICE`）。flash-attention の kernel は metal-flash-attention にならっています（`native_extensions/mfa_attn/MFA_LICENSE.txt`）。
