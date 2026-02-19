# dcnv2-onnx-test

deepctr-torch の DCN V2 (DCNMix) モデルを ONNX に変換し、onnxruntime で推論できるか検証するプロジェクト。

## セットアップ

```bash
uv sync
```

## 使い方

### 1. 学習 & ONNX エクスポート

```bash
uv run python train_and_export.py
```

ダミーデータで DCNMix を学習し、`dcnv2.onnx` と検証用の `reference.json` を出力する。

### 2. ONNX で推論

```bash
uv run python infer_onnx.py
```

`dcnv2.onnx` を onnxruntime で読み込み、スパース特徴量 (user_id, item_id, category) と dense 特徴量 (price, rating) を入力して推論。PyTorch の出力との差分を比較する。

### 3. エクスポーター比較テスト

```bash
uv run python test_dcnv2_onnx.py
```

Dynamo / Legacy TorchScript / torch.jit.trace の 3 種類のエクスポーターで変換を試し、それぞれの結果を比較する。

### 4. CPU 推論レイテンシ計測

```bash
uv run python bench_cpu_infer.py
```

GPU で学習 → ONNX エクスポート → CPU-only onnxruntime で推論し、レイテンシを計測する。

## 検証結果

- スパース特徴量の embedding lookup は ONNX 上で正常に動作する
- PyTorch との出力差は最大 2.98e-08 (float32 精度内)
- 3 種類のエクスポーター全てで変換に成功
- **batch_size=1 は NG** — CrossNetMix 内部の squeeze で rank が崩れる。batch_size>=2 で使うか、ダミー行でパディングして回避する

## ONNX CPU 推論レイテンシ

GPU (CUDA) で学習・エクスポートした ONNX モデルを CPU-only onnxruntime で推論した結果 (1000 回実行):

| Batch Size | Mean | Median | P95 | P99 |
|---:|---:|---:|---:|---:|
| 1 (pad→2) | 0.130 ms | 0.129 ms | 0.136 ms | 0.140 ms |
| 8 | 0.221 ms | 0.235 ms | 0.247 ms | 0.252 ms |
| 32 | 0.701 ms | 0.702 ms | 0.723 ms | — |
| 128 | 0.932 ms | 0.931 ms | 0.955 ms | — |
| 512 | 1.949 ms | 1.881 ms | 2.010 ms | — |
| 1024 | 3.519 ms | 3.355 ms | 4.339 ms | — |

PyTorch GPU 出力との最大差: 5.96e-08 (float32 精度内)
