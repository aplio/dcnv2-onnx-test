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

## 検証結果

- スパース特徴量の embedding lookup は ONNX 上で正常に動作する
- PyTorch との出力差は最大 2.98e-08 (float32 精度内)
- 3 種類のエクスポーター全てで変換に成功
- **batch_size=1 は NG** — CrossNetMix 内部の squeeze で rank が崩れる。batch_size>=2 で使うか、ダミー行でパディングして回避する
