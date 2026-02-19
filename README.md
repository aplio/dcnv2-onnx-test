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
uv run python bench_cpu_infer.py                # light + heavy + 100x100 全部
uv run python bench_cpu_infer.py --config light
uv run python bench_cpu_infer.py --config heavy
uv run python bench_cpu_infer.py --config 100x100
uv run python bench_cpu_infer.py --runs 5000    # 計測回数を変更
```

GPU で学習 → ONNX エクスポート → CPU-only onnxruntime で推論し、レイテンシを計測する。

### 5. CPU vs GPU 推論比較

```bash
uv run python bench_cpu_vs_gpu.py                # 全 config
uv run python bench_cpu_vs_gpu.py --config light
uv run python bench_cpu_vs_gpu.py --config heavy
uv run python bench_cpu_vs_gpu.py --config 100x100
```

ONNX CPU / ONNX GPU / PyTorch GPU の 3 方式を同一条件で比較する。
`LD_LIBRARY_PATH` に NVIDIA ライブラリパスが必要 (GPU provider 利用時)。

## 検証結果

- スパース特徴量の embedding lookup は ONNX 上で正常に動作する
- PyTorch との出力差は最大 2.98e-08 (float32 精度内)
- 3 種類のエクスポーター全てで変換に成功
- **batch_size=1 は NG** — CrossNetMix 内部の squeeze で rank が崩れる。batch_size>=2 で使うか、ダミー行でパディングして回避する

## 計測環境

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 3950X 16-Core (32 threads) @ 4.76 GHz |
| RAM | 94 GB DDR4 |
| GPU 0 | NVIDIA GeForce RTX 2060 SUPER (8 GB) |
| GPU 1 | NVIDIA GeForce RTX 4070 Ti (12 GB) |
| Driver | 590.48.01 / CUDA 13.1 |
| OS | Linux 6.17.0-14-generic |
| Python | 3.12 |
| PyTorch | 2.10.0+cu126 |
| ONNX Runtime | 1.24.1 (GPU build, CPU provider で計測) |

## ONNX CPU 推論レイテンシ

GPU (CUDA) で学習・エクスポートした ONNX モデルを CPU-only onnxruntime で推論した結果 (1000 回実行)。

### Light モデル (26,393 params / 0.1 MB)

- Sparse: 3 特徴量 (embedding_dim 4–8)
- Dense: 2 特徴量
- DNN: (32, 16) / Cross layers: 2

| Batch Size | Mean | Median | P95 | P99 |
|---:|---:|---:|---:|---:|
| 2 | 0.129 ms | 0.128 ms | 0.137 ms | 0.143 ms |
| 8 | 0.230 ms | 0.229 ms | 0.241 ms | 0.248 ms |
| 32 | 0.715 ms | 0.711 ms | 0.732 ms | 0.746 ms |
| 128 | 0.932 ms | 0.929 ms | 0.949 ms | 0.977 ms |
| 512 | 2.001 ms | 1.888 ms | 2.143 ms | 4.899 ms |
| 1024 | 4.014 ms | 3.426 ms | 6.386 ms | 7.365 ms |

### Heavy モデル (39,771,765 params / 151.8 MB)

- Sparse: 8 特徴量 (embedding_dim 8–64, vocab 最大 500k)
- Dense: 6 特徴量
- DNN: (512, 256, 128) / Cross layers: 4

| Batch Size | Mean | Median | P95 | P99 |
|---:|---:|---:|---:|---:|
| 2 | 0.239 ms | 0.232 ms | 0.279 ms | 0.302 ms |
| 8 | 0.573 ms | 0.569 ms | 0.614 ms | 0.671 ms |
| 32 | 1.812 ms | 1.799 ms | 1.842 ms | 1.927 ms |
| 128 | 3.289 ms | 3.257 ms | 3.333 ms | 3.861 ms |
| 512 | 7.142 ms | 6.318 ms | 10.294 ms | 13.202 ms |
| 1024 | 13.895 ms | 12.388 ms | 19.293 ms | 23.692 ms |

### 100x100 モデル (34,541,749 params / 132.0 MB)

- Sparse: 100 特徴量 (embedding_dim 16, vocab 50–100k)
- Dense: 100 特徴量
- DNN: (512, 256, 128) / Cross layers: 4

| Batch Size | Mean | Median | P95 | P99 |
|---:|---:|---:|---:|---:|
| 2 | 0.868 ms | 0.867 ms | 0.886 ms | 0.894 ms |
| 8 | 1.339 ms | 1.321 ms | 1.349 ms | 1.539 ms |
| 32 | 4.160 ms | 4.079 ms | 4.228 ms | 7.086 ms |
| 128 | 11.408 ms | 10.343 ms | 15.872 ms | 18.695 ms |
| 512 | 50.635 ms | 49.302 ms | 69.123 ms | 75.071 ms |
| 1024 | 164.950 ms | 163.909 ms | 198.008 ms | 207.533 ms |

全モデルとも PyTorch GPU 出力との最大差: 5.96e-08 (float32 精度内)

## ONNX CPU vs GPU vs PyTorch GPU 推論比較

ONNX Runtime の CPU / CUDA プロバイダおよび PyTorch GPU 推論の Median レイテンシ (ms) を比較 (1000 回実行)。

### Light モデル (26,393 params / 0.1 MB)

| Batch | ORT CPU | ORT GPU | PyTorch GPU | Winner |
|---:|---:|---:|---:|---|
| 2 | **0.107** | 0.856 | 1.668 | CPU |
| 8 | **0.246** | 0.865 | 1.663 | CPU |
| 32 | **0.701** | 0.857 | 1.923 | CPU |
| 128 | 0.910 | **0.896** | 1.813 | ORT-GPU |
| 512 | 1.897 | **0.980** | 1.831 | ORT-GPU |
| 1024 | 3.382 | **0.999** | 1.828 | ORT-GPU |

### Heavy モデル (39,771,765 params / 151.8 MB)

| Batch | ORT CPU | ORT GPU | PyTorch GPU | Winner |
|---:|---:|---:|---:|---|
| 2 | **0.229** | 1.632 | 3.194 | CPU |
| 8 | **0.569** | 1.641 | 3.219 | CPU |
| 32 | 2.127 | **1.680** | 3.548 | ORT-GPU |
| 128 | 3.452 | **1.574** | 3.433 | ORT-GPU |
| 512 | 6.736 | **1.677** | 3.304 | ORT-GPU |
| 1024 | 12.772 | **2.101** | 3.335 | ORT-GPU |

### 100x100 モデル (34,541,749 params / 132.0 MB)

| Batch | ORT CPU | ORT GPU | PyTorch GPU | Winner |
|---:|---:|---:|---:|---|
| 2 | **0.994** | 2.507 | 8.186 | CPU |
| 8 | **1.485** | 2.531 | 8.496 | CPU |
| 32 | 4.162 | **2.538** | 9.328 | ORT-GPU |
| 128 | 10.152 | **2.483** | 9.206 | ORT-GPU |
| 512 | 43.134 | **4.602** | 9.634 | ORT-GPU |
| 1024 | 151.631 | **7.606** | 9.527 | ORT-GPU |

### 考察

- **小バッチ (batch <= 8–32) では ONNX CPU が最速**。GPU はカーネル起動・データ転送のオーバーヘッドがあるため、計算量が少ないと不利
- **バッチが大きくなると ORT GPU が逆転**。クロスオーバーポイントは light で batch~128、heavy/100x100 で batch~32
- **ORT GPU は PyTorch GPU より常に高速** (2–4 倍)。ONNX Runtime のグラフ最適化が効いている
- リアルタイム推論 (batch=1–8) のユースケースでは **GPU 不要で ONNX CPU が最適解**
