"""Train on GPU → export ONNX → infer on CPU-only ONNX Runtime with latency."""

import time
import json
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DCNMix

# ── 1. Feature definition ──
sparse_features = ["user_id", "item_id", "category"]
dense_features = ["price", "rating"]
all_features = sparse_features + dense_features

feature_columns = [
    SparseFeat(name="user_id",  vocabulary_size=100, embedding_dim=8),
    SparseFeat(name="item_id",  vocabulary_size=500, embedding_dim=8),
    SparseFeat(name="category", vocabulary_size=20,  embedding_dim=4),
    DenseFeat(name="price",     dimension=1),
    DenseFeat(name="rating",    dimension=1),
]

# ── 2. Dummy training data ──
n = 256
np.random.seed(42)
data = pd.DataFrame({
    "user_id":  np.random.randint(0, 100, n),
    "item_id":  np.random.randint(0, 500, n),
    "category": np.random.randint(0, 20, n),
    "price":    np.random.rand(n).astype(np.float32) * 100,
    "rating":   np.random.rand(n).astype(np.float32) * 5,
})
labels = np.random.randint(0, 2, n).astype(np.float32)

# ── 3. Train on GPU ──
print("=" * 65)
print("PHASE 1: Train on GPU")
print("=" * 65)
model = DCNMix(
    linear_feature_columns=feature_columns,
    dnn_feature_columns=feature_columns,
    cross_num=2,
    dnn_hidden_units=(32, 16),
    task="binary",
    device="cuda",
)
model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy"])
model_input = {name: data[name].values for name in all_features}
model.fit(model_input, labels, batch_size=32, epochs=3, verbose=1)

# ── 4. Export to ONNX ──
print("\n" + "=" * 65)
print("PHASE 2: Export to ONNX")
print("=" * 65)
model.eval()

test_data = pd.DataFrame({
    "user_id":  [0, 50, 99, 1, 42, 77, 10, 88],
    "item_id":  [0, 250, 499, 3, 100, 400, 55, 321],
    "category": [0, 10, 19, 5, 2, 15, 8, 11],
    "price":    [9.99, 0.0, 100.0, 50.5, 25.0, 75.0, 1.0, 99.99],
    "rating":   [4.5, 0.0, 5.0, 2.5, 3.0, 1.0, 4.0, 3.5],
})

tensor_parts = []
for name in all_features:
    vals = test_data[name].values
    dtype = torch.long if name in sparse_features else torch.float32
    t = torch.tensor(vals, dtype=dtype).unsqueeze(1)
    tensor_parts.append(t)
dummy_input = torch.cat([t.float() for t in tensor_parts], dim=-1).to("cuda")

with torch.no_grad():
    pt_preds = model(dummy_input).cpu().numpy().flatten()
print(f"PyTorch (GPU) predictions: {pt_preds}")

onnx_path = "dcnv2_bench.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
    dynamo=False,
)
print(f"ONNX exported: {onnx_path}")

# ── 5. CPU-only ONNX inference + latency ──
print("\n" + "=" * 65)
print("PHASE 3: ONNX CPU-only inference + latency")
print("=" * 65)

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
print(f"Active providers: {sess.get_providers()}")

input_name = sess.get_inputs()[0].name
X = dummy_input.cpu().numpy()

# Warmup
for _ in range(10):
    sess.run(None, {input_name: X})

# ── Batch inference (batch_size=8) ──
n_runs = 1000
times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    sess.run(None, {input_name: X})
    times.append(time.perf_counter() - t0)
times = np.array(times) * 1000  # ms

onnx_preds = sess.run(None, {input_name: X})[0].flatten()
max_diff = np.abs(pt_preds - onnx_preds).max()

print(f"\nBatch inference (batch_size={X.shape[0]}, {n_runs} runs):")
print(f"  Mean:   {times.mean():.3f} ms")
print(f"  Median: {np.median(times):.3f} ms")
print(f"  P95:    {np.percentile(times, 95):.3f} ms")
print(f"  P99:    {np.percentile(times, 99):.3f} ms")
print(f"  Min:    {times.min():.3f} ms")
print(f"  Max:    {times.max():.3f} ms")
print(f"  Max diff vs PyTorch: {max_diff:.2e}")

# ── Single-sample inference (padded to batch=2) ──
single = X[0:1]
padded = np.concatenate([single, single], axis=0)

for _ in range(10):
    sess.run(None, {input_name: padded})

times_single = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    sess.run(None, {input_name: padded})
    times_single.append(time.perf_counter() - t0)
times_single = np.array(times_single) * 1000

print(f"\nSingle-sample inference (padded to batch=2, {n_runs} runs):")
print(f"  Mean:   {times_single.mean():.3f} ms")
print(f"  Median: {np.median(times_single):.3f} ms")
print(f"  P95:    {np.percentile(times_single, 95):.3f} ms")
print(f"  P99:    {np.percentile(times_single, 99):.3f} ms")
print(f"  Min:    {times_single.min():.3f} ms")
print(f"  Max:    {times_single.max():.3f} ms")

# ── Larger batch test ──
for batch_size in [32, 128, 512, 1024]:
    np.random.seed(0)
    X_large = np.random.rand(batch_size, X.shape[1]).astype(np.float32)
    # Fill sparse cols with valid IDs
    X_large[:, 0] = np.random.randint(0, 100, batch_size).astype(np.float32)
    X_large[:, 1] = np.random.randint(0, 500, batch_size).astype(np.float32)
    X_large[:, 2] = np.random.randint(0, 20, batch_size).astype(np.float32)

    for _ in range(10):
        sess.run(None, {input_name: X_large})

    times_large = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: X_large})
        times_large.append(time.perf_counter() - t0)
    times_large = np.array(times_large) * 1000

    print(f"\nBatch size={batch_size} ({n_runs} runs):")
    print(f"  Mean:   {times_large.mean():.3f} ms")
    print(f"  Median: {np.median(times_large):.3f} ms")
    print(f"  P95:    {np.percentile(times_large, 95):.3f} ms")
