"""Train on GPU → export ONNX → infer on CPU-only ONNX Runtime with latency.

Runs model configs: "light", "heavy", and "100x100".
"""

import argparse
import time
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DCNMix

# ── Model configs ──
CONFIGS = {
    "light": {
        "sparse": [
            SparseFeat("user_id",  vocabulary_size=100,  embedding_dim=8),
            SparseFeat("item_id",  vocabulary_size=500,  embedding_dim=8),
            SparseFeat("category", vocabulary_size=20,   embedding_dim=4),
        ],
        "dense": [
            DenseFeat("price",  dimension=1),
            DenseFeat("rating", dimension=1),
        ],
        "cross_num": 2,
        "dnn_hidden_units": (32, 16),
    },
    "heavy": {
        "sparse": [
            SparseFeat("user_id",    vocabulary_size=100_000, embedding_dim=64),
            SparseFeat("item_id",    vocabulary_size=500_000, embedding_dim=64),
            SparseFeat("category",   vocabulary_size=1_000,   embedding_dim=32),
            SparseFeat("brand",      vocabulary_size=5_000,   embedding_dim=32),
            SparseFeat("city",       vocabulary_size=500,     embedding_dim=16),
            SparseFeat("device",     vocabulary_size=50,      embedding_dim=8),
            SparseFeat("os",         vocabulary_size=10,      embedding_dim=8),
            SparseFeat("channel",    vocabulary_size=200,     embedding_dim=16),
        ],
        "dense": [
            DenseFeat("price",       dimension=1),
            DenseFeat("rating",      dimension=1),
            DenseFeat("age",         dimension=1),
            DenseFeat("hist_ctr",    dimension=1),
            DenseFeat("hist_cvr",    dimension=1),
            DenseFeat("page_score",  dimension=1),
        ],
        "cross_num": 4,
        "dnn_hidden_units": (512, 256, 128),
    },
    "100x100": {
        "sparse": [
            SparseFeat(f"s{i:03d}", vocabulary_size=np.random.RandomState(i).choice(
                [50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
            ), embedding_dim=16)
            for i in range(100)
        ],
        "dense": [
            DenseFeat(f"d{i:03d}", dimension=1)
            for i in range(100)
        ],
        "cross_num": 4,
        "dnn_hidden_units": (512, 256, 128),
    },
}


def run_bench(config_name: str, n_runs: int = 1000):
    cfg = CONFIGS[config_name]
    feature_columns = cfg["sparse"] + cfg["dense"]
    sparse_names = [f.name for f in cfg["sparse"]]
    dense_names = [f.name for f in cfg["dense"]]
    all_names = sparse_names + dense_names

    # ── Training data ──
    n = 1024
    np.random.seed(42)
    data = {}
    for sf in cfg["sparse"]:
        data[sf.name] = np.random.randint(0, sf.vocabulary_size, n)
    for df in cfg["dense"]:
        data[df.name] = np.random.rand(n).astype(np.float32)
    data = pd.DataFrame(data)
    labels = np.random.randint(0, 2, n).astype(np.float32)

    # ── Train on GPU ──
    print(f"\n{'=' * 65}")
    print(f"CONFIG: {config_name}")
    print(f"  Sparse features: {len(sparse_names)}  Dense features: {len(dense_names)}")
    print(f"  DNN: {cfg['dnn_hidden_units']}  Cross layers: {cfg['cross_num']}")
    print(f"{'=' * 65}")

    print("\n[Train on GPU]")
    model = DCNMix(
        linear_feature_columns=feature_columns,
        dnn_feature_columns=feature_columns,
        cross_num=cfg["cross_num"],
        dnn_hidden_units=cfg["dnn_hidden_units"],
        task="binary",
        device="cuda",
    )
    model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy"])
    model_input = {name: data[name].values for name in all_names}
    model.fit(model_input, labels, batch_size=64, epochs=3, verbose=1)

    # ── Export ──
    model.eval()
    batch_size_export = 8
    tensor_parts = []
    for name in all_names:
        vals = data[name].values[:batch_size_export]
        dtype = torch.long if name in sparse_names else torch.float32
        t = torch.tensor(vals, dtype=dtype).unsqueeze(1)
        tensor_parts.append(t)
    dummy_input = torch.cat([t.float() for t in tensor_parts], dim=-1).to("cuda")

    with torch.no_grad():
        pt_preds = model(dummy_input).cpu().numpy().flatten()

    onnx_path = f"dcnv2_bench_{config_name}.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17, dynamo=False,
    )
    onnx_size_mb = torch.tensor(0).new_empty(0).element_size()  # dummy
    import os
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX exported: {onnx_path} ({onnx_size_mb:.1f} MB)")

    # count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── CPU inference + latency ──
    print(f"\n[ONNX CPU-only inference — {n_runs} runs per batch size]")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"Active providers: {sess.get_providers()}")
    input_name = sess.get_inputs()[0].name

    X_base = dummy_input.cpu().numpy()

    # Verify correctness
    onnx_preds = sess.run(None, {input_name: X_base})[0].flatten()
    max_diff = np.abs(pt_preds - onnx_preds).max()
    print(f"Max diff vs PyTorch GPU: {max_diff:.2e}")

    # Bench various batch sizes
    batch_sizes = [2, 8, 32, 128, 512, 1024]
    n_features = X_base.shape[1]

    print(f"\n{'Batch':>6} {'Mean':>9} {'Median':>9} {'P95':>9} {'P99':>9} {'Min':>9} {'Max':>9}")
    print("-" * 63)

    for bs in batch_sizes:
        np.random.seed(0)
        X = np.random.rand(bs, n_features).astype(np.float32)
        for i, name in enumerate(all_names):
            if name in sparse_names:
                cap = next(sf.vocabulary_size for sf in cfg["sparse"] if sf.name == name)
                X[:, i] = np.random.randint(0, cap, bs).astype(np.float32)

        # warmup
        for _ in range(20):
            sess.run(None, {input_name: X})

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sess.run(None, {input_name: X})
            times.append(time.perf_counter() - t0)
        t = np.array(times) * 1000

        print(f"{bs:>6} {t.mean():>8.3f}ms {np.median(t):>8.3f}ms "
              f"{np.percentile(t, 95):>8.3f}ms {np.percentile(t, 99):>8.3f}ms "
              f"{t.min():>8.3f}ms {t.max():>8.3f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()

    configs = list(CONFIGS.keys()) if args.config == "all" else [args.config]
    for cfg in configs:
        run_bench(cfg, n_runs=args.runs)
