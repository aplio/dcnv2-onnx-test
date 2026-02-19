"""Compare ONNX CPU vs GPU inference latency.

Train on GPU → export ONNX → benchmark both CPUExecutionProvider and
CUDAExecutionProvider side by side.
"""

import argparse
import os
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


def make_input(cfg, sparse_names, all_names, batch_size):
    """Generate a random input array for the given config."""
    n_features = len(all_names)
    X = np.random.rand(batch_size, n_features).astype(np.float32)
    for i, name in enumerate(all_names):
        if name in sparse_names:
            cap = next(sf.vocabulary_size for sf in cfg["sparse"] if sf.name == name)
            X[:, i] = np.random.randint(0, cap, batch_size).astype(np.float32)
    return X


def bench_session(sess, input_name, X, n_warmup, n_runs):
    """Benchmark a single ORT session. Returns times in ms."""
    for _ in range(n_warmup):
        sess.run(None, {input_name: X})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: X})
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000


def run_comparison(config_name: str, n_runs: int = 1000):
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
    print(f"\n{'=' * 70}")
    print(f"CONFIG: {config_name}")
    print(f"  Sparse: {len(sparse_names)}  Dense: {len(dense_names)}")
    print(f"  DNN: {cfg['dnn_hidden_units']}  Cross layers: {cfg['cross_num']}")
    print(f"{'=' * 70}")

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

    # ── Export ONNX ──
    model.eval()
    export_bs = 8
    tensor_parts = []
    for name in all_names:
        vals = data[name].values[:export_bs]
        dtype = torch.long if name in sparse_names else torch.float32
        t = torch.tensor(vals, dtype=dtype).unsqueeze(1)
        tensor_parts.append(t)
    dummy_input = torch.cat([t.float() for t in tensor_parts], dim=-1).to("cuda")

    onnx_path = f"dcnv2_cmp_{config_name}.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17, dynamo=False,
    )
    total_params = sum(p.numel() for p in model.parameters())
    onnx_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX: {onnx_path} ({onnx_mb:.1f} MB, {total_params:,} params)")

    # ── Create sessions ──
    sess_cpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"CPU session providers: {sess_cpu.get_providers()}")

    gpu_available = "CUDAExecutionProvider" in ort.get_available_providers()
    sess_gpu = None
    if gpu_available:
        sess_gpu = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        print(f"GPU session providers: {sess_gpu.get_providers()}")
    else:
        print("WARNING: CUDAExecutionProvider not available, skipping GPU bench")

    # Also benchmark PyTorch GPU for reference
    input_name = sess_cpu.get_inputs()[0].name

    # ── Benchmark ──
    batch_sizes = [2, 8, 32, 128, 512, 1024]

    header = f"{'Batch':>6}  {'CPU Mean':>10} {'CPU Med':>10} {'CPU P95':>10}"
    if sess_gpu:
        header += f"  {'GPU Mean':>10} {'GPU Med':>10} {'GPU P95':>10}"
    header += f"  {'PT GPU Mean':>12} {'PT GPU Med':>12}"
    header += f"  {'Winner':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for bs in batch_sizes:
        np.random.seed(bs)
        X = make_input(cfg, sparse_names, all_names, bs)

        # CPU ONNX
        tc = bench_session(sess_cpu, input_name, X, n_warmup=20, n_runs=n_runs)

        # GPU ONNX
        tg = None
        if sess_gpu:
            tg = bench_session(sess_gpu, input_name, X, n_warmup=20, n_runs=n_runs)

        # PyTorch GPU
        X_torch = torch.tensor(X, dtype=torch.float32).to("cuda")
        # warmup
        with torch.no_grad():
            for _ in range(20):
                model(X_torch)
            torch.cuda.synchronize()

        pt_times = []
        with torch.no_grad():
            for _ in range(n_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(X_torch)
                torch.cuda.synchronize()
                pt_times.append(time.perf_counter() - t0)
        tp = np.array(pt_times) * 1000

        # Determine winner
        candidates = {"CPU": np.median(tc), "PT-GPU": np.median(tp)}
        if tg is not None:
            candidates["ORT-GPU"] = np.median(tg)
        winner = min(candidates, key=candidates.get)

        row = (f"{bs:>6}  {tc.mean():>9.3f}ms {np.median(tc):>9.3f}ms "
               f"{np.percentile(tc, 95):>9.3f}ms")
        if tg is not None:
            row += (f"  {tg.mean():>9.3f}ms {np.median(tg):>9.3f}ms "
                    f"{np.percentile(tg, 95):>9.3f}ms")
        row += (f"  {tp.mean():>11.3f}ms {np.median(tp):>11.3f}ms")
        row += f"  {winner:>8}"
        print(row)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()

    configs = list(CONFIGS.keys()) if args.config == "all" else [args.config]
    for cfg_name in configs:
        run_comparison(cfg_name, n_runs=args.runs)
