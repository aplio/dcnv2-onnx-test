"""Minimal DCN V2 model from deepctr-torch → ONNX conversion test.

Tests both the new dynamo exporter and the legacy TorchScript exporter.
"""

import numpy as np
import pandas as pd
import torch
import onnx
import onnxruntime as ort

from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DCNMix  # DCNMix = DCN V2

# ── 1. Define minimal feature columns ──
sparse_features = ["C1", "C2"]
dense_features = ["D1", "D2"]

feature_columns = [
    SparseFeat(name="C1", vocabulary_size=10, embedding_dim=4),
    SparseFeat(name="C2", vocabulary_size=8, embedding_dim=4),
    DenseFeat(name="D1", dimension=1),
    DenseFeat(name="D2", dimension=1),
]

# ── 2. Create dummy data ──
n = 32
np.random.seed(42)
data = pd.DataFrame({
    "C1": np.random.randint(0, 10, n),
    "C2": np.random.randint(0, 8, n),
    "D1": np.random.randn(n),
    "D2": np.random.randn(n),
})
labels = np.random.randint(0, 2, n).astype(np.float32)

# ── 3. Build & briefly train DCN V2 (DCNMix) ──
model = DCNMix(
    linear_feature_columns=feature_columns,
    dnn_feature_columns=feature_columns,
    cross_num=2,
    dnn_hidden_units=(16, 8),
    task="binary",
    device="cuda",
)

model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy"])

model_input = {name: data[name].values for name in sparse_features + dense_features}
model.fit(model_input, labels, batch_size=16, epochs=1, verbose=0)

# ── 4. Prepare dummy input tensor for export ──
model.eval()

# Build input tensor (batch_size=2 to avoid dim squeeze issues)
batch_size = 2
test_tensor_list = []
for name in sparse_features + dense_features:
    t = torch.tensor(data[name].values[:batch_size],
                     dtype=torch.long if name in sparse_features else torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(1)
    test_tensor_list.append(t)
dummy_input = torch.cat([t.float() for t in test_tensor_list], dim=-1).to("cuda")

print(f"Input shape: {dummy_input.shape}")
print("=" * 60)


def validate_and_compare(onnx_path, dummy_input, model):
    """Validate ONNX model and compare outputs with PyTorch."""
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model validation passed.")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    ort_input = {sess.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_out = sess.run(None, ort_input)
    print(f"  ONNX inference output: {ort_out[0].flatten()}")

    with torch.no_grad():
        pt_out = model(dummy_input).cpu().numpy()
    print(f"  PyTorch output:        {pt_out.flatten()}")
    diff = np.abs(pt_out - ort_out[0]).max()
    print(f"  Max abs diff: {diff:.6e}")


# ── 5a. Try dynamo exporter (new, default in PyTorch 2.x) ──
print("\n[1] Dynamo exporter (torch.onnx.export, dynamo=True)")
onnx_path_dynamo = "dcnv2_dynamo.onnx"
try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path_dynamo,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"  Export SUCCEEDED: {onnx_path_dynamo}")
    validate_and_compare(onnx_path_dynamo, dummy_input, model)
except Exception as e:
    print(f"  Export FAILED: {type(e).__name__}: {e}")

# ── 5b. Try legacy TorchScript exporter ──
print("\n[2] Legacy TorchScript exporter (dynamo=False)")
onnx_path_legacy = "dcnv2_legacy.onnx"
try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path_legacy,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"  Export SUCCEEDED: {onnx_path_legacy}")
    validate_and_compare(onnx_path_legacy, dummy_input, model)
except Exception as e:
    print(f"  Export FAILED: {type(e).__name__}: {e}")

# ── 5c. Try torch.jit.trace + export ──
print("\n[3] TorchScript trace → ONNX")
onnx_path_trace = "dcnv2_trace.onnx"
try:
    traced = torch.jit.trace(model, dummy_input)
    torch.onnx.export(
        traced,
        dummy_input,
        onnx_path_trace,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"  Export SUCCEEDED: {onnx_path_trace}")
    validate_and_compare(onnx_path_trace, dummy_input, model)
except Exception as e:
    print(f"  Export FAILED: {type(e).__name__}: {e}")
