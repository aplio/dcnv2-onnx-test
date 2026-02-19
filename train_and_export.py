"""Train a minimal DCN V2 (DCNMix) and export to ONNX."""

import json
import numpy as np
import pandas as pd
import torch

from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DCNMix

# ── 1. Feature definition ──
sparse_features = ["user_id", "item_id", "category"]
dense_features = ["price", "rating"]

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

# ── 3. Train ──
model = DCNMix(
    linear_feature_columns=feature_columns,
    dnn_feature_columns=feature_columns,
    cross_num=2,
    dnn_hidden_units=(32, 16),
    task="binary",
    device="cuda",
)
model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy"])

all_features = sparse_features + dense_features
model_input = {name: data[name].values for name in all_features}
model.fit(model_input, labels, batch_size=32, epochs=3, verbose=1)

# ── 4. Build test samples & get PyTorch predictions ──
model.eval()

# Various sparse ID patterns to test embedding lookup coverage
test_data = pd.DataFrame({
    "user_id":  [0, 50, 99, 1, 42, 77, 10, 88],
    "item_id":  [0, 250, 499, 3, 100, 400, 55, 321],
    "category": [0, 10, 19, 5, 2, 15, 8, 11],
    "price":    [9.99, 0.0, 100.0, 50.5, 25.0, 75.0, 1.0, 99.99],
    "rating":   [4.5, 0.0, 5.0, 2.5, 3.0, 1.0, 4.0, 3.5],
})

# Build input tensor: columns in feature_columns order
# [user_id, item_id, category, price, rating]
tensor_parts = []
for name in all_features:
    vals = test_data[name].values
    dtype = torch.long if name in sparse_features else torch.float32
    t = torch.tensor(vals, dtype=dtype).unsqueeze(1)
    tensor_parts.append(t)
dummy_input = torch.cat([t.float() for t in tensor_parts], dim=-1).to("cuda")

with torch.no_grad():
    pt_preds = model(dummy_input).cpu().numpy().flatten()

print(f"\nInput shape: {dummy_input.shape}")
print(f"PyTorch predictions: {pt_preds}")

# ── 5. Export to ONNX (model & input must be on same device) ──
onnx_path = "dcnv2.onnx"
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

# ── 6. Save test data & reference predictions for the inference script ──
ref = {
    "feature_order": all_features,
    "sparse_features": sparse_features,
    "test_data": test_data.to_dict(orient="list"),
    "pytorch_predictions": pt_preds.tolist(),
}
with open("reference.json", "w") as f:
    json.dump(ref, f, indent=2)
print("Reference saved: reference.json")
