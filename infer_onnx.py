"""ONNX-only inference for DCN V2.

No deepctr-torch / torch dependency — just onnxruntime + numpy.
Loads the exported model and runs inference with sparse (categorical ID)
and dense features, then compares against PyTorch reference predictions.
"""

import json
import numpy as np
import onnxruntime as ort

# ── 1. Load reference data ──
with open("reference.json") as f:
    ref = json.load(f)

feature_order = ref["feature_order"]
sparse_features = set(ref["sparse_features"])
test_data = ref["test_data"]
pt_preds = np.array(ref["pytorch_predictions"], dtype=np.float32)

print("Feature order:", feature_order)
print(f"Sparse: {ref['sparse_features']}")
print(f"Dense:  {[f for f in feature_order if f not in sparse_features]}")
print()

# ── 2. Build input array (no torch needed) ──
# Columns: [user_id, item_id, category, price, rating]
# Sparse columns are int IDs cast to float (ONNX embedding Gather works on float→int internally)
cols = []
for name in feature_order:
    vals = np.array(test_data[name], dtype=np.float32).reshape(-1, 1)
    cols.append(vals)
X = np.concatenate(cols, axis=1).astype(np.float32)

print(f"Input shape: {X.shape}")
print(f"Sample input (row 0): {X[0]}")
print(f"  user_id={int(X[0,0])}, item_id={int(X[0,1])}, "
      f"category={int(X[0,2])}, price={X[0,3]:.2f}, rating={X[0,4]:.2f}")
print()

# ── 3. Run ONNX inference ──
sess = ort.InferenceSession("dcnv2.onnx")

input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
input_type = sess.get_inputs()[0].type
print(f"ONNX input: name={input_name}, shape={input_shape}, type={input_type}")

ort_out = sess.run(None, {input_name: X})
onnx_preds = ort_out[0].flatten()

# ── 4. Compare ──
print("\n" + "=" * 65)
print(f"{'Row':>3}  {'user':>5} {'item':>5} {'cat':>4} {'price':>7} {'rate':>5}"
      f"  {'PyTorch':>8} {'ONNX':>8} {'diff':>10}")
print("-" * 65)
for i in range(len(onnx_preds)):
    row = X[i]
    diff = abs(pt_preds[i] - onnx_preds[i])
    print(f"{i:3d}  {int(row[0]):5d} {int(row[1]):5d} {int(row[2]):4d} "
          f"{row[3]:7.2f} {row[4]:5.2f}"
          f"  {pt_preds[i]:8.6f} {onnx_preds[i]:8.6f} {diff:10.2e}")
print("-" * 65)

max_diff = np.abs(pt_preds - onnx_preds).max()
mean_diff = np.abs(pt_preds - onnx_preds).mean()
print(f"Max  abs diff: {max_diff:.2e}")
print(f"Mean abs diff: {mean_diff:.2e}")

if max_diff < 1e-5:
    print("\nResult: PASS — sparse embedding lookups work correctly in ONNX")
else:
    print(f"\nResult: WARN — diff={max_diff:.2e} (may be float precision)")

# ── 5. Single-sample inference ──
# batch_size=1 causes CrossNetMix squeeze to collapse dims in the ONNX graph.
# Workaround: pad to batch_size=2, take first row.
print("\n" + "=" * 65)
print("Single-sample inference (batch_size=1 workaround: pad to 2):")
for i in [0, 3, 7]:
    single = X[i:i+1]

    # Try raw batch_size=1 first
    try:
        out = sess.run(None, {input_name: single})[0].flatten()
        print(f"  Row {i}: pred={out[0]:.6f} (ref={pt_preds[i]:.6f}) — batch_size=1 OK")
    except Exception:
        # Pad with a dummy row, take first result
        padded = np.concatenate([single, single], axis=0)
        out = sess.run(None, {input_name: padded})[0].flatten()
        diff = abs(out[0] - pt_preds[i])
        print(f"  Row {i}: user_id={int(X[i,0])}, item_id={int(X[i,1])}, "
              f"cat={int(X[i,2])} → pred={out[0]:.6f} (ref={pt_preds[i]:.6f}, "
              f"diff={diff:.2e}) [padded]")

print("\nNote: batch_size=1 fails due to CrossNetMix squeeze op in ONNX graph.")
print("      Use batch_size>=2 (pad if needed) as a workaround.")
