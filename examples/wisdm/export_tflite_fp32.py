#!/usr/bin/env python3
"""Export the locked WISDM Encoder as a split-MHA FP32 TFLite graph and gate it."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
HEADER = ROOT / "wisdm_weights.hpp"
OUT_DIR = ROOT / "tflite"
L, D, H, FF, N_CLS, DK = 80, 16, 2, 32, 6, 8
EPS = 1e-5


def parse_header(path: Path) -> dict[str, np.ndarray]:
    text = path.read_text()
    arrays: dict[str, np.ndarray] = {}
    for name, body in re.findall(
        r"static const float (\w+)\[[^\]]+\] = \{([^;]+)\}", text, flags=re.S
    ):
        vals = np.fromstring(body.replace("f", " ").replace(",", " "), sep=" ", dtype=np.float32)
        arrays[name] = vals
    enums = dict(re.findall(r"WISDM_(\w+) = (\d+)", text))
    assert int(enums["L"]) == L and int(enums["PROBE"]) == 32
    return arrays


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * gamma + beta


def numpy_forward(x: np.ndarray, w: dict[str, np.ndarray]) -> np.ndarray:
    """x: [L, 3] → logits [6]. Matches PicoTorch / PyTorch MHA packed in_proj."""
    h = x @ w["W_PROJ"].reshape(D, 3).T + w["B_PROJ"]
    h = h + w["W_POS"].reshape(L, D)
    qkv = h @ w["W_IN"].reshape(3 * D, D).T + w["B_IN"]
    q, k, v = np.split(qkv, 3, axis=-1)
    scale = 1.0 / np.sqrt(DK)
    heads = []
    for hd in range(H):
        sl = slice(hd * DK, (hd + 1) * DK)
        scores = (q[:, sl] @ k[:, sl].T) * scale
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        heads.append(attn @ v[:, sl])
    combo = np.concatenate(heads, axis=-1)
    a = combo @ w["W_OUT"].reshape(D, D).T + w["B_OUT"]
    h = layernorm(h + a, w["W_N1"], w["B_N1"])
    ff = np.maximum(h @ w["W_FF1"].reshape(FF, D).T + w["B_FF1"], 0.0)
    h = layernorm(h + ff @ w["W_FF2"].reshape(D, FF).T + w["B_FF2"], w["W_N2"], w["B_N2"])
    pooled = h.mean(axis=0)
    return pooled @ w["W_CLS"].reshape(N_CLS, D).T + w["B_CLS"]


def dense(x, kernel_out_in: np.ndarray, bias: np.ndarray, out_dim: int, in_dim: int):
    kernel = tf.constant(kernel_out_in.reshape(out_dim, in_dim).T)  # [in, out]
    return tf.linalg.matmul(x, kernel) + tf.constant(bias)


def tf_layernorm(x, gamma, beta):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * tf.math.rsqrt(var + EPS) * tf.constant(gamma) + tf.constant(beta)


def build_tf_fn(w: dict[str, np.ndarray]):
    @tf.function(input_signature=[tf.TensorSpec([1, L, 3], tf.float32)], autograph=False)
    def wisdm(x):
        x = tf.squeeze(x, 0)
        h = dense(x, w["W_PROJ"], w["B_PROJ"], D, 3) + tf.constant(w["W_POS"].reshape(L, D))
        qkv = dense(h, w["W_IN"], w["B_IN"], 3 * D, D)
        q, k, v = tf.split(qkv, 3, axis=-1)
        scale = tf.constant(1.0 / np.sqrt(DK), tf.float32)
        parts = []
        for hd in range(H):
            qh = tf.slice(q, [0, hd * DK], [L, DK])
            kh = tf.slice(k, [0, hd * DK], [L, DK])
            vh = tf.slice(v, [0, hd * DK], [L, DK])
            scores = tf.raw_ops.MatMul(a=qh, b=kh, transpose_a=False, transpose_b=True) * scale
            attn = tf.nn.softmax(scores, axis=-1)
            # attn @ V as weighted sum so the converter stays off BATCH_MATMUL (TFLM resolver).
            cols = []
            for col in range(DK):
                cols.append(tf.reduce_sum(attn * vh[:, col], axis=1))
            parts.append(tf.stack(cols, axis=1))
        a = dense(tf.concat(parts, axis=-1), w["W_OUT"], w["B_OUT"], D, D)
        h = tf_layernorm(h + a, w["W_N1"], w["B_N1"])
        ff = tf.nn.relu(dense(h, w["W_FF1"], w["B_FF1"], FF, D))
        h = tf_layernorm(h + dense(ff, w["W_FF2"], w["B_FF2"], D, FF), w["W_N2"], w["B_N2"])
        pooled = tf.reduce_mean(h, axis=0, keepdims=True)
        logits = dense(pooled, w["W_CLS"], w["B_CLS"], N_CLS, D)
        return tf.reshape(logits, [1, N_CLS])

    return wisdm


def xxd_header(blob: bytes, path: Path) -> None:
    lines = [
        "#pragma once\n",
        "#include <cstddef>\n",
        f"enum {{ WISDM_TFLITE_LEN = {len(blob)} }};\n",
        "alignas(16) static const unsigned char WISDM_TFLITE[] = {\n",
    ]
    for i in range(0, len(blob), 16):
        chunk = ", ".join(f"0x{b:02x}" for b in blob[i : i + 16])
        lines.append(f"  {chunk},\n")
    lines.append("};\n")
    path.write_text("".join(lines))


def main() -> int:
    w = parse_header(HEADER)
    probe_x = w["PROBE_X"].reshape(32, L, 3)
    probe_logits = w["PROBE_LOGITS"].reshape(32, N_CLS)

    np_err = 0.0
    agree = 0
    for i in range(32):
        y = numpy_forward(probe_x[i], w)
        np_err = max(np_err, float(np.max(np.abs(y - probe_logits[i]))))
        agree += int(y.argmax() == probe_logits[i].argmax())
    print(f"numpy vs header logit_maxabs={np_err:.6e} top1_agree={agree / 32:.3f}")
    if np_err > 1e-4 or agree != 32:
        print("FAIL numpy")
        return 1

    fn = build_tf_fn(w)
    concrete = fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], fn)
    converter.optimizations = []
    converter.target_spec.supported_types = [tf.float32]
    blob = converter.convert()
    OUT_DIR.mkdir(exist_ok=True)
    tflite_path = OUT_DIR / "wisdm_fp32.tflite"
    tflite_path.write_bytes(blob)
    xxd_header(blob, OUT_DIR / "wisdm_tflite_model.hpp")
    print(f"wrote {tflite_path} bytes={len(blob)}")

    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    tfl_err = 0.0
    tfl_agree = 0
    for i in range(32):
        interp.set_tensor(inp["index"], probe_x[i][None].astype(np.float32))
        interp.invoke()
        y = interp.get_tensor(out["index"])[0]
        tfl_err = max(tfl_err, float(np.max(np.abs(y - probe_logits[i]))))
        tfl_agree += int(y.argmax() == probe_logits[i].argmax())
        if i == 0:
            print("tflite probe0", " ".join(f"{v:.4f}" for v in y), f"pred={int(y.argmax())}")
    rate = tfl_agree / 32
    print(f"tflite probe n=32 logit_maxabs={tfl_err:.6e} top1_agree={rate:.3f}")
    ok = tfl_err <= 1e-4 and tfl_agree == 32
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
