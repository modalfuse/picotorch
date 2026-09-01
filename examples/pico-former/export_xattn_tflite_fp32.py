#!/usr/bin/env python3
"""Export Pico-Former cross-attention + heads as a LiteRT FP32 graph and gate the golden window."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # ESP-PPQ env has no TF
    tf = None

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "tflite"


def resolve_weights(cli: Path | None) -> Path:
    path = cli if cli is not None else Path(os.environ.get("PICO_FORMER_WEIGHTS", ""))
    if not path or not path.is_file():
        raise SystemExit("set --weights or PICO_FORMER_WEIGHTS to Pico-Former ceqt_weights.h")
    return path


N_CGM, N_MARK, N_TOK, D, H, DK = 72, 24, 96, 48, 4, 12
N_OUT, N_EVT, FF = 48, 16, 96
EPS = 1e-5
GELU_K = 0.7978845608


def parse_header(path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    text = path.read_text()
    scalars = {k: float(v) for k, v in re.findall(r"#define CEQT_(\w+) ([0-9.eE+-]+)f?", text)}
    arrays: dict[str, np.ndarray] = {}
    for name, body in re.findall(r"static const float (\w+)\[[^\]]+\] = \{([^;]+)\}", text, flags=re.S):
        arrays[name] = np.fromstring(body.replace("f", " ").replace(",", " "), sep=" ", dtype=np.float32)
    return scalars, arrays


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(GELU_K * (x + 0.044715 * x * x * x)))


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * gamma + beta


def linear(x: np.ndarray, w_out_in: np.ndarray, b: np.ndarray, out_d: int, in_d: int) -> np.ndarray:
    return x @ w_out_in.reshape(out_d, in_d).T + b


def mha(q: np.ndarray, kv: np.ndarray, w_in: np.ndarray, b_in: np.ndarray, w_out: np.ndarray, b_out: np.ndarray) -> np.ndarray:
    nq, nk = q.shape[0], kv.shape[0]
    packed = w_in.reshape(3 * D, D)
    wq, wkv = packed[:D], packed[D:]
    bq, bkv = b_in[:D], b_in[D:]
    Q = q @ wq.T + bq
    kv_p = kv @ wkv.T + bkv
    K, V = kv_p[:, :D], kv_p[:, D:]
    scale = 1.0 / np.sqrt(DK)
    heads = []
    for hd in range(H):
        sl = slice(hd * DK, (hd + 1) * DK)
        scores = (Q[:, sl] @ K[:, sl].T) * scale
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        heads.append(attn @ V[:, sl])
    return np.concatenate(heads, axis=-1) @ w_out.reshape(D, D).T + b_out


def encoder_layer(h: np.ndarray, w: dict[str, np.ndarray], li: int) -> np.ndarray:
    a = mha(h, h, w[f"W_E{li}_IN"], w[f"B_E{li}_IN"], w[f"W_E{li}_OUT"], w[f"B_E{li}_OUT"])
    h = layernorm(h + a, w[f"W_E{li}_N1"], w[f"B_E{li}_N1"])
    ff = np.maximum(linear(h, w[f"W_E{li}_L1"], w[f"B_E{li}_L1"], FF, D), 0.0)
    return layernorm(h + linear(ff, w[f"W_E{li}_L2"], w[f"B_E{li}_L2"], D, FF), w[f"W_E{li}_N2"], w[f"B_E{li}_N2"])


def numpy_full(w: dict[str, np.ndarray], y_mean: float, y_std: float) -> dict[str, np.ndarray]:
    y_hist = w["G_Y_HIST"]
    marks = w["G_MARKS"].reshape(N_MARK, 3)
    mask = w["G_MASK"]
    tok = np.zeros((N_TOK, D), np.float32)
    for i in range(N_CGM):
        yn = (y_hist[i] - y_mean) / y_std
        tok[i] = w["W_CGM"] * yn + w["B_CGM"] + w["W_POS"].reshape(N_CGM, D)[i]
    for j in range(N_MARK):
        typ = int(np.clip(marks[j, 0], 0, 3))
        h = gelu(linear(marks[j, 1:3], w["W_MARK1"], w["B_MARK1"], D, 2))
        m2 = linear(h, w["W_MARK2"], w["B_MARK2"], D, D)
        tok[N_CGM + j] = (w["W_TYPE"].reshape(4, D)[typ] + m2) * mask[j]
    tok = encoder_layer(tok, w, 0)
    tok = encoder_layer(tok, w, 1)
    curve = mha(w["W_CURVE_Q"].reshape(N_OUT, D), tok, w["W_CROSS_IN"], w["B_CROSS_IN"], w["W_CROSS_OUT"], w["B_CROSS_OUT"])
    event = mha(w["W_EVENT_Q"].reshape(N_EVT, D), tok, w["W_CROSS_IN"], w["B_CROSS_IN"], w["W_CROSS_OUT"], w["B_CROSS_OUT"])
    head = gelu(linear(curve, w["W_DELTA1"], w["B_DELTA1"], D, D))
    delta = 40.0 * np.tanh(linear(head, w["W_DELTA2"], w["B_DELTA2"], 1, D) / 40.0) * y_std
    delta = delta.reshape(N_OUT)
    head_e = gelu(linear(event, w["W_EVT1"], w["B_EVT1"], D, D))
    logits = linear(head_e, w["W_EVT2"], w["B_EVT2"], 1, D).reshape(N_EVT)
    last = next((float(v) for v in y_hist[::-1] if np.isfinite(v)), float(y_hist[0]))
    yhat = last + 1.0 * delta
    return {"tok": tok, "curve": curve, "event": event, "delta": delta, "logits": logits, "yhat": yhat, "last": np.float32(last)}


def tf_dense(x, kernel_out_in, bias, out_d, in_d):
    return tf.linalg.matmul(x, tf.constant(kernel_out_in.reshape(out_d, in_d).T)) + tf.constant(bias)


def tf_gelu(x):
    k = tf.constant(GELU_K, tf.float32)
    return 0.5 * x * (1.0 + tf.tanh(k * (x + 0.044715 * x * x * x)))


def tf_cross(q, kv, w_in, b_in, w_out, b_out, nq):
    packed = w_in.reshape(3 * D, D)
    Q = tf_dense(q, packed[:D], b_in[:D], D, D)
    kv_p = tf_dense(kv, packed[D:], b_in[D:], 2 * D, D)
    k, v = tf.split(kv_p, 2, axis=-1)
    scale = tf.constant(1.0 / np.sqrt(DK), tf.float32)
    parts = []
    for hd in range(H):
        qh = tf.slice(Q, [0, hd * DK], [nq, DK])
        kh = tf.slice(k, [0, hd * DK], [N_TOK, DK])
        vh = tf.slice(v, [0, hd * DK], [N_TOK, DK])
        scores = tf.raw_ops.MatMul(a=qh, b=kh, transpose_a=False, transpose_b=True) * scale
        attn = tf.nn.softmax(scores, axis=-1)
        cols = [tf.reduce_sum(attn * vh[:, col], axis=1) for col in range(DK)]
        parts.append(tf.stack(cols, axis=1))
    return tf_dense(tf.concat(parts, axis=-1), w_out, b_out, D, D)


def build_tf_fn(w: dict[str, np.ndarray], y_std: float):
    @tf.function(input_signature=[tf.TensorSpec([1, N_TOK, D], tf.float32)], autograph=False)
    def xattn_heads(tok):
        tok = tf.squeeze(tok, 0)
        curve = tf_cross(
            tf.constant(w["W_CURVE_Q"].reshape(N_OUT, D)),
            tok,
            w["W_CROSS_IN"],
            w["B_CROSS_IN"],
            w["W_CROSS_OUT"],
            w["B_CROSS_OUT"],
            N_OUT,
        )
        event = tf_cross(
            tf.constant(w["W_EVENT_Q"].reshape(N_EVT, D)),
            tok,
            w["W_CROSS_IN"],
            w["B_CROSS_IN"],
            w["W_CROSS_OUT"],
            w["B_CROSS_OUT"],
            N_EVT,
        )
        head = tf_gelu(tf_dense(curve, w["W_DELTA1"], w["B_DELTA1"], D, D))
        delta = 40.0 * tf.tanh(tf_dense(head, w["W_DELTA2"], w["B_DELTA2"], 1, D) / 40.0) * tf.constant(y_std, tf.float32)
        head_e = tf_gelu(tf_dense(event, w["W_EVT1"], w["B_EVT1"], D, D))
        logits = tf_dense(head_e, w["W_EVT2"], w["B_EVT2"], 1, D)
        return tf.reshape(delta, [1, N_OUT]), tf.reshape(logits, [1, N_EVT])

    return xattn_heads


def xxd_header(blob: bytes, path: Path) -> None:
    lines = [
        "#pragma once\n",
        "#include <cstddef>\n",
        f"enum {{ XATTN_TFLITE_LEN = {len(blob)} }};\n",
        "alignas(16) static const unsigned char XATTN_TFLITE[] = {\n",
    ]
    for i in range(0, len(blob), 16):
        chunk = ", ".join(f"0x{b:02x}" for b in blob[i : i + 16])
        lines.append(f"  {chunk},\n")
    lines.append("};\n")
    path.write_text("".join(lines))


def c_array(name: str, a: np.ndarray) -> str:
    flat = np.asarray(a, dtype=np.float32).reshape(-1)
    body = ", ".join(f"{float(v):.8e}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{ {body} }};\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=None, help="Pico-Former ceqt_weights.h")
    args = ap.parse_args()
    scalars, w = parse_header(resolve_weights(args.weights))
    y_mean = scalars["Y_MEAN"]
    y_std = scalars["Y_STD"]
    gold = numpy_full(w, y_mean, y_std)
    dmax = float(np.max(np.abs(gold["delta"] - w["G_DELTA"])))
    lmax = float(np.max(np.abs(gold["logits"] - w["G_LOGITS"])))
    ymax = float(np.max(np.abs(gold["yhat"] - w["G_YHAT"])))
    print(f"numpy golden dMax={dmax:.4f} lMax={lmax:.6f} yMax={ymax:.4f}")
    if ymax > 0.0095 + 1e-4 or lmax > 5.8e-4 + 1e-4:
        print("FAIL numpy golden")
        return 1

    fn = build_tf_fn(w, y_std)
    concrete = fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], fn)
    converter.optimizations = []
    converter.target_spec.supported_types = [tf.float32]
    blob = converter.convert()
    OUT_DIR.mkdir(exist_ok=True)
    tflite_path = OUT_DIR / "xattn_fp32.tflite"
    tflite_path.write_bytes(blob)
    xxd_header(blob, OUT_DIR / "xattn_tflite_model.hpp")

    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    interp.set_tensor(inp["index"], gold["tok"][None].astype(np.float32))
    interp.invoke()
    # outputs may not be in declaration order
    got = {tuple(d["shape"]): interp.get_tensor(d["index"])[0] for d in outs}
    t_delta = got[(1, N_OUT)]
    t_logits = got[(1, N_EVT)]
    last = float(gold["last"])
    t_yhat = last + t_delta
    td = float(np.max(np.abs(t_delta - w["G_DELTA"])))
    tl = float(np.max(np.abs(t_logits - w["G_LOGITS"])))
    ty = float(np.max(np.abs(t_yhat - w["G_YHAT"])))
    print(f"tflite golden dMax={td:.4f} lMax={tl:.6f} yMax={ty:.4f} bytes={len(blob)}")

    probe = [
        "#pragma once\n",
        f"enum {{ XATTN_N_TOK = {N_TOK}, XATTN_D = {D}, XATTN_N_OUT = {N_OUT}, XATTN_N_EVT = {N_EVT}, XATTN_N_CGM = {N_CGM} }};\n",
        f"#define XATTN_LAST {last:.8e}f\n",
        c_array("G_TOK", gold["tok"]),
        c_array("G_DELTA", w["G_DELTA"]),
        c_array("G_LOGITS", w["G_LOGITS"]),
        c_array("G_YHAT", w["G_YHAT"]),
        c_array("G_Y_HIST", w["G_Y_HIST"]),
    ]
    (OUT_DIR / "xattn_probe.hpp").write_text("".join(probe))
    ok = ty <= 0.0095 + 1e-3 and tl <= 5.8e-4 + 1e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
