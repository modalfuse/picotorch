#!/usr/bin/env python3
"""NumPy reference for PicoTorch Linear, LayerNorm, Softmax, and MHA (FP32)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EPS = 1e-5


def linear(x: np.ndarray, W: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    y = x @ W.T
    if b is not None:
        y = y + b
    return y.astype(np.float32)


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    return ((x - mean) * inv * gamma + beta).astype(np.float32)


def softmax_row(s: np.ndarray) -> np.ndarray:
    m = s.max(axis=-1, keepdims=True)
    e = np.exp(s - m)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def mha(q_src, kv_src, n_head, Wq, bq, Wk, bk, Wv, bv, Wo, bo, self_attn=True):
    Q = linear(q_src, Wq, bq)
    src = q_src if self_attn else kv_src
    K = linear(src, Wk, bk)
    V = linear(src, Wv, bv)
    nq, d = Q.shape
    dk = d // n_head
    scale = 1.0 / np.sqrt(np.float32(dk))
    combo = np.zeros((nq, d), dtype=np.float32)
    for h in range(n_head):
        qh = Q[:, h * dk : (h + 1) * dk]
        kh = K[:, h * dk : (h + 1) * dk]
        vh = V[:, h * dk : (h + 1) * dk]
        scores = (qh @ kh.T) * scale
        attn = softmax_row(scores)
        combo[:, h * dk : (h + 1) * dk] = attn @ vh
    return linear(combo, Wo, bo)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("host/ref_case.json"))
    args = p.parse_args()

    rng = np.random.default_rng(0)
    L, D, H = 8, 16, 2
    x = (0.01 * np.arange(1, L * D + 1, dtype=np.float32)).reshape(L, D)
    I = np.eye(D, dtype=np.float32)
    z = np.zeros(D, dtype=np.float32)
    y = mha(x, x, H, I, z, I, z, I, z, I, z, True)

    W = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    b = np.array([0.10, 0.20], dtype=np.float32)
    xin = np.array([[1, 2, 3]], dtype=np.float32)
    ylin = linear(xin, W, b)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "linear": ylin.reshape(-1).tolist(),
                "mha_row0": y[0].tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"linear {ylin.reshape(-1)}")
    print(f"mha_row0 {y[0]}")


if __name__ == "__main__":
    main()
