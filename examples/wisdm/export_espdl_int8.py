#!/usr/bin/env python3
"""Rebuild the locked WISDM Encoder and export ESP-PPQ INT8 .espdl (S3)."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
HEADER = ROOT / "wisdm_weights.hpp"
OUT_DIR = ROOT / "espdl"
RAW = ROOT / "data" / "WISDM_ar_v1.1_raw.txt"
L, C_IN, D, H, FF, N_CLS, DK = 80, 3, 16, 2, 32, 6, 8
EPS = 1e-5


def parse_header(path: Path) -> dict[str, np.ndarray]:
    text = path.read_text()
    arrays: dict[str, np.ndarray] = {}
    for name, body in re.findall(r"static const float (\w+)\[[^\]]+\] = \{([^;]+)\}", text, flags=re.S):
        arrays[name] = np.fromstring(body.replace("f", " ").replace(",", " "), sep=" ", dtype=np.float32)
    return arrays


def load_weights(w: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    t = {k: torch.from_numpy(v.copy()) for k, v in w.items()}
    return {
        "proj.weight": t["W_PROJ"].reshape(D, C_IN),
        "proj.bias": t["B_PROJ"],
        "pos": t["W_POS"].reshape(1, L, D),
        "w_in": t["W_IN"].reshape(3 * D, D),
        "b_in": t["B_IN"],
        "w_out": t["W_OUT"].reshape(D, D),
        "b_out": t["B_OUT"],
        "n1w": t["W_N1"],
        "n1b": t["B_N1"],
        "n2w": t["W_N2"],
        "n2b": t["B_N2"],
        "w_ff1": t["W_FF1"].reshape(FF, D),
        "b_ff1": t["B_FF1"],
        "w_ff2": t["W_FF2"].reshape(D, FF),
        "b_ff2": t["B_FF2"],
        "w_cls": t["W_CLS"].reshape(N_CLS, D),
        "b_cls": t["B_CLS"],
    }


class WisdmSplit(nn.Module):
    """Same numbers as TinyHAREnc / PicoTorch, MHA unrolled for ONNX / ESP-PPQ."""

    def __init__(self, p: dict[str, torch.Tensor]):
        super().__init__()
        self.proj = nn.Linear(C_IN, D)
        self.proj.weight = nn.Parameter(p["proj.weight"].clone())
        self.proj.bias = nn.Parameter(p["proj.bias"].clone())
        self.pos = nn.Parameter(p["pos"].clone())
        self.register_buffer("w_in", p["w_in"].clone())
        self.register_buffer("b_in", p["b_in"].clone())
        self.register_buffer("w_out", p["w_out"].clone())
        self.register_buffer("b_out", p["b_out"].clone())
        self.register_buffer("n1w", p["n1w"].clone())
        self.register_buffer("n1b", p["n1b"].clone())
        self.register_buffer("n2w", p["n2w"].clone())
        self.register_buffer("n2b", p["n2b"].clone())
        self.register_buffer("w_ff1", p["w_ff1"].clone())
        self.register_buffer("b_ff1", p["b_ff1"].clone())
        self.register_buffer("w_ff2", p["w_ff2"].clone())
        self.register_buffer("b_ff2", p["b_ff2"].clone())
        self.register_buffer("w_cls", p["w_cls"].clone())
        self.register_buffer("b_cls", p["b_cls"].clone())

    @staticmethod
    def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + EPS) * gamma + beta

    def mha(self, h: torch.Tensor) -> torch.Tensor:
        qkv = F.linear(h, self.w_in, self.b_in)
        q, k, v = qkv.split(D, dim=-1)
        scale = 1.0 / math.sqrt(DK)
        parts = []
        for hd in range(H):
            sl = slice(hd * DK, (hd + 1) * DK)
            scores = torch.matmul(q[..., sl], k[..., sl].transpose(-1, -2)) * scale
            attn = torch.softmax(scores, dim=-1)
            parts.append(torch.matmul(attn, v[..., sl]))
        return F.linear(torch.cat(parts, dim=-1), self.w_out, self.b_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos
        h = self.layernorm(h + self.mha(h), self.n1w, self.n1b)
        ff = F.relu(F.linear(h, self.w_ff1, self.b_ff1))
        h = self.layernorm(h + F.linear(ff, self.w_ff2, self.b_ff2), self.n2w, self.n2b)
        return F.linear(h.mean(dim=1), self.w_cls, self.b_cls)


def calib_windows(n: int = 128) -> np.ndarray:
    sys.path.insert(0, str(ROOT))
    from train_export import parse_raw, windows

    users, labs, xyz = parse_raw(RAW)
    x_tr, _, _, _ = windows(users, labs, xyz, set(range(1, 29)))
    rng = np.random.default_rng(20260901)
    pick = rng.choice(len(x_tr), size=min(n, len(x_tr)), replace=False)
    return x_tr[pick]


def main() -> int:
    from esp_ppq.api import espdl_quantize_torch
    from esp_ppq.executor.torch import TorchExecutor

    w = parse_header(HEADER)
    p = load_weights(w)
    net = WisdmSplit(p).eval()
    probe_x = w["PROBE_X"].reshape(32, L, C_IN)
    probe_logits = w["PROBE_LOGITS"].reshape(32, N_CLS)

    with torch.no_grad():
        y = net(torch.from_numpy(probe_x)).numpy()
    fp_err = float(np.max(np.abs(y - probe_logits)))
    agree = int((y.argmax(-1) == probe_logits.argmax(-1)).sum())
    print(f"fp32 split vs header logit_maxabs={fp_err:.6e} top1_agree={agree / 32:.3f}")
    if fp_err > 1e-4 or agree != 32:
        print("FAIL fp32 rebuild")
        return 1

    x_cal = calib_windows(128)
    ds = TensorDataset(torch.from_numpy(x_cal))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    def collate_fn(batch):
        return batch[0].to("cpu")

    OUT_DIR.mkdir(exist_ok=True)
    espdl_path = str(OUT_DIR / "wisdm_int8.espdl")
    graph = espdl_quantize_torch(
        model=net,
        espdl_export_file=espdl_path,
        calib_dataloader=loader,
        calib_steps=min(32, len(ds)),
        input_shape=[1, L, C_IN],
        target="esp32s3",
        num_of_bits=8,
        collate_fn=collate_fn,
        device="cpu",
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
        opset_version=17,
    )
    executor = TorchExecutor(graph=graph, device="cpu")
    q_err = 0.0
    q_agree = 0
    for i in range(32):
        xin = torch.from_numpy(probe_x[i : i + 1])
        yq = executor(xin)[0].detach().cpu().numpy().reshape(N_CLS)
        q_err = max(q_err, float(np.max(np.abs(yq - probe_logits[i]))))
        q_agree += int(yq.argmax() == probe_logits[i].argmax())
        if i == 0:
            print("int8 probe0", " ".join(f"{v:.4f}" for v in yq), f"pred={int(yq.argmax())}")
    print(f"int8 vs fp32 header logit_maxabs={q_err:.6e} top1_agree={q_agree / 32:.3f}")
    print(f"wrote {espdl_path} bytes={Path(espdl_path).stat().st_size}")
    (OUT_DIR / "host_gate.json").write_text(
        "{\n"
        f'  "fp32_logit_maxabs": {fp_err:.8e},\n'
        f'  "int8_logit_maxabs": {q_err:.8e},\n'
        f'  "int8_top1_agree": {q_agree / 32:.6f},\n'
        '  "target": "esp32s3",\n'
        '  "bits": 8\n'
        "}\n"
    )
    print("PASS host INT8 export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
