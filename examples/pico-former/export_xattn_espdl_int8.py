#!/usr/bin/env python3
"""Pico-Former cross-attention + heads → ESP-PPQ INT8 .espdl (S3), golden-window gate."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "espdl"


def resolve_weights(cli: Path | None) -> Path:
    path = cli if cli is not None else Path(os.environ.get("PICO_FORMER_WEIGHTS", ""))
    if not path or not path.is_file():
        raise SystemExit("set --weights or PICO_FORMER_WEIGHTS to Pico-Former ceqt_weights.h")
    return path


N_TOK, D, H, DK = 96, 48, 4, 12
N_OUT, N_EVT = 48, 16
GELU_K = 0.7978845608


def parse_header(path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    text = path.read_text()
    scalars = {k: float(v) for k, v in re.findall(r"#define CEQT_(\w+) ([0-9.eE+-]+)f?", text)}
    arrays: dict[str, np.ndarray] = {}
    for name, body in re.findall(r"static const float (\w+)\[[^\]]+\] = \{([^;]+)\}", text, flags=re.S):
        arrays[name] = np.fromstring(body.replace("f", " ").replace(",", " "), sep=" ", dtype=np.float32)
    return scalars, arrays


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(GELU_K * (x + 0.044715 * x * x * x)))


class XattnHeads(nn.Module):
    def __init__(self, w: dict[str, np.ndarray], y_std: float):
        super().__init__()
        self.y_std = y_std
        self.register_buffer("curve_q", torch.from_numpy(w["W_CURVE_Q"].reshape(N_OUT, D).copy()))
        self.register_buffer("event_q", torch.from_numpy(w["W_EVENT_Q"].reshape(N_EVT, D).copy()))
        packed = w["W_CROSS_IN"].reshape(3 * D, D)
        self.register_buffer("wq", torch.from_numpy(packed[:D].copy()))
        self.register_buffer("bq", torch.from_numpy(w["B_CROSS_IN"][:D].copy()))
        self.register_buffer("wkv", torch.from_numpy(packed[D:].copy()))
        self.register_buffer("bkv", torch.from_numpy(w["B_CROSS_IN"][D:].copy()))
        self.register_buffer("wo", torch.from_numpy(w["W_CROSS_OUT"].reshape(D, D).copy()))
        self.register_buffer("bo", torch.from_numpy(w["B_CROSS_OUT"].copy()))
        self.register_buffer("wd1", torch.from_numpy(w["W_DELTA1"].reshape(D, D).copy()))
        self.register_buffer("bd1", torch.from_numpy(w["B_DELTA1"].copy()))
        self.register_buffer("wd2", torch.from_numpy(w["W_DELTA2"].reshape(1, D).copy()))
        self.register_buffer("bd2", torch.from_numpy(w["B_DELTA2"].copy()))
        self.register_buffer("we1", torch.from_numpy(w["W_EVT1"].reshape(D, D).copy()))
        self.register_buffer("be1", torch.from_numpy(w["B_EVT1"].copy()))
        self.register_buffer("we2", torch.from_numpy(w["W_EVT2"].reshape(1, D).copy()))
        self.register_buffer("be2", torch.from_numpy(w["B_EVT2"].copy()))

    def cross(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        Q = F.linear(q, self.wq, self.bq)
        kv_p = F.linear(kv, self.wkv, self.bkv)
        k, v = kv_p.split(D, dim=-1)
        scale = 1.0 / math.sqrt(DK)
        parts = []
        for hd in range(H):
            sl = slice(hd * DK, (hd + 1) * DK)
            scores = torch.matmul(Q[..., sl], k[..., sl].transpose(-1, -2)) * scale
            attn = torch.softmax(scores, dim=-1)
            parts.append(torch.matmul(attn, v[..., sl]))
        return F.linear(torch.cat(parts, dim=-1), self.wo, self.bo)

    def forward(self, tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        curve = self.cross(self.curve_q.unsqueeze(0).expand(tok.size(0), -1, -1), tok)
        event = self.cross(self.event_q.unsqueeze(0).expand(tok.size(0), -1, -1), tok)
        delta = 40.0 * torch.tanh(F.linear(gelu_tanh(F.linear(curve, self.wd1, self.bd1)), self.wd2, self.bd2) / 40.0)
        delta = delta.squeeze(-1) * self.y_std
        logits = F.linear(gelu_tanh(F.linear(event, self.we1, self.be1)), self.we2, self.be2).squeeze(-1)
        return delta, logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=None, help="Pico-Former ceqt_weights.h")
    args = ap.parse_args()

    from export_xattn_tflite_fp32 import numpy_full

    from esp_ppq.api import espdl_quantize_torch
    from esp_ppq.executor.torch import TorchExecutor

    scalars, w = parse_header(resolve_weights(args.weights))
    gold = numpy_full(w, scalars["Y_MEAN"], scalars["Y_STD"])
    net = XattnHeads(w, scalars["Y_STD"]).eval()
    tok = torch.from_numpy(gold["tok"][None].copy())
    with torch.no_grad():
        d, lg = net(tok)
    dmax = float(torch.max(torch.abs(d[0] - torch.from_numpy(w["G_DELTA"]))))
    lmax = float(torch.max(torch.abs(lg[0] - torch.from_numpy(w["G_LOGITS"]))))
    print(f"fp32 torch dMax={dmax:.4f} lMax={lmax:.6f}")
    if dmax > 0.01 or lmax > 1e-3:
        print("FAIL torch rebuild")
        return 1

    noise = [gold["tok"] + rng.normal(0, 0.02, gold["tok"].shape).astype(np.float32) for rng in [np.random.default_rng(i) for i in range(16)]]
    calib = np.stack([gold["tok"]] + noise)
    ds = TensorDataset(torch.from_numpy(calib))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    def collate_fn(batch):
        return batch[0].to("cpu")

    OUT_DIR.mkdir(exist_ok=True)
    espdl_path = str(OUT_DIR / "xattn_int8.espdl")
    graph = espdl_quantize_torch(
        model=net,
        espdl_export_file=espdl_path,
        calib_dataloader=loader,
        calib_steps=min(17, len(ds)),
        input_shape=[1, N_TOK, D],
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
    yq = executor(tok)
    dq = yq[0].detach().cpu().numpy().reshape(N_OUT)
    lq = yq[1].detach().cpu().numpy().reshape(N_EVT)
    last = float(gold["last"])
    yhat = last + dq
    td = float(np.max(np.abs(dq - w["G_DELTA"])))
    tl = float(np.max(np.abs(lq - w["G_LOGITS"])))
    ty = float(np.max(np.abs(yhat - w["G_YHAT"])))
    print(f"int8 golden dMax={td:.4f} lMax={tl:.6f} yMax={ty:.4f}")
    print(f"wrote {espdl_path} bytes={Path(espdl_path).stat().st_size}")
    (OUT_DIR / "host_gate.json").write_text(
        "{\n"
        f'  "int8_dMax": {td:.8e},\n'
        f'  "int8_lMax": {tl:.8e},\n'
        f'  "int8_yMax": {ty:.8e}\n'
        "}\n"
    )
    print("PASS host INT8 xattn export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
