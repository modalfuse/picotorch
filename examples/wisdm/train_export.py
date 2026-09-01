#!/usr/bin/env python3
"""Train a 6-class WISDM Encoder and export PicoTorch weights + a 32-window probe."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
L, C_IN, D, H, FF, N_CLS = 80, 3, 16, 2, 32, 6
STRIDE = 40
LABELS = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
LAB2I = {n: i for i, n in enumerate(LABELS)}


def parse_raw(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users, labs, xyz = [], [], []
    with path.open("r", errors="replace") as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                u = int(parts[0])
                lab = parts[1]
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            except ValueError:
                continue
            if lab not in LAB2I:
                continue
            users.append(u)
            labs.append(LAB2I[lab])
            xyz.append((x, y, z))
    return np.asarray(users, np.int32), np.asarray(labs, np.int32), np.asarray(xyz, np.float32)


def windows(users, labs, xyz, train_users):
    x_tr, y_tr, x_te, y_te = [], [], [], []
    for u in np.unique(users):
        idx = np.where(users == u)[0]
        acc = xyz[idx]
        lab = labs[idx]
        for s in range(0, len(acc) - L + 1, STRIDE):
            sl = lab[s : s + L]
            if sl.min() != sl.max():
                continue
            win = acc[s : s + L]
            if u in train_users:
                x_tr.append(win)
                y_tr.append(int(sl[0]))
            else:
                x_te.append(win)
                y_te.append(int(sl[0]))
    return (
        np.stack(x_tr).astype(np.float32),
        np.asarray(y_tr, np.int64),
        np.stack(x_te).astype(np.float32),
        np.asarray(y_te, np.int64),
    )


class TinyHAREnc(nn.Module):
    """Linear proj + one PicoTorch-shaped Encoder layer + mean pool + linear."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(C_IN, D)
        self.pos = nn.Parameter(torch.zeros(1, L, D))
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.n1 = nn.LayerNorm(D)
        self.n2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D)
        self.cls = nn.Linear(D, N_CLS)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.n1(h + a)
        h = self.n2(h + self.ff2(F.relu(self.ff1(h))))
        return h.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.encode(x))


def f1_macro(pred: np.ndarray, y: np.ndarray) -> float:
    scores = []
    for c in range(N_CLS):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(scores))


def c_array(name: str, a: np.ndarray) -> str:
    flat = np.asarray(a, dtype=np.float32).reshape(-1)
    body = ", ".join(f"{float(v):.8e}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{ {body} }};\n"


def export_header(net: TinyHAREnc, probe_x: np.ndarray, probe_y: np.ndarray, probe_logits: np.ndarray, path: Path) -> None:
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    # nn.Linear: weight is out × in; MHA in_proj is 3d × d
    chunks = [
        "#pragma once\n",
        f"enum {{ WISDM_L = {L}, WISDM_D = {D}, WISDM_H = {H}, WISDM_FF = {FF}, WISDM_N = {N_CLS}, WISDM_PROBE = {len(probe_y)} }};\n",
        c_array("W_PROJ", sd["proj.weight"]),
        c_array("B_PROJ", sd["proj.bias"]),
        c_array("W_POS", sd["pos"][0]),
        c_array("W_IN", sd["attn.in_proj_weight"]),
        c_array("B_IN", sd["attn.in_proj_bias"]),
        c_array("W_OUT", sd["attn.out_proj.weight"]),
        c_array("B_OUT", sd["attn.out_proj.bias"]),
        c_array("W_N1", sd["n1.weight"]),
        c_array("B_N1", sd["n1.bias"]),
        c_array("W_N2", sd["n2.weight"]),
        c_array("B_N2", sd["n2.bias"]),
        c_array("W_FF1", sd["ff1.weight"]),
        c_array("B_FF1", sd["ff1.bias"]),
        c_array("W_FF2", sd["ff2.weight"]),
        c_array("B_FF2", sd["ff2.bias"]),
        c_array("W_CLS", sd["cls.weight"]),
        c_array("B_CLS", sd["cls.bias"]),
        c_array("PROBE_X", probe_x),
        "static const int PROBE_Y[] = { " + ", ".join(str(int(v)) for v in probe_y) + " };\n",
        c_array("PROBE_LOGITS", probe_logits),
    ]
    path.write_text("".join(chunks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "WISDM_ar_v1.1_raw.txt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()
    if not args.raw.exists():
        raise SystemExit(f"missing {args.raw}; download classic WISDM Actitracker first")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    users, labs, xyz = parse_raw(args.raw)
    train_users = set(range(1, 29))
    x_tr, y_tr, x_te, y_te = windows(users, labs, xyz, train_users)
    print(f"windows train={len(y_tr)} test={len(y_te)} users={np.unique(users).size}")

    net = TinyHAREnc()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x_tr_t = torch.from_numpy(x_tr)
    y_tr_t = torch.from_numpy(y_tr)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(y_tr))
        loss_sum, n = 0.0, 0
        for i in range(0, len(perm), 64):
            b = perm[i : i + 64]
            opt.zero_grad()
            logits = net(x_tr_t[b])
            loss = F.cross_entropy(logits, y_tr_t[b])
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(b)
            n += len(b)
        net.eval()
        with torch.no_grad():
            pred = net(torch.from_numpy(x_te)).argmax(-1).numpy()
        acc = float((pred == y_te).mean())
        print(f"epoch {ep+1} loss={loss_sum/n:.4f} test_acc={acc:.4f} f1={f1_macro(pred, y_te):.4f}")

    net.eval()
    with torch.no_grad():
        te_log = net(torch.from_numpy(x_te)).numpy()
    pred = te_log.argmax(-1)
    acc = float((pred == y_te).mean())
    f1 = f1_macro(pred, y_te)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(y_te), size=32, replace=False)
    export_header(net, x_te[pick], y_te[pick], te_log[pick], ROOT / "wisdm_weights.hpp")
    (ROOT / "gates.json").write_text(
        "{\n"
        f'  "test_top1": {acc:.6f},\n'
        f'  "test_macro_f1": {f1:.6f},\n'
        f'  "probe": 32,\n'
        '  "logit_maxabs": 1e-4,\n'
        '  "probe_top1_agree": 1.0,\n'
        '  "split": "users 1-28 train, 29-36 test",\n'
        '  "window": "L=80 stride=40, 20 Hz, 6-class Kwapisz 2010"\n'
        "}\n"
    )
    print(f"exported {ROOT / 'wisdm_weights.hpp'} test_top1={acc:.4f} macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
