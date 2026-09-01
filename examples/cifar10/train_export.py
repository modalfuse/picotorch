#!/usr/bin/env python3
"""Train a CIFAR-10 Conv+Encoder hybrid and export PicoTorch weights + a 32-image probe."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
L, D, H, FF, N_CLS = 16, 16, 2, 32, 10


class ConvEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.n1 = nn.LayerNorm(D)
        self.n2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D)
        self.cls = nn.Linear(D, N_CLS)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h.permute(0, 2, 3, 1).reshape(x.size(0), L, D)

    def encode(self, tok: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(tok, tok, tok, need_weights=False)
        h = self.n1(tok + a)
        h = self.n2(h + self.ff2(F.relu(self.ff1(h))))
        return h.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.encode(self.tokens(x)))


def c_array(name: str, a: np.ndarray) -> str:
    flat = np.asarray(a, dtype=np.float32).reshape(-1)
    body = ", ".join(f"{float(v):.8e}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{ {body} }};\n"


CIFAR_TGZ_BYTES = 170498071


def _img_to_nchw(img) -> np.ndarray:
    if isinstance(img, dict):
        raw = img.get("bytes") or img.get("path")
        if isinstance(raw, (bytes, bytearray)):
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            im = Image.open(raw).convert("RGB")
    elif isinstance(img, Image.Image):
        im = img.convert("RGB")
    else:
        im = Image.fromarray(np.asarray(img)).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def load_hf_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    name = "train-00000-of-00001.parquet" if split == "train" else "test-00000-of-00001.parquet"
    dest = root / "hf"
    dest.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="cifar10",
        filename=f"plain_text/{name}",
        repo_type="dataset",
        local_dir=str(dest),
    )
    table = pq.read_table(path)
    cols = {c.lower(): c for c in table.column_names}
    img_col = cols.get("img") or cols.get("image")
    lab_col = cols.get("label") or cols.get("fine_label")
    if img_col is None or lab_col is None:
        raise RuntimeError(f"unexpected parquet columns: {table.column_names}")
    xs, ys = [], []
    for img, lab in zip(table[img_col].to_pylist(), table[lab_col].to_pylist()):
        xs.append(_img_to_nchw(img))
        ys.append(int(lab))
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def _tarball_ready(data_root: Path) -> bool:
    extracted = data_root / "cifar-10-batches-py" / "data_batch_1"
    if extracted.is_file():
        return True
    tgz = data_root / "cifar-10-python.tar.gz"
    if not tgz.is_file() or tgz.stat().st_size != CIFAR_TGZ_BYTES:
        return False
    try:
        import tarfile

        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(data_root)
        return (data_root / "cifar-10-batches-py" / "data_batch_1").is_file()
    except Exception:
        return False


def load_arrays(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if _tarball_ready(data_root):
        tfm = transforms.ToTensor()
        train = datasets.CIFAR10(data_root, train=True, download=False, transform=tfm)
        test = datasets.CIFAR10(data_root, train=False, download=False, transform=tfm)
        xt = np.stack([train[i][0].numpy() for i in range(len(train))]).astype(np.float32)
        yt = np.asarray([train[i][1] for i in range(len(train))], dtype=np.int64)
        xv = np.stack([test[i][0].numpy() for i in range(len(test))]).astype(np.float32)
        yv = np.asarray([test[i][1] for i in range(len(test))], dtype=np.int64)
        return xt, yt, xv, yv
    print("official tarball incomplete; loading CIFAR-10 official split from Hugging Face")
    xt, yt = load_hf_split(data_root, "train")
    xv, yv = load_hf_split(data_root, "test")
    return xt, yt, xv, yv


def export_header(net: ConvEnc, probe_x: np.ndarray, probe_y: np.ndarray, probe_logits: np.ndarray, path: Path) -> None:
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    chunks = [
        "#pragma once\n",
        f"enum {{ CIFAR_L = {L}, CIFAR_D = {D}, CIFAR_H = {H}, CIFAR_FF = {FF}, CIFAR_N = {N_CLS}, CIFAR_PROBE = {len(probe_y)} }};\n",
        c_array("W_C1", sd["conv1.weight"]),
        c_array("B_C1", sd["conv1.bias"]),
        c_array("W_C2", sd["conv2.weight"]),
        c_array("B_C2", sd["conv2.bias"]),
        c_array("W_C3", sd["conv3.weight"]),
        c_array("B_C3", sd["conv3.bias"]),
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
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    xt, yt, xv, yv = load_arrays(args.data)
    train = torch.utils.data.TensorDataset(torch.from_numpy(xt), torch.from_numpy(yt))
    test = torch.utils.data.TensorDataset(torch.from_numpy(xv), torch.from_numpy(yv))
    loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=0)

    net = ConvEnc()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        net.train()
        loss_sum, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(net(xb), yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(yb)
            n += len(yb)
        net.eval()
        correct, tot = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = net(xb).argmax(-1)
                correct += int((pred == yb).sum())
                tot += len(yb)
        acc = correct / tot
        print(f"epoch {ep+1} loss={loss_sum/n:.4f} test_top1={acc:.4f}")

    net.eval()
    xs, ys = [], []
    with torch.no_grad():
        for i in range(32):
            x, y = test[i]
            xs.append(x.numpy())
            ys.append(int(y))
        probe_x = np.stack(xs).astype(np.float32)
        probe_y = np.asarray(ys, np.int64)
        probe_log = net(torch.from_numpy(probe_x)).numpy()
        correct, tot = 0, 0
        for xb, yb in test_loader:
            pred = net(xb).argmax(-1)
            correct += int((pred == yb).sum())
            tot += len(yb)
        acc = correct / tot
    export_header(net, probe_x, probe_y, probe_log, ROOT / "cifar_weights.hpp")
    (ROOT / "gates.json").write_text(
        "{\n"
        f'  "test_top1": {acc:.6f},\n'
        '  "probe": 32,\n'
        '  "logit_maxabs": 1e-4,\n'
        '  "probe_top1_agree": 1.0,\n'
        '  "graph": "3x conv stride2 + Encoder L=16 d=16 h=2 + 10-way",\n'
        '  "split": "CIFAR-10 official train/test"\n'
        "}\n"
    )
    print(f"exported {ROOT / 'cifar_weights.hpp'} test_top1={acc:.4f}")


if __name__ == "__main__":
    main()
