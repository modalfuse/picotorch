# CIFAR-10 Conv + Encoder

**An open-source machine learning framework that accelerates the path from research prototyping to deployment on constrained embedded platforms.**

This example is the third PicoTorch graph: three stride-2 \(3\times3\) convolutions in the example, then `TransformerEncoderLayer` + mean pool + a 10-way head. Training stays on PC PyTorch. Convolution stays in the example (`cifar_ops.hpp`); the Encoder path is PicoTorch hardware FP32.

Data: official CIFAR-10 train/test split, \(32\times32\times3\). Place the official tarball under `data/` (see `data/README.md`), or let `train_export.py` load the same split from Hugging Face. The published tree keeps `cifar_weights.hpp`; it does not ship the images.

## Gates (locked at start)

| Check | Door |
| --- | --- |
| Graph | three \(3\times3\) Conv (FP32) + Encoder (\(L=16,d=16,h=2,d_{\mathrm{ff}}=32\)) + mean pool + 10-way |
| 32-image probe, C vs PyTorch logits | max-abs \(\le 10^{-4}\) |
| Probe Top-1 | agreement \(= 1.0\) |
| Official test set | report this graph's Top-1 |

TinyFormer-300K's CIFAR-10 Top-1 96.1% is the literature model on STM32F746. It is a cross-chip magnitude reference, not this graph's score.

## Host

```bash
python examples/cifar10/train_export.py --epochs 8
cmake --build build --target cifar_infer
./build/cifar_infer
```

Host probe (same weights): logit max-abs \(1.25\times10^{-6}\), Top-1 agreement \(1.000\). Official test set: Top-1 **0.4744**.

## ESP32-S3 (hardware FP32)

```bash
cd examples/cifar10/firmware
ln -sfn ../../.. lib/PicoTorch   # local symlink; not in the archive
pio run -t upload -e s3_n16r8
```

Board report (2026-09-01, ESP32-S3, hardware FP32):

| Column | Value |
| --- | --- |
| Latency | burn median **49.6 ms**, p95 **49.6 ms**; 32-image probe median **49.8 ms** |
| Work SRAM | **31 872 B** (arena 9 280 B + Conv / token buffers, internal SRAM) |
| Flash | firmware image **698 193 B** |
| Numeric | logit max-abs **\(1.19\times10^{-6}\)**; Top-1 agreement **1.000** |
