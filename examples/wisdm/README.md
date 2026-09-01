# WISDM 6-class encoder

**An open-source machine learning framework that accelerates the path from research prototyping to deployment on constrained embedded platforms.**

This example is the second PicoTorch graph: a short-sequence Encoder on the classic 6-class WISDM Actitracker set (Kwapisz, Weiss, Moore, SensorKDD 2010 / *SIGKDD Explorations* 2011). Training stays on PC PyTorch. The on-device API is `Linear` + `TransformerEncoderLayer` + mean pool + `Linear`, one `forward` per window.

Data: [Fordham WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php), activity-prediction / Actitracker v1.1, 20 Hz phone accelerometer, classes Walking / Jogging / Upstairs / Downstairs / Sitting / Standing. This is the Kwapisz 2011 set, not UCI 507. Download the archive into `data/` (see `data/README.md`). The published tree keeps `wisdm_weights.hpp`; it does not ship the raw text.

## Gates (locked at start)

| Check | Door |
| --- | --- |
| 32-window probe, C vs PyTorch logits | max-abs \(\le 10^{-4}\) |
| Probe Top-1 | agreement \(= 1.0\) |
| Held-out users 29–36 | report this graph's Top-1 / macro-F1 |

TinyHAR-Net's WISDM F1 96.75% is the literature model on STM32F446RE. It is a cross-chip magnitude reference, not this graph's score.

## Host

```bash
# optional: refresh weights after placing WISDM_ar_v1.1_raw.txt (data/README.md)
python examples/wisdm/train_export.py --raw examples/wisdm/data/WISDM_ar_v1.1_raw.txt
cmake --build build --target wisdm_infer
./build/wisdm_infer
```

Host probe (same weights): logit max-abs \(2.38\times10^{-6}\), Top-1 agreement \(1.000\). Held-out users 29–36: Top-1 **0.8125**, macro-F1 **0.7731**.

## ESP32-S3 (hardware FP32)

```bash
cd examples/wisdm/firmware
ln -sfn ../../.. lib/PicoTorch   # local symlink; not in the archive
pio run -t upload -e s3_n16r8
```

ESP-DL INT8 contrast firmware: `firmware_espdl/`. Run `idf.py` in that directory so ESP-IDF writes `managed_components/` locally. Do not copy that tree into the archive.

Board report (2026-09-01, ESP32-S3, hardware FP32):

| Column | Value |
| --- | --- |
| Latency | burn median **30.9 ms**, p95 **30.9 ms**; 32-window probe median **31.0 ms** |
| Work SRAM | **47 488 B** (arena 42 304 B + token buffer, internal SRAM) |
| Flash | firmware image **319 281 B** |
| Numeric | logit max-abs **\(2.38\times10^{-6}\)**; Top-1 agreement **1.000** |

Same-weight contrast firmware: LiteRT Micro FP32 in `firmware_tflm/` (burn 169.7 ms); ESP-DL INT8 official path in `firmware_espdl/` (burn 9.8 ms, logit max-abs 2.39, Top-1 0.906).
