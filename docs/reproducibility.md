# Reproducibility

## Hardware

The reported board numbers used an ESP32-S3 (dual-core 240 MHz, hardware FP32 FPU). Working tensors stay in internal SRAM. PSRAM is unused for the WISDM and CIFAR-10 PicoTorch graphs.

## Host

CMake 3.16+ and a C++17 compiler. Python 3.13 is sufficient for the optional export scripts.

```bash
cmake -S . -B build
cmake --build build
./build/01_linear
./build/test_mha_maxabs
./build/wisdm_infer
./build/cifar_infer
```

## Firmware

PlatformIO 6.x with Arduino-ESP32. Production path is hardware FP32.

```bash
cd examples/wisdm/firmware
ln -sfn ../../.. lib/PicoTorch
pio run -e s3_n16r8 -t upload
```

LiteRT Micro contrast firmware lives in `examples/wisdm/firmware_tflm/`. ESP-DL INT8 official-path firmware lives in `examples/wisdm/firmware_espdl/`; run `idf.py` there so ESP-IDF writes `managed_components/` locally.

## Reported numbers

Same ESP32-S3, same WISDM graph and weights (2026-09-01):

| Stack | Burn median | Working SRAM | Numeric |
| --- | --- | --- | --- |
| PicoTorch FP32 | 30.9 ms | 47.5 KB (47 488 B) | logit max-abs \(2.38\times10^{-6}\); Top-1 1.000 |
| LiteRT Micro FP32 | 169.7 ms | 88.5 KB | logit max-abs \(\sim10^{-6}\); Top-1 1.000 |
| Espressif INT8 official path | 9.8 ms | listed as a separate precision tier | logit max-abs 2.387; Top-1 0.906 |

Host probe (same WISDM weights): logit max-abs \(2.38\times10^{-6}\), Top-1 1.000. Held-out users 29–36: Top-1 0.8125, macro-F1 0.7731.

CIFAR-10 PicoTorch FP32 (same board, 2026-09-01): burn median 49.6 ms, working SRAM 31 872 B, host probe Top-1 1.000, official test Top-1 0.4744.

TinyHAR-Net WISDM F1 96.75% and TinyFormer-300K CIFAR-10 Top-1 96.1% are literature models on STM32 hosts. They are cross-chip magnitude references, not this graph's scores.

Do not commit `.pio/`, `managed_components/`, `build/`, or the raw dataset directories.
