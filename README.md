# PicoTorch: an open-source machine learning framework for constrained embedded platforms

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22231092.svg)](https://doi.org/10.5281/zenodo.22231092)

Repository: [https://github.com/modalfuse/picotorch](https://github.com/modalfuse/picotorch)  
Release: [v1.0.2](https://github.com/modalfuse/picotorch/releases/tag/v1.0.2)  
Zenodo: [https://doi.org/10.5281/zenodo.22231092](https://doi.org/10.5281/zenodo.22231092)

PicoTorch is an open-source unified machine-learning framework that shortens the path from a research prototype to a constrained embedded platform. The on-device interface assembles a `Module` graph (`Linear`, `LayerNorm`, multi-head attention, feed-forward, `Sequential`) and runs one `forward`. Fused self-attention and cross-attention are implemented in this library in hardware FP32. The first host is ESP32-S3. Training stays on PC PyTorch. PicoTorch does not call Espressif ESP-DL, ESP-NN, or ESP-PPQ.

This package contains the public headers and kernels, five host call programs, WISDM / CIFAR-10 / glucose example graphs with exported weights, contrast firmware for LiteRT Micro and ESP-DL INT8, and the programming guide. Raw WISDM and CIFAR-10 archives, PlatformIO `.pio/` trees, and ESP-IDF `managed_components/` are not distributed.

## Companion archives

- Glucose counterpart application: [Pico-Former](https://github.com/modalfuse/pico-former), Zenodo [10.5281/zenodo.22218446](https://doi.org/10.5281/zenodo.22218446). Pico-Former is an independently developed glucose application.
- Host protocol, model code, and locked splits: [RICE-Former](https://github.com/modalfuse/rice-former), Zenodo [10.5281/zenodo.22171786](https://doi.org/10.5281/zenodo.22171786).

## Repository contents

- `include/picotorch/`, `src/`: public headers and kernels.
- `examples/minimal/`: five host programs that assemble a small graph and run one `forward`.
- `examples/wisdm/`: 6-class IMU Encoder, exported FP32 weights, and contrast firmware.
- `examples/cifar10/`: Conv + Encoder on CIFAR-10, exported FP32 weights, and PicoTorch firmware.
- `examples/pico-former/`: PicoTorch glucose application (counterpart: Pico-Former).
- `host/`: NumPy reference and max-abs checks.
- `docs/`: programming guide, data access, reproducibility, and GitHub–Zenodo release notes.

## Build the host examples

```bash
cmake -S . -B build
cmake --build build
./build/01_linear
./build/test_mha_maxabs
./build/wisdm_infer
```

See `docs/programming-guide.md` for the layer API.

## Build the firmware

1. Install [PlatformIO](https://platformio.org/).
2. Link this tree as a local library (the symlink is not in the archive):

```bash
cd examples/wisdm/firmware
ln -sfn ../../.. lib/PicoTorch
pio run -e s3_n16r8 -t upload
```

On one ESP32-S3 and one WISDM weight set, PicoTorch FP32 reports a burn-in median of 30.9 ms and 47.5 KB of working SRAM against LiteRT Micro FP32 at 169.7 ms and 88.5 KB, with Top-1 agreement 1.000 on both sides. The Espressif INT8 official path reports 9.8 ms and is listed as a separate precision tier. Details are in `docs/reproducibility.md`.

## Host check

```bash
cmake --build build --target wisdm_infer
./build/wisdm_infer
```

Host probe on the shipped WISDM weights: logit max-abs \(2.38\times10^{-6}\), Top-1 agreement \(1.000\). Refreshing those headers from raw data is optional; see `docs/data-access.md`.

## Data and model artifacts

This repository ships exported weight headers and contrast `.tflite` / `.espdl` graphs. It does not ship raw WISDM text, CIFAR-10 images, PlatformIO `.pio/` trees, or ESP-IDF `managed_components/`.

WISDM Actitracker v1.1 remains available from the [Fordham WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php). CIFAR-10 remains available from its [official page](https://www.cs.toronto.edu/~kriz/cifar.html). Access terms are set by those providers.

## Citation

Cite this software archive; cite the companions when the glucose application or the host protocol is used:

> Liu Q, Yang M, Wang Z, Wang S, An X, Lu S, Yang Q, Liu M, Wu Z, Huang D. PicoTorch: an open-source machine learning framework for constrained embedded platforms. Zenodo. https://doi.org/10.5281/zenodo.22231092
>
> Liu Q, Yang M, Wang Z, Wang S, An X, Lu S, Yang Q, Liu M, Wu Z, Huang D. Pico-Former: a curve–event glucose Transformer on ESP32-S3. Zenodo. https://doi.org/10.5281/zenodo.22218446
>
> Liu Q, Yang M, Wang Z, Wang S, An X, Lu S, Yang Q, Liu M, Wu Z, Huang D. RICE-Former: a curve–event Transformer. Zenodo. https://doi.org/10.5281/zenodo.22171786

Citation metadata are also provided in `CITATION.cff`. Please cite the accompanying manuscript when referring to the scientific findings.

## Licensing

Source code: MIT License (`LICENSE`).
