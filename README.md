# PicoTorch

**An open-source machine learning framework that accelerates the path from research prototyping to deployment on constrained embedded platforms.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22231092.svg)](https://doi.org/10.5281/zenodo.22231092)

Repository: [https://github.com/modalfuse/picotorch](https://github.com/modalfuse/picotorch)  
Release: [v1.0.1](https://github.com/modalfuse/picotorch/releases/tag/v1.0.1)  
Zenodo: [https://doi.org/10.5281/zenodo.22231092](https://doi.org/10.5281/zenodo.22231092)

PicoTorch belongs to the same class of on-device inference frameworks as the Espressif INT8/INT16 stack and LiteRT Micro. The on-device interface assembles a `Module` graph and runs one `forward`. Fused self-attention and cross-attention are implemented in this library in hardware FP32 on ESP32-S3.

## Scope

- Programming framework: assemble a `Module` graph and run one `forward`.
- On-device API: `Linear` / `LayerNorm` / `MHA` / `FFN` / `Sequential`.
- Own kernels: `Linear` / `LayerNorm` / Softmax / fused self-attention and cross-attention, ESP32-S3 hardware FP32.
- Independent of Espressif ESP-DL, ESP-NN, and ESP-PPQ (those stacks are INT8/INT16; PicoTorch does not call them).
- Training stays on PC PyTorch. Windowing, inertia, gates, and frozen \(\tau\) stay in firmware.

## Layout

```text
picotorch/
├── include/picotorch/      # public headers
├── src/                    # kernels and module runtime
├── examples/minimal/       # five host call programs
├── examples/pico-former/   # PicoTorch glucose application (counterpart: Pico-Former)
├── examples/wisdm/         # PicoTorch example: 6-class IMU Encoder
├── examples/cifar10/       # PicoTorch example: Conv + Encoder on CIFAR-10
├── docs/programming-guide.md
└── host/                   # NumPy reference and max-abs checks
```

## Public tree

The published snapshot is source, exported weights, and firmware glue. It is a few megabytes. Local `.pio/`, ESP-IDF `managed_components/`, `build/`, and raw datasets stay on the machine that builds; they are listed in `.gitignore` and are not part of a GitHub or Zenodo archive.

| Keep in the archive | Obtain locally |
| --- | --- |
| `include/`, `src/`, `docs/`, `host/`, `examples/minimal/` | PlatformIO `.pio/` after `pio run` |
| Weight headers (`wisdm_weights.hpp`, `cifar_weights.hpp`) and exported `.tflite` / `.espdl` | ESP-IDF `managed_components/` after `idf.py` (see `idf_component.yml`) |
| Firmware `src/`, `platformio.ini`, `CMakeLists.txt`, `sdkconfig.defaults*` | WISDM / CIFAR-10 archives under `examples/*/data/` |
| Export and train scripts | CMake `build/` at the repo root |

Link PicoTorch into a PlatformIO project as a library (do not commit the symlink):

```bash
cd examples/wisdm/firmware
ln -sfn ../../.. lib/PicoTorch
pio run -t upload -e s3_n16r8
```

Dataset download steps: `examples/wisdm/data/README.md`, `examples/cifar10/data/README.md`. On-device inference reads the weight headers, not those directories.

## Companion

- Glucose counterpart application: [Pico-Former](https://github.com/modalfuse/pico-former), Zenodo [10.5281/zenodo.22218446](https://doi.org/10.5281/zenodo.22218446)
- Host protocol: [RICE-Former](https://github.com/modalfuse/rice-former), Zenodo [10.5281/zenodo.22171786](https://doi.org/10.5281/zenodo.22171786)

## Citation

> Liu Q, Yang M, Wang Z, Wang S, An X, Lu S, Yang Q, Liu M, Wu Z, Huang D. PicoTorch: an open-source machine learning framework for constrained embedded platforms. Zenodo. https://doi.org/10.5281/zenodo.22231092

Citation metadata are also in `CITATION.cff`. Please cite the accompanying manuscript when referring to the scientific findings.

## Licensing

Source code: MIT License (`LICENSE`).
