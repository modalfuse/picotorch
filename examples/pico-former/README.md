# PicoTorch glucose application

**An open-source machine learning framework that accelerates the path from research prototyping to deployment on constrained embedded platforms.**

This example is the PicoTorch glucose application: Encoder + cross-attention + dual heads, assembled with `TransformerEncoderLayer` and `MultiHeadAttention`, one `forward` per window. The same-task counterpart is Pico-Former, an independently developed glucose application.

Firmware glue (windowing, inertia, gates, frozen thresholds) lives in the Pico-Former firmware tree and includes `<picotorch/encoder.hpp>` / `<picotorch/attention.hpp>`.

Host check (weights stay in the Pico-Former firmware include path):

```bash
cmake -S . -B build -DRICE_INCLUDE=/path/to/rice-s3/firmware/include
cmake --build build --target host_golden
./build/host_golden
```

On ESP32-S3, flash `rice-s3/firmware` with PicoTorch linked as a PlatformIO library. The boot line reports golden-window max-abs and the 1 h / 4 h point values.

To refresh the contrast exports, point at `ceqt_weights.h`:

```bash
export PICO_FORMER_WEIGHTS=/path/to/rice-s3/firmware/include/ceqt_weights.h
python examples/pico-former/export_xattn_tflite_fp32.py
python examples/pico-former/export_xattn_espdl_int8.py
```

Cross-attention + heads subgraph (same golden window): LiteRT Micro FP32 in `firmware_tflm/` (\(dMax=0.0095\)); ESP-DL INT8 in `firmware_espdl/`. `idf.py` writes `managed_components/` locally; that tree is not in the archive.
