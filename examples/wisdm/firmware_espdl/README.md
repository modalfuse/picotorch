# WISDM Encoder on ESP-DL INT8

Same graph and weights as `examples/wisdm/firmware` (PicoTorch), exported by `export_espdl_int8.py` (`glucobench` + `esp-ppq`, target `esp32s3`, opset 17).

```bash
. $IDF_PATH/export.sh
idf.py set-target esp32s3
idf.py flash
```

`idf.py` writes `managed_components/` and `build/` in this directory. Those trees stay local.

Board report (2026-09-01, ESP32-S3, INT8 official path): burn median **9.8 ms**, `variable` **74.61 KB**, Flash **1 534 944 B**, logit max-abs \(2.39\), Top-1 agreement 0.906. Read as a quantized-precision column, not against PicoTorch FP32 milliseconds.
