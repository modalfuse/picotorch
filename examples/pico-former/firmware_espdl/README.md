# Pico-Former cross-attention on ESP-DL INT8

Same split graph as `firmware_tflm`, quantized by `export_xattn_espdl_int8.py` (`glucobench` + `esp-ppq`).

Board report (2026-09-01, ESP32-S3, INT8): **完成一次前向**；黄金窗 \(dMax=10.38\)，\(lMax=0.590\)，\(yMax=10.38\)（与宿主机 INT8 同数）；burn 中位 **26.4 ms**；`variable` **121.12 KB**；Flash **1 577 344 B**.
