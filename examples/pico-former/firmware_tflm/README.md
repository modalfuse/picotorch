# Pico-Former cross-attention on LiteRT Micro (FP32)

Split graph: golden-window encoder tokens `G_TOK` → fused-equivalent cross-attention + delta / event heads. Host export: `export_xattn_tflite_fp32.py` (`tf_battery`).

Board report (2026-09-01, ESP32-S3, FP32): **完成一次前向**；黄金窗 \(dMax=0.0095\)，\(lMax=5.79\times10^{-4}\)，\(yMax=0.0095\)；burn 中位 **485.1 ms**；`arena_used` **154 908 B**；Flash **477 520 B**. Timing is a functional note, not a speed contest against the full Pico-Former graph.
