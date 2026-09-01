# WISDM LiteRT Micro (FP32)

Same graph and weights as `examples/wisdm/firmware` (PicoTorch). Attention is a split TFLite graph. Host export: `python examples/wisdm/export_tflite_fp32.py` (use an environment with working TensorFlow).

Board report (2026-09-01, ESP32-S3, FP32): burn median **169.7 ms**, arena used **88 496 B**, Flash **419 280 B**, logit max-abs \(2.86\times10^{-6}\), Top-1 agreement 1.000.
