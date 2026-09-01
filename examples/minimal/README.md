# Minimal call examples

Five host programs that assemble a small graph and run one `forward`. Weights are compile-time arrays. Build with the reference backend (see `docs/programming-guide.md`).

| Program | API surface |
| --- | --- |
| `01_linear` | `Linear` |
| `02_sequential` | `Linear` + `ReLU` + `Sequential` |
| `03_self_attention` | `MultiHeadAttention::forward` |
| `04_cross_attention` | `MultiHeadAttention::forward_cross` |
| `05_encoder_layer` | `TransformerEncoderLayer` |

Shapes: \(L=4\) or \(L=8\), \(d=16\). The Pico-Former on-device graph lives in `examples/pico-former/`.
