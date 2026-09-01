# PicoTorch Programming Guide

**An open-source machine learning framework that accelerates the path from research prototyping to deployment on constrained embedded platforms.**

PicoTorch is the C++ programming framework for that path: assemble a `Module` graph, run one `forward`, and deploy the same graph on the host reference backend or on ESP32-S3 hardware FP32. `Linear`, `LayerNorm`, Softmax, and fused attention are implemented in this library. PicoTorch does not call ESP-DL, ESP-NN, or ESP-PPQ. Training stays on PC PyTorch.

This guide matches the public headers under `include/picotorch/`. Worked call sites live in `examples/minimal/`.

## Build

Host (reference backend):

```bash
cmake -S . -B build
cmake --build build
./build/01_linear
./build/02_sequential
./build/03_self_attention
./build/04_cross_attention
./build/05_encoder_layer
./build/test_mha_maxabs
```

On ESP32-S3, add `picotorch-release` as an ESP-IDF extra component or a PlatformIO `lib/` entry, set `Backend::S3Fp32`, and keep windowing, inertia, gates, and frozen \(\tau\) in the firmware project.

## Tensor and Context

A `Tensor` is a row-major `rows × cols` view. Sequence layout is \(L \times d\). The buffer is owned by the caller.

```cpp
picotorch::Tensor x{data, /*rows=*/8, /*cols=*/16};
```

`Context` holds one arena and the backend. Allocate the arena once; kernels borrow tiles from it.

```cpp
alignas(16) static float arena[48 * 1024];
picotorch::Context ctx{arena, sizeof(arena), picotorch::Backend::Ref};
```

Shapes are fixed at compile time or set once on `Context`. The working envelope is \(L \le 128\), \(d \le 64\).

## Linear, LayerNorm, Sequential

Weights stay in caller memory. `Linear` is `out × in` row-major. `LayerNorm` uses \(\varepsilon = 10^{-5}\) unless you pass another value.

```cpp
picotorch::Linear fc(/*out=*/2, /*in=*/3, W, b);
picotorch::LayerNorm ln(/*normalized_shape=*/16, gamma, beta);
picotorch::ReLU relu;
picotorch::Sequential net({&fc, &relu});
net.forward(ctx, x, y);
```

`01_linear.cpp` and `02_sequential.cpp` are the shortest compile-and-run paths.

## Attention

`MultiHeadAttention::forward` is self-attention (Encoder). `forward_cross` is cross-attention: a shorter query sequence against a memory of keys and values.

```cpp
picotorch::MultiHeadAttention mha(/*n_head=*/2, /*d_model=*/16, /*self=*/true);
mha.forward(ctx, x, y);

picotorch::MultiHeadAttention xattn(/*n_head=*/2, /*d_model=*/16, /*self=*/false);
xattn.forward_cross(ctx, q, kv, y);
```

Load Q/K/V/out projections with `set_weights(Wq, bq, Wk, bk, Wv, bv, Wo, bo)`. Each `W*` is `d_model × d_model` row-major; each bias is length `d_model` or null. Packed Pico-Former weights use `set_in_proj` (`3d × d`) and `set_out_proj`. See `03_self_attention.cpp` and `04_cross_attention.cpp`.

Self-attention and cross-attention share one kernel. Padding and masks are applied as large positive or negative additives before Softmax.

## Encoder layer

```cpp
picotorch::TransformerEncoderLayer layer(/*d_model=*/16, /*n_head=*/2, /*d_ff=*/32);
layer.forward(ctx, x, y);
```

Stack two layers with `Sequential`. FFN activation follows the exported graph (ReLU or Swish). `05_encoder_layer.cpp` runs one layer at \(L=8\), \(d=16\).

## Custom module

```cpp
struct AddNorm : picotorch::Module {
    picotorch::LayerNorm ln;
    void forward(picotorch::Context &ctx,
                 const picotorch::Tensor &x,
                 picotorch::Tensor &y) override {
        // residual add into y, then ln.forward(ctx, y, y);
    }
};
```

Implement one `forward`. Borrow workspace from `ctx.arena`. Do not call `malloc` inside the hot path.

## On-device

The S3 path uses hardware FP32 for weights, activations, and Softmax, matching the Pico-Former firmware grade. Firmware includes `<picotorch/attention.hpp>` and `<picotorch/encoder.hpp>`. Task glue stays in the application: window assembly, inertia \(y_{\mathrm{iner}}\), gate \(g\), and frozen thresholds \(\tau\).

The PicoTorch glucose application is `examples/pico-former` (golden-window check; counterpart: the independently developed Pico-Former). The IMU example is `examples/wisdm` (6-class short-sequence Encoder). The vision example is `examples/cifar10` (three stride-2 \(3\times3\) convolutions in the example, then the same Encoder API). The five files under `examples/minimal/` are the API teaching set.
