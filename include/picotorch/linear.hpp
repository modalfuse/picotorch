#pragma once

#include <picotorch/module.hpp>

namespace picotorch {

inline void linear(const float *x, const float *W, const float *b, float *y, int n, int out_d, int in_d) {
    for (int i = 0; i < n; ++i) {
        const float *xi = x + i * in_d;
        float *yo = y + i * out_d;
        for (int o = 0; o < out_d; ++o) {
            float s = b ? b[o] : 0.f;
            const float *w = W + o * in_d;
            for (int k = 0; k < in_d; ++k) {
                s += w[k] * xi[k];
            }
            yo[o] = s;
        }
    }
}

struct Linear : Module {
    int out_features;
    int in_features;
    const float *weight;
    const float *bias;

    Linear() : out_features(0), in_features(0), weight(nullptr), bias(nullptr) {}

    Linear(int out_f, int in_f, const float *W, const float *b)
        : out_features(out_f), in_features(in_f), weight(W), bias(b) {}

    void set_weights(const float *W, const float *b) {
        weight = W;
        bias = b;
    }

    int out_cols(int) const override { return out_features; }

    void forward(Context &ctx, const Tensor &x, Tensor &y) override;
};

}  // namespace picotorch
