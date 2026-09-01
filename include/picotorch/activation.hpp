#pragma once

#include <cmath>

#include <picotorch/module.hpp>

namespace picotorch {

inline void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.f) {
            x[i] = 0.f;
        }
    }
}

inline void gelu_inplace(float *x, int n) {
    const float k = 0.7978845608f;
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.f + tanhf(k * (v + 0.044715f * v * v * v)));
    }
}

inline void swish_inplace(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = v / (1.f + expf(-v));
    }
}

inline void softmax_row(float *s, int n) {
    float m = s[0];
    for (int i = 1; i < n; ++i) {
        if (s[i] > m) {
            m = s[i];
        }
    }
    float z = 0.f;
    for (int i = 0; i < n; ++i) {
        s[i] = expf(s[i] - m);
        z += s[i];
    }
    const float inv = 1.f / z;
    for (int i = 0; i < n; ++i) {
        s[i] *= inv;
    }
}

struct ReLU : Module {
    void forward(Context &ctx, const Tensor &x, Tensor &y) override;
};

struct GELU : Module {
    void forward(Context &ctx, const Tensor &x, Tensor &y) override;
};

struct Add : Module {
    void forward(Context &ctx, const Tensor &x, Tensor &y) override;
};

}  // namespace picotorch
