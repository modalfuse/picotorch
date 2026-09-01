#pragma once

#include <cmath>

#include <picotorch/module.hpp>

namespace picotorch {

inline void layernorm(float *x, const float *gamma, const float *beta, int n, int d, float eps = 1e-5f) {
    for (int i = 0; i < n; ++i) {
        float *row = x + i * d;
        float mean = 0.f;
        for (int j = 0; j < d; ++j) {
            mean += row[j];
        }
        mean /= static_cast<float>(d);
        float var = 0.f;
        for (int j = 0; j < d; ++j) {
            const float t = row[j] - mean;
            var += t * t;
        }
        const float inv = 1.f / sqrtf(var / static_cast<float>(d) + eps);
        for (int j = 0; j < d; ++j) {
            row[j] = (row[j] - mean) * inv * gamma[j] + beta[j];
        }
    }
}

struct LayerNorm : Module {
    int normalized_shape;
    const float *gamma;
    const float *beta;
    float eps;

    LayerNorm() : normalized_shape(0), gamma(nullptr), beta(nullptr), eps(1e-5f) {}

    LayerNorm(int dim, const float *g, const float *b, float e = 1e-5f)
        : normalized_shape(dim), gamma(g), beta(b), eps(e) {}

    void set_weights(const float *g, const float *b) {
        gamma = g;
        beta = b;
    }

    void forward(Context &ctx, const Tensor &x, Tensor &y) override;
};

}  // namespace picotorch
