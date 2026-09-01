#pragma once

#include <cstring>

#include <picotorch/context.hpp>
#include <picotorch/encoder.hpp>
#include <picotorch/linear.hpp>
#include <picotorch/tensor.hpp>

#include "cifar_ops.hpp"
#include "cifar_weights.hpp"

inline void cifar_forward(picotorch::Context &ctx, picotorch::TransformerEncoderLayer &enc, picotorch::Linear &cls,
                          const float *image, float *y16, float *y8, float *y4, float *tok, float *pooled,
                          float *logits) {
    conv2d_relu(image, W_C1, B_C1, y16, 3, 16, 32, 32, 2, 1, 3);
    conv2d_relu(y16, W_C2, B_C2, y8, 16, 16, 16, 16, 2, 1, 3);
    conv2d_relu(y8, W_C3, B_C3, y4, 16, 16, 8, 8, 2, 1, 3);
    nchw_to_tokens(y4, tok, 4, 4, CIFAR_D);
    ctx.reset();
    picotorch::Tensor tt{tok, CIFAR_L, CIFAR_D};
    enc.forward(ctx, tt, tt);
    memset(pooled, 0, sizeof(float) * CIFAR_D);
    for (int i = 0; i < CIFAR_L; ++i) {
        for (int d = 0; d < CIFAR_D; ++d) {
            pooled[d] += tok[i * CIFAR_D + d];
        }
    }
    const float inv = 1.f / static_cast<float>(CIFAR_L);
    for (int d = 0; d < CIFAR_D; ++d) {
        pooled[d] *= inv;
    }
    picotorch::Tensor tp{pooled, 1, CIFAR_D};
    picotorch::Tensor tl{logits, 1, CIFAR_N};
    cls.forward(ctx, tp, tl);
}

inline int cifar_argmax10(const float *logits) {
    int pred = 0;
    for (int c = 1; c < CIFAR_N; ++c) {
        if (logits[c] > logits[pred]) {
            pred = c;
        }
    }
    return pred;
}
