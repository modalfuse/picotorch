#pragma once

#include <cstring>

#include <picotorch/context.hpp>
#include <picotorch/encoder.hpp>
#include <picotorch/linear.hpp>
#include <picotorch/tensor.hpp>

#include "wisdm_weights.hpp"

inline void wisdm_forward(picotorch::Context &ctx, picotorch::Linear &proj, picotorch::TransformerEncoderLayer &enc,
                          picotorch::Linear &cls, const float *x, float *tok, float *pooled, float *logits) {
    ctx.reset();
    picotorch::Tensor xin{const_cast<float *>(x), WISDM_L, 3};
    picotorch::Tensor ttok{tok, WISDM_L, WISDM_D};
    proj.forward(ctx, xin, ttok);
    for (int i = 0; i < WISDM_L * WISDM_D; ++i) {
        tok[i] += W_POS[i];
    }
    enc.forward(ctx, ttok, ttok);
    memset(pooled, 0, sizeof(float) * WISDM_D);
    for (int i = 0; i < WISDM_L; ++i) {
        for (int d = 0; d < WISDM_D; ++d) {
            pooled[d] += tok[i * WISDM_D + d];
        }
    }
    const float inv = 1.f / static_cast<float>(WISDM_L);
    for (int d = 0; d < WISDM_D; ++d) {
        pooled[d] *= inv;
    }
    picotorch::Tensor tp{pooled, 1, WISDM_D};
    picotorch::Tensor tl{logits, 1, WISDM_N};
    cls.forward(ctx, tp, tl);
}

inline int wisdm_argmax6(const float *logits) {
    int pred = 0;
    for (int c = 1; c < WISDM_N; ++c) {
        if (logits[c] > logits[pred]) {
            pred = c;
        }
    }
    return pred;
}
