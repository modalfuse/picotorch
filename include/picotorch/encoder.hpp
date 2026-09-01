#pragma once

#include <picotorch/attention.hpp>
#include <picotorch/layernorm.hpp>
#include <picotorch/linear.hpp>

namespace picotorch {

struct TransformerEncoderLayer : Module {
    int d_model;
    int n_head;
    int d_ff;
    MultiHeadAttention self_attn;
    Linear ff1;
    Linear ff2;
    LayerNorm n1;
    LayerNorm n2;

    TransformerEncoderLayer(int d, int heads, int ff)
        : d_model(d), n_head(heads), d_ff(ff), self_attn(heads, d, true), ff1(ff, d, nullptr, nullptr),
          ff2(d, ff, nullptr, nullptr), n1(d, nullptr, nullptr), n2(d, nullptr, nullptr) {}

    void set_weights(const float *win, const float *bin, const float *wout, const float *bout, const float *wl1,
                     const float *bl1, const float *wl2, const float *bl2, const float *n1w, const float *n1b,
                     const float *n2w, const float *n2b) {
        self_attn.set_in_proj(win, bin);
        self_attn.set_out_proj(wout, bout);
        ff1 = Linear(d_ff, d_model, wl1, bl1);
        ff2 = Linear(d_model, d_ff, wl2, bl2);
        n1 = LayerNorm(d_model, n1w, n1b);
        n2 = LayerNorm(d_model, n2w, n2b);
    }

    void forward(Context &ctx, const Tensor &x, Tensor &y) override;
};

}  // namespace picotorch
