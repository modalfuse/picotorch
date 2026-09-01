#pragma once

#include <picotorch/module.hpp>

namespace picotorch {

struct MultiHeadAttention : Module {
    int n_head;
    int d_model;
    int d_k;
    bool self;

    const float *in_w;
    const float *in_b;
    const float *Wq;
    const float *bq;
    const float *Wk;
    const float *bk;
    const float *Wv;
    const float *bv;
    const float *Wo;
    const float *bo;

    MultiHeadAttention(int heads, int d, bool is_self)
        : n_head(heads),
          d_model(d),
          d_k(d / heads),
          self(is_self),
          in_w(nullptr),
          in_b(nullptr),
          Wq(nullptr),
          bq(nullptr),
          Wk(nullptr),
          bk(nullptr),
          Wv(nullptr),
          bv(nullptr),
          Wo(nullptr),
          bo(nullptr) {}

    void set_in_proj(const float *packed_w, const float *packed_b) {
        in_w = packed_w;
        in_b = packed_b;
        Wq = Wk = Wv = nullptr;
        bq = bk = bv = nullptr;
    }

    void set_out_proj(const float *out_w, const float *out_b) {
        Wo = out_w;
        bo = out_b;
    }

    void set_weights(const float *wq, const float *bq_, const float *wk, const float *bk_, const float *wv,
                     const float *bv_, const float *wo, const float *bo_) {
        Wq = wq;
        bq = bq_;
        Wk = wk;
        bk = bk_;
        Wv = wv;
        bv = bv_;
        Wo = wo;
        bo = bo_;
        in_w = nullptr;
        in_b = nullptr;
    }

    void forward(Context &ctx, const Tensor &x, Tensor &y) override;
    void forward_cross(Context &ctx, const Tensor &q, const Tensor &kv, Tensor &y);
};

}  // namespace picotorch
