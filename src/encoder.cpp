#include <picotorch/activation.hpp>
#include <picotorch/encoder.hpp>

#include <cstring>

namespace picotorch {

void TransformerEncoderLayer::forward(Context &ctx, const Tensor &x, Tensor &y) {
    const size_t mark = ctx.used;
    const int n = x.rows;
    const int d = d_model;
    float *work = y.data;
    if (x.data != y.data) {
        memcpy(y.data, x.data, static_cast<size_t>(n) * static_cast<size_t>(d) * sizeof(float));
    }
    y.rows = n;
    y.cols = d;

    float *attn = ctx.alloc_f32(static_cast<size_t>(n) * static_cast<size_t>(d));
    if (!attn) {
        ctx.used = mark;
        return;
    }

    Tensor xin{work, n, d};
    Tensor t_attn{attn, n, d};
    self_attn.forward(ctx, xin, t_attn);
    for (int i = 0; i < n * d; ++i) {
        work[i] += attn[i];
    }
    n1.forward(ctx, xin, xin);

    float *ff = ctx.alloc_f32(static_cast<size_t>(n) * static_cast<size_t>(d_ff));
    if (!ff) {
        ctx.used = mark;
        return;
    }
    Tensor t_ff{ff, n, d_ff};
    ff1.forward(ctx, xin, t_ff);
    relu_inplace(ff, n * d_ff);
    ff2.forward(ctx, t_ff, t_attn);
    for (int i = 0; i < n * d; ++i) {
        work[i] += attn[i];
    }
    n2.forward(ctx, xin, xin);
    ctx.used = mark;
}

}  // namespace picotorch
