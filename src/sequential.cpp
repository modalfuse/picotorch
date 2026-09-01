#include <picotorch/sequential.hpp>

#include <cstring>

namespace picotorch {

void Sequential::forward(Context &ctx, const Tensor &x, Tensor &y) {
    if (n_layers == 0) {
        if (x.data != y.data) {
            memcpy(y.data, x.data, static_cast<size_t>(x.numel()) * sizeof(float));
        }
        y.rows = x.rows;
        y.cols = x.cols;
        return;
    }
    if (n_layers == 1) {
        layers[0]->forward(ctx, x, y);
        return;
    }

    int cols = x.cols;
    for (int i = 0; i < n_layers; ++i) {
        const int next = layers[i]->out_cols(cols);
        if (next > cols) {
            cols = next;
        }
    }
    if (y.cols > cols) {
        cols = y.cols;
    }

    const size_t mark = ctx.used;
    float *a = ctx.alloc_f32(static_cast<size_t>(x.rows) * static_cast<size_t>(cols));
    float *b = ctx.alloc_f32(static_cast<size_t>(x.rows) * static_cast<size_t>(cols));
    if (!a || !b) {
        ctx.used = mark;
        return;
    }
    memcpy(a, x.data, static_cast<size_t>(x.numel()) * sizeof(float));
    Tensor cur{a, x.rows, x.cols};
    for (int i = 0; i < n_layers; ++i) {
        const int oc = layers[i]->out_cols(cur.cols);
        Tensor nxt{b, cur.rows, oc};
        layers[i]->forward(ctx, cur, nxt);
        cur = nxt;
        float *tmp = a;
        a = b;
        b = tmp;
        cur.data = a;
    }
    y.rows = cur.rows;
    y.cols = cur.cols;
    memcpy(y.data, cur.data, static_cast<size_t>(cur.numel()) * sizeof(float));
    ctx.used = mark;
}

}  // namespace picotorch
