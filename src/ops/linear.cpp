#include <picotorch/linear.hpp>

#include <cstring>

namespace picotorch {

void Linear::forward(Context &ctx, const Tensor &x, Tensor &y) {
    const float *src = x.data;
    float *tmp = nullptr;
    if (x.data == y.data && in_features != out_features) {
        tmp = ctx.alloc_f32(static_cast<size_t>(x.rows) * static_cast<size_t>(in_features));
        if (!tmp) {
            return;
        }
        memcpy(tmp, x.data, static_cast<size_t>(x.rows) * static_cast<size_t>(in_features) * sizeof(float));
        src = tmp;
    }
    linear(src, weight, bias, y.data, x.rows, out_features, in_features);
    y.rows = x.rows;
    y.cols = out_features;
}

}  // namespace picotorch
