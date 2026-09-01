#include <picotorch/layernorm.hpp>

#include <cstring>

namespace picotorch {

void LayerNorm::forward(Context &ctx, const Tensor &x, Tensor &y) {
    (void)ctx;
    if (x.data != y.data) {
        memcpy(y.data, x.data, static_cast<size_t>(x.numel()) * sizeof(float));
    }
    y.rows = x.rows;
    y.cols = x.cols;
    layernorm(y.data, gamma, beta, x.rows, normalized_shape, eps);
}

}  // namespace picotorch
