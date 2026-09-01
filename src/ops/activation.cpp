#include <picotorch/activation.hpp>

#include <cstring>

namespace picotorch {

void ReLU::forward(Context &ctx, const Tensor &x, Tensor &y) {
    (void)ctx;
    if (x.data != y.data) {
        memcpy(y.data, x.data, static_cast<size_t>(x.numel()) * sizeof(float));
    }
    y.rows = x.rows;
    y.cols = x.cols;
    relu_inplace(y.data, y.numel());
}

void GELU::forward(Context &ctx, const Tensor &x, Tensor &y) {
    (void)ctx;
    if (x.data != y.data) {
        memcpy(y.data, x.data, static_cast<size_t>(x.numel()) * sizeof(float));
    }
    y.rows = x.rows;
    y.cols = x.cols;
    gelu_inplace(y.data, y.numel());
}

void Add::forward(Context &ctx, const Tensor &x, Tensor &y) {
    (void)ctx;
    const int n = x.numel();
    for (int i = 0; i < n; ++i) {
        y.data[i] += x.data[i];
    }
}

}  // namespace picotorch
