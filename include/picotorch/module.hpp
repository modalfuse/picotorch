#pragma once

#include <picotorch/context.hpp>
#include <picotorch/tensor.hpp>

namespace picotorch {

struct Module {
    virtual void forward(Context &ctx, const Tensor &x, Tensor &y) = 0;
    virtual int out_cols(int in_cols) const { return in_cols; }
    virtual ~Module() = default;
};

}  // namespace picotorch
