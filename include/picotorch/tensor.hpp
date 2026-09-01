#pragma once

#include <cstddef>

namespace picotorch {

struct Tensor {
    float *data;
    int rows;
    int cols;

    int numel() const { return rows * cols; }
};

}  // namespace picotorch
