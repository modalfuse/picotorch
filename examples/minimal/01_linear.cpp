#include <cstdio>

#include <picotorch/context.hpp>
#include <picotorch/linear.hpp>
#include <picotorch/tensor.hpp>

int main() {
    using namespace picotorch;

    // y = x W^T + b, W is 2×3 row-major.
    static const float W[] = {
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f,
    };
    static const float b[] = {0.10f, 0.20f};
    static float x[] = {1.f, 2.f, 3.f};
    static float y[2];
    alignas(16) static float arena[1024];

    Context ctx{arena, sizeof(arena), Backend::Ref};
    Linear fc(/*out_features=*/2, /*in_features=*/3, W, b);

    Tensor xin{x, /*rows=*/1, /*cols=*/3};
    Tensor yout{y, /*rows=*/1, /*cols=*/2};
    fc.forward(ctx, xin, yout);

    std::printf("01_linear: %.4f %.4f\n", y[0], y[1]);
    return 0;
}
