#include <cstdio>

#include <picotorch/activation.hpp>
#include <picotorch/context.hpp>
#include <picotorch/linear.hpp>
#include <picotorch/sequential.hpp>
#include <picotorch/tensor.hpp>

int main() {
    using namespace picotorch;

    static const float W1[] = {
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f,
        0.f, 0.f, 1.f,
    };
    static const float b1[] = {-0.5f, -0.5f, -0.5f};
    static const float W2[] = {
        1.f, 1.f, 0.f,
        0.f, 1.f, 1.f,
    };
    static const float b2[] = {0.f, 0.f};
    static float x[] = {1.f, 2.f, 3.f};
    static float y[2];
    alignas(16) static float arena[2048];

    Context ctx{arena, sizeof(arena), Backend::Ref};
    Linear fc1(/*out=*/3, /*in=*/3, W1, b1);
    ReLU relu;
    Linear fc2(/*out=*/2, /*in=*/3, W2, b2);
    Sequential net({&fc1, &relu, &fc2});

    Tensor xin{x, 1, 3};
    Tensor yout{y, 1, 2};
    net.forward(ctx, xin, yout);

    std::printf("02_sequential: %.4f %.4f\n", y[0], y[1]);
    return 0;
}
