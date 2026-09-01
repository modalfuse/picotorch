#include <cstdio>

#include <picotorch/attention.hpp>
#include <picotorch/context.hpp>
#include <picotorch/tensor.hpp>

#include "id_weights.hpp"

int main() {
    using namespace picotorch;

    constexpr int L = 8;
    constexpr int D = 16;
    static float x[L * D];
    static float y[L * D];
    static float Wq[D * D], Wk[D * D], Wv[D * D], Wo[D * D];
    static float bq[D], bk[D], bv[D], bo[D];
    alignas(16) static float arena[16 * 1024];

    for (int i = 0; i < L * D; ++i) {
        x[i] = 0.01f * static_cast<float>(i + 1);
    }
    fill_identity(Wq, D);
    fill_identity(Wk, D);
    fill_identity(Wv, D);
    fill_identity(Wo, D);
    fill_zero(bq, D);
    fill_zero(bk, D);
    fill_zero(bv, D);
    fill_zero(bo, D);

    Context ctx{arena, sizeof(arena), Backend::Ref};
    MultiHeadAttention mha(/*n_head=*/2, /*d_model=*/D, /*self=*/true);
    mha.set_weights(Wq, bq, Wk, bk, Wv, bv, Wo, bo);

    Tensor xin{x, L, D};
    Tensor yout{y, L, D};
    mha.forward(ctx, xin, yout);

    std::printf("03_self_attention: ");
    for (int c = 0; c < D; ++c) {
        std::printf("%.4f%s", y[c], c + 1 == D ? "\n" : " ");
    }
    return 0;
}
