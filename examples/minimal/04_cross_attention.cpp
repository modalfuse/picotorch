#include <cstdio>

#include <picotorch/attention.hpp>
#include <picotorch/context.hpp>
#include <picotorch/tensor.hpp>

#include "id_weights.hpp"

int main() {
    using namespace picotorch;

    constexpr int NQ = 4;
    constexpr int NK = 8;
    constexpr int D = 16;
    static float q[NQ * D];
    static float kv[NK * D];
    static float y[NQ * D];
    static float Wq[D * D], Wk[D * D], Wv[D * D], Wo[D * D];
    static float bq[D], bk[D], bv[D], bo[D];
    alignas(16) static float arena[16 * 1024];

    for (int i = 0; i < NQ * D; ++i) {
        q[i] = 0.02f * static_cast<float>(i + 1);
    }
    for (int i = 0; i < NK * D; ++i) {
        kv[i] = 0.01f * static_cast<float>(i + 1);
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
    MultiHeadAttention xattn(/*n_head=*/2, /*d_model=*/D, /*self=*/false);
    xattn.set_weights(Wq, bq, Wk, bk, Wv, bv, Wo, bo);

    Tensor tq{q, NQ, D};
    Tensor tkv{kv, NK, D};
    Tensor yout{y, NQ, D};
    xattn.forward_cross(ctx, tq, tkv, yout);

    std::printf("04_cross_attention: ");
    for (int c = 0; c < D; ++c) {
        std::printf("%.4f%s", y[c], c + 1 == D ? "\n" : " ");
    }
    return 0;
}
