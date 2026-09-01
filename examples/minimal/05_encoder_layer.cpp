#include <cstdio>

#include <picotorch/context.hpp>
#include <picotorch/encoder.hpp>
#include <picotorch/tensor.hpp>

#include "id_weights.hpp"

int main() {
    using namespace picotorch;

    constexpr int L = 8;
    constexpr int D = 16;
    constexpr int FF = 32;
    static float x[L * D];
    static float y[L * D];
    static float win[3 * D * D], bin[3 * D];
    static float wout[D * D], bout[D];
    static float wl1[FF * D], bl1[FF];
    static float wl2[D * FF], bl2[D];
    static float n1w[D], n1b[D], n2w[D], n2b[D];
    alignas(16) static float arena[32 * 1024];

    for (int i = 0; i < L * D; ++i) {
        x[i] = 0.01f * static_cast<float>(i + 1);
    }
    fill_zero(win, 3 * D * D);
    fill_identity(win, D);
    fill_identity(win + D * D, D);
    fill_identity(win + 2 * D * D, D);
    fill_zero(bin, 3 * D);
    fill_identity(wout, D);
    fill_zero(bout, D);
    fill_zero(wl1, FF * D);
    for (int i = 0; i < D; ++i) {
        wl1[i * D + i] = 1.f;
    }
    fill_zero(bl1, FF);
    fill_zero(wl2, D * FF);
    for (int i = 0; i < D; ++i) {
        wl2[i * FF + i] = 1.f;
    }
    fill_zero(bl2, D);
    fill_ones(n1w, D);
    fill_zero(n1b, D);
    fill_ones(n2w, D);
    fill_zero(n2b, D);

    Context ctx{arena, sizeof(arena), Backend::Ref};
    TransformerEncoderLayer layer(/*d_model=*/D, /*n_head=*/2, /*d_ff=*/FF);
    layer.set_weights(win, bin, wout, bout, wl1, bl1, wl2, bl2, n1w, n1b, n2w, n2b);

    Tensor xin{x, L, D};
    Tensor yout{y, L, D};
    layer.forward(ctx, xin, yout);

    std::printf("05_encoder_layer: ");
    for (int c = 0; c < D; ++c) {
        std::printf("%.4f%s", y[c], c + 1 == D ? "\n" : " ");
    }
    return 0;
}
