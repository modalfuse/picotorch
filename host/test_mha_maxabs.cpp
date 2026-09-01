#include <cmath>
#include <cstdio>
#include <cstring>

#include <picotorch/activation.hpp>
#include <picotorch/attention.hpp>
#include <picotorch/context.hpp>
#include <picotorch/layernorm.hpp>
#include <picotorch/linear.hpp>
#include <picotorch/tensor.hpp>

static float maxabs(const float *a, const float *b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) {
        const float d = fabsf(a[i] - b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

static void fill_identity(float *W, int n) {
    memset(W, 0, static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float));
    for (int i = 0; i < n; ++i) {
        W[i * n + i] = 1.f;
    }
}

int main() {
    using namespace picotorch;

    // Linear: y = [1.1, 2.2]
    const float W[] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    const float b[] = {0.10f, 0.20f};
    float x1[] = {1.f, 2.f, 3.f};
    float y1[2];
    const float y1_ref[] = {1.10f, 2.20f};
    alignas(16) float arena[32 * 1024];
    Context ctx{arena, sizeof(arena), Backend::Ref};
    Linear fc(2, 3, W, b);
    Tensor xin{x1, 1, 3};
    Tensor yout{y1, 1, 2};
    fc.forward(ctx, xin, yout);
    const float lin_err = maxabs(y1, y1_ref, 2);

    // LayerNorm on [1,2,3] with γ=1, β=0
    float row[] = {1.f, 2.f, 3.f};
    const float g[] = {1.f, 1.f, 1.f};
    const float be[] = {0.f, 0.f, 0.f};
    layernorm(row, g, be, 1, 3, 1e-5f);
    const float mean = 2.f;
    const float var = ((1.f - mean) * (1.f - mean) + (2.f - mean) * (2.f - mean) + (3.f - mean) * (3.f - mean)) / 3.f;
    const float inv = 1.f / sqrtf(var + 1e-5f);
    const float ln_ref[] = {(1.f - mean) * inv, (2.f - mean) * inv, (3.f - mean) * inv};
    const float ln_err = maxabs(row, ln_ref, 3);

    // Identity MHA, L=8, d=16, h=2 — same fill as examples/minimal/03
    constexpr int L = 8;
    constexpr int D = 16;
    float x[L * D];
    float y[L * D];
    float Wq[D * D], Wk[D * D], Wv[D * D], Wo[D * D];
    float bq[D], bk[D], bv[D], bo[D];
    for (int i = 0; i < L * D; ++i) {
        x[i] = 0.01f * static_cast<float>(i + 1);
    }
    fill_identity(Wq, D);
    fill_identity(Wk, D);
    fill_identity(Wv, D);
    fill_identity(Wo, D);
    memset(bq, 0, sizeof(bq));
    memset(bk, 0, sizeof(bk));
    memset(bv, 0, sizeof(bv));
    memset(bo, 0, sizeof(bo));

    ctx.reset();
    MultiHeadAttention mha(2, D, true);
    mha.set_weights(Wq, bq, Wk, bk, Wv, bv, Wo, bo);
    Tensor xt{x, L, D};
    Tensor yt{y, L, D};
    mha.forward(ctx, xt, yt);

    // RICE shape smoke: L=96, d=48, h=4 — random-ish deterministic, finite output
    constexpr int LR = 96;
    constexpr int DR = 48;
    static float xr[LR * DR];
    static float yr[LR * DR];
    static float Wqr[DR * DR], Wkr[DR * DR], Wvr[DR * DR], Wor[DR * DR];
    static float bqr[DR], bkr[DR], bvr[DR], bor[DR];
    for (int i = 0; i < LR * DR; ++i) {
        xr[i] = 0.001f * static_cast<float>((i % 17) + 1);
    }
    fill_identity(Wqr, DR);
    fill_identity(Wkr, DR);
    fill_identity(Wvr, DR);
    fill_identity(Wor, DR);
    memset(bqr, 0, sizeof(bqr));
    memset(bkr, 0, sizeof(bkr));
    memset(bvr, 0, sizeof(bvr));
    memset(bor, 0, sizeof(bor));
    ctx.reset();
    MultiHeadAttention mha_r(4, DR, true);
    mha_r.set_weights(Wqr, bqr, Wkr, bkr, Wvr, bvr, Wor, bor);
    Tensor xtr{xr, LR, DR};
    Tensor ytr{yr, LR, DR};
    mha_r.forward(ctx, xtr, ytr);
    float rice_abs = 0.f;
    for (int i = 0; i < LR * DR; ++i) {
        if (!isfinite(yr[i])) {
            std::printf("FAIL rice MHA non-finite\n");
            return 1;
        }
        rice_abs = fmaxf(rice_abs, fabsf(yr[i]));
    }

    std::printf("linear_maxabs=%.6e ln_maxabs=%.6e mha_y00=%.6f rice_absmax=%.6f\n", lin_err, ln_err, y[0], rice_abs);
    if (lin_err > 1e-5f || ln_err > 1e-5f) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
