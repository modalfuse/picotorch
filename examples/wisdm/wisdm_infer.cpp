#include <cmath>
#include <cstdio>

#include <picotorch/context.hpp>
#include <picotorch/encoder.hpp>
#include <picotorch/linear.hpp>

#include "wisdm_graph.hpp"

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

int main() {
    using namespace picotorch;

    alignas(16) static float arena[32 * 1024];
    Context ctx{arena, sizeof(arena), Backend::Ref};
    Linear proj(WISDM_D, 3, W_PROJ, B_PROJ);
    TransformerEncoderLayer enc(WISDM_D, WISDM_H, WISDM_FF);
    enc.set_weights(W_IN, B_IN, W_OUT, B_OUT, W_FF1, B_FF1, W_FF2, B_FF2, W_N1, B_N1, W_N2, B_N2);
    Linear cls(WISDM_N, WISDM_D, W_CLS, B_CLS);

    static float tok[WISDM_L * WISDM_D];
    static float pooled[WISDM_D];
    static float logits[WISDM_N];
    int agree = 0;
    float worst = 0.f;

    for (int p = 0; p < WISDM_PROBE; ++p) {
        wisdm_forward(ctx, proj, enc, cls, PROBE_X + p * WISDM_L * 3, tok, pooled, logits);
        const float err = maxabs(logits, PROBE_LOGITS + p * WISDM_N, WISDM_N);
        if (err > worst) {
            worst = err;
        }
        if (wisdm_argmax6(logits) == wisdm_argmax6(PROBE_LOGITS + p * WISDM_N)) {
            ++agree;
        }
        if (p == 0) {
            std::printf("wisdm probe0 logits:");
            for (int c = 0; c < WISDM_N; ++c) {
                std::printf(" %.4f", logits[c]);
            }
            std::printf(" pred=%d pt=%d\n", wisdm_argmax6(logits), wisdm_argmax6(PROBE_LOGITS));
        }
    }

    const float agree_rate = static_cast<float>(agree) / static_cast<float>(WISDM_PROBE);
    std::printf("wisdm probe n=%d logit_maxabs=%.6e top1_agree=%.3f\n", WISDM_PROBE, worst, agree_rate);
    const bool ok = worst <= 1e-4f && agree_rate >= 1.f;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
