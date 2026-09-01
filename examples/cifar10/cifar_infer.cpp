#include <cmath>
#include <cstdio>
#include <cstring>

#include <picotorch/context.hpp>
#include <picotorch/encoder.hpp>
#include <picotorch/linear.hpp>
#include <picotorch/tensor.hpp>

#include "cifar_graph.hpp"

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
    alignas(16) static float arena[16 * 1024];
    Context ctx{arena, sizeof(arena), Backend::Ref};
    TransformerEncoderLayer enc(CIFAR_D, CIFAR_H, CIFAR_FF);
    enc.set_weights(W_IN, B_IN, W_OUT, B_OUT, W_FF1, B_FF1, W_FF2, B_FF2, W_N1, B_N1, W_N2, B_N2);
    Linear cls(CIFAR_N, CIFAR_D, W_CLS, B_CLS);

    static float y16[16 * 16 * 16];
    static float y8[16 * 8 * 8];
    static float y4[16 * 4 * 4];
    static float tok[CIFAR_L * CIFAR_D];
    static float pooled[CIFAR_D];
    static float logits[CIFAR_N];

    int agree = 0;
    float worst = 0.f;
    for (int p = 0; p < CIFAR_PROBE; ++p) {
        cifar_forward(ctx, enc, cls, PROBE_X + p * 3 * 32 * 32, y16, y8, y4, tok, pooled, logits);
        const float err = maxabs(logits, PROBE_LOGITS + p * CIFAR_N, CIFAR_N);
        if (err > worst) {
            worst = err;
        }
        if (cifar_argmax10(logits) == cifar_argmax10(PROBE_LOGITS + p * CIFAR_N)) {
            ++agree;
        }
        if (p == 0) {
            std::printf("cifar probe0 logits:");
            for (int c = 0; c < CIFAR_N; ++c) {
                std::printf(" %.4f", logits[c]);
            }
            std::printf(" pred=%d pt=%d\n", cifar_argmax10(logits), cifar_argmax10(PROBE_LOGITS));
        }
    }
    const float rate = static_cast<float>(agree) / static_cast<float>(CIFAR_PROBE);
    std::printf("cifar probe n=%d logit_maxabs=%.6e top1_agree=%.3f\n", CIFAR_PROBE, worst, rate);
    const bool ok = worst <= 1e-4f && rate >= 1.f;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
