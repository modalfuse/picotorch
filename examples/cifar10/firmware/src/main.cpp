#include <Arduino.h>
#include <esp_heap_caps.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <picotorch/context.hpp>
#include <picotorch/encoder.hpp>
#include <picotorch/linear.hpp>

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

static int cmp_u32(const void *a, const void *b) {
    const uint32_t x = *static_cast<const uint32_t *>(a);
    const uint32_t y = *static_cast<const uint32_t *>(b);
    return (x > y) - (x < y);
}

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println("PicoTorch CIFAR-10 S3");

    const size_t arena_bytes = picotorch::workspace_bytes(CIFAR_L, CIFAR_D, CIFAR_FF);
    float *arena = static_cast<float *>(heap_caps_malloc(arena_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *y16 = static_cast<float *>(heap_caps_malloc(16 * 16 * 16 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *y8 = static_cast<float *>(heap_caps_malloc(16 * 8 * 8 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *y4 = static_cast<float *>(heap_caps_malloc(16 * 4 * 4 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *tok = static_cast<float *>(heap_caps_malloc((size_t)CIFAR_L * CIFAR_D * sizeof(float),
                                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!arena || !y16 || !y8 || !y4 || !tok) {
        Serial.println("alloc FAIL");
        return;
    }
    const size_t work_bytes = arena_bytes + (16 * 16 * 16 + 16 * 8 * 8 + 16 * 4 * 4 + CIFAR_L * CIFAR_D + CIFAR_D) *
                                                sizeof(float);

    picotorch::Context ctx{arena, arena_bytes, picotorch::Backend::S3Fp32};
    picotorch::TransformerEncoderLayer enc(CIFAR_D, CIFAR_H, CIFAR_FF);
    enc.set_weights(W_IN, B_IN, W_OUT, B_OUT, W_FF1, B_FF1, W_FF2, B_FF2, W_N1, B_N1, W_N2, B_N2);
    picotorch::Linear cls(CIFAR_N, CIFAR_D, W_CLS, B_CLS);

    float pooled[CIFAR_D];
    float logits[CIFAR_N];
    int agree = 0;
    float worst = 0.f;
    uint32_t us[CIFAR_PROBE];

    for (int p = 0; p < CIFAR_PROBE; ++p) {
        const uint32_t t0 = micros();
        cifar_forward(ctx, enc, cls, PROBE_X + p * 3 * 32 * 32, y16, y8, y4, tok, pooled, logits);
        us[p] = micros() - t0;
        const float err = maxabs(logits, PROBE_LOGITS + p * CIFAR_N, CIFAR_N);
        if (err > worst) {
            worst = err;
        }
        if (cifar_argmax10(logits) == cifar_argmax10(PROBE_LOGITS + p * CIFAR_N)) {
            ++agree;
        }
    }

    uint32_t burn[50];
    for (int i = 0; i < 50; ++i) {
        const uint32_t t0 = micros();
        cifar_forward(ctx, enc, cls, PROBE_X, y16, y8, y4, tok, pooled, logits);
        burn[i] = micros() - t0;
    }
    qsort(us, CIFAR_PROBE, sizeof(uint32_t), cmp_u32);
    qsort(burn, 50, sizeof(uint32_t), cmp_u32);

    Serial.printf("probe n=%d logit_maxabs=%.6e top1_agree=%.3f\n", CIFAR_PROBE, worst,
                  agree / (float)CIFAR_PROBE);
    Serial.printf("ms_probe_median=%.1f ms_burn_median=%.1f ms_burn_p95=%.1f\n", us[CIFAR_PROBE / 2] / 1000.0,
                  burn[25] / 1000.0, burn[47] / 1000.0);
    Serial.printf("work_sram=%u arena=%u heap_internal=%u\n", (unsigned)work_bytes, (unsigned)arena_bytes,
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    Serial.println("PicoTorch CIFAR-10 ready");
}

void loop() { delay(1000); }
