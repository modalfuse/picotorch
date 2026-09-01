#include <Arduino.h>
#include <esp_heap_caps.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

static int cmp_u32(const void *a, const void *b) {
    const uint32_t x = *static_cast<const uint32_t *>(a);
    const uint32_t y = *static_cast<const uint32_t *>(b);
    return (x > y) - (x < y);
}

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println("PicoTorch WISDM S3");

    const size_t arena_bytes = picotorch::workspace_bytes(WISDM_L, WISDM_D, WISDM_FF);
    float *arena = static_cast<float *>(heap_caps_malloc(arena_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    float *tok = static_cast<float *>(heap_caps_malloc((size_t)WISDM_L * WISDM_D * sizeof(float),
                                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!arena || !tok) {
        Serial.println("alloc FAIL");
        return;
    }
    const size_t work_bytes = arena_bytes + (size_t)WISDM_L * WISDM_D * sizeof(float) + WISDM_D * sizeof(float);

    picotorch::Context ctx{arena, arena_bytes, picotorch::Backend::S3Fp32};
    picotorch::Linear proj(WISDM_D, 3, W_PROJ, B_PROJ);
    picotorch::TransformerEncoderLayer enc(WISDM_D, WISDM_H, WISDM_FF);
    enc.set_weights(W_IN, B_IN, W_OUT, B_OUT, W_FF1, B_FF1, W_FF2, B_FF2, W_N1, B_N1, W_N2, B_N2);
    picotorch::Linear cls(WISDM_N, WISDM_D, W_CLS, B_CLS);

    float pooled[WISDM_D];
    float logits[WISDM_N];
    int agree = 0;
    float worst = 0.f;
    uint32_t us[WISDM_PROBE];

    for (int p = 0; p < WISDM_PROBE; ++p) {
        const uint32_t t0 = micros();
        wisdm_forward(ctx, proj, enc, cls, PROBE_X + p * WISDM_L * 3, tok, pooled, logits);
        us[p] = micros() - t0;
        const float err = maxabs(logits, PROBE_LOGITS + p * WISDM_N, WISDM_N);
        if (err > worst) {
            worst = err;
        }
        if (wisdm_argmax6(logits) == wisdm_argmax6(PROBE_LOGITS + p * WISDM_N)) {
            ++agree;
        }
    }

    uint32_t burn[50];
    for (int i = 0; i < 50; ++i) {
        const uint32_t t0 = micros();
        wisdm_forward(ctx, proj, enc, cls, PROBE_X, tok, pooled, logits);
        burn[i] = micros() - t0;
    }
    qsort(us, WISDM_PROBE, sizeof(uint32_t), cmp_u32);
    qsort(burn, 50, sizeof(uint32_t), cmp_u32);

    Serial.printf("probe n=%d logit_maxabs=%.6e top1_agree=%.3f\n", WISDM_PROBE, worst,
                  agree / (float)WISDM_PROBE);
    Serial.printf("ms_probe_median=%.1f ms_burn_median=%.1f ms_burn_p95=%.1f\n", us[WISDM_PROBE / 2] / 1000.0,
                  burn[25] / 1000.0, burn[47] / 1000.0);
    Serial.printf("work_sram=%u arena=%u heap_internal=%u\n", (unsigned)work_bytes, (unsigned)arena_bytes,
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    Serial.println("PicoTorch WISDM ready");
}

void loop() { delay(1000); }
