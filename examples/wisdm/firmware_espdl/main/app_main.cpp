#include <cmath>
#include <cstdlib>
#include <cstring>

#include "dl_model_base.hpp"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "wisdm_weights.hpp"

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

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

static int argmax6(const float *logits) {
    int pred = 0;
    for (int c = 1; c < WISDM_N; ++c) {
        if (logits[c] > logits[pred]) {
            pred = c;
        }
    }
    return pred;
}

static int cmp_u64(const void *a, const void *b) {
    const int64_t x = *static_cast<const int64_t *>(a);
    const int64_t y = *static_cast<const int64_t *>(b);
    return (x > y) - (x < y);
}

extern "C" void app_main(void) {
    printf("ESP-DL WISDM S3 INT8\n");
    dl::Model *model = new dl::Model(reinterpret_cast<const char *>(model_espdl), fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    if (!model) {
        printf("model FAIL\n");
        return;
    }

    auto inputs = model->get_inputs();
    auto outputs = model->get_outputs();
    dl::TensorBase *model_input = inputs.begin()->second;
    dl::TensorBase *model_output = outputs.begin()->second;
    dl::TensorBase in_f({1, WISDM_L, 3}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    dl::TensorBase out_f({1, WISDM_N}, nullptr, 0, dl::DATA_TYPE_FLOAT);

    int agree = 0;
    float worst = 0.f;
    int64_t us[WISDM_PROBE];
    for (int p = 0; p < WISDM_PROBE; ++p) {
        memcpy(in_f.data, PROBE_X + p * WISDM_L * 3, sizeof(float) * WISDM_L * 3);
        model_input->assign(&in_f);
        const int64_t t0 = esp_timer_get_time();
        model->run();
        us[p] = esp_timer_get_time() - t0;
        out_f.assign(model_output);
        const float *logits = static_cast<const float *>(out_f.data);
        const float err = maxabs(logits, PROBE_LOGITS + p * WISDM_N, WISDM_N);
        if (err > worst) {
            worst = err;
        }
        if (argmax6(logits) == argmax6(PROBE_LOGITS + p * WISDM_N)) {
            ++agree;
        }
    }

    int64_t burn[50];
    for (int i = 0; i < 50; ++i) {
        memcpy(in_f.data, PROBE_X, sizeof(float) * WISDM_L * 3);
        model_input->assign(&in_f);
        const int64_t t0 = esp_timer_get_time();
        model->run();
        burn[i] = esp_timer_get_time() - t0;
    }
    qsort(us, WISDM_PROBE, sizeof(int64_t), cmp_u64);
    qsort(burn, 50, sizeof(int64_t), cmp_u64);

    model->profile_memory();
    printf("probe n=%d logit_maxabs=%.6e top1_agree=%.3f\n", WISDM_PROBE, worst, agree / (float)WISDM_PROBE);
    printf("ms_probe_median=%.1f ms_burn_median=%.1f ms_burn_p95=%.1f\n", us[WISDM_PROBE / 2] / 1000.0,
           burn[25] / 1000.0, burn[47] / 1000.0);
    printf("heap_internal=%u heap_psram=%u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    printf("ESP-DL WISDM ready\n");
    vTaskDelay(portMAX_DELAY);
}
