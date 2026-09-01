#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

#include "dl_model_base.hpp"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "xattn_probe.hpp"

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

static int cmp_u64(const void *a, const void *b) {
    const int64_t x = *static_cast<const int64_t *>(a);
    const int64_t y = *static_cast<const int64_t *>(b);
    return (x > y) - (x < y);
}

extern "C" void app_main(void) {
    printf("ESP-DL Pico-Former xattn S3 INT8\n");
    dl::Model *model = new dl::Model(reinterpret_cast<const char *>(model_espdl), fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    if (!model) {
        printf("model FAIL\n");
        return;
    }

    auto inputs = model->get_inputs();
    auto outputs = model->get_outputs();
    dl::TensorBase *model_input = inputs.begin()->second;
    dl::TensorBase in_f({1, XATTN_N_TOK, XATTN_D}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    memcpy(in_f.data, G_TOK, sizeof(float) * XATTN_N_TOK * XATTN_D);
    model_input->assign(&in_f);

    const int64_t t0 = esp_timer_get_time();
    model->run();
    const int64_t us0 = esp_timer_get_time() - t0;

    dl::TensorBase *delta_t = nullptr;
    dl::TensorBase *logit_t = nullptr;
    for (auto &kv : outputs) {
        if (kv.second->get_size() == XATTN_N_OUT) {
            delta_t = kv.second;
        } else if (kv.second->get_size() == XATTN_N_EVT) {
            logit_t = kv.second;
        }
    }
    dl::TensorBase delta_f({1, XATTN_N_OUT}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    dl::TensorBase logit_f({1, XATTN_N_EVT}, nullptr, 0, dl::DATA_TYPE_FLOAT);
    delta_f.assign(delta_t);
    logit_f.assign(logit_t);
    const float *delta = static_cast<const float *>(delta_f.data);
    const float *logits = static_cast<const float *>(logit_f.data);
    float yhat[XATTN_N_OUT];
    for (int i = 0; i < XATTN_N_OUT; ++i) {
        yhat[i] = XATTN_LAST + delta[i];
    }
    const float dmax = maxabs(delta, G_DELTA, XATTN_N_OUT);
    const float lmax = maxabs(logits, G_LOGITS, XATTN_N_EVT);
    const float ymax = maxabs(yhat, G_YHAT, XATTN_N_OUT);

    int64_t burn[50];
    for (int i = 0; i < 50; ++i) {
        memcpy(in_f.data, G_TOK, sizeof(float) * XATTN_N_TOK * XATTN_D);
        model_input->assign(&in_f);
        const int64_t t = esp_timer_get_time();
        model->run();
        burn[i] = esp_timer_get_time() - t;
    }
    qsort(burn, 50, sizeof(int64_t), cmp_u64);

    model->profile_memory();
    printf("dMax=%.4f lMax=%.6f yMax=%.4f\n", dmax, lmax, ymax);
    printf("ms_probe=%.1f ms_burn_median=%.1f ms_burn_p95=%.1f\n", us0 / 1000.0, burn[25] / 1000.0, burn[47] / 1000.0);
    printf("heap_internal=%u heap_psram=%u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    printf("ESP-DL xattn ready\n");
    vTaskDelay(portMAX_DELAY);
}
