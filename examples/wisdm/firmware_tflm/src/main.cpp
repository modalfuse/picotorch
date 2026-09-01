#include <Arduino.h>
#include <esp_heap_caps.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "wisdm_tflite_model.hpp"
#include "wisdm_weights.hpp"

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

static int cmp_u32(const void *a, const void *b) {
    const uint32_t x = *static_cast<const uint32_t *>(a);
    const uint32_t y = *static_cast<const uint32_t *>(b);
    return (x > y) - (x < y);
}

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.println("LiteRT Micro WISDM S3 FP32");

    const size_t arena_bytes = 96 * 1024;
    uint8_t *arena = static_cast<uint8_t *>(heap_caps_malloc(arena_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!arena) {
        Serial.println("alloc FAIL");
        return;
    }

    static tflite::MicroMutableOpResolver<16> resolver;
    resolver.AddFullyConnected();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddSoftmax();
    resolver.AddMean();
    resolver.AddRsqrt();
    resolver.AddSquaredDifference();
    resolver.AddSplit();
    resolver.AddStridedSlice();
    resolver.AddConcatenation();
    resolver.AddSlice();
    resolver.AddPack();
    resolver.AddSum();

    const tflite::Model *model = tflite::GetModel(WISDM_TFLITE);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("schema %d != %d\n", (int)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    tflite::MicroInterpreter interpreter(model, resolver, arena, arena_bytes);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors FAIL");
        return;
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);
    int agree = 0;
    float worst = 0.f;
    uint32_t us[WISDM_PROBE];

    for (int p = 0; p < WISDM_PROBE; ++p) {
        memcpy(input->data.f, PROBE_X + p * WISDM_L * 3, sizeof(float) * WISDM_L * 3);
        const uint32_t t0 = micros();
        if (interpreter.Invoke() != kTfLiteOk) {
            Serial.println("Invoke FAIL");
            return;
        }
        us[p] = micros() - t0;
        const float err = maxabs(output->data.f, PROBE_LOGITS + p * WISDM_N, WISDM_N);
        if (err > worst) {
            worst = err;
        }
        if (argmax6(output->data.f) == argmax6(PROBE_LOGITS + p * WISDM_N)) {
            ++agree;
        }
    }

    uint32_t burn[50];
    for (int i = 0; i < 50; ++i) {
        memcpy(input->data.f, PROBE_X, sizeof(float) * WISDM_L * 3);
        const uint32_t t0 = micros();
        interpreter.Invoke();
        burn[i] = micros() - t0;
    }
    qsort(us, WISDM_PROBE, sizeof(uint32_t), cmp_u32);
    qsort(burn, 50, sizeof(uint32_t), cmp_u32);

    const size_t used = interpreter.arena_used_bytes();
    Serial.printf("probe n=%d logit_maxabs=%.6e top1_agree=%.3f\n", WISDM_PROBE, worst, agree / (float)WISDM_PROBE);
    Serial.printf("ms_probe_median=%.1f ms_burn_median=%.1f ms_burn_p95=%.1f\n", us[WISDM_PROBE / 2] / 1000.0,
                  burn[25] / 1000.0, burn[47] / 1000.0);
    Serial.printf("work_sram=%u arena=%u arena_used=%u heap_internal=%u\n", (unsigned)arena_bytes, (unsigned)arena_bytes,
                  (unsigned)used, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    Serial.println("LiteRT WISDM ready");
}

void loop() { delay(1000); }
