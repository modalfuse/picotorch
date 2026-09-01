#include <Arduino.h>
#include <esp_heap_caps.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "xattn_probe.hpp"
#include "xattn_tflite_model.hpp"

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
    Serial.println("LiteRT Micro Pico-Former xattn S3 FP32");
    Serial.flush();

    const size_t arena_bytes = 1024 * 1024;
    uint8_t *arena = static_cast<uint8_t *>(heap_caps_malloc(arena_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!arena) {
        arena = static_cast<uint8_t *>(heap_caps_malloc(arena_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    }
    if (!arena) {
        Serial.println("alloc FAIL");
        Serial.flush();
        return;
    }
    Serial.printf("arena_ok bytes=%u ptr=%p\n", (unsigned)arena_bytes, arena);
    Serial.flush();

    static tflite::MicroMutableOpResolver<16> resolver;
    resolver.AddFullyConnected();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddSoftmax();
    resolver.AddTanh();
    resolver.AddSlice();
    resolver.AddStridedSlice();
    resolver.AddSum();
    resolver.AddPack();
    resolver.AddConcatenation();
    resolver.AddSplit();
    resolver.AddReshape();

    const tflite::Model *model = tflite::GetModel(XATTN_TFLITE);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("schema %d != %d\n", (int)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    tflite::MicroInterpreter interpreter(model, resolver, arena, arena_bytes);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors FAIL");
        Serial.flush();
        return;
    }
    Serial.printf("alloc_ok used=%u\n", (unsigned)interpreter.arena_used_bytes());
    Serial.flush();

    TfLiteTensor *input = interpreter.input(0);
    memcpy(input->data.f, G_TOK, sizeof(float) * XATTN_N_TOK * XATTN_D);
    const uint32_t t0 = micros();
    if (interpreter.Invoke() != kTfLiteOk) {
        Serial.println("Invoke FAIL");
        Serial.flush();
        return;
    }
    Serial.println("invoke0_ok");
    Serial.flush();
    const uint32_t us0 = micros() - t0;

    TfLiteTensor *o0 = interpreter.output(0);
    TfLiteTensor *o1 = interpreter.output(1);
    const float *delta = o0->bytes == sizeof(float) * XATTN_N_OUT ? o0->data.f : o1->data.f;
    const float *logits = o0->bytes == sizeof(float) * XATTN_N_EVT ? o0->data.f : o1->data.f;
    float yhat[XATTN_N_OUT];
    for (int i = 0; i < XATTN_N_OUT; ++i) {
        yhat[i] = XATTN_LAST + delta[i];
    }
    const float dmax = maxabs(delta, G_DELTA, XATTN_N_OUT);
    const float lmax = maxabs(logits, G_LOGITS, XATTN_N_EVT);
    const float ymax = maxabs(yhat, G_YHAT, XATTN_N_OUT);

    uint32_t burn[50];
    for (int i = 0; i < 50; ++i) {
        memcpy(input->data.f, G_TOK, sizeof(float) * XATTN_N_TOK * XATTN_D);
        const uint32_t t = micros();
        interpreter.Invoke();
        burn[i] = micros() - t;
    }
    qsort(burn, 50, sizeof(uint32_t), cmp_u32);

    const size_t used = interpreter.arena_used_bytes();
    Serial.printf("dMax=%.4f lMax=%.6f yMax=%.4f\n", dmax, lmax, ymax);
    Serial.printf("ms_probe=%.1f ms_burn_median=%.1f ms_burn_p95=%.1f\n", us0 / 1000.0, burn[25] / 1000.0,
                  burn[47] / 1000.0);
    Serial.printf("work_sram=%u arena=%u arena_used=%u heap_internal=%u\n", (unsigned)used, (unsigned)arena_bytes,
                  (unsigned)used, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    Serial.println("LiteRT xattn ready");
}

void loop() { delay(1000); }
