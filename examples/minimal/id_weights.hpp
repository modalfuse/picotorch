#pragma once

#include <cstring>

inline void fill_identity(float *W, int n) {
    memset(W, 0, static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float));
    for (int i = 0; i < n; ++i) {
        W[i * n + i] = 1.f;
    }
}

inline void fill_ones(float *v, int n, float s = 1.f) {
    for (int i = 0; i < n; ++i) {
        v[i] = s;
    }
}

inline void fill_zero(float *v, int n) { memset(v, 0, static_cast<size_t>(n) * sizeof(float)); }
