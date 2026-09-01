#pragma once

#include <cstddef>
#include <cstdint>

namespace picotorch {

enum class Backend { Ref, S3Fp32 };

struct Context {
    void *arena;
    size_t arena_bytes;
    Backend backend;
    size_t used;

    Context(void *a, size_t n, Backend b) : arena(a), arena_bytes(n), backend(b), used(0) {}

    void reset() { used = 0; }

    void *alloc(size_t bytes, size_t align = 16) {
        if (!arena || bytes == 0) {
            return nullptr;
        }
        const uintptr_t base = reinterpret_cast<uintptr_t>(arena) + used;
        const uintptr_t mask = static_cast<uintptr_t>(align - 1);
        const uintptr_t aligned = (base + mask) & ~mask;
        const size_t pad = static_cast<size_t>(aligned - reinterpret_cast<uintptr_t>(arena));
        if (pad + bytes > arena_bytes) {
            return nullptr;
        }
        used = pad + bytes;
        return reinterpret_cast<void *>(aligned);
    }

    float *alloc_f32(size_t n) { return static_cast<float *>(alloc(n * sizeof(float))); }
};

size_t workspace_bytes(int max_tokens, int d_model, int d_ff);

}  // namespace picotorch
