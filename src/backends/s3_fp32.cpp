#include <picotorch/backends/s3_fp32.hpp>

#include <cstdlib>

#if defined(ARDUINO) || defined(ESP_PLATFORM)
#include <esp_heap_caps.h>
#endif

namespace picotorch {

void *s3_malloc(size_t bytes) {
#if defined(ARDUINO) || defined(ESP_PLATFORM)
    return heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
#else
    return std::malloc(bytes);
#endif
}

void s3_free(void *p) {
#if defined(ARDUINO) || defined(ESP_PLATFORM)
    heap_caps_free(p);
#else
    std::free(p);
#endif
}

}  // namespace picotorch
