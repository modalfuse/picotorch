#pragma once

#include <cstddef>

namespace picotorch {

void *s3_malloc(size_t bytes);
void s3_free(void *p);

}  // namespace picotorch
