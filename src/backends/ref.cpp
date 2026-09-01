#include <picotorch/context.hpp>

namespace picotorch {

size_t workspace_bytes(int max_tokens, int d_model, int d_ff) {
    const int n = max_tokens;
    const int d = d_model;
    // Q, K, V, combo, packed tmp (3d), score tile, attn, ff, sequential pair.
    // Peak: encoder attn + MHA (Q,K,V,combo,score tile, packed tmp). FF is after MHA.
    const size_t floats = static_cast<size_t>(n) * static_cast<size_t>(d) +      // attn
                          static_cast<size_t>(n) * static_cast<size_t>(d) +      // Q
                          static_cast<size_t>(n) * static_cast<size_t>(d) +      // K
                          static_cast<size_t>(n) * static_cast<size_t>(d) +      // V
                          static_cast<size_t>(n) * static_cast<size_t>(d) +      // combo
                          static_cast<size_t>(n) +                               // score tile
                          static_cast<size_t>(n) * static_cast<size_t>(3 * d) +  // packed tmp
                          256;
    return floats * sizeof(float);
}

}  // namespace picotorch
