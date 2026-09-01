#include <picotorch/activation.hpp>
#include <picotorch/attention.hpp>
#include <picotorch/linear.hpp>

#include <cmath>
#include <cstring>

namespace picotorch {

static void project_qkv(Context &ctx, const float *q_src, const float *kv_src, int nq, int nk, int d, bool is_self,
                        const float *packed_w, const float *packed_b, const float *Wq, const float *bq,
                        const float *Wk, const float *bk, const float *Wv, const float *bv, float *Q, float *K,
                        float *V) {
    float *tmp = ctx.alloc_f32(static_cast<size_t>(nq > nk ? nq : nk) * static_cast<size_t>(3 * d));
    if (!tmp) {
        return;
    }

    if (packed_w) {
        if (is_self) {
            linear(q_src, packed_w, packed_b, tmp, nq, 3 * d, d);
            for (int i = 0; i < nq; ++i) {
                memcpy(Q + i * d, tmp + i * 3 * d, static_cast<size_t>(d) * sizeof(float));
                memcpy(K + i * d, tmp + i * 3 * d + d, static_cast<size_t>(d) * sizeof(float));
                memcpy(V + i * d, tmp + i * 3 * d + 2 * d, static_cast<size_t>(d) * sizeof(float));
            }
        } else {
            linear(q_src, packed_w, packed_b, Q, nq, d, d);
            linear(kv_src, packed_w + d * d, packed_b + d, tmp, nk, 2 * d, d);
            for (int i = 0; i < nk; ++i) {
                memcpy(K + i * d, tmp + i * 2 * d, static_cast<size_t>(d) * sizeof(float));
                memcpy(V + i * d, tmp + i * 2 * d + d, static_cast<size_t>(d) * sizeof(float));
            }
        }
        return;
    }

    linear(q_src, Wq, bq, Q, nq, d, d);
    linear(is_self ? q_src : kv_src, Wk, bk, K, is_self ? nq : nk, d, d);
    linear(is_self ? q_src : kv_src, Wv, bv, V, is_self ? nq : nk, d, d);
}

static void mha_core(Context &ctx, const float *q_src, const float *kv_src, int nq, int nk, int d, int n_head,
                     bool is_self, const float *packed_w, const float *packed_b, const float *Wq, const float *bq,
                     const float *Wk, const float *bk, const float *Wv, const float *bv, const float *out_w,
                     const float *out_b, float *y) {
    const int dk = d / n_head;
    const float scale = 1.f / sqrtf(static_cast<float>(dk));
    float *Q = ctx.alloc_f32(static_cast<size_t>(nq) * static_cast<size_t>(d));
    float *K = ctx.alloc_f32(static_cast<size_t>(nk) * static_cast<size_t>(d));
    float *V = ctx.alloc_f32(static_cast<size_t>(nk) * static_cast<size_t>(d));
    float *combo = ctx.alloc_f32(static_cast<size_t>(nq) * static_cast<size_t>(d));
    float *score = ctx.alloc_f32(static_cast<size_t>(nk));
    if (!Q || !K || !V || !combo || !score) {
        return;
    }

    project_qkv(ctx, q_src, kv_src, nq, nk, d, is_self, packed_w, packed_b, Wq, bq, Wk, bk, Wv, bv, Q, K, V);

    memset(y, 0, static_cast<size_t>(nq) * static_cast<size_t>(d) * sizeof(float));
    memset(combo, 0, static_cast<size_t>(nq) * static_cast<size_t>(d) * sizeof(float));

    for (int head = 0; head < n_head; ++head) {
        for (int i = 0; i < nq; ++i) {
            const float *qi = Q + i * d + head * dk;
            for (int j = 0; j < nk; ++j) {
                const float *kj = K + j * d + head * dk;
                float s = 0.f;
                for (int k = 0; k < dk; ++k) {
                    s += qi[k] * kj[k];
                }
                score[j] = s * scale;
            }
            softmax_row(score, nk);
            float acc[64];
            memset(acc, 0, static_cast<size_t>(dk) * sizeof(float));
            for (int j = 0; j < nk; ++j) {
                const float *vj = V + j * d + head * dk;
                const float a = score[j];
                for (int k = 0; k < dk; ++k) {
                    acc[k] += a * vj[k];
                }
            }
            memcpy(combo + i * d + head * dk, acc, static_cast<size_t>(dk) * sizeof(float));
        }
    }
    linear(combo, out_w, out_b, y, nq, d, d);
}

void MultiHeadAttention::forward(Context &ctx, const Tensor &x, Tensor &y) {
    const size_t mark = ctx.used;
    mha_core(ctx, x.data, x.data, x.rows, x.rows, d_model, n_head, true, in_w, in_b, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
             y.data);
    y.rows = x.rows;
    y.cols = d_model;
    ctx.used = mark;
}

void MultiHeadAttention::forward_cross(Context &ctx, const Tensor &q, const Tensor &kv, Tensor &y) {
    const size_t mark = ctx.used;
    mha_core(ctx, q.data, kv.data, q.rows, kv.rows, d_model, n_head, false, in_w, in_b, Wq, bq, Wk, bk, Wv, bv, Wo,
             bo, y.data);
    y.rows = q.rows;
    y.cols = d_model;
    ctx.used = mark;
}

}  // namespace picotorch
