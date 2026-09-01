#pragma once

#include <cstring>

inline void conv2d_relu(const float *x, const float *W, const float *b, float *y, int in_c, int out_c, int in_h,
                        int in_w, int stride, int pad, int k) {
    const int out_h = (in_h + 2 * pad - k) / stride + 1;
    const int out_w = (in_w + 2 * pad - k) / stride + 1;
    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float s = b ? b[oc] : 0.f;
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < k; ++kh) {
                        const int ih = oh * stride + kh - pad;
                        for (int kw = 0; kw < k; ++kw) {
                            const int iw = ow * stride + kw - pad;
                            if (ih < 0 || iw < 0 || ih >= in_h || iw >= in_w) {
                                continue;
                            }
                            const float xv = x[(ic * in_h + ih) * in_w + iw];
                            const float wv = W[(((oc * in_c + ic) * k) + kh) * k + kw];
                            s += wv * xv;
                        }
                    }
                }
                y[(oc * out_h + oh) * out_w + ow] = s > 0.f ? s : 0.f;
            }
        }
    }
}

inline void nchw_to_tokens(const float *x, float *tok, int h, int w, int d) {
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            for (int ch = 0; ch < d; ++ch) {
                tok[(r * w + c) * d + ch] = x[(ch * h + r) * w + c];
            }
        }
    }
}
