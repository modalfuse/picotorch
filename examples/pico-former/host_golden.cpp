#include <cstdio>

#include "ceqt_infer.h"

int main() {
    CeqtResult out;
    ceqt_infer_golden(&out);
    std::printf("host_golden dMax=%.4f lMax=%.5f yMax=%.4f 1h=%.1f 4h=%.1f\n", out.maxabs_delta, out.maxabs_logits,
                out.maxabs_yhat, out.pred_1h, out.pred_4h);
    const bool ok = out.maxabs_yhat <= 0.0095f + 1e-4f && out.maxabs_logits <= 5.8e-4f + 1e-4f;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
