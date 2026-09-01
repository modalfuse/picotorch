#pragma once

#include <initializer_list>

#include <picotorch/module.hpp>

namespace picotorch {

struct Sequential : Module {
    Module *layers[8];
    int n_layers;

    Sequential() : n_layers(0) {
        for (int i = 0; i < 8; ++i) {
            layers[i] = nullptr;
        }
    }

    Sequential(std::initializer_list<Module *> xs) : Sequential() {
        for (Module *m : xs) {
            if (n_layers < 8) {
                layers[n_layers++] = m;
            }
        }
    }

    void forward(Context &ctx, const Tensor &x, Tensor &y) override;
};

}  // namespace picotorch
