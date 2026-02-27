#include <torch/extension.h>
#include "kernel.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", &square, "square");
    m.def("relu", &relu, "relu");
    m.def("quant_relu", &quant_relu, "quant_relu");
    m.def("quant_add", &quant_add, "quant_add");
    m.def("quant_linear", &quant_linear, "quant_linear");
};
