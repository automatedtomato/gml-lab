#include "kernel.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_relu", &quant_relu, "quant_relu");
    m.def("quant_add", &quant_add, "quant_add");
    m.def("quant_linear", &quant_linear, "quant_linear");
};
