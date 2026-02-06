#include <torch/extension.h>
#include "kernel.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", &square, "square");
};