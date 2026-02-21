#include "fun/sum.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(colss, m) { m.def("sigma", &sigma, py::arg("expr")); }
