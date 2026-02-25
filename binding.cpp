#include "fun/mean.hpp"
#include "fun/median.hpp"
#include "fun/prod.hpp"
#include "fun/query.hpp"
#include "fun/sigma.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_colss, m) {
    m.def("query", &query, py::arg("expr"));
    m.def("sigma", &sigma, py::arg("expr"));
    m.def("prod", &prod, py::arg("expr"));
    m.def("mean", &mean, py::arg("expr:"));
    m.def("median", &median, py::arg("expr"));
}