
#pragma once
#include "../include/exprtk.hpp"
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace py = pybind11;

inline py::array_t<double> query(std::string expr, py::kwargs arrays) {
    using T = double;

    size_t n_vars = arrays.size();
    if (n_vars == 0)
        throw std::runtime_error("No arrays provided");

    std::vector<std::string> names;
    std::vector<const double *> ptrs;
    std::vector<ssize_t> sizes;

    names.reserve(n_vars);
    ptrs.reserve(n_vars);
    sizes.reserve(n_vars);

    for (auto item : arrays) {
        std::string name = item.first.cast<std::string>();

        using arr_t =
            py::array_t<double, py::array::c_style | py::array::forcecast>;
        auto arr = item.second.cast<arr_t>();

        if (arr.ndim() != 1)
            throw std::runtime_error("Array must be 1D");

        names.push_back(name);
        ptrs.push_back(arr.data());
        sizes.push_back(arr.size());
    }

    ssize_t n = sizes[0];
    for (size_t j = 1; j < n_vars; ++j)
        if (sizes[j] != n)
            throw std::runtime_error("Array size mismatch");

    // Allocate output NumPy array
    py::array_t<double> result(n);
    auto buf = result.mutable_unchecked<1>();

#pragma omp parallel
    {
        exprtk::symbol_table<T> symbol_table;
        exprtk::expression<T> expression;
        exprtk::parser<T> parser;

        std::vector<T> variables(n_vars, 0.0);

        for (size_t j = 0; j < n_vars; ++j)
            symbol_table.add_variable(names[j], variables[j]);

        symbol_table.add_constants();
        expression.register_symbol_table(symbol_table);

        if (!parser.compile(expr, expression))
            throw std::runtime_error("Expression compile failed");

#pragma omp for schedule(static)
        for (ssize_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n_vars; ++j)
                variables[j] = ptrs[j][i];

            buf(i) = expression.value();
        }
    }

    return result;
}
