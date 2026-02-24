#pragma once
#include "../include/compact.hpp"
#include "../include/eval.hpp"
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
namespace py = pybind11;

inline py::array_t<double> query(std::string expr, py::dict scalar_dict,
                                 py::kwargs arrays) {
    using T = double;

    // --- Parse scalars ---
    std::unordered_map<std::string, double> scalars;
    for (auto item : scalar_dict)
        scalars[item.first.cast<std::string>()] = item.second.cast<double>();

    // --- Parse arrays ---
    size_t n_vars = arrays.size();
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

    // If no arrays at all (pure scalar expr), n=1
    ssize_t n = n_vars > 0 ? sizes[0] : 1;
    for (size_t j = 1; j < n_vars; ++j)
        if (sizes[j] != n)
            throw std::runtime_error("Array size mismatch");

    py::array_t<double> result(n);
    auto buf = result.mutable_unchecked<1>();

#pragma omp parallel
    {
        exprtk::symbol_table<T> symbol_table;
        exprtk::expression<T> expression;
        exprtk::parser<T> parser;

        // Register array variables (thread-local copies)
        std::vector<T> variables(n_vars, 0.0);
        for (size_t j = 0; j < n_vars; ++j)
            symbol_table.add_variable(names[j], variables[j]);

        // Register scalars — added once, never change per-iteration = free
        std::unordered_map<std::string, T> scalar_locals(scalars.begin(),
                                                         scalars.end());
        for (auto &[name, val] : scalar_locals)
            symbol_table.add_variable(name, scalar_locals[name]);

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
