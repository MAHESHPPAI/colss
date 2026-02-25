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

inline double var(std::string expr, py::dict scalar_dict, py::kwargs arrays) {
    using T = double;

    unordered_map<string, double> scalars;
    for (auto item : scalar_dict)
        scalars[item.first.cast<string>()] = item.second.cast<double>();

    size_t n_vars = arrays.size();
    vector<string> names;
    vector<const double *> ptrs;
    vector<ssize_t> sizes;
    names.reserve(n_vars);
    ptrs.reserve(n_vars);
    sizes.reserve(n_vars);

    for (auto item : arrays) {
        string name = item.first.cast<string>();
        using arr_t =
            py::array_t<double, py::array::c_style | py::array::forcecast>;
        auto arr = item.second.cast<arr_t>();
        if (arr.ndim() != 1)
            throw runtime_error("Array must be 1D");
        names.push_back(name);
        ptrs.push_back(arr.data());
        sizes.push_back(arr.size());
    }

    ssize_t n = n_vars ? sizes[0] : 1;
    for (size_t j = 1; j < n_vars; ++j)
        if (sizes[j] != n)
            throw runtime_error("Array size mismatch");

    if (n <= 0)
        throw runtime_error("Invalid input size");

    vector<T> results(n);
    double sum = 0.0;

#pragma omp parallel
    {
        exprtk::symbol_table<T> symbol_table;
        exprtk::expression<T> expression;
        exprtk::parser<T> parser;

        vector<T> variables(n_vars, 0.0);
        for (size_t j = 0; j < n_vars; ++j)
            symbol_table.add_variable(names[j], variables[j]);

        unordered_map<string, T> scalar_locals(scalars.begin(), scalars.end());
        for (auto &p : scalar_locals)
            symbol_table.add_variable(p.first, p.second);

        symbol_table.add_constants();
        expression.register_symbol_table(symbol_table);

        if (!parser.compile(expr, expression))
            throw runtime_error("Expression compile failed");

#pragma omp for reduction(+ : sum)
        for (ssize_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n_vars; ++j)
                variables[j] = ptrs[j][i];
            double val = expression.value();
            results[i] = val;
            sum += val;
        }
    }

    double mean = sum / n;
    double var_sum = 0.0;

#pragma omp parallel for simd reduction(+ : var_sum)
    for (ssize_t i = 0; i < n; ++i) {
        double d = results[i] - mean;
        var_sum += d * d;
    }

    return var_sum / n;
}
