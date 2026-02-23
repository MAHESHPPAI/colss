#pragma once

#include "../include/exprtk.hpp"
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

using namespace std;
namespace py = pybind11;

inline double sigma(string expr, py::args headers_and_arr) {
    using T = double;

    size_t n_vars = headers_and_arr.size();

    vector<string> names;
    vector<py::detail::unchecked_reference<T, 1>> buffers;

    names.reserve(n_vars);
    buffers.reserve(n_vars);

    for (auto arg : headers_and_arr) {
        auto t = arg.cast<py::tuple>();

        string name = t[0].cast<string>();
        py::array_t<T> arr = t[1].cast<py::array_t<T>>();

        names.push_back(name);
        buffers.push_back(arr.unchecked<1>());
    }

    size_t n = buffers[0].shape(0);

    for (size_t i = 1; i < buffers.size(); ++i)
        if (buffers[i].shape(0) != n)
            throw runtime_error("Array size mismatch");

    double share = 0.0;

#pragma omp parallel reduction(+ : share)
    {
        // 🔥 Thread-local engine
        exprtk::symbol_table<T> symbol_table;
        exprtk::expression<T> expression;
        exprtk::parser<T> parser;

        vector<T> variables(n_vars, 0.0);

        for (size_t i = 0; i < n_vars; ++i)
            symbol_table.add_variable(names[i], variables[i]);

        symbol_table.add_constants();
        expression.register_symbol_table(symbol_table);

        if (!parser.compile(expr, expression))
            throw runtime_error("Expression compile failed");

#pragma omp for
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n_vars; ++j)
                variables[j] = buffers[j](i);

            share += expression.value();
        }
    }

    return share;
}
