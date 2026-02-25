#pragma once

#include <cstdlib>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

inline long partition(double *arr, long low, long high) {
    double pivot = arr[high];
    long i = low;

    for (long j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            std::swap(arr[i], arr[j]);
            i++;
        }
    }
    std::swap(arr[i], arr[high]);
    return i;
}

inline double quickselect(double *arr, long low, long high, long k) {
    while (low <= high) {
        long pivotIndex = partition(arr, low, high);

        if (pivotIndex == k)
            return arr[pivotIndex];
        else if (pivotIndex > k)
            high = pivotIndex - 1;
        else
            low = pivotIndex + 1;
    }
    return arr[k]; // fallback (shouldn't reach here)
}

inline py::array_t<double> quick(py::array_t<double> array, long k) {

    auto buf = array.request();

    if (buf.ndim != 1)
        throw std::runtime_error("Input must be 1D");

    long n = buf.shape[0];

    if (k < 0 || k >= n)
        throw std::runtime_error("k out of bounds");

    // copying the array
    py::array_t<double> result = py::array_t<double>(1);
    double *ptr = static_cast<double *>(buf.ptr);

    std::vector<double> temp(ptr, ptr + n);

    double val = quickselect(temp.data(), 0, n - 1, k);

    result.mutable_at(0) = val;

    return result;
}
