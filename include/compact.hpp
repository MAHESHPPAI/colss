// include/compat.hpp
#pragma once
#include <cstddef>

// ssize_t is POSIX; MSVC doesn't provide it. Make a portable alias on MSVC.
#if defined(_MSC_VER)
using ssize_t = std::ptrdiff_t;
#endif
