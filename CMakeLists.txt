cmake_minimum_required(VERSION 3.16)
project(colss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(_colss binding.cpp)

target_link_libraries(_colss PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(_colss PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_options(_colss PRIVATE
    -O3
    -march=native
    -ffast-math
    -funroll-loops
)

install(TARGETS _colss DESTINATION colss)
install(FILES ${CMAKE_SOURCE_DIR}/colss/__init__.py DESTINATION colss)
