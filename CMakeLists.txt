cmake_minimum_required(VERSION 3.16)
project(colss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(_colss binding.cpp)

# Link OpenMP properly (portable way)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_colss PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(_colss PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Compiler-specific optimization (portable)
if(MSVC)
    target_compile_options(_colss PRIVATE /O2)
else()
    target_compile_options(_colss PRIVATE -O3)
endif()

install(TARGETS _colss DESTINATION colss)
install(FILES ${CMAKE_SOURCE_DIR}/colss/__init__.py DESTINATION colss)
