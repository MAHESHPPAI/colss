cmake_minimum_required(VERSION 3.16)
project(colss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 REQUIRED)

# Better OpenMP handling for macOS
if(APPLE)
    # Check if we're using AppleClang
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        # For AppleClang, we need to use OpenMP from Homebrew
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "omp")
        
        # Try to find Homebrew's libomp
        find_library(OpenMP_omp_LIBRARY
            NAMES omp
            HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib
            PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib
            NO_DEFAULT_PATH
        )
        
        if(OpenMP_omp_LIBRARY)
            message(STATUS "Found OpenMP library: ${OpenMP_omp_LIBRARY}")
            set(OpenMP_CXX_LIBRARIES ${OpenMP_omp_LIBRARY})
            
            # Set include directory
            find_path(OpenMP_CXX_INCLUDE_DIR
                NAMES omp.h
                HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include
                PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include
                NO_DEFAULT_PATH
            )
            if(OpenMP_CXX_INCLUDE_DIR)
                set(OpenMP_CXX_INCLUDE_DIRS ${OpenMP_CXX_INCLUDE_DIR})
            endif()
        endif()
    endif()
endif()

# Add Windows-specific OpenMP SIMD support BEFORE find_package(OpenMP)
if(MSVC)
    # Enable OpenMP with SIMD support for MSVC
    add_compile_options(/openmp:experimental)
    # Optional: Set OpenMP version (if needed)
    # add_compile_options(/openmp:experimental /openmp:llvm)  # For newer MSVC versions
endif()

find_package(OpenMP REQUIRED)

pybind11_add_module(_colss binding.cpp)

# Link OpenMP properly (portable way)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_colss PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE AND OpenMP_omp_LIBRARY)
    # Fallback for older CMake versions or tricky setups
    target_compile_options(_colss PRIVATE -Xpreprocessor -fopenmp)
    target_include_directories(_colss PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(_colss PRIVATE ${OpenMP_omp_LIBRARY})
endif()

# Ensure Windows OpenMP SIMD flags are applied to the target
if(MSVC)
    target_compile_options(_colss PRIVATE /openmp:experimental)
endif()

target_include_directories(_colss PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Compiler-specific optimization (portable)
if(MSVC)
    target_compile_options(_colss PRIVATE /O2 /openmp:experimental)
else()
    target_compile_options(_colss PRIVATE -O3 -march=native -mtune=native)
endif()

install(TARGETS _colss DESTINATION colss)
install(FILES ${CMAKE_SOURCE_DIR}/colss/__init__.py DESTINATION colss)
