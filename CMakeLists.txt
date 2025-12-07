cmake_minimum_required(VERSION 3.12)
project(ImageProcessing VERSION 1.0 LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_OPENMP "Build OpenMP parallel implementations" ON)
option(BUILD_MPI "Build MPI parallel implementations" OFF)
option(BUILD_CUDA "Build CUDA parallel implementations" OFF)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmark tools" ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find OpenCV (required)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files for core library
set(CORE_SOURCES
    src/image_filters.cpp
)

# Core library (serial implementation)
add_library(iproc_core STATIC ${CORE_SOURCES})
target_link_libraries(iproc_core ${OpenCV_LIBS})
target_include_directories(iproc_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Serial demo executable
add_executable(iproc_demo src/main.cpp)
target_link_libraries(iproc_demo iproc_core ${OpenCV_LIBS})

# OpenMP implementation
if(BUILD_OPENMP)
    find_package(OpenMP REQUIRED)
    
    add_library(iproc_omp STATIC src/image_filters_omp.cpp)
    target_link_libraries(iproc_omp iproc_core OpenMP::OpenMP_CXX ${OpenCV_LIBS})
    target_include_directories(iproc_omp PUBLIC ${CMAKE_SOURCE_DIR}/include)
    
    add_executable(iproc_omp_demo src/cli_omp.cpp)
    target_link_libraries(iproc_omp_demo iproc_omp iproc_core OpenMP::OpenMP_CXX ${OpenCV_LIBS})
endif()

# MPI implementation
if(BUILD_MPI)
    find_package(MPI REQUIRED)
    
    add_library(iproc_mpi STATIC src/image_filters_mpi.cpp)
    target_link_libraries(iproc_mpi iproc_core MPI::MPI_CXX ${OpenCV_LIBS})
    target_include_directories(iproc_mpi PUBLIC ${CMAKE_SOURCE_DIR}/include)
    
    add_executable(iproc_mpi_demo src/cli_mpi.cpp)
    target_link_libraries(iproc_mpi_demo iproc_mpi iproc_core MPI::MPI_CXX ${OpenCV_LIBS})
endif()

# CUDA implementation
if(BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    set(CUDA_SOURCES
        cuda/kernels.cu
        cuda/cuda_wrapper.cpp
    )
    
    add_library(iproc_cuda STATIC ${CUDA_SOURCES})
    set_target_properties(iproc_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(iproc_cuda iproc_core CUDA::cudart ${OpenCV_LIBS})
    target_include_directories(iproc_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include)
    
    add_executable(iproc_cuda_demo src/cli_cuda.cpp)
    target_link_libraries(iproc_cuda_demo iproc_cuda iproc_core CUDA::cudart ${OpenCV_LIBS})
endif()

# Unit tests
if(BUILD_TESTS)
    # Download and setup Catch2
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.0
    )
    FetchContent_MakeAvailable(Catch2)
    
    enable_testing()
    
    # Test executable
    add_executable(unit_tests
        tests/test_main.cpp
        tests/test_io.cpp
        tests/test_point_ops.cpp
        tests/test_noise.cpp
        tests/test_smoothing.cpp
        tests/test_edge_detection.cpp
        tests/test_morphology.cpp
        tests/test_geometric.cpp
        tests/test_color.cpp
        tests/test_bonus.cpp
    )
    target_link_libraries(unit_tests PRIVATE iproc_core Catch2::Catch2WithMain ${OpenCV_LIBS})
    
    if(BUILD_OPENMP)
        target_link_libraries(unit_tests PRIVATE iproc_omp OpenMP::OpenMP_CXX)
    endif()
    
    # Register tests
    add_test(NAME UnitTests COMMAND unit_tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_executable(benchmark_serial tools/benchmark/benchmark_serial.cpp)
    target_link_libraries(benchmark_serial iproc_core ${OpenCV_LIBS})
    
    if(BUILD_OPENMP)
        add_executable(benchmark_openmp tools/benchmark/benchmark_openmp.cpp)
        target_link_libraries(benchmark_openmp iproc_omp iproc_core OpenMP::OpenMP_CXX ${OpenCV_LIBS})
    endif()
    
    if(BUILD_MPI)
        add_executable(benchmark_mpi tools/benchmark/benchmark_mpi.cpp)
        target_link_libraries(benchmark_mpi iproc_mpi iproc_core MPI::MPI_CXX ${OpenCV_LIBS})
    endif()
    
    if(BUILD_CUDA)
        add_executable(benchmark_cuda tools/benchmark/benchmark_cuda.cpp)
        target_link_libraries(benchmark_cuda iproc_cuda iproc_core CUDA::cudart ${OpenCV_LIBS})
    endif()
endif()

# Installation rules
install(TARGETS iproc_demo DESTINATION bin)
if(BUILD_OPENMP)
    install(TARGETS iproc_omp_demo DESTINATION bin)
endif()
if(BUILD_MPI)
    install(TARGETS iproc_mpi_demo DESTINATION bin)
endif()
if(BUILD_CUDA)
    install(TARGETS iproc_cuda_demo DESTINATION bin)
endif()
