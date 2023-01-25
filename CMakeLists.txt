cmake_minimum_required(VERSION 3.24)
project(ramsey CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)

add_executable(ramsey main.cu kernels.cu definitions.cuh utils.cu)

find_package(OpenMP REQUIRED)
target_compile_options(ramsey PRIVATE -Xcompiler -fopenmp)
target_link_libraries(ramsey PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(ramsey PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
