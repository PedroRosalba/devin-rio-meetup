cmake_minimum_required(VERSION 3.18)
project(gpt2_cuda_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GPT2_BUILD_CUDA "Build NVIDIA CUDA kernels" ON)

add_library(gpt2_host STATIC src/weights.cpp)
target_include_directories(gpt2_host PUBLIC include)

find_package(CUDAToolkit)
if(GPT2_BUILD_CUDA AND CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  add_library(gpt2_cuda_kernels STATIC
    cuda/vecadd.cu
    cuda/gemm_naive.cu
  )
  target_link_libraries(gpt2_cuda_kernels PUBLIC CUDA::cudart)
  target_include_directories(gpt2_cuda_kernels PUBLIC include)

  add_executable(gpt2_cuda_cpp src/main.cpp)
  target_compile_definitions(gpt2_cuda_cpp PRIVATE GPT2_HAS_CUDA=1)
  target_link_libraries(gpt2_cuda_cpp PRIVATE gpt2_host gpt2_cuda_kernels CUDA::cudart)
else()
  message(STATUS "CUDA not found — building host-only gpt2_cuda_cpp (weight load + manifest check)")
  add_executable(gpt2_cuda_cpp src/main.cpp)
  target_link_libraries(gpt2_cuda_cpp PRIVATE gpt2_host)
endif()
