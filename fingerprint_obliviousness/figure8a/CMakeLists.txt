#if (NOT CMAKE_CUDA_COMPILER)
#        set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.6/bin/nvcc")
#endif()

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)
set(Torch_DIR /home/xian/libtorch17/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(main oblifp_vgg.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
