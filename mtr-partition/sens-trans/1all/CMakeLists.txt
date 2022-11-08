cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

set(Torch_DIR /home/xian/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
