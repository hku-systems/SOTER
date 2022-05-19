cmake_minimum_required(VERSION 3.5.1)

project(MLCapsule C CXX)

# Generated sources
find_package(Torch REQUIRED PATHS "/home/ubuntu/libtorch17/libtorch/share/cmake/Torch")

add_executable(vgg1         "../models/src/vgg19_various_partition/vgg19-1.cpp")
add_executable(vgg3         "../models/src/vgg19_various_partition/vgg19-3.cpp")
add_executable(vgg6         "../models/src/vgg19_various_partition/vgg19-6.cpp")

foreach(_target
  vgg1 vgg3 vgg6)
  target_link_libraries(${_target} "${TORCH_LIBRARIES}")
  set_property(TARGET ${_target}  PROPERTY CXX_STANDARD 14)
  set_target_properties(${_target} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
  install(TARGETS ${_target}
          DESTINATION /bin/)  
endforeach()

