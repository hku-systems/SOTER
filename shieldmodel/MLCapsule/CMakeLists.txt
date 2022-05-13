cmake_minimum_required(VERSION 3.5.1)

project(MLCapsule C CXX)

# Generated sources
find_package(Torch REQUIRED PATHS "/home/ubuntu/libtorch17/libtorch/share/cmake/Torch")

add_executable(vgg          "../models/src/vgg19_various_partition/vgg19.cpp")
add_executable(alexnet      "../models/src/alexnet/alexnet.cpp")
add_executable(resnet       "../models/src/resnet152/res152.cpp")
add_executable(densenet     "../models/src/densenet121/densenet121.cpp")
add_executable(mlp          "../models/src/mlp/mlp.cpp")
add_executable(transformer  "../models/src/transformer_verioua_partion/tf.cpp")

foreach(_target
  vgg alexnet resnet densenet mlp transformer)
  target_link_libraries(${_target} "${TORCH_LIBRARIES}")
  set_property(TARGET ${_target}  PROPERTY CXX_STANDARD 14)
  set_target_properties(${_target} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
  install(TARGETS ${_target}
          DESTINATION /bin/)  
endforeach()

