rm -rf cmake
mkdir -p cmake/build
cd cmake/build
cmake -DCMAKE_PREFIX_PATH=$MY_INSTALL_DIR ../..
make -j
./gpu_server