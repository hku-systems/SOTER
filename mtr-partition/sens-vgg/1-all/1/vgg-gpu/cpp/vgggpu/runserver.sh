rm -rf cmake
mkdir -p cmake/build
cd cmake/build
cmake -DCMAKE_PREFIX_PATH=$MY_INSTALL_DIR ../..
make -j
nohup ./gpu_server &
# ./gpu_server