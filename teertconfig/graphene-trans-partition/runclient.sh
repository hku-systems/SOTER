pushd ../../mtr-partition/trans-partition/cpp/soter-graphene-trans/
sed -i "s/xian/ubuntu/g" ./CMakeLists.txt

rm -rf cmake
mkdir -p cmake/build
cd cmake/build
cmake -DCMAKE_PREFIX_PATH=$MY_INSTALL_DIR ../..
make -j

sudo rm -rf /bin/tee_client
sudo cp ./tee_client /bin/tee_client

popd

make clean
SGX=1 make
graphene-sgx ./tee_client --target=10.22.1.26:50051 > ~/atc22-artifact/SOTER/script/data/sotertrans.txt 