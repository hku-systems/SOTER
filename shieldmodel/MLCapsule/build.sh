#!/bin/bash

mkdir -p build
cd build
cmake ../ && make -j4 && sudo make install
cd ..
make clean
SGX=1 make

