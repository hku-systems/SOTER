if [ ! -f "./build" ];then
    rm -rf ./build
fi
mkdir build
cd build
cmake ..
make
./main 1 > ../1.txt
./main 3 > ../3.txt
./main 6 > ../6.txt