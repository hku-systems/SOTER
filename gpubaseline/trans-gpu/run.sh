if [ ! -f "./build" ];then
    rm -rf ./build
fi
mkdir build
cd build
cmake ..
make
./main 1024 > ~/atc22-artifact/SOTER/script/data/gputrans.txt