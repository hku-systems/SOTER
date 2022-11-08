if [ ! -f "./build" ];then
    rm -rf ./build
fi
mkdir build
cd build
cmake ..
make
./main 1 > ./00-1.txt
./main 3 > ./00-3.txt
./main 6 > ./00-6.txt
scp ./00-* xian@202.45.128.185:~/atc22-artifact/SOTER/script/sensitivity/
ssh xian@202.45.128.185 "cd ~/atc22-artifact/SOTER/script;sed -i 's/0/1/g' signal"