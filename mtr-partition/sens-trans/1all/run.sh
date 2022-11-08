if [ ! -f "./build" ];then
    rm -rf ./build
fi
mkdir build
cd build
cmake ..
make
./main 512 > ./1-1.txt
echo "Token 512 complete!"
./main 1024 > ./1-3.txt
echo "Token 1024 complete!"
./main 2048 > ./1-6.txt
echo "Token 2048 complete!"
scp ./1-* xian@202.45.128.185:~/atc22-artifact/SOTER/script/sensitivity-trans/
ssh xian@202.45.128.185 "cd ~/atc22-artifact/SOTER/script;sed -i 's/0/1/g' signal"