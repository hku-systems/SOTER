cp ../CMakeLists.txt ./
sed -i "s/vgg19_80/fixedfp_vgg/g" ./CMakeLists.txt

if [ ! -f "./build" ];then
    rm -rf ./build
fi
if [ ! -f "*.dat" ];then
    rm -rf *.dat
fi
if [ ! -f "*.pdf" ];then
    rm -rf *.pdf
fi

mkdir build
cd build
cmake ..
make
./main 1
cd ..

python3 fixed_dist.py
cp ./figure8b-fixedfp.pdf /home/xian/atc22-artifact/SOTER/figure/

