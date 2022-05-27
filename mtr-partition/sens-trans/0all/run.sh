# if [ ! -f "./build" ];then
#     rm -rf ./build
# fi
# mkdir build
cd build
echo "Start!"
# cmake ..
# make
./main 512 > ./00-1.txt
result=0
tmp=0
for file in ./00-*
do 
    str=$(grep 'consuming' $file)
    str_select=$(echo ${str: 27:8})
    str_filter=$(echo $str_select | tr -d "a-zA-Z")
    tmp=$str_filter
    if (( $(echo "$tmp > $result" |bc -l) )); then
        result=$tmp
    fi
done    
echo "Fetch here. Time consuming: $result ms per inference." > ./00-1.txt
echo "Token 512 complete!"
./main 1024 > ./00-3.txt
echo "Token 1024 complete!"
./main 2048 > ./00-6.txt
echo "Token 2048 complete!"
scp ./00-1.txt xian@202.45.128.185:~/atc22-artifact/SOTER/script/sensitivity-trans/
scp ./00-3.txt xian@202.45.128.185:~/atc22-artifact/SOTER/script/sensitivity-trans/
scp ./00-6.txt xian@202.45.128.185:~/atc22-artifact/SOTER/script/sensitivity-trans/
ssh xian@202.45.128.185 "cd ~/atc22-artifact/SOTER/script;sed -i 's/0/1/g' signal"

