# clean up old data and old figures
rm -rf *.pdf
rm -rf ./sensitivity-trans/tmp
rm -rf ./sensitivity-trans/CMakeCache.txt

rm -rf ./latency-sen-trans.txt
touch latency-sen-trans.txt
target=./latency-sen-trans.txt
for file in ./sensitivity-trans/*
do 
    str=$(grep 'Fetch' $file)
    str_select=$(echo ${str: 28:6})
    str_filter=$(echo $str_select | tr -d "a-zA-Z")
    echo $str_filter >> $target
done
python3 sentrans.py
mv ./sentrans.pdf /home/xian/atc22-artifact/SOTER/figure/
touch ./sensitivity-trans/tmp
rm -rf ./latency-sen-trans.txt

