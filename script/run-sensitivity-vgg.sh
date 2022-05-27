# clean up old data and old figures
rm -rf *.pdf
rm -rf ./sensitivity/tmp

rm -rf ./latency-sen.txt
touch latency-sen.txt
target=./latency-sen.txt
for file in ./sensitivity/*
do 
    str=$(grep 'Fetch' $file)
    str_select=$(echo ${str: 28:6})
    str_filter=$(echo $str_select | tr -d "a-zA-Z")
    echo $str_filter >> $target
done
python3 senvgg.py
mv ./senvgg.pdf /home/xian/atc22-artifact/SOTER/figure/
touch ./sensitivity/tmp
rm -rf ./latency-sen.txt

