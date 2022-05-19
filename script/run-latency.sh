# clean up old data and old figures
rm -rf *.pdf
rm -rf ./data/tmp

rm -rf ./latency.txt
touch latency.txt
target=./latency.txt
for file in ./data/*
do 
    str=$(grep 'Fetch' $file)
    str_select=$(echo ${str: 28:6})
    str_filter=$(echo $str_select | tr -d "a-zA-Z")
    echo $str_filter >> $target
done
python3 check.py
rm -rf ./latency.txt
touch latency.txt
target=./latency.txt
for file in ./data/*
do 
    str=$(grep 'Fetch' $file)
    str_select=$(echo ${str: 28:6})
    str_filter=$(echo $str_select | tr -d "a-zA-Z")
    echo $str_filter >> $target
done
# produce new experimental figures

python3 normalized_latency.py
mv ./normalized_latency.pdf /home/xian/atc22-artifact/SOTER/figure/
touch ./data/tmp
# rm -rf ./latency.txt

