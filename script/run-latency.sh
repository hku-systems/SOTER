# clean up old data and old figures
rm -rf ./latency.txt
rm -rf *.pdf
rm -rf ./data/tmp

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
touch ./data/tmp

