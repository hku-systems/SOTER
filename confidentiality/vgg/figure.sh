rm -rf accuracy.txt
rm -rf acc.txt
target=./log
grep 'accuracy' $target > acc.txt

while read line
do 
    # echo $line
    str_select=$(echo ${line: 15:6})
    echo $str_select >> accuracy.txt
done < acc.txt
rm -rf acc.txt
rm -rf log
python3 figure7a-vgg.py
rm -rf /home/xian/atc22-artifact/SOTER/figure/figure7a-vgg.pdf
mv ./figure7a-vgg.pdf /home/xian/atc22-artifact/SOTER/figure/