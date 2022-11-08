rm -rf bleu.txt
touch bleu.txt
target=./bleu.txt

for file in ./*.log
do 
    echo $file
    str=$(tail -n 1 $file)
    str_select=$(echo ${str: 35:5})
    # echo $str_select
    echo $str_select >> $target
done
python3 figure7b-trans.py
rm -rf /home/xian/atc22-artifact/SOTER/figure/figure7b-trans.pdf
rm -rf bleu.txt
rm -rf ./*.log
mv ./figure7b-trans.pdf /home/xian/atc22-artifact/SOTER/figure/