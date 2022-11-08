# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/tianxiang/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/tianxiang/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/tianxiang/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/tianxiang/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate fairseq

fairseq-generate data-bin4w/iwslt14.tokenized.de-en-40000/ --path aegisdnn0-4w.pt  --batch-size 128 --beam 6 > aegisdnn0.log
fairseq-generate data-bin5w/iwslt14.tokenized.de-en-50000/ --path aegisdnn20-5w.pt  --batch-size 128 --beam 6 > aegisdnn20.log
fairseq-generate data-bin4w/iwslt14.tokenized.de-en-40000/ --path aegisdnn40-4w.pt  --batch-size 128 --beam 6 > aegisdnn40.log
fairseq-generate data-bin5w/iwslt14.tokenized.de-en-50000/ --path aegisdnn60-5w.pt --batch-size 128 --beam 6 > aegisdnn60.log
fairseq-generate data-bin4w/iwslt14.tokenized.de-en-40000/ --path aegisdnn80-4w.pt  --batch-size 128 --beam 6 > aegisdnn80.log
fairseq-generate data-bin4w/iwslt14.tokenized.de-en-40000/ --path aegisdnn100-4w.pt  --batch-size 128 --beam 6 > aegisdnn9100.log

fairseq-generate data-bin4w/iwslt14.tokenized.de-en-40000/ --path mlcapsule-4w.pt  --batch-size 128 --beam 6 > mlcapsule.log
fairseq-generate data-bin8w/iwslt14.tokenized.de-en-80000/ --path standard-8w.pt  --batch-size 128 --beam 6 > standard.log

fairseq-generate data-bin5w/iwslt14.tokenized.de-en-50000/ --path soter0-5w.pt --batch-size 128 --beam 6 > soter0.log
fairseq-generate data-bin4w/iwslt14.tokenized.de-en-40000/ --path soter20-4w.pt --batch-size 128 --beam 6 > soter20.log
fairseq-generate data-bin5w/iwslt14.tokenized.de-en-50000/ --path soter40-5w.pt --batch-size 128 --beam 6 > soter40.log
fairseq-generate data-bin4w/iwslt14.tokenized.de-en-40000/ --path soter60-4w.pt --batch-size 128 --beam 6 > soter60.log
fairseq-generate data-bin5w/iwslt14.tokenized.de-en-50000/ --path soter80-5w.pt --batch-size 128 --beam 6 > soter80.log
fairseq-generate data-bin4w/iwslt14.tokenized.de-en-40000/ --path soter100-4w.pt --batch-size 128 --beam 6 > soter9100.log