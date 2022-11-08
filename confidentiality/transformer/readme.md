# Steps for training

### Directly test the BLEU score after stealing attacks

```shell
conda activate fairseq
cd ~/fairseq/result

TEST_DATA=data-bin4w/iwslt14.tokenized.de-en-40000/
#TEST_DATA=data-bin5w/iwslt14.tokenized.de-en-50000/
#MODEL=[soter/aegisdnn/mlcapsule][p]-[4w/5w].pt
MODEL=soter100-4w.pt

fairseq-generate $TEST_DATA --path $MODEL  --batch-size 128 --beam 6 > test.log
```

Then, in the end of test.log, you can get the bleu with the format of:

```log
**ATTENTION: THE SIZE OF TEST_DATA NEED TO MATCH THE SIZE OF *.PT **

Generate test with beam=5: BLEU4 = 9.27, 42.6/15.6/6.9/3.2 (BP=0.844, ratio=0.855, syslen=129139, reflen=150967)
```



### Train each data point by yourself

```shell
conda activate fairseq
DATA_PATH=iwslt14.tokenized.de-en-40000
DATA_PATH2=data-bin4w/iwslt14.tokenized.de-en-40000
SAVE_PATH=checkpoints/aegisdnn-30
TEXT=~/fairseq/examples/translation/$DATA_PATH
mkdir -p $SAVE_PATH
```



#### First, you need the dataset

1. ```shell
   cd ~/fairseq/examples/translation
   vim pre-iwslt14-2.sh
   (change preg, change 【awk '{if (NR%23 == 0 && NR<40000)】)
   bash pre-iwslt14-2.sh
   ```

2. ```shell
   cd ~/fairseq
   fairseq-preprocess --source-lang de --target-lang en \
       --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
       --destdir $DATA_PATH2
   ```



#### Second, train the data

```shell
CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_PATH2 \
    --optimizer nag --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir $SAVE_PATH --max-epoch 30
```

This is for standard line and MLCapsule

```shell
CUDA_VISIBLE_DEVICES=0 python train_my.py $DATA_PATH2 \
    --optimizer nag --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir $SAVE_PATH --max-epoch 30
```

This is for AegisDNN and SOTER. For AegisDNN & SOTER, you need to config ``train_my.py``

```python
if __name__ == "__main__":
    cli_main(filename="standard-5w.pt",partition=0.6,morph_n=0.8)
    # 0 <= partition <= 1, for example, 0.0 ,0.2 ,0.4 ,0.6 ,0.8 ,1.0
    # morph_n < 1, like 0.8, 0.5
```

After training, you can get  **\*.pt** in $SAVE_PATH

#### Last, test the result

```shell
fairseq-generate $DATA_PATH2 \
    --path $SAVE_PATH/checkpoint_best.pt \
    --batch-size 128 --beam 5
```





