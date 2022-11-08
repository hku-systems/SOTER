ssh tianxiang@202.45.128.181 "source /etc/profile;source ~/.profile;source ~/.bashrc;cd ~/soter-graphene/figure7/7a;echo 'VGG stealing experiment starts!';bash run.sh;scp ./log xian@202.45.128.185:/home/xian/atc22-artifact/SOTER/confidentiality/vgg/"
pushd /home/xian/atc22-artifact/SOTER/confidentiality/vgg
bash figure.sh
popd
sleep 5s
ssh tianxiang@202.45.128.187 "source /etc/profile;source ~/.profile;source ~/.bashrc;cd /home/tianxiang/fairseq/result;echo 'Transformer stealing experiment starts!';bash run.sh;scp *.log xian@202.45.128.185:/home/xian/atc22-artifact/SOTER/confidentiality/transformer/"
pushd /home/xian/atc22-artifact/SOTER/confidentiality/transformer
bash figure.sh
popd