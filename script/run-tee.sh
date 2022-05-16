echo '** This is TEE client script **'

if [ "$1" == "vgg" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-vgg-partition
    bash runclient.sh
    echo '** VGG inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 