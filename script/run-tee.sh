echo '** This is TEE client script **'

if [ "$1" == "vggsoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "vggennclave" ];then
    cd ~/atc22-artifact/SOTER/other-partition/ennclave/graphene-vgg-partition
    echo '** VGG-ennclave inference started **'
    bash runclient.sh
    echo '** VGG-ennclave inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "vggag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/vgg-partition/graphene-vgg-partition
    echo '** VGG-ag inference started **'
    bash runclient.sh
    echo '** VGG-ag inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "mlcapsule" ];then
    cd ~/atc22-artifact/SOTER/shieldmodel/MLCapsule
    echo '** MLcapsule inference started **'
    bash run.sh
    echo '** MLcapsule inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "alexsoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-alex-partition
    echo '** Alexnet-soter inference started **'
    bash runclient.sh
    echo '** Alexnet-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "alexag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/alex-partition/graphene-an-partition
    echo '** Alexnet-ag inference started **'
    bash runclient.sh
    echo '** Alexnet-ag inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "ressoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-res-partition
    echo '** Resnet-soter inference started **'
    bash runclient.sh
    echo '** Resnet-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "densesoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-dense-partition
    echo '** Densenet-soter inference started **'
    bash runclient.sh
    echo '** Densenet-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "mlpsoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-mlp-partition
    echo '** MLP-soter inference started **'
    bash runclient.sh
    echo '** MLP-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "resag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/res-partition/graphene-res-partition
    echo '** Resnet-ag inference started **'
    bash runclient.sh
    echo '** Resnet-ag inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi
if [ "$1" == "denseag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/dense-partition/graphene-dense-partition
    echo '** Densenet-ag inference started **'
    bash runclient.sh
    echo '** Densenet-ag inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi
if [ "$1" == "mlpag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/mlp-partition/graphene-mlp-partition
    echo '** MLP-ag inference started **'
    bash runclient.sh
    echo '** MLP-ag inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi
if [ "$1" == "scp" ];then
    cd ~/atc22-artifact/SOTER/script/data
    scp ./* jianyu@10.22.1.16:/home/jianyu/atc22-artifact/SOTER/script/data
    echo '** Send back experimental results to 185 **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script/data;scp ./* xian@202.45.128.185:~/atc22-artifact/SOTER/script/data"
fi
if [ "$1" == "transsoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "transag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/trans-partition/graphene-trans-partition
    echo '** Transformer-ag inference started **'
    bash runclient.sh
    echo '** Transformer-ag inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi
if [ "$1" == "02-1" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/02-1/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "02-3" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/02-3/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "02-6" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/02-6/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "04-1" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/04-1/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "04-3" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/04-3/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "04-6" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/04-6/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "06-1" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/06-1/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "06-3" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/06-3/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "06-6" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/06-6/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "08-1" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/08-1/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "08-3" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/08-3/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "08-6" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/08-6/normal/vgg-partition/graphene-vgg-partition
    echo '** VGG-soter inference started **'
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "scp-sens" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg
    scp ./*.txt jianyu@10.22.1.16:/home/jianyu/atc22-artifact/SOTER/script/sensitivity
    echo '** Send back experimental results to 185 **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script/sensitivity;scp ./* xian@202.45.128.185:~/atc22-artifact/SOTER/script/sensitivity"
fi
if [ "$1" == "02-1-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/02-1/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "02-3-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/02-3/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "02-6-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/02-6/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "04-1-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/04-1/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "04-3-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/04-3/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "04-6-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/04-6/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "06-1-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/06-1/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "06-3-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/06-3/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "06-6-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/06-6/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "08-1-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/08-1/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "08-3-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/08-3/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "08-6-trans" ];then
    cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/08-6/normal/trans-partition/graphene-trans-partition
    echo '** Transformer-soter inference started **'
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "scp-sens-trans" ];then
    cd ~/atc22-artifact/SOTER/script/sensitivity-trans
    scp ./*.txt jianyu@10.22.1.16:/home/jianyu/atc22-artifact/SOTER/script/sensitivity-trans
    echo '** Send back experimental results to 185 **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script/sensitivity-trans;scp ./* xian@202.45.128.185:~/atc22-artifact/SOTER/script/sensitivity-trans"
fi