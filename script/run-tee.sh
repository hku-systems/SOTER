echo '** This is TEE client script **'

if [ "$1" == "vggsoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-vgg-partition
    bash runclient.sh
    echo '** VGG-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "vggennclave" ];then
    cd ~/atc22-artifact/SOTER/other-partition/ennclave/graphene-vgg-partition
    bash runclient.sh
    echo '** VGG-ennclave inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "vggag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/vgg-partition/graphene-vgg-partition
    bash runclient.sh
    echo '** VGG-ag inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "mlcapsule" ];then
    cd ~/atc22-artifact/SOTER/shieldmodel/MLCapsule
    bash run.sh
    echo '** MLcapsule inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "alexsoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-alex-partition
    bash runclient.sh
    echo '** Alexnet-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "alexag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/alex-partition/graphene-an-partition
    bash runclient.sh
    echo '** Alexnet-ag inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "ressoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-res-partition
    bash runclient.sh
    echo '** Resnet-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "densesoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-dense-partition
    bash runclient.sh
    echo '** Densenet-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "mlpsoter" ];then
    cd ~/atc22-artifact/SOTER/teertconfig/graphene-mlp-partition
    bash runclient.sh
    echo '** MLP-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 
if [ "$1" == "resag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/res-partition/graphene-res-partition
    bash runclient.sh
    echo '** Resnet-ag inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi
if [ "$1" == "denseag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/dense-partition/graphene-dense-partition
    bash runclient.sh
    echo '** Densenet-ag inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi
if [ "$1" == "mlpag" ];then
    cd ~/atc22-artifact/SOTER/other-partition/aegisdnn/mlp-partition/graphene-mlp-partition
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
    bash runclient.sh
    bash runclient.sh
    bash runclient.sh
    echo '** Transformer-soter inference completed **'
    ssh jianyu@10.22.1.16 "cd ~/atc22-artifact/SOTER/script;bash run-relay-back.sh"
fi 