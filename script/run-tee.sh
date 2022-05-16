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