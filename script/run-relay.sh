echo '** This is Relay server script **'

if [ "$1" == "vgg" ];then
    ssh ubuntu@192.168.122.158 "cd ~/atc22-artifact/SOTER/teertconfig/graphene-vgg-partition;bash runclient.sh"
    echo '** VGG inference completed **'
    ssh xian@202.45.128.185 "cd ~/atc22-artifact/SOTER/script;sed -i 's/0/1/g' signal"
fi