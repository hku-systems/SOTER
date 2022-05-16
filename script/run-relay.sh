echo '** This is Relay server script **'

if [ "$1" == "vgg" ];then
    ssh xian@202.45.128.185 "cd ~/atc22-artifact/SOTER/script;sed -i 's/0/1/g' signal"
fi