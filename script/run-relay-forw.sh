echo '** This is Relay server (for) script **'

if [ "$1" == "vgg" ];then
    ssh ubuntu@192.168.122.158 "cd ~/atc22-artifact/SOTER/script;bash run-tee.sh $1"
fi  