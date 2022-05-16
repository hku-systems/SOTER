echo '** This is Relay server (for) script **'

ssh ubuntu@192.168.122.158 "source /etc/profile;source ~/.profile;source ~/.bashrc;bash ~/atc22-artifact/SOTER/script/run-tee.sh $1"

# if [ "$1" == "vggsoter" ];then
#     ssh ubuntu@192.168.122.158 "source /etc/profile;source ~/.profile;source ~/.bashrc;bash ~/atc22-artifact/SOTER/script/run-tee.sh $1"
# fi  
# if [ "$1" == "vggennclave" ];then
#     ssh ubuntu@192.168.122.158 "source /etc/profile;source ~/.profile;source ~/.bashrc;bash ~/atc22-artifact/SOTER/script/run-tee.sh $1"
# fi  
# if [ "$1" == "vggag" ];then
#     ssh ubuntu@192.168.122.158 "source /etc/profile;source ~/.profile;source ~/.bashrc;bash ~/atc22-artifact/SOTER/script/run-tee.sh $1"
# fi  