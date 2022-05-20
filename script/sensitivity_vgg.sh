echo '** This is GPU server script **'
sed -i 's/1/0/g' signal
# model=("0all" "1all")
model=("1all")
# model=($@)
for ((i = 0 ; i < ${#model[@]} ; i++))
do
    if [ "${model[$i]}" == "0all" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        ssh xian@202.45.128.183 "cd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/0-all;bash run.sh"
        while :
        do
            sleep 1s
            var=$(head -n +1 signal)
            if [ "$var" == "1" ]
            then
                echo "[After running] Signal reset to 1 by relay script"
                sed -i 's/1/0/g' signal
                var=$(head -n +1 signal)
                if [ "$var" == "0" ];then
                    echo "[After running] Signal reset to 0 by gpu script"  
                fi
                break
            else 
                echo "[After running] Signal = 0 "  
            fi
        done
    fi
    if [ "${model[$i]}" == "1all" ];then
        cd /home/xian/atc22-artifact/SOTER/mtr-partition/sens-vgg/1-all/1/vgg-gpu/cpp/vgggpu/cmake/build
        nohup ./gpu_server &
        sleep 7
        ./tee_client > ~/atc22-artifact/SOTER/script/sensitivity/1-1.txt
        sleep 5
        proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
        kill -9 $proc

        cd /home/xian/atc22-artifact/SOTER/mtr-partition/sens-vgg/1-all/3/vgg-gpu/cpp/vgggpu/cmake/build
        nohup ./gpu_server &
        sleep 7
        ./tee_client > ~/atc22-artifact/SOTER/script/sensitivity/1-3.txt
        sleep 5
        proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
        kill -9 $proc

        cd /home/xian/atc22-artifact/SOTER/mtr-partition/sens-vgg/1-all/6/vgg-gpu/cpp/vgggpu/cmake/build
        nohup ./gpu_server &
        sleep 7
        ./tee_client > ~/atc22-artifact/SOTER/script/sensitivity/1-6.txt
        sleep 5
        proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
        kill -9 $proc
    fi
    
done
