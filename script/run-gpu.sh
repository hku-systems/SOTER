echo '** This is GPU server script **'
sed -i 's/1/0/g' signal
# model=("vggsoter" "vggennclave" "vggag")
model=("vggalex")
for ((i = 0 ; i < ${#model[@]} ; i++))
do
    if [ "${model[$i]}" == "vggsoter" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/mtr-partition/vgg-partition/cpp/soter-graphene-vgg
        bash runserver.sh
        popd
        ssh jianyu@10.22.1.16 "bash ~/atc22-artifact/SOTER/script/run-relay-forw.sh ${model[$i]}"
        while :
        do
            sleep 1s
            var=$(head -n +1 signal)
            if [ "$var" == "1" ]
            then
                echo "[After running] Signal reset to 1 by relay script"
                proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
                kill -9 $proc  
                echo "[After running] VGG completed. Process killed. Exit"
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
    if [ "${model[$i]}" == "vggennclave" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/other-partition/ennclave/vgg-partition/cpp/ennclave-vgg
        bash runserver.sh
        popd
        ssh jianyu@10.22.1.16 "bash ~/atc22-artifact/SOTER/script/run-relay-forw.sh ${model[$i]}"
        while :
        do
            sleep 1s
            var=$(head -n +1 signal)
            if [ "$var" == "1" ]
            then
                echo "[After running] Signal reset to 1 by relay script"
                proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
                kill -9 $proc  
                echo "[After running] VGG completed. Process killed. Exit"
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
    if [ "${model[$i]}" == "vggag" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/other-partition/aegisdnn/vgg-partition/cpp/ag-graphene-vgg
        bash runserver.sh
        popd
        ssh jianyu@10.22.1.16 "bash ~/atc22-artifact/SOTER/script/run-relay-forw.sh ${model[$i]}"
        while :
        do
            sleep 1s
            var=$(head -n +1 signal)
            if [ "$var" == "1" ]
            then
                echo "[After running] Signal reset to 1 by relay script"
                proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
                kill -9 $proc  
                echo "[After running] VGG completed. Process killed. Exit"
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
    if [ "${model[$i]}" == "mlcapsule" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        ssh jianyu@10.22.1.16 "bash ~/atc22-artifact/SOTER/script/run-relay-forw.sh ${model[$i]}"
        while :
        do
            sleep 1s
            var=$(head -n +1 signal)
            if [ "$var" == "1" ]
            then
                echo "[After running] Signal reset to 1 by relay script"
                echo "[After running] VGG completed. Process killed. Exit"
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
    if [ "${model[$i]}" == "vggalex" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/mtr-partition/alex-partition/cpp/soter-graphene-an
        bash runserver.sh
        popd
        ssh jianyu@10.22.1.16 "bash ~/atc22-artifact/SOTER/script/run-relay-forw.sh ${model[$i]}"
        while :
        do
            sleep 1s
            var=$(head -n +1 signal)
            if [ "$var" == "1" ]
            then
                echo "[After running] Signal reset to 1 by relay script"
                proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
                kill -9 $proc  
                echo "[After running] VGG completed. Process killed. Exit"
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
done

