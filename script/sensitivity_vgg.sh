echo '** This is GPU server script **'
sed -i 's/1/0/g' signal
# model=("0all" "1all" "02-1" "02-3" "02-6" "04-1" "04-3" "04-6" "06-1" "06-3" "06-6" "08-1" "08-3" "08-6" "scp-sens")
# model=("1all" "02-1")
model=($@)
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
    if [ "${model[$i]}" == "02-1" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/02-1/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "02-3" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/02-3/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "02-6" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/02-6/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "04-1" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/04-1/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "04-3" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/04-3/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "04-6" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/04-6/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "06-1" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/06-1/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "06-3" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/06-3/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "06-6" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/06-6/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "08-1" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/08-1/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "08-3" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/08-3/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "08-6" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd ~/atc22-artifact/SOTER/mtr-partition/sens-vgg/08-6/normal/vgg-partition/cpp/soter-graphene-vgg
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
                echo "[After running] Process killed. Exit"
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
    if [ "${model[$i]}" == "scp-sens" ];then
         ssh jianyu@10.22.1.16 "bash ~/atc22-artifact/SOTER/script/run-relay-forw.sh ${model[$i]}"
    fi
done
