echo '** This is GPU server script **'
sed -i 's/1/0/g' signal
# model=("vggsoter" "vggennclave" "vggag" "mlcapsule" "alexsoter" "alexag" "ressoter" "densesoter" "mlpsoter" "transsoter" "transag" "resag" "denseag" "mlpag" "gpubaseline" "scp")
# model=("alexsoter" "scp")
model=($@)
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
    if [ "${model[$i]}" == "alexsoter" ];then
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
                ps aux | grep gpu_server | awk 'NR==1{print $2}'
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
    if [ "${model[$i]}" == "alexag" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/other-partition/aegisdnn/alex-partition/cpp/ag-graphene-an
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
    if [ "${model[$i]}" == "ressoter" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/mtr-partition/res-partition/cpp/soter-graphene-res
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
    if [ "${model[$i]}" == "densesoter" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/mtr-partition/dense-partition/cpp/soter-graphene-dense
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
    if [ "${model[$i]}" == "mlpsoter" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/mtr-partition/mlp-partition/cpp/soter-graphene-mlp
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
    if [ "${model[$i]}" == "resag" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/other-partition/aegisdnn/res-partition/cpp/ag-graphene-res
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
    if [ "${model[$i]}" == "denseag" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/other-partition/aegisdnn/dense-partition/cpp/ag-graphene-dense
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
    if [ "${model[$i]}" == "mlpag" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/other-partition/aegisdnn/mlp-partition/cpp/ag-graphene-mlp
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
    if [ "${model[$i]}" == "gpubaseline" ];then
        cd /home/xian/atc22-artifact/SOTER/gpubaseline/alex-gpu/cpp/alexgpu/cmake/build
        nohup ./gpu_server &
        sleep 7
        ./tee_client > ~/atc22-artifact/SOTER/script/data/gpualex.txt
        sleep 5
        proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
        kill -9 $proc

        cd /home/xian/atc22-artifact/SOTER/gpubaseline/dense-gpu/cpp/densegpu/cmake/build
        nohup ./gpu_server &
        sleep 7
        ./tee_client > ~/atc22-artifact/SOTER/script/data/gpudense.txt
        sleep 15
        proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
        kill -9 $proc

        cd /home/xian/atc22-artifact/SOTER/gpubaseline/mlp-gpu/cpp/mlpgpu/cmake/build
        nohup ./gpu_server &
        sleep 7
        ./tee_client > ~/atc22-artifact/SOTER/script/data/gpumlp.txt
        sleep 5
        proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
        kill -9 $proc

        cd /home/xian/atc22-artifact/SOTER/gpubaseline/res-gpu/cpp/resnetgpu/cmake/build
        nohup ./gpu_server &
        sleep 12
        ./tee_client > ~/atc22-artifact/SOTER/script/data/gpures.txt
        sleep 60
        proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
        kill -9 $proc

        cd /home/xian/atc22-artifact/SOTER/gpubaseline/vgg-gpu/cpp/vgggpu/cmake/build
        nohup ./gpu_server &
        sleep 7
        ./tee_client > ~/atc22-artifact/SOTER/script/data/gpuvgg.txt
        sleep 8
        proc=$(ps aux | grep gpu_server | awk 'NR==1{print $2}')
        kill -9 $proc
        
        cd /home/xian/atc22-artifact/SOTER/gpubaseline/trans-gpu
        bash run.sh
    fi
    if [ "${model[$i]}" == "scp" ];then
         ssh jianyu@10.22.1.16 "bash ~/atc22-artifact/SOTER/script/run-relay-forw.sh ${model[$i]}"
    fi
    if [ "${model[$i]}" == "transsoter" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/mtr-partition/trans-partition/cpp/soter-graphene-trans
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
    if [ "${model[$i]}" == "transag" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        pushd /home/xian/atc22-artifact/SOTER/other-partition/aegisdnn/trans-partition/cpp/ag-graphene-trans
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
