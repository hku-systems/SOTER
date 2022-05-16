echo '** This is GPU server script **'
sed -i 's/1/0/g' signal
# model=("vgg" "alex" "res" "dense" "mlp")
model=("vgg")

for ((i = 0 ; i < ${#model[@]} ; i++))
do
    echo -n "Run model: " 
    echo ${model[$i]}

    if [ "${model[$i]}" == "vgg" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var

        pushd /home/xian/atc22-artifact/SOTER/mtr-partition/vgg-partition/cpp/soter-graphene-vgg
        bash runserver.sh
        popd

        ssh jianyu@10.22.1.16 "bash ~/atc22-artifact/SOTER/script/run-relay.sh ${model[$i]}"

        while :
        do
            sleep 1s
            var=$(head -n +1 signal)
            if [ "$var" == "1" ]
            then
                echo "[After running] Signal = 1 by relay script"  
                sed -i 's/1/0/g' signal
                var=$(head -n +1 signal)
                if [ "$var" == "0" ];then
                    echo "[After running] Signal = 0 by gpu script"  
                fi
                break
            else 
                echo "[After running] Signal = 0 "  
            fi
        done
    fi

    
done

