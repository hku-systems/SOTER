echo '** This is GPU server script **'
sed -i 's/1/0/g' signal
# model=("0all" "1all" "02-1" "02-3" "02-6" "04-1" "04-3" "04-6" "06-1" "06-3" "06-6" "08-1" "08-3" "08-6" "scp-sens")
model=("0all" "1all")
# model=($@)
for ((i = 0 ; i < ${#model[@]} ; i++))
do
    if [ "${model[$i]}" == "0all" ];then
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        ssh xian@202.45.128.183 "cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/0all;bash run.sh"
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
        var=$(head -n +1 signal)
        echo -n "[Before running] signal =  "
        echo $var
        ssh xian@202.45.128.183 "cd ~/atc22-artifact/SOTER/mtr-partition/sens-trans/1all;bash run.sh"
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
    
    # if [ "${model[$i]}" == "scp-sens" ];then
    #      ssh jianyu@10.22.1.16 "bash ~/atc22-artifact/SOTER/script/run-relay-forw.sh ${model[$i]}"
    # fi
done
