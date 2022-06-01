# Experiment 1
kill -9 $(ps aux | grep gpu_server | awk '{print $2}')
bash ./latency_fig5.sh
# Experiment 2
sleep 5s
kill -9 $(ps aux | grep gpu_server | awk '{print $2}')
bash ./sensitivity_fig6.sh
# Experiment 3
sleep 5s
kill -9 $(ps aux | grep gpu_server | awk '{print $2}')
bash ./conf.sh
# Experiment 4
sleep 5s 
kill -9 $(ps aux | grep gpu_server | awk '{print $2}')
bash ./run-fpcheck.sh
sleep 5s 
kill -9 $(ps aux | grep gpu_server | awk '{print $2}')
echo "Completed Successfully!"