bash sensitivity_vgg.sh 0all 1all
bash sensitivity_vgg.sh 02-1 02-3 02-6 04-1 04-3 04-6 06-1 06-3 06-6 08-1 08-3 08-6 scp-sens
echo "Prepare for drawing sensitivity figures ..."
sleep 5s
bash run-sensitivity.sh