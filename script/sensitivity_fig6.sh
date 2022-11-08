bash sensitivity_vgg.sh 0all 1all
bash sensitivity_vgg.sh 02-1 02-3 02-6 04-1 04-3 04-6 06-1 06-3 06-6 08-1 08-3 08-6 scp-sens
echo "Prepare for drawing sensitivity vgg figure ..."
sleep 5s
bash run-sensitivity-vgg.sh
sleep 5s
bash sensitivity_trans.sh 0all 1all
bash sensitivity_trans.sh 02-1-trans 02-3-trans 02-6-trans 04-1-trans 04-3-trans 04-6-trans 06-1-trans 06-3-trans 06-6-trans 08-1-trans 08-3-trans 08-6-trans scp-sens-trans
echo "Prepare for drawing sensitivity transformer figure ..."
sleep 5s
bash run-sensitivity-trans.sh
echo "Completed!"