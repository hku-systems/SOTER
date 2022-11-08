cd ~/atc22-artifact/SOTER/figure/
email=$@
echo "Figure5" | mail -s "end2end latency" $email -A normalized_latency.pdf
echo "Figure6a" | mail -s "sensitivity 1" $email -A senvgg.pdf
echo "Figure6b" | mail -s "sensitivity 2" $email -A sentrans.pdf
echo "Figure7a" | mail -s "confidentiality 1" $email -A figure7a-vgg.pdf
echo "Figure7b" | mail -s "confidentiality 2" $email -A figure7b-trans.pdf
echo "Figure8a" | mail -s "fingerprint 1" $email -A figure8a-oblifp.pdf
echo "Figure8b" | mail -s "fingerprint 2" $email -A figure8b-fixedfp.pdf