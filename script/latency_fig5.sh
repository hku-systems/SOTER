# compile, build, and run the data
bash run-gpu.sh vggsoter vggennclave vggag mlcapsule alexsoter alexag ressoter densesoter mlpsoter transsoter transag resag denseag mlpag gpubaseline scp
echo "Prepare for drawing latency figures ..."
sleep 5s
# draw figures
bash run-latency.sh
