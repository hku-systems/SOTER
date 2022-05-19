from config import *
import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib.lines as mlines

# aegisdnn: alex dense mlp res trans vgg 
# ennclave: vgg
# gpu: alex dense mlp res trans vgg
# mlcapsule: alex dense mlp res trans vgg
# soter: alex dense mlp res trans vgg

exper = []

dist = open('latency.txt','r')
content = dist.readlines()
for line in content:
    charac = line.strip()
    exper.append(charac)
# print(exper)

# plt.figure(figsize=(10, 5.5))
fig=plt.figure(figsize=(7,8.5))

# soter aegisdnn ennclave mlcap 
# vgg alex res dense mlp trans
data = [
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]
]
# vgg: 
try:
    data[0][0] = float(exper[24])/float(exper[12])
except:
    os.system('bash run-gpu.sh vggsoter scp')

try:
    data[0][1] = float(exper[5])/float(exper[12])
except:
    os.system('bash run-gpu.sh vggag scp') 

try:
    data[0][2] = float(exper[6])/float(exper[12])
except:
    os.system('bash run-gpu.sh vggennclave scp') 

try:
    data[0][3] = float(exper[18])/float(exper[12])
except:
    os.system('bash run-gpu.sh mlcapsule scp') 

# model=("vggsoter" "vggennclave" "vggag" "mlcapsule" "alexsoter" "alexag"
# "ressoter" "densesoter" "mlpsoter" "transsoter" "transag" "resag" "denseag"
# "mlpag" "gpubaseline" "scp")

# aegisdnn: alex dense mlp res trans vgg 0 - 5
# ennclave: vgg 6
# gpu: alex dense mlp res trans vgg 7 - 12
# mlcapsule: alex dense mlp res trans vgg 13 - 18
# soter: alex dense mlp res trans vgg 19 - 24

# alex: 
try:
    data[1][0] = float(exper[19])/float(exper[7])
except:
    os.system('bash run-gpu.sh alexsoter scp')
try:
    data[1][1] = float(exper[0])/float(exper[7])
except:
    os.system('bash run-gpu.sh alexag scp')
try:
    data[1][3] = float(exper[13])/float(exper[7])
except:
    os.system('bash run-gpu.sh mlcapsule scp')
         
# res: 
try:
    data[2][0] = float(exper[22])/float(exper[10])
except:
    os.system('bash run-gpu.sh ressoter scp')
try:
    data[2][1] = float(exper[3])/float(exper[10])
except:
    os.system('bash run-gpu.sh resag scp')
try:
    data[2][3] = float(exper[16])/float(exper[10])
except:
    os.system('bash run-gpu.sh mlcapsule scp')

# dense: 
try:
    data[3][0] = float(exper[20])/float(exper[8])
except:
    os.system('bash run-gpu.sh densesoter scp')
try:
    data[3][1] = float(exper[1])/float(exper[8])
except:
    os.system('bash run-gpu.sh denseag scp')
try:
    data[3][3] = float(exper[14])/float(exper[8])
except:
    os.system('bash run-gpu.sh mlcapsule scp')

# mlp: 
try:
    data[4][0] = float(exper[21])/float(exper[9])
except:
    os.system('bash run-gpu.sh mlpsoter scp')
try:
    data[4][1] = float(exper[2])/float(exper[9])
except:
    os.system('bash run-gpu.sh mlpag scp')
try:
    data[4][3] = float(exper[15])/float(exper[9])
except:
    os.system('bash run-gpu.sh mlcapsule scp')

# trans: 
try:
    data[5][0] = float(exper[23])/float(exper[11])
except:
    os.system('bash run-gpu.sh transsoter scp')
try:
    data[5][1] = float(exper[4])/float(exper[11])
except:
    os.system('bash run-gpu.sh transag scp')
try:
    data[5][3] = float(exper[17])/float(exper[11])
except:
    os.system('bash run-gpu.sh mlcapsule scp')


