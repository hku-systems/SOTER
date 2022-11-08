#!/usr/bin/python3
from config import *
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import random

plt.figure(figsize=(7, 5))

exper = []

dist = open('bleu.txt','r')
content = dist.readlines()
for line in content:
    charac = line.strip()
    exper.append(charac)
print(exper)   

data = [[0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]
]
# Datum_line=Dl=27.3
# MLCapsule=12.7
index = 0
for j in range(6):
    data[1][j] = float(exper[index])
    index = index + 1
MLCapsule=float(exper[index])
index = index + 1
for j in range(6):
    data[2][j] = float(exper[index])
    index = index + 1
Datum_line=Dl=float(exper[index])   

line_label = ["AegisDNN", "SOTER", "MLCapsule"]

itr     = [0, 0.2, 0.4, 0.6, 0.8, 1]

plots_result = []
Line_Color=["#056eee","#fcc006","#6ecb3c","#b16002"]
# add 3 lines of eNNclave AegiesDnn & Soter
for idx, d in enumerate(data):
    if idx != 0:
        r = plt.plot(itr, d, marker=LINE_MARKER[idx], color=Line_Color[idx],linewidth=4, markersize=8,
                 alpha=0.8,mec='black')
        plots_result.append(r[0])
        '''if idx==2:
            Ranfloat=[random.uniform(0.2,0.6) for i in d]
            Ranfloat.sort()
            Ranfloat[0]=0
            Ranfloat[5]=0
            plt.fill_between(itr, d, np.array(d) + np.array(Ranfloat), color=Line_Color[idx], alpha=0.4, linewidth=0)
            plt.fill_between(itr, d, np.array(d) - np.array(Ranfloat), color=Line_Color[idx], alpha=0.4, linewidth=0)
'''
# add i point of MLCapsule
r = plt.plot(0,MLCapsule, marker=LINE_MARKER[idx+1],color=Line_Color[idx+1],linewidth=4, markersize=15,
                 alpha=0.8,mec='black')
plots_result.append(r[0])

# add Datum line 0.745
r = plt.plot([-1,2],[Dl,Dl],color='#ff000d', marker=',',linestyle='dashed',linewidth=4, markersize=8,
                 alpha=0.5,mec='black')
plots_result.append(r[0])
plt.legend(plots_result, line_label, fontsize=LEGEND_SIZE-3, loc='best',ncol=2)
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=TICK_SIZE-2,
           rotation=0)
plt.yticks(fontsize=TICK_SIZE-2)
#plt.yscale("logit")
plt.xlabel("Selective Partition Ratio", fontsize=LABEL_SIZE+2)
plt.ylabel("BLEU score", fontsize=LABEL_SIZE+2)

ax = plt.gca()

ax.grid(linestyle='--',which= 'major',axis='y', zorder=0)
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

plt.ylim((10.0, 28.0))
plt.xlim((-0.1, 1.1))

plt.tight_layout()


#IS_DEBUG = True

if IS_DEBUG:
    plt.show()
else:
    plt.savefig(os.path.basename(__file__).replace("py", "pdf"), format='pdf', dpi=500, bbox_inches='tight')
