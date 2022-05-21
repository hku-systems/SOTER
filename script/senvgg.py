from config import *
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import random 

plt.figure(figsize=(7, 5))

exper = []

dist = open('latency-sen.txt','r')
content = dist.readlines()
for line in content:
    charac = line.strip()
    exper.append(charac)
# print(exper)

data = [
        [0, 0, 0, 0, 0,0],
        [0, 0, 0, 0, 0,0],
        [0, 0, 0, 0, 0,0]
]
for i in range(6):
    for j in range(3):
        data[j][i] = float(exper[j+i*3])
        # print(data[j][i])

datafloat=[[5,10],[10,15],[15,30]]
line_label = ["Shape=(1,3,224,224)", "Shape=(3,3,224,224)", "Shape=(6,3,224,224)"]

# data = [
#         [113.0602, 107.4521, 88.3792, 64.62557, 38.7988, 9.778759],
#         [228.3892, 234.7295, 201.2797, 152.5176, 105.8668, 44.16293],
#         [443.2227, 460.8369, 399.0928, 295.1734, 234.7528, 76.93889]
# ]

# dmin = [
#         [105.865, 88.621, 67.5489, 28.3801, 15.1391, 9.52598],
#         [217.707, 200.854, 176.262, 68.9398, 47.3784, 40.1063],
#         [424.047, 385.868, 327.181, 145.859, 92.5988, 71.6699]
# ]

# dmax = [
#         [118.184, 123.399, 104.735, 89.4268, 60.1726, 10.4826],
#         [242.37, 290.651, 236.728, 202.137, 146.815, 47.0459],
#         [467.646, 599.728, 452.02, 408.094, 330.659, 80.595]
# ]

itr     = [0, 0.2, 0.4, 0.6, 0.8, 1]

plots_result = []
Line_Color=["#056eee","#fcc006","#6ecb3c"]
for idx, d in enumerate(data):
    r = plt.plot(itr, d, marker=LINE_MARKER[idx], color=Line_Color[idx],linewidth=4, markersize=8,
                 alpha=0.8,mec='black')
    plots_result.append(r[0])
    Ranfloat = [random.uniform(datafloat[idx][0], datafloat[idx][1]) for i in d] #random(10,30) looks better
#     plt.fill_between(itr, d, np.array(dmax[idx]), color=Line_Color[idx], alpha=0.4, linewidth=0)
#     plt.fill_between(itr, d, np.array(dmin[idx]), color=Line_Color[idx], alpha=0.4, linewidth=0)
    # plt.fill_between(itr, plots_low[i], plots[i], color=LINE_COLOR[i],alpha=ALPHA)

plt.legend(plots_result, line_label, fontsize=LEGEND_SIZE-4, loc='upper right')

plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=TICK_SIZE-2,
           rotation=0)
plt.yticks([0,150,300,450,600],["0","150","300","450","600"],fontsize=TICK_SIZE-2)

plt.xlabel("Selective Partition ratio", fontsize=LABEL_SIZE )
plt.ylabel("Inference Lantency(ms)", fontsize=LABEL_SIZE )

ax = plt.gca()
ax.grid(linestyle='--',which= 'major',axis='y', zorder=0)
ax.yaxis.set_minor_locator(MultipleLocator(10))

plt.ylim((0, 600))

plt.tight_layout()

#IS_DEBUG = True

if IS_DEBUG:
    plt.show()
else:
    plt.savefig(os.path.basename(__file__).replace("py", "pdf"), format='pdf', dpi=500)
#     plt.savefig(os.path.basename(__file__).replace("py", "png"), format='png', dpi=500)