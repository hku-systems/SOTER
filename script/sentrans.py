from config import *
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import random

plt.figure(figsize=(7, 5))
line_label = ["Token size=512", "Token size=1024", "Token size=2048"]

exper = []
dist = open('latency-sen-trans.txt','r')
content = dist.readlines()
for line in content:
    charac = line.strip()
    exper.append(charac)
# print(exper)

data = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
]
for i in range(6):
    for j in range(3):
        data[j][i] = float(exper[j+i*3])

# data = [
#         [724.227, 671.5703, 624.5523, 577.5857, 525.9023, 417.8437],
#         [1373.08, 1157.661, 1035.244, 888.6163, 799.9023, 556.471],
#         [3472.19, 2818.987, 2331.037, 1854.73, 1437.827, 961.5973]
#         ]
# dmin = [
#         [712.367, 615.932, 615.173, 521.126, 511.314, 409.285],
#         [1370.7, 1137.263, 1031.103, 832.637, 783.12, 550.46],
#         [3405.93, 2811.78, 2256.96, 1779.42, 1346.6, 899.597]
#         ]
# dmax = [
#         [731.494, 732.379, 636.884, 618.569, 538.749, 432.926],
#         [1375.46, 1187.73, 1043.108, 932.34, 828.672, 563.921],
#         [3538.45, 2828.58, 2383.06, 1908.95, 1569.86, 1041.12]
#         ]

itr     = [0, 0.2, 0.4, 0.6, 0.8, 1]

plots_result = []
Line_Color=["#056eee","#fcc006","#6ecb3c"]
for idx, d in enumerate(data):
    r = plt.plot(itr, d, marker=LINE_MARKER[idx], color=Line_Color[idx],linewidth=4, markersize=8,
                 alpha=0.8,mec='black')
    plots_result.append(r[0])
    # Ranfloat = [random.uniform(datafloat[idx][0], datafloat[idx][1]) for i in d] #random(10,30) looks better
#     plt.fill_between(itr, d, np.array(dmax[idx]), color=Line_Color[idx], alpha=0.4, linewidth=0)
#     plt.fill_between(itr, d, np.array(dmin[idx]), color=Line_Color[idx], alpha=0.4, linewidth=0)
    # plt.fill_between(itr, plots_low[i], plots[i], color=LINE_COLOR[i],alpha=ALPHA)

plt.legend(plots_result, line_label, fontsize=LEGEND_SIZE-4, loc='upper right')

plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=TICK_SIZE-2,
           rotation=0)
plt.yticks([0,1000,2000,3000,4000],["0","1000","2000","3000","4000"],fontsize=TICK_SIZE-2)

plt.xlabel("Selective Partition ratio", fontsize=LABEL_SIZE )
plt.ylabel("Inference Lantency(ms)", fontsize=LABEL_SIZE )

ax = plt.gca()
ax.grid(linestyle='--',which= 'major',axis='y', zorder=0)
ax.yaxis.set_minor_locator(MultipleLocator(100))

plt.ylim((0, 4000))

plt.tight_layout()

#IS_DEBUG = True

if IS_DEBUG:
    plt.show()
else:
    plt.savefig(os.path.basename(__file__).replace("py", "pdf"), format='pdf', dpi=500)