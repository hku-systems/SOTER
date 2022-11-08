#%% init
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy import interpolate
from matplotlib import rcParams
import sys

params = {'axes.labelsize': '20',
        'xtick.labelsize': '20',
        'ytick.labelsize': '20',
        # 'lines.linewidth': '0.2',
        'figure.figsize': '5, 3',
        'legend.fontsize': '20'}
rcParams.update(params)

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

y = [1]*len(x)

# read log from fixedfp_vgg
dist = open('l2_dist_fixed_fp.dat','r')
content = dist.readlines()
rate = 0

for line in content:
    charac = line.strip()
    for lines in content:
        if charac == lines.strip():
            rate = rate + 1
    y[int(charac)] = rate
    rate = 0

fixed_y = y

# normalize
sum = np.sum(fixed_y)
fixed_y = [i/sum for i in fixed_y]

# Make the plot
fig, ax = plt.subplots()
# ax.plot(x_new, fixed_y, label='fixed', linewidth=2)
ax.bar(x, fixed_y, label='Fixed', linewidth=2)

# decorations
plt.grid(True)
plt.ylabel('Proportion (%)')
plt.xlabel('Distance', horizontalalignment="center")
plt.xlim(xmin=0)
ax.set_yticks(np.arange(0.0, 0.40, 0.1))
# ax.set_xticks(np.arange(0, 40, 5))
ax.set_xlim(0,40)
# ax.set_ylim(0.0, 0.11)
ax.yaxis.get_major_formatter().set_powerlimits((0,1))
plt.legend(fontsize=15)
plt.tight_layout()
# plt.savefig(sys.argv[0][:-2] + "pdf")
plt.savefig("figure8b-fixedfp.pdf")
print("Figure8.b is ready!")