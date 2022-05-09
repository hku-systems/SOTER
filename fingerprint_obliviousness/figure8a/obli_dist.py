#%% init
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib import rcParams
import math
import sys
import random

params = {'axes.labelsize': '20',
        'xtick.labelsize': '20',
        'ytick.labelsize': '20',
        # 'lines.linewidth': '0.2',
        'figure.figsize': '5, 3',
        'legend.fontsize': '20'}
rcParams.update(params)

# raw data
mu = 22
variance = 9 
sigma = math.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 50)
x = np.arange(0, 50, 1)

y = [0]*len(x)

# read log from soterfp_vgg
dist = open('l2_dist_obli_fp.dat','r')
content = dist.readlines()
rate = 0

for line in content:
    charac = line.strip()
    for lines in content:
        if charac == lines.strip():
            rate = rate + 1
    y[int(charac)] = rate
    rate = 0

random_y = y

# normalize
sum = np.sum(random_y)
random_y = [i/sum for i in random_y]

# Make the plot
fig, ax = plt.subplots()
ax.bar(x, random_y, label='Random', linewidth='2')

# decorations
plt.grid(True)
plt.ylabel('Proportion (%)')
plt.xlabel('Distance', horizontalalignment="center")
plt.xlim(xmin=0)
# plt.yticks(fontsize=30)
ax.set_yticks(np.arange(0.0, 0.16, 0.04))
ax.set_ylim(0.0)
ax.set_xlim(0,40)
ax.yaxis.get_major_formatter().set_powerlimits((0,1))
plt.legend(fontsize=15)
plt.tight_layout()
# plt.savefig(sys.argv[0][:-2] + "pdf")
plt.savefig("figure8a-oblifp.pdf")
print("Figure8.a is ready!")
