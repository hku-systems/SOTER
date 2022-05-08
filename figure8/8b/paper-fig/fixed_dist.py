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

# raw data
# x = np.arange(0, 40, 4)
# x = [0, 2, 4, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28, 32, 36]
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
y = [0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.5, 1.0, 7.5, 5.8, 1.0, 0.58, 0.18, 0.15, 0.20, 0.38, 0.22, 0.22, 0.11, 0.11, 0.11, 0.11, 0.11, 0.52, 1.75, 0.70, 0.35, 0.15, 0.15, 0.11, 0.11, 0.11, 0.10, 0.08, 0.08, 0.08, 0.00]
# y = [0.0, 0.1, 0.2, 1.1, 2.0, 5.5, 4.8, 2.0, 0.38, 0.22, 0.22, 0.75, 0.20, 0.15, 0.00]
# y = [0.0, 0.1, 0.2, 0.5, 0.7, 4.8, 4.0, 0.8, 0.59, 0.58, 0.25, 0.75, 0.22, 0.20, 0.15, 0.00]
fixed_y = y

# new data
# x_new = np.linspace(min(x), max(x), 25)
# func = interpolate.interp1d(np.array(x), y, kind='linear')
# fixed_y = func(x_new)

# normalize
sum = np.sum(fixed_y)
fixed_y = [i/sum for i in fixed_y]

temp = [i*10 for i in y]
print(temp)
# val = 0
# # 16~29 -> 19 ~32 (25 max)
# for i in range(16,29):
#         val = val + temp[i]

for i in range(6,11):
        print(" #",i, " need: ", temp[i])
print(" #",15, " need: ", temp[15])
for i in range(23,26):
        print(" #",i, " need: ", temp[i])

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
ax.set_xlim(0, 40)
# ax.set_ylim(0.0, 0.11)
ax.yaxis.get_major_formatter().set_powerlimits((0,1))
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(sys.argv[0][:-2] + "pdf")
