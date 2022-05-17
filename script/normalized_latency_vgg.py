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
fig=plt.figure(figsize=(7,8.5))
# soter aegisdnn ennclave mlcap 
# vgg alex res dense mlp trans
data = [[0,0,0,0]]
# vgg: 
data[0][0] = float(exper[24])/float(exper[12])
data[0][1] = float(exper[5])/float(exper[12])
data[0][2] = float(exper[6])/float(exper[12])
data[0][3] = float(exper[18])/float(exper[12])
soter = [round(float(exper[24]),2)]
myticks = [
    [0,10,20,40]
]
myticks_labels = [
    ['0x','10x','20x','40x']
]
bar_label = ['SOTER','AegisDNN', 'eNNclave', 'MLCapsule' ]
bar_width = 0.57
program_label = ['VGG19']

for index in range(len(data)):
    fig=plt.subplot(3,2,index+1)
    xx = np.array([i for i in range(len(data[index]))])
    plots = []
    for idx, d in enumerate(data[index]):
        left=idx
        r = plt.bar(left, d, bar_width, label="hey", **bar_format_args, hatch=hatches[0], edgecolor=colors[idx])
        plt.bar(left, d, bar_width, color='none', edgecolor='k', zorder=2, lw=1.5)
        if idx == 0:
            if index ==0:
                plt.text(idx-bar_width/2-0.08, d+0.8, str(soter[index]), fontweight='bold')
        plots.append(r)
    r = plt.plot([-1, 4], [1, 1], color='#ff000d', marker=',',alpha=0.6,linestyle='dashed')
    ax = plt.gca()
    ax.grid(linestyle='--',which= 'major',axis='y', zorder=0)
    if index==0:
        num1,num2,num3,num4=-0.05,1.08,3,0
        plt.legend(plots, bar_label, ncol=4,bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4,fontsize=LEGEND_SIZE-9)
       
    labels=[]
    for i in data[index]:
        if i==0:
            labels.append("N/A")
        else:
            labels.append("")
    plt.xticks(xx, labels, fontsize=TICK_SIZE-7)
    ax.tick_params(axis='x',pad=-30)
    plt.yticks(myticks[index],myticks_labels[index],fontsize=TICK_SIZE-6)
    plt.ylim((myticks[index][0],myticks[index][3]))
    plt.xlim(-0.5,3.5)
    if index%2 == 0:
        plt.ylabel("Norm-latency", fontsize=LABEL_SIZE-8,labelpad=10)
    plt.xlabel(program_label[index], fontsize=LABEL_SIZE-8,labelpad=10)
    
    plt.plot()

#plt.title("Latency",loc='left center')
#IS_DEBUG = True#False
if IS_DEBUG:
    plt.show()
else:
    plt.savefig(os.path.basename(__file__).replace(".py",".pdf"), format='pdf',dpi=500, bbox_inches='tight')