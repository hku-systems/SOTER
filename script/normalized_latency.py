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
data[0][0] = round(float(exper[24])/float(exper[12]),2) 
data[0][1] = round(float(exper[5])/float(exper[12]),2) 
data[0][2] = round(float(exper[6])/float(exper[12]),2) 
data[0][3] = round(float(exper[18])/float(exper[12]),2) 
# alex: 
data[1][0] = round(float(exper[19])/float(exper[7]) ,2)   
data[1][1] = round(float(exper[0])/float(exper[7]) ,2)   
data[1][2] = 0
data[1][3] = round(float(exper[13])/float(exper[7]) ,2)   
# res: 
data[2][0] = round(float(exper[22])/float(exper[10]) ,2)   
data[2][1] = round(float(exper[3])/float(exper[10]) ,2)  
data[2][2] = 0
data[2][3] = round(float(exper[16])/float(exper[10]) ,2)  
# aegisdnn: alex dense mlp res trans vgg 0 - 5
# ennclave: vgg 6
# gpu: alex dense mlp res trans vgg 7 - 12
# mlcapsule: alex dense mlp res trans vgg 13 - 18
# soter: alex dense mlp res trans vgg 19 - 24

# dense: 
data[3][0] = round(float(exper[20])/float(exper[8]) ,2)  
data[3][1] = round(float(exper[1])/float(exper[8]) ,2) 
data[3][2] = 0
data[3][3] = round(float(exper[14])/float(exper[8]) ,2) 
# mlp: 
data[4][0] = round(float(exper[21])/float(exper[9]) ,2) 
data[4][1] = round(float(exper[2])/float(exper[9]) ,2) 
data[4][2] = 0
data[4][3] = round(float(exper[15])/float(exper[9]) ,2) 
# trans: 
data[5][0] = round(float(exper[23])/float(exper[11]) ,2) 
data[5][1] = round(float(exper[4])/float(exper[11]) ,2) 
data[5][2] = 0
data[5][3] = round(float(exper[17])/float(exper[11]) ,2) 

print (data)
soter = [round(float(exper[24]),2),round(float(exper[19]),2),round(float(exper[22]),2),round(float(exper[20]),2),round(float(exper[21]),2),round(float(exper[23]),2)]

myticks = [
    [0,10,20,40],
    [0,5,10,25],
    [0,10,15,25],
    [0,3,6,9],
    [0,1,2,6],
    [0,15,30,45]
]

myticks_labels = [
    ['0x','10x','20x','40x'],
    ['0x','5x','10x','25x'],
    ['0x','10x','15x','25x'],
    ['0x','3x','6x','9x'],
    ['0x','1x','2x','6x'],
    ['0x','15x','30x','45x']
]

# data = [
#     [10.54,10.32433875,10.23013209,24.71098875],
#     [3.21, 2.59,  0, 13.80138875],
#     [13.26, 12.58,0, 16.04],
#     [5.64,5.35   ,0, 7.05],
#     [2.24,2.175524543,0,2.71537693],
#     [13.58, 11.31, 0, 36.08]
# ]
# print (data)

bar_label = ['SOTER','AegisDNN', 'eNNclave', 'MLCapsule' ]
bar_width = 0.57
program_label = ['VGG19', 'Alexnet', 'Resnet152', 'Densenet121', 'MLP', 'Transformer']

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
            elif index ==1:
                plt.text(idx-bar_width/2, d+0.4, str(soter[index]), fontweight='bold')
            elif index ==2:
                plt.text(idx-bar_width/2-0.08, d+0.8, str(soter[index]), fontweight='bold')
            elif index ==3:
                plt.text(idx-bar_width/2-0.04, d+0.3, str(soter[index]), fontweight='bold')
            elif index ==4:
                plt.text(idx-bar_width/2, d+0.1, str(soter[index]), fontweight='bold')
            elif index ==5:
                plt.text(idx-bar_width/2-0.08, d+2, str(soter[index]), fontweight='bold')
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

    #plt.ylabel("Normalized Latency", fontsize=LABEL_SIZE-4)
    
    plt.plot()


#plt.title("Latency",loc='left center')
#IS_DEBUG = True#False
if IS_DEBUG:
    plt.show()
else:
    plt.savefig(os.path.basename(__file__).replace(".py",".pdf"), format='pdf',dpi=500, bbox_inches='tight')
    # plt.savefig(os.path.basename(__file__).replace(".py", ".png"), format='png', dpi=500, bbox_inches='tight')