import sys
from tkinter import font
sys.path.append('./')
from turtle import color
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
import matplotlib.patches as mpatches

import numpy as np

import tool.algorithm

fontsize = 15 # label_fontsize, title_fontsize, tick_fontsize
title_fontsize = fontsize -2 
legend_fontsize = 12
tick_fontsize = 15-2
label_fontsize = tick_fontsize + 1 
color_mine = '#3685fe' # 蓝 54,133,254 
color1 = '#50c48f' # 绿 80,196,143 hslc
color2 = '#f5616f' # 红 245,97,111 hshc
color3 = '#ffa510' # 黄 255,165,16 lslc
color4 = '#9977ef' # 紫 153,119,239 lshc
color5 = '#009db2' # 墨绿 2,75,81
color6 = '#555555' # 灰 85,85,85
color7 = '#943c39' # 棕 148,60,57
color8 = '#c82d31' # 玫红 200,45,49
color9 = '#f05326' # 橙 240,83,38
# 微调
color4 = '#898989'
color5 = color6
color = [color_mine, color1, color2, color3, color4, color5, color6, color7, color8, color9]
# color = ['#194f97', '#9c9800', '#c82d31', '#625ba1', '#898989', '#555555']

marker = ['o', 'X', '+', '*', '.', '//'] # *+-./OX\ox|

labels = ['SDES', 'ORIGAMI-M', 'ORIGAMI-A', 'ORIGAMI\n-M-BS', 'DIRECT\n-MIN-COV'] 
# labels = ['SDES', 'ORIGAMI-M', 'ORIGAMI-A', 'ORIGAMI\n-M-BS', 'DIRECT\n-MIN-COV']  # x-axis
legend = ['rank-1', 'rank-2', 'rank-3', 'rank-4', 'rank-5', 'TIMEOUT']

# HV
rank_1 = tool.algorithm.excelsplit('24	1	1	0	0')
rank_2 = tool.algorithm.excelsplit('0	9	8	3	6')
rank_3 = tool.algorithm.excelsplit('2	5	3	3	9')
rank_4 = tool.algorithm.excelsplit('0	3	3	6	6')
rank_5 = tool.algorithm.excelsplit('0	0	0	10	2')
timeout = tool.algorithm.excelsplit('0	8	11	4	3')
HV = np.array([rank_1, rank_2, rank_3, rank_4, rank_5, timeout])
bottomHV = np.zeros_like(HV)

# IGD+
rank_1 = tool.algorithm.excelsplit('22	2	3	0	0')
rank_2 = tool.algorithm.excelsplit('3	5	9	5	3')
rank_3 = tool.algorithm.excelsplit('0	11	3	3	5')
rank_4 = tool.algorithm.excelsplit('1	0	0	7	10')
rank_5 = tool.algorithm.excelsplit('0	0	0	7	5')
timeout = tool.algorithm.excelsplit('0	8	11	4	3')
IGDPlus = np.array([rank_1, rank_2, rank_3, rank_4, rank_5, timeout])
bottomIGDPlus = np.zeros_like(IGDPlus)

for i in range(1, HV.shape[0]):
    bottomHV[i] = HV[i-1] + bottomHV[i-1]
    bottomIGDPlus[i] = IGDPlus[i-1] + bottomIGDPlus[i-1]

# men_std = [2, 3, 4, 1, 2] # 显示在柱顶部的方差
# women_std = [3, 5, 2, 3, 3]

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()
x = np.array([1,2,3, 4, 5])  # the label locations
# ax.bar(labels, men_means, width, yerr=men_std, label='Men')
# ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,

for i in range(HV.shape[0]):
    # barHV = ax.bar(x - width/2, HV[i], width, bottom=bottomHV[i], label=legend[i], color=color[i], hatch='/', alpha=0.95)  # ParallowBar
    # barIGDPlus = ax.bar(x + width/2, IGDPlus[i], width, bottom=bottomIGDPlus[i], color=color[i], hatch='\\', alpha=0.95)
    barHV = ax.bar(x - width/2 - 0.02, HV[i], width, bottom=bottomHV[i], label=legend[i], color=color[i], alpha=0.95)  # ParallowBar
    barIGDPlus = ax.bar(x + width/2 + 0.02, IGDPlus[i], width, bottom=bottomIGDPlus[i], color=color[i], alpha=0.95)
plt.xticks(x, labels)
plt.yticks(fontsize=tick_fontsize)

barlabelHV = ['HV'] * len(labels)
barlabelIGDPlus = ['IGD$^+$'] * len(labels)
ax.bar_label(barHV, barlabelHV, padding=1.5)
ax.bar_label(barIGDPlus, barlabelIGDPlus)
ax.set_ylabel('Rank distribution', fontsize=label_fontsize)
# ax.set_title('Scores by group and gender')
# plt.tick_params(labelsize=tick_fontsize) # 同时调节两个轴

bwith = 2
ax.spines['top'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)

patches = []
for i in range(len(legend)):
    patches.append(mpatches.Patch(color=color[i], label=legend[i]))
ax.legend(handles=patches)

plt.show()
plt.savefig('Rankbar')
plt.savefig('Rankbar.pdf')