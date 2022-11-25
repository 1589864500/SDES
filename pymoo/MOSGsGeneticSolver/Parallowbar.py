import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='serif')


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
color = [color_mine, color1, color2, color3, color4, color5, color6, color7, color8, color9]

labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3) # 柱子顶部的text
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
plt.savefig('test')