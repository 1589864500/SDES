import sys
from tkinter import font
sys.path.append('./')
from turtle import color
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
import matplotlib.patches as mpatches

import numpy as np

import tool.algorithm

linestyle = ['-', '--', '-.', ':']
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
color_2 = [color_mine, color1, color2, color3, color4, color5, color6, color7, color8, color9]
marker = ['P', 'd', '>', '|', 'x', 'X', ',', 'o', '*', 'D', 'h', 'H', 's', '1', '2']

linewidth = 2
markersize = 25

## 作图
## 所有节点
# SDES
# 关键节点 (3,1000) (4,1000) (5,400) (18,200) (20,100)
# 折点 (4,400) (5,200) (18,100)
type = 0
plt.plot([2, 4],[1000, 1000],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([4, 4],[1000, 400],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([4, 5],[400, 400],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([5, 5],[400, 200],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([5, 18],[200, 200],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([18, 18],[200, 100],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([18, 20],[100, 100],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.scatter([3,4,5,18,20], [1000,1000,400,200,100], marker=marker[type], s=markersize)
# ORIGAMI-M
# 关键节点 (3,1000) (4,400) (5,100)
# 折点 (3,400) (4,100) (5,0)
# DIRECT-MIN-COV
# 关键节点 (3,600) (4,200) (5,100) (6,100) (7,50) (8,25)
# 折点 (3,200) (4,100) (6,50) (8,0)
type = 4
plt.plot([2, 3],[600, 600],linewidth=linewidth,linestyle=linestyle[type-4],color=color_2[type])
plt.plot([3, 3],[600, 200],linewidth=linewidth,linestyle=linestyle[type-4],color=color_2[type])
plt.plot([3, 4],[200, 200],linewidth=linewidth,linestyle=linestyle[type-4],color=color_2[type])
plt.plot([4, 4],[200, 100],linewidth=linewidth,linestyle=linestyle[type-4],color=color_2[type])
plt.plot([4, 6],[100, 100],linewidth=linewidth,linestyle=linestyle[type-4],color=color_2[type])
plt.plot([6, 6],[100, 50],linewidth=linewidth,linestyle=linestyle[type-4],color=color_2[type])
plt.plot([6, 7],[50, 50],linewidth=linewidth,linestyle=linestyle[type-4],color=color_2[type])
plt.plot([7, 7],[50, 25],linewidth=linewidth,linestyle=linestyle[type-4],color=color_2[type])
plt.plot([7, 8],[25, 25],linewidth=linewidth,linestyle=linestyle[type-4],color=color_2[type])
plt.plot([8, 8],[25, 0],linewidth=linewidth,linestyle=linestyle[type-4],color=color_2[type])
plt.scatter([3,4,5,6,7,8], [600,200,100,100,50,25], marker=marker[type], s=markersize)
type = 1
plt.plot([2, 3],[1000, 1000],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([3, 3],[1000, 400],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([3, 4],[400, 400],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([4, 4],[400, 100],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([4, 5],[100, 100],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([5, 5],[100, 0],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.scatter([3,4,5], [1000,400,100], marker=marker[type], s=markersize)
# ORIGAMI-A
# 关键节点 (3,200) (4,100) (5,75) (6,25) (8,25) 
# 折点 (3,100) (4,75) (5,25) (6,0)
type = 2
plt.plot([2, 3],[200, 200],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([3, 3],[200, 100],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([3, 4],[100, 100],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([4, 4],[100, 75],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([4, 5],[75, 75],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([5, 5],[75, 25],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([5, 6],[25, 25],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([6, 6],[25, 0],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.scatter([3,4,5,6,8], [200,100,75,25,25], marker=marker[type], s=markersize)
# ORIGAMI-M-BS
# 关键节点 (3,1000) (4,200) (5,100) (6,75) (7,50)
# 折点 (3,200) (4,100) (5,75) (6,50) (7,0)
type = 3
plt.plot([2, 3],[1000, 1000],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([3, 3],[1000, 200],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([3, 4],[200, 200],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([4, 4],[200, 100],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([4, 5],[100, 100],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([5, 5],[100, 75],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([5, 6],[75, 75],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([6, 6],[75, 50],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([6, 7],[50, 50],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.plot([7, 7],[50, 0],linewidth=linewidth,linestyle=linestyle[type],color=color_2[type])
plt.scatter([3,4,5,6,7], [1000,200,100,75,50], marker=marker[type], s=markersize)


# 设置 x 和 y 轴名称
plt.xlabel('Attacker')
plt.ylabel('Target')
# 设置 x 和 y 轴标尺
# plt.yticks(np.arange(1, 9, 1), ['3', '4', '5', '6', '7', '8', '9~18', '19~20'])
plt.xticks(np.arange(3, 21, 1), np.arange(3, 21, 1))
plt.yticks([100, 200, 400, 600, 800, 1000], ['100', '200', '400', '600', '800', '1000'])
#自定义坐标轴
plt.xlim((2,20))
plt.ylim((0,1100))
# save
plt.savefig('Scalability')
plt.savefig('Scalability.pdf')
