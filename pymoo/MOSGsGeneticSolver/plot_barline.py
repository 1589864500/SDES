from cProfile import label
import sys
sys.path.append('./')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick    #导入百分比
import matplotlib
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
plt.rc('font',family='serif')
import tool.algorithm
# 遇到数据中有中文的时候，一定要先设置中文字体
import os

RES_DIR = os.path.join(os.getcwd(), 'Results', 'demo')
RES_DIR = os.getcwd()

import numpy as np
import matplotlib.pyplot as plt
from typing import *


# NOTE 配色
color2 = '#0c84c6'	# blue
color1 = '#f74d4d'  # red
color3 = '#8ECFC9'  # green

# color1 = '#c82d31'
# color2 = '#194f97'

# 距离高自由度的代码还差一段距离，但是勉强用下
def barline_twinx(res:List[np.ndarray], std:List[np.ndarray], fname, yticks:np.ndarray=None, 
                bwith=2, title=None, ncol=1, loc='best', linewidth=3, elinewidth=3, markersize=9, 
                 grid=False, legend_title:str='Indicators', legend_fontsize=12, framealpha=False,
                fontsize=12, labelsize=15, capsize=4):
    res1, res2 = res[0], res[1]
    std1, std2 = std[0], std[1]

    '''绘制子图'''
    # NOTE legend和figure合并 (默认情况)
    fig = plt.figure(figsize=(4,4))
    # fig = plt.figure()
    ax1_1 = fig.add_subplot(111)   #
    # NOTE legend和figure分开 (特殊情况)
    # fig = plt.figure(figsize=(6,3))  # 作为做单独图例
    # ax1_1 = fig.add_subplot(121) # 作为做单独图例
    # NOTE 不绘制子图(与第一种情况类似)
    # fig, ax1_1 = plt.subplots()

    color = 'red' # color1
    ax1_1.set_xlabel('Population size', fontsize=fontsize+4)
    # ax1_1.set_ylabel('$HV$', color=color)
    # ax1_1.plot(t, res1[0], color=color1, marker='o', markersize=8, linewidth=2, label="$HV$")
    ax1_1.errorbar(t, res1[0] ,yerr=std1[0], fmt='o-', markersize=markersize, ecolor=color1, color=color1, elinewidth=elinewidth, linewidth=linewidth,
        capsize=capsize, label="HV")
    ax1_1.tick_params(labelsize=labelsize)

    ax1_2 = ax1_1.twinx()  # 实例化一个新的坐标轴，共享同一个x轴
    # ax1_2.set_ylabel('$IGD^+$')  # 共享x坐标轴，这里设置其y轴坐标标签
    # ax1_2.plot(t, res2[0], color=color2, marker='x', markersize=8, linewidth=2, label="$IGD^+$")#绘制第二个曲线
    ax1_2.errorbar(t, res2[0] ,yerr=std2[0], fmt='', markersize=markersize, ecolor=color2, color=color2, elinewidth=elinewidth, linewidth=linewidth,
        capsize=capsize, label="IGD$^+$")
    # ax1_2.tick_params(axis='y')
    ax1_2.tick_params(labelsize=labelsize)
    
    # NOTE 由于所有子图使用的图例都是相同的，因此将子图的图例单独拿出来绘制
    # NOTE 下面输出两次的方法是不正确的
    # ax1_1.legend(loc='center left') # wrong
    # ax1_2.legend(loc='center right')

    '''绘制legend 图例，legend和figure分来的情况'''
    # NOTE (特殊情况) legend和figure分开的情况
    # ax2_1 = fig.add_subplot(122)
    # #初始化labels和线型数组
    # lines=[]
    # labels=[]
    # #通过循环获取线型和labels
    # axLine, axLabel = ax1_1.get_legend_handles_labels()
    # lines.extend(axLine)
    # labels.extend(axLabel)
    # axLine, axLabel = ax1_2.get_legend_handles_labels()
    # lines.extend(axLine)
    # labels.extend(axLabel)
    # # NOTE 用于做单独图例
    # ax2_1.legend(lines, labels,loc='center',
    #            ncol=1,framealpha=True, fontsize=10, title='Indicator')
    # ax2_1.axis('off')

    # NOTE (一般情况) legend和figure合并的情况
    # 初始化labels和线型数组
    lines=[]
    labels=[]
    #通过循环获取线型和labels
    axLine, axLabel = ax1_1.get_legend_handles_labels() # 获取ax1_1上的所有legend
    lines.extend(axLine)
    labels.extend(axLabel)
    axLine, axLabel = ax1_2.get_legend_handles_labels() # 获取ax1_2上的所有legend
    lines.extend(axLine)
    labels.extend(axLabel)
    # NOTE 用于做图内图例
    ax1_1.legend(lines, labels,loc=loc,
            ncol=ncol,framealpha=framealpha, fontsize=legend_fontsize-2, title=legend_title, title_fontsize=legend_fontsize)

    ax1_1.spines['top'].set_linewidth(bwith)
    ax1_1.spines['bottom'].set_linewidth(bwith)
    ax1_1.spines['right'].set_linewidth(bwith)
    ax1_1.spines['left'].set_linewidth(bwith)
    plt.subplots_adjust(wspace=0.1,hspace=0.4)
    # NOTE 设置坐标轴刻度(一般不变)
    x_major_locator = MultipleLocator(50)
    y2_major_locator = MultipleLocator(.05)
    ax1_1.xaxis.set_major_locator(x_major_locator)
    ax1_2.yaxis.set_major_locator(y2_major_locator)

    # NOTE 刻度没有写成自动化，每次绘制新图片都需要修改(上界无法取到，下界是小于min的一格，上界是大于max的一格))
    if yticks is not None:
        ax1_2.set_yticks(yticks)  # for N=5 T=25
    # NOTE 科学计数法表示刻度
    # matplotlib.rcParams['font.size']=50
    ax1_1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')

    fig.tight_layout()  # 保证图像被完整显示
    ax1_1.grid(alpha=0.4)  # 网格
    # plt.show()
    plt.savefig(os.path.join(RES_DIR, 'barline'+fname))
    if pdf:
        plt.savefig(os.path.join(RES_DIR, 'barline'+fname)+'.pdf')

res1 = []
res2 = []
std1 = []
std2 = []
pdf = True

# 创建数据
# N=5, T=25-75
t = [350, 400, 450, 500]
N=5
T=100
if T==25:
    data1 = '537714 503351  547880  508308'
    data1 = tool.algorithm.excelsplit(data1)
    res1.append(data1)
# NOTE T=50
if T==50:
    data1 = '536764 558756 544382 554427'
    data1 = tool.algorithm.excelsplit(data1)
    res1.append(data1)
# T=75
if T==75:
    data1 = '820919 768335 770930 777721'
    data1 = tool.algorithm.excelsplit(data1)
    res1.append(data1)
# T=100
if T==100:
    data1 = '953971 755922 1072355 958332'
    data1 = tool.algorithm.excelsplit(data1)
    res1.append(data1)
# NOTE std
# NOTE T=25
if T==25:
    data1 = '228069 25924 228111 36846'
    data1 = tool.algorithm.excelsplit(data1)
    std1.append(data1)
# T=50
if T==50:
    data1 = '25471 123555 38792 55070'
    data1 = tool.algorithm.excelsplit(data1)
    std1.append(data1)
# T=75
if T==75:
    data1 = '164780 22742 54051 49157'
    data1 = tool.algorithm.excelsplit(data1)
    std1.append(data1)
# T=100
if T==100:
    data1 = '708302 157727 957851 708613'
    data1 = tool.algorithm.excelsplit(data1)
    std1.append(data1)
# NOTE IGD N=5
if T==25:
    data2 = '0.83 0.85 0.8 0.82'
    data2 = tool.algorithm.excelsplit(data2)
    res2.append(data2)
# NOTE T=50
if T==50:
    data2 = '0.69 0.67 0.67 0.67'
    data2 = tool.algorithm.excelsplit(data2)
    res2.append(data2)
# T=75
if T==75:
    data2 = '0.55 0.55 0.55 0.55'
    data2 = tool.algorithm.excelsplit(data2)
    res2.append(data2)
# T=100
if T==100:
    data2 = '0.59 0.63 0.69 0.61'
    data2 = tool.algorithm.excelsplit(data2)
    res2.append(data2)
# NOTE std
# NOTE T=25
if T==25:
    data2 = '0.15 0.03 0.14 0.04'
    data2 = tool.algorithm.excelsplit(data2)
    std2.append(data2)
# T=50
if T==50:
    data2 = '0.02 0.06 0.03 0.05'
    data2 = tool.algorithm.excelsplit(data2)
    std2.append(data2)
# T=75
if T==75:
    data2 = '0.06 0.01 0.02 0.02'
    data2 = tool.algorithm.excelsplit(data2)
    std2.append(data2)
# T=100
if T==100:
    data2 = '0.18 0.1 0.2 0.15'
    data2 = tool.algorithm.excelsplit(data2)
    std2.append(data2)

'''函数式改版'''
res = [res1, res2]
std = [std1, std2]
# NOTE 刻度没有写成自动化，每次绘制新图片都需要修改(上界无法取到，下界是小于min的一格，上界是大于max的一格))
if T==25:
    ytick = np.arange(0.60,1.1,0.10)  # for N=5 T=25
elif T==50:
    ytick = np.arange(0.6, 0.75,0.05)  # T=50
elif T==75:
    ytick = np.arange(0.4,0.7,0.10)  # T=75
elif T==100:
    ytick = np.arange(0.40,1.0,0.10)  # T=100
para = {"N":N}
if T==25:
    fname = tool.algorithm.paras2Str(para) + '_1'
elif T==50:
    fname = tool.algorithm.paras2Str(para) + '_2'
elif T==75:
    fname = tool.algorithm.paras2Str(para) + '_3'
elif T==100:
    fname = tool.algorithm.paras2Str(para) + '_4'
barline_twinx(res, std, fname, yticks=ytick, loc='upper right')





'''原版、保留备份'''
# # 绘制子图
# # legend和figure合并
# fig = plt.figure(figsize=(3,3))
# ax1_1 = fig.add_subplot(111)   #
# # NOTE legend和figure分开
# # fig = plt.figure(figsize=(6,3))  # 作为做单独图例
# # ax1_1 = fig.add_subplot(121) # 作为做单独图例
# # 不绘制子图
# # fig, ax1_1 = plt.subplots()

# color = 'red' # color1
# ax1_1.set_xlabel('Pop_size')
# # ax1_1.set_ylabel('$HV$', color=color)
# # ax1_1.plot(t, res1[0], color=color1, marker='o', markersize=8, linewidth=2, label="$HV$")
# ax1_1.errorbar(t, res1[0] ,yerr=std1[0], fmt='o-', ecolor=color1, color=color1, elinewidth=2, capsize=4, label="$HV$")
# ax1_1.tick_params(axis='y')

# ax1_2 = ax1_1.twinx()  # 实例化一个新的坐标轴，共享同一个x轴
# # ax1_2.set_ylabel('$IGD^+$')  # 共享x坐标轴，这里设置其y轴坐标标签
# # ax1_2.plot(t, res2[0], color=color2, marker='x', markersize=8, linewidth=2, label="$IGD^+$")#绘制第二个曲线
# ax1_2.errorbar(t, res2[0] ,yerr=std2[0], fmt='', ecolor=color2, color=color2, elinewidth=2, capsize=4, label="$IGD^+$")
# ax1_2.tick_params(axis='y')
# # NOTE 由于所有子图使用的图例都是相同的，因此将子图的图例单独拿出来绘制,下面输出两次的方法是不正确的
# # ax1_1.legend(loc='center left')
# # ax1_2.legend(loc='center right')

# # ax2_1 = fig.add_subplot(142)  # for N=4
# # # ax2_1 = fig.add_subplot(132)  # for N=5
# # color = 'red'
# # ax2_1.set_xlabel('Pop_size')
# # ax2_1.set_ylabel('$HV$', color=color)
# # ax2_1.plot(t, res1[1], color=color, marker='o', markersize=8, linewidth=2, label="$HV$")
# # ax2_1.tick_params(axis='y')

# # ax2_2 = ax2_1.twinx()  # 实例化一个新的坐标轴，共享同一个x轴
# # ax2_2.set_ylabel('$IGD^+$', color='blue')  # 共享x坐标轴，这里设置其y轴坐标标签
# # ax2_2.plot(t, res2[1], color='blue', marker='x', markersize=8, linewidth=2, label="$IGD^+$")#绘制第二个曲线
# # ax2_2.tick_params(axis='y')

# # ax3_1 = fig.add_subplot(143)  # for N=4
# # # ax3_1 = fig.add_subplot(133)  # for N=5
# # color = 'red'
# # ax3_1.set_xlabel('Pop_size')
# # ax3_1.set_ylabel('$HV$', color=color)
# # ax3_1.plot(t, res1[2], color=color, marker='o', markersize=8, linewidth=2, label="$HV$")
# # ax3_1.tick_params(axis='y')

# # ax3_2 = ax3_1.twinx()  # 实例化一个新的坐标轴，共享同一个x轴
# # ax3_2.set_ylabel('$IGD^+$', color='blue')  # 共享x坐标轴，这里设置其y轴坐标标签
# # ax3_2.plot(t, res2[2], color='blue', marker='x', markersize=8, linewidth=2, label="$IGD^+$")#绘制第二个曲线
# # ax3_2.tick_params(axis='y')

# # NOTE 绘制legend 图例，legend和figure分来的情况
# # ax2_1 = fig.add_subplot(122)
# # #初始化labels和线型数组
# # lines=[]
# # labels=[]
# # #通过循环获取线型和labels
# # axLine, axLabel = ax1_1.get_legend_handles_labels()
# # lines.extend(axLine)
# # labels.extend(axLabel)
# # axLine, axLabel = ax1_2.get_legend_handles_labels()
# # lines.extend(axLine)
# # labels.extend(axLabel)
# # # NOTE 用于做单独图例
# # ax2_1.legend(lines, labels,loc='center',
# #            ncol=1,framealpha=True, fontsize=10, title='Indicator')
# # ax2_1.axis('off')

# # NOTE 绘制legend 图例，legend和figure合并的情况
# #初始化labels和线型数组
# lines=[]
# labels=[]
# #通过循环获取线型和labels
# axLine, axLabel = ax1_1.get_legend_handles_labels()
# lines.extend(axLine)
# labels.extend(axLabel)
# axLine, axLabel = ax1_2.get_legend_handles_labels()
# lines.extend(axLine)
# labels.extend(axLabel)
# # NOTE 用于做图内图例
# ax1_1.legend(lines, labels,loc='best',
#            ncol=1,framealpha=True, fontsize=10, title='Indicator')

# # NOTE 子图4
# # ax4_1 = fig.add_subplot(144)  # for N=4
# # color = 'red'
# # ax4_1.set_xlabel('Pop_size')
# # ax4_1.set_ylabel('$HV$', color=color)
# # ax4_1.plot(t, res1[3], color=color, marker='o', markersize=8, linewidth=2, label="$HV$")
# # ax4_1.tick_params(axis='y')

# # ax4_2 = ax4_1.twinx()  # 实例化一个新的坐标轴，共享同一个x轴
# # ax4_2.set_ylabel('$IGD^+$', color='blue')  # 共享x坐标轴，这里设置其y轴坐标标签
# # ax4_2.plot(t, res2[3], color='blue', marker='x', markersize=8, linewidth=2, label="$IGD^+$")#绘制第二个曲线
# # ax4_2.tick_params(axis='y')


# plt.subplots_adjust(wspace=0.1,hspace=0.4)
# # NOTE 设置坐标轴刻度(一般不变)
# x_major_locator = MultipleLocator(50)
# y2_major_locator = MultipleLocator(.05)
# ax1_1.xaxis.set_major_locator(x_major_locator)
# ax1_2.yaxis.set_major_locator(y2_major_locator)
# # ax2_1.xaxis.set_major_locator(x_major_locator)
# # ax2_2.yaxis.set_major_locator(y2_major_locator)
# # ax3_1.xaxis.set_major_locator(x_major_locator)
# # ax3_2.yaxis.set_major_locator(y2_major_locator)
# # NOTE 子图4
# # ax4_1.xaxis.set_major_locator(x_major_locator)
# # ax4_2.yaxis.set_major_locator(y2_major_locator)

# # NOTE 刻度没有写成自动化，每次绘制新图片都需要修改(上界无法取到，下界是小于min的一格，上界是大于max的一格))
# if T==25:
#     ax1_2.set_yticks(np.arange(0.6,1.1,0.1))  # for N=5 T=25
# elif T==50:
#     ax1_2.set_yticks(np.arange(0.6, 0.75,0.05))  # T=50
# elif T==75:
#     ax1_2.set_yticks(np.arange(0.5,0.7,0.05))  # T=75
# elif T==100:
#     ax1_2.set_yticks(np.arange(0.15,0.3,0.05))  # T=100
# # NOTE 科学计数法表示刻度
# ax1_1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')

# fig.tight_layout()  # 保证图像被完整显示
# plt.show()

# para = {"N":N}
# if T==25:
#     para_str = tool.algorithm.paras2Str(para) + '_1'
# elif T==50:
#     para_str = tool.algorithm.paras2Str(para) + '_2'
# elif T==75:
#     para_str = tool.algorithm.paras2Str(para) + '_3'
# elif T==100:
#     para_str = tool.algorithm.paras2Str(para) + '_4'
# plt.savefig(os.path.join(RES_DIR, 'barline'+para_str))
# if pdf:
#     plt.savefig(os.path.join(RES_DIR, 'barline'+para_str)+'.pdf')


print("__main__ finish!")