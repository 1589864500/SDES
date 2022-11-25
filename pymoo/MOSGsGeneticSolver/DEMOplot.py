from asyncio import FastChildWatcher
import sys
sys.path.append('./')

from webbrowser import get
import matplotlib.pyplot as plt
import os

import numpy as np

from pymoo.factory import get_problem, get_reference_directions
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


'''绘制函数图'''
#  二元一次函数图像
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
x = np.arange(-100, 100, 1)
y = np.arange(-100, 100, 1)
X, Y = np.meshgrid(x, y)  # 网格的创建，生成二维数组，这个是关键
Z = X + Y
plt.xlabel('x')
plt.ylabel('y')
# 将函数显示为3d,rstride和cstride代表row(行)和column(列)的跨度cmap为色图分类
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.savefig('1')
#  二元二次函数图像
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax = Axes3D(fig)
x = np.arange(-100, 100, 1)
y = np.arange(-100, 100, 1)
X, Y = np.meshgrid(x, y)  # 网格的创建，生成二维数组，这个是关键
Z = X*X + Y*Y
plt.xlabel('x')
plt.ylabel('y')
# 将函数显示为3d,rstride和cstride代表row(行)和column(列)的跨度cmap为色图分类
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.savefig('2')

'''多折线图'''
# #对比两天内同一时刻温度的变化情况
# x = [5, 8, 12, 14, 16, 18, 20]
# y1 = [18, 21, 29, 31, 26, 24, 20]
# y2 = [15, 18, 24, 30, 31, 25, 24]
# z1 = [item * -1 for item in y1]
# z2 = [item * -1 for item in y2]
# #绘制折线图，添加数据点，设置点的大小
# # * 表示绘制五角星；此处也可以不设置线条颜色，matplotlib会自动为线条添加不同的颜色
# plt.plot(x, y1, 'r', marker='*', markersize=10, label='a')
# plt.plot(x, z1, 'r', marker='*', markersize=10, label='b')
# plt.plot(x, z2, 'b', marker='P', markersize=10, label='c')
# plt.plot(x, y2, 'b', marker='P', markersize=10)
# # plt.title('温度对比折线图')  # 折线图标题
# # plt.xlabel('时间(h)')  # x轴标题
# # plt.ylabel('温度(℃)')  # y轴标题
# #给图像添加注释，并设置样式
# for a, b in zip(x, y1):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# #绘制图例
# plt.legend()
# #显示图像
# plt.savefig('1')

'''柱状图'''
# if __name__ == '__main__':
# 	max_lst_of_all = {}  #一个字典，value是四季最大阵风的风速值，key是年份
# 	max_lst_of_all[2010] = [29.7, 34.3, 29.7, 26.3]
# 	max_lst_of_all[2011] = [36.0, 30.2, 27.3, 30.9]
# 	max_lst_of_all[2012] = [27.3, 32.3, 40.4, 27.8]
# 	max_lst_of_all[2013] = [35.9, 29.9, 40.1, 33.3]
# 	max_lst_of_all[2014] = [26.3, 30.6, 28.6, 34.3]
# 	max_lst_of_all[2015] = [33.1, 27.0, 25.4, 30.7]
# 	max_lst_of_all[2016] = [41.3, 31.3, 41.1, 38.0]
# 	max_lst_of_all[2017] = [27.5, 31.2, 43.2, 41.2]
# 	fig = plt.figure()
# 	for key in max_lst_of_all.keys():
# 		print(max_lst_of_all[key])
# 		x = np.arange(key-0.3, key+0.31, 0.2)  #一年有四季，此行指定四季对应的bar的位置，比如2010年：2009.7,2009.9,2010.1,2010.3
# 		y = max_lst_of_all[key]  #此行决定了bar的高度(风速值）
# 		#bar_width = 0.2
# 		color = ['lightskyblue', 'lime', 'red', 'gold']  #指定bar的颜色
# 		for x1, y1, c1 in zip(x, y, color):  #遍历以上三者，每一次生成一条bar
# 			plt.bar(x1, y1, width=0.2, color=c1)
# 	#我试过这里不能直接生成legend，解决方法就是自己定义，创建legend
# 	labels = ['winter', 'spring', 'summer', 'autumn']  #legend标签列表，上面的color即是颜色列表
# 	#用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
# 	patches = [ mpatches.Patch(color=color[i], label="{:s}".format(labels[i]) ) for i in range(len(color)) ] 
# 	ax=plt.gca()
# 	box = ax.get_position()
# 	ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
# #下面一行中bbox_to_anchor指定了legend的位置
# 	ax.legend(handles=patches, bbox_to_anchor=(0.95,1.12), ncol=4) #生成legend
# 	plt.savefig('1')
# create a figure with one subplot

'''额外生成legend'''
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot([1,2,3,4,5], [1,2,3,4,5], 'r', label='test')
# # save it *without* adding a legend
# fig.savefig('image.png')

# # then create a new image
# # adjust the figure size as necessary
# figsize = (3, 3)
# fig_leg = plt.figure(figsize=figsize)
# ax_leg = fig_leg.add_subplot(111)
# # add the legend from the previous axes
# ax_leg.legend(*ax.get_legend_handles_labels(), loc='center')
# # hide the axes frame and the x/y labels
# ax_leg.axis('off')
# fig_leg.savefig('legend.png')

'''多子图'''
# fname='test'
# xscale = 'log'
# yscale = 'log'
# x = [10, 100, 1000]
# y = np.random.random(size=[3,4])
# path = os.path.join(os.getcwd(), 'pymoo', 'MOSGsGeneticSolver', 'DEMO')
# # n_evals = np.array([e.evaluator.n_eval for e in res.history])
# # opt = np.array([e.opt[0].F for e in res.history])
# # x = n_evals
# # y = opt
# yscale='log'
# fname='Convergence_n_evals-fitness'
# lstyle='--'  # 用什么API接口设置？
# # fig = plt.figure()
# # fig.set_title(fname)
# ax1 = plt.subplot2grid((2,2), (0,0))
# ax2 = plt.subplot2grid((2,2), (0,1))
# ax3 = plt.subplot2grid((2,2), (1,0))
# ax4 = plt.subplot2grid((2,2), (1,1))
# ax1.set_yscale(yscale)
# ax2.set_yscale(yscale)
# ax3.set_yscale(yscale)
# ax4.set_yscale(yscale)
# ax1.plot(x, y[:,0])
# ax2.plot(x, y[:,1])
# ax3.plot(x, y[:,2])
# ax4.plot(x, y[:,3])
# ax1.set_title('F1')
# ax2.set_title('F2')
# ax3.set_title('F3')
# ax4.set_title('F4')
# plt.show()
# path = os.path.join(path, fname)
# plt.savefig(path)