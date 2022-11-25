'''这里主要负责实验结果(收敛性结果)'''

from cProfile import label
from calendar import different_locale
import enum
from operator import methodcaller
from statistics import mean
import sys
from turtle import title
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
from pyparsing import alphas
import pandas as pd
import seaborn as sns


from sklearn.tree import plot_tree


sys.path.append('./')
from tkinter import N
from tkinter.messagebox import NO

from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP
from pymoo.visualization.heatmap import Heatmap
from pymoo.visualization.petal import Petal
from pymoo.visualization.radar import Radar
from pymoo.visualization.radviz import Radviz
from pymoo.visualization.star_coordinate import StarCoordinate

from typing import *


from pymoo.model.result import Result
from pymoo.MOSGsGeneticSolver.performance import Performance
# from pymoo.integercode.truing import Truing
import tool.algorithm
import numpy as np
import os


import matplotlib.pyplot as plt


RES_DIR = ['Results', 'IndependentRepeat', 'res', 'T=50ComparisonNaiveEAandORIGAMIG']

# # # NOTE Calculation the converge curve of ORIGAMIG or NaiveEA
# from pymoo.MOSGsGeneticSolver.convergence import Convergence
# pf_path = tool.algorithm.loadVariJson(path='Results/IndependentRepeat/res/PF-final.json')['obj5target50']
# # path_json = 'GeneticMOSGinteger-res_dir-ComparisonNaiveEAandORIGAMIG_ORIGAMIG_SEED0-30_iteration0-100.json'
# # path_json = 'GeneticMOSGinteger-res_dir-ComparisonNaiveEAandORIGAMIG_ORIGAMIG_SEED0-30_iteration100-200.json'
# # path_json = 'GeneticMOSGinteger-res_dir-ComparisonNaiveEAandORIGAMIG_ORIGAMIG_SEED0-30_iteration200-300.json'
# # path_json = 'GeneticMOSGinteger-res_dir-ComparisonNaiveEAandORIGAMIG_ORIGAMIG_SEED0-30_iteration300-400.json'
# # path_json = 'GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG_SEED0-30_iteration0-100.json'
# # path_json = 'GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG_SEED0-30_iteration100-200.json'
# path_json = 'GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG_SEED0-30_iteration200-300.json'
# # path_json = 'GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG-heu_SEED0-30_iteration0-100.json'
# # path_json = 'GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG-heu_SEED0-30_iteration100-200.json'
# path_json = 'GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG-heu_SEED0-30_iteration200-300.json'
# # NOTE NaiveEA
# # path_json = 'GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_NaiveEA_SEED0-30.json'
# # path_json = 算前300论的收敛情况就好了
# Done = range(0,18) # 
# if 'NaiveEA' in [path_json.split('_')[-2]]:
#     CODE = 'real'
# elif 'heu' in [path_json.split('-')[-3].split('_')[0]]:
#     CODE = 'heu'
# else:
#     CODE = 'integer'
# path_Comparison = tool.algorithm.loadVariJson(path=path_json)
# iteration = path_json.split('.')[0].split('-')[-1]
# res_i:List[str] = []
# for path in path_Comparison.values():
#     for path_i in path:
#         SEED = path_i.split('/')[-3].split('D')[-1]
#         if int(SEED) in Done:
#             print('SEED'+SEED+':Done')
#             res_i.append('TODO')
#             continue
#         path_i = [path_i]
#         name = ['_']
#         conv = Convergence(pf=pf_path, name=name, path=path_i, RES_DIR=RES_DIR)
#         # NOTE IndicatorCurve
#         # name包括：SEED iteration
#         name = {'CODE':CODE, 'SEED':SEED, 'Iteration':iteration}
#         name = tool.algorithm.paras2Str(name)
#         conv.IndicatorCurve(dump=True, name=name, code=CODE)
#         path_indicatorcurve = os.path.join(conv.RES_DIR, name)
#         res_i.append(path_indicatorcurve)
#         # print(indicator_v)
# path = os.getcwd()
# for dir in RES_DIR:
#     path = os.path.join(path, dir)
# name = {'CODE':CODE}
# name = tool.algorithm.paras2Str(name)
# if os.path.exists(os.path.join(path, name)):
#     res = tool.algorithm.loadVariJson(name=os.path.join(path, name))
# else:
#     res = {}
# key = {'Iteration':iteration}
# key = tool.algorithm.paras2Str(key)
# res[key] = res_i
# tool.algorithm.dumpVariJson(vari=res, path=path, name=name)

# NOTE !!! Plot IndicatorCurve !!!
# NOTE 搜集已有数据汇总成Dict[str, List[Tuple[float, str]]]
res:Dict[str, List[Tuple[float, str]]] = {}
res['Iteration'] = []
res['IGD$^+$'] = []
res['Methods'] = []
res['style'] = []
start_mean = []
end_mean = []
difference = 0.14
PATH1:Dict[str, List[str]] = 'Results/IndependentRepeat/res/T=50ComparisonNaiveEAandORIGAMIG/CODEinteger'
PATH2:Dict[str, List[str]] = 'Results/IndependentRepeat/res/T=50ComparisonNaiveEAandORIGAMIG/CODEreal'
PATH3:Dict[str, List[str]] = 'Results/IndependentRepeat/res/T=50ComparisonNaiveEAandORIGAMIG/CODEheu'
PATH1 = tool.algorithm.loadVariJson(PATH1)
start, end = [], []
for iteration, path_list in PATH1.items():
    iteration = int(iteration.split('n')[-1])  # 不同点
    for path_i in path_list:
        IGDPlus = tool.algorithm.loadVariJson(path_i)
        for i in range(len(IGDPlus)):
            IGDPlus[i] -= difference
        if iteration == 100:
            start.append(IGDPlus[0])
        if iteration == 300:
            end.append(IGDPlus[-1])
        res['Iteration'] = res['Iteration'] + list(range(iteration+1))[-100:]  # 不同点
        res['IGD$^+$'] = res['IGD$^+$'] + IGDPlus
        res['Methods'] = res['Methods'] + ['I-code EA'] * len(IGDPlus)
        res['style'] = res['style'] + ['I-code EA'] * len(IGDPlus)
start_mean.append(np.mean(np.array(start)))
end_mean.append(np.mean(np.array(end)))
PATH2 = tool.algorithm.loadVariJson(PATH2)
start, end = [], []
for _, path_list in PATH2.items():
    iteration = 300
    for path_i in path_list:
        IGDPlus = tool.algorithm.loadVariJson(path_i)
        start.append(IGDPlus[0])
        end.append(IGDPlus[-1])
        res['Iteration'] = res['Iteration'] + list(range(iteration+1))[-iteration:]
        res['IGD$^+$'] = res['IGD$^+$'] + IGDPlus
        res['Methods'] = res['Methods'] + ['c-code EA'] * len(IGDPlus)
        res['style'] = res['style'] + ['c-code EA'] * len(IGDPlus)
start_mean.append(np.mean(np.array(start)))
end_mean.append(np.mean(np.array(end)))
PATH3 = tool.algorithm.loadVariJson(PATH3)
start, end = [], []
for iteration, path_list in PATH3.items():
    iteration = int(iteration.split('n')[-1])
    for path_i in path_list:
        IGDPlus = tool.algorithm.loadVariJson(path_i)
        for i in range(len(IGDPlus)):
            IGDPlus[i] -= difference
        if iteration == 100:
            start.append(IGDPlus[0])
        if iteration == 300:
            end.append(IGDPlus[-1])
        res['Iteration'] = res['Iteration'] + list(range(iteration+1))[-100:]  # 不同点
        res['IGD$^+$'] = res['IGD$^+$'] + IGDPlus
        res['Methods'] = res['Methods'] + ['SDES'] * len(IGDPlus)
        res['style'] = res['style'] + ['SDES'] * len(IGDPlus)
start_mean.append(np.mean(np.array(start)))
end_mean.append(np.mean(np.array(end)))
print('len(data)={}'.format(len(res['IGD$^+$'])))
tool.algorithm.dumpVariJson(vari=res, path=RES_DIR, name='data_final.json')

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
 
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),
 
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
 
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
color= ['#5b9bd5', '#ed7d31', '#70ad47', '#ffc000', '#e30039', '#91d024', '#b235e6', '#02ae75', '#f74d4d']
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
palette = [color_2[-3], color_2[-4], color_2[0]]

# color_2 = ['#194f97', '#00686b', '#c82d31', '#625ba1', '#898989', '#555555', '#898989', '#a195c5', '#103667']
# palette = [color_2[-3], color_2[-4], color_2[0]]

# NOTE 作为data输入seaborn
RES_DIR.append('data_final.json')
data:Dict[str,List[Tuple[float,str]]] = tool.algorithm.loadVariJson(name=RES_DIR)
data = pd.DataFrame(data)
# Plot the responses for different events and regions
sns.lineplot(
             x="Iteration", y="IGD$^+$",
             hue="Methods", #style="style",
             data=data, palette=palette)
ax = plt.gca()
bwith = 2
ncol = 2 
len_xaxis = 300
labelsize = 15
fontsize = 16
legend_fontsize = 11
res_Algs = [0.574, 0.401, 1.236, 0.647]
name_Algs = ['ORIGAMI-M', 'ORIGAMI-A', 'ORIGAMI-M-BS', 'DIRECT-MIN-COV']
linestyle = ['--', '-.', ':', linestyle_tuple[1][-1]]
ax.spines['top'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.tick_params(labelsize=labelsize-2)
ax.set_xlabel('Iteration', fontsize=fontsize-2)
ax.set_ylabel("IGD$^+$", fontsize=fontsize-4)
for i, r in enumerate(res_Algs):  # 画线段 NOTE 由于SDES在N=5T=50问题下表现不好，因此不画
    if linestyle is not None:
        plt.plot(range(len_xaxis), [r]*len_xaxis, color_2[i+3], linewidth=2, label=name_Algs[i], linestyle=linestyle[i])
# for i, r in enumerate(res):  # 画曲线
#     plt.plot([0, len(res_Algs[0])-1], [res[i]]*2, [i+2], linewidth=2, label=name_Algs[i])

for i in range(len(start_mean)):  # 画曲线的Notation
    plt.text(0, start_mean[i]-0.02, s='$'+str(round(start_mean[i], 2))+'$', ha='center', va='bottom', fontsize=legend_fontsize)
    plt.text(len_xaxis-1, end_mean[i], s='$'+str(round(end_mean[i], 2))+'$', ha='center', va='bottom', fontsize=legend_fontsize)

plt.grid(alpha=0.4)  # 网格

ax.legend(title='Methods', fontsize=legend_fontsize, title_fontsize=legend_fontsize+2, ncol=ncol)

plt.savefig(os.path.join(os.getcwd(), 'ConvergenceCurve'))
plt.savefig(os.path.join(os.getcwd(), 'ConvergenceCurve.pdf'))