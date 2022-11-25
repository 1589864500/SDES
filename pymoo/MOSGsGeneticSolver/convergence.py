'''这里负责可视化（实验结果）'''

from cProfile import label
import enum
from locale import T_FMT
from operator import methodcaller
from statistics import mean
from turtle import title
import matplotlib
import matplotlib.pyplot as plt
# plt.rc('font',family='Times New Roman')
plt.rc('font', family='serif') # NOTE serif字体看起来确实好看
from pyparsing import alphas
# import pandas as pd
import seaborn as sns
import time

from sklearn.tree import plot_tree

from tkinter import N, font
from tkinter.messagebox import NO
from typing import *


import sys
sys.path.append('./')
from pymoo import model
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP
from pymoo.visualization.heatmap import Heatmap
from pymoo.visualization.petal import Petal
from pymoo.visualization.radar import Radar
from pymoo.visualization.radviz import Radviz
from pymoo.visualization.star_coordinate import StarCoordinate
from pymoo.model.result import Result
from pymoo.MOSGsGeneticSolver.performance import Performance
# from pymoo.integercode.truing import Truing  # 名字变了
import tool.algorithm
import numpy as np
import os


import matplotlib.pyplot as plt

# 最优值边界函数
class IdealPoint():
    def __init__(self, res:Result, mosg_flag=False, name='', RES_DIR:List[str]=['Results','figures']):
        self.RES_DIR = RES_DIR
        self.res = res
        self.name = name
        self.mosg_flag = mosg_flag
        if isinstance(res, np.ndarray):
            self.fit = res
        else:
            self.fit = res.F
        self.path = os.getcwd()
        for dir in RES_DIR:
            self.path = os.path.join(self.path, dir)
        self.N = self.fit.shape[-1]
        # self.popsize = 
    
    def plot_bound(self): 
        # 统计MOSG的数据
        fit_min:np.ndarray = np.min(self.fit, axis=0)
        print(self.name, '最优解下界：', fit_min)

# 结果散点图
class ScatterPlot(Scatter):
    def __init__(self, **kwargs):
        super(ScatterPlot, self).__init__(**kwargs)
        # 统计GMOSG的数据
        # n_evals = np.array([e.evaluator.n_eval for e in self.history])
        # opt = np.array([e.opt[0].F for e in res.history])
        # g_opt:np.ndarray = np.empty(shape=[self.popsize, self.N])

        # for i, gen in enumerate(self.history):
        #     gmosg_min = gen.opt[0].F
        #     for opt in gen.opt:
        #         min_idx = gmosg_min > opt.F
        #         gmosg_min[min_idx] = opt.F[min_idx]
        #     g_opt[i] = gmosg_min

        # x1 = n_evals
        # y1 = g_opt
        # # 只取部分，绘制局部图
        # # x1 = x1[:5]
        # # y1 = y1[:5,:]
        # # 统计MOSG的数据
        # x2 = x1
        # y2 = mosg_min[np.newaxis,:].repeat(y1.shape[0], axis=0)

        # # yscale='log'
        # fname='Convergence_n_evals-fitness'
        # lstyle='--'  # 用什么API接口设置？
        # # fig = plt.figure()
        # # fig.set_title(fname)
        
        # ax1 = plt.subplot2grid((2,3), (0,0))
        # ax2 = plt.subplot2grid((2,3), (0,1))
        # ax3 = plt.subplot2grid((2,3), (0,2))
        # ax4 = plt.subplot2grid((2,3), (1,0))
        # ax5 = plt.subplot2grid((2,3), (1,1))
        # ax6 = plt.subplot2grid((2,3), (1,2))
        # # ax1.set_yscale(yscale)
        # # ax2.set_yscale(yscale)
        # # ax3.set_yscale(yscale)
        # # ax4.set_yscale(yscale)
        # # ax5.set_yscale(yscale)
        # ax1.plot(x1, y1[:,0])
        # ax1.plot(x2, y2[:,0])

        # ax2.plot(x1, y1[:,1])
        # ax2.plot(x2, y2[:,1])

        # ax3.plot(x1, y1[:,2])
        # ax3.plot(x2, y2[:,2])

        # ax4.plot(x1, y1[:,3])
        # ax4.plot(x2, y2[:,3])

        # ax5.plot(x1, y1[:,4])
        # ax5.plot(x2, y2[:,4])

        # ax6.plot(x1, y1)
        # ax6.plot(x2, y2)

        # ax1.set_title(r'$F_1$')
        # ax2.set_title(r'$F_2$')
        # ax3.set_title(r'$F_3$')
        # ax4.set_title(r'$F_4$')
        # ax5.set_title(r'$F_5$')
        # ax6.set_title(r'$F_1-F_5$')
        
        # plt.ylim(np.min(y1)-0.2, np.max(y1))
        # plt.show()
        # path = os.path.join(self.path, fname)
        # plt.savefig(path)

    # def plot():
    #     # plot = ()
    #     pass

class PCPPlot(PCP):
    def __init__(self, **kwargs):
        super(PCPPlot, self).__init__(**kwargs)
        
class HeatmapPlot(Heatmap):
    def __init__(self, **kwargs):
        super(HeatmapPlot, self).__init__(**kwargs)

class PetalPlot(Petal):
    def __init__(self, **kwargs):
        super(PetalPlot, self).__init__(**kwargs)

class RadarPlot(Radar):
    def __init__(self, **kwargs):
        super(RadarPlot, self).__init__(**kwargs)

class RadvizPlot(Radviz):
    def __init__(self, **kwargs):
        super(RadvizPlot, self).__init__(**kwargs)

class StarCoordinatePlot(StarCoordinate):
    def __init__(self, **kwargs):
        super(StarCoordinatePlot, self).__init__(**kwargs)

class HeatmapAnnotation():
    def __init__(self, RES_DIR:List[str]=['Results','figures'], fname=None):
        self.RES_DIR = os.getcwd()
        for d in RES_DIR: self.RES_DIR = os.path.join(self.RES_DIR, d) 
        self.fname = fname

    def heatmap(self, data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Arguments:
            data       : A 2D numpy array of shape (N,M)
            row_labels : A list or array of length N with the labels
                        for the rows
            col_labels : A list or array of length M with the labels
                        for the columns
        Optional arguments:
            ax         : A matplotlib.axes.Axes instance to which the heatmap
                        is plotted. If not provided, use current axes or
                        create a new one.
            cbar_kw    : A dictionary with arguments to
                        :meth:`matplotlib.Figure.colorbar`.
            cbarlabel  : The label for the colorbar
        All other arguments are directly passed on to the imshow call.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        # ax.set_xticklabels(col_labels)
        # ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar


    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                        textcolors=["black", "white"],
                        threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Arguments:
            im         : The AxesImage to be labeled.
        Optional arguments:
            data       : Data used to annotate. If None, the image's data is used.
            valfmt     : The format of the annotations inside the heatmap.
                        This should either use the string format method, e.g.
                        "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
            textcolors : A list or array of two color specifications. The first is
                        used for values below a threshold, the second for those
                        above.
            threshold  : Value in data units according to which the colors from
                        textcolors are applied. If None (the default) uses the
                        middle of the colormap as separation.

        Further arguments are passed on to the created text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts
    
    def heatmapAnnotation(self, z, x, y, figsize=(7,8)):
        fig, ax = plt.subplots()
        # im, cbar = self.heatmap(z, x, y, ax=ax,
        #                 cmap="YlGn", cbarlabel="harvest [t/year]")
        im, cbar = self.heatmap(z, x, y,
                        cmap="YlGn", cbarlabel="harvest [t/year]")
        texts = self.annotate_heatmap(im, valfmt="{x:.1f}")
        # texts = annotate_heatmap(im, valfmt="{x:.1f} t")
        fig.tight_layout()
        fname = tool.algorithm.getTime() + self.fname
        fname = os.path.join(self.RES_DIR, fname)
        plt.savefig(figsize=figsize, fname=fname)


class Convergence():

    '''初始化负责：
        1、读取pf
        2、读取对比算法（List）
        3、记录持久化目录
    初始化过后：
        1、pf记作self.pf_res，对比算法基座self.res，持久化目录记为self.RES_DIR
        2、展示时能用到的图例和颜色记为self.marker,self.color'''
    def __init__(self, pf:str=None ,name:List[str]=None, path:List[str]=None, res:List[np.ndarray]=None, pf_name='PF', RES_DIR:List[str]=['Results','figures']):
        '''Convergence同时接收pf_true和pf_methods
    其他参数包含：pf_name:有时候pf参数传入的不一定是pf_true，则可以用pf_name标记
    RES_DIR表示图片存储的位置'''
        # 对比方法的结果
        self.res:List[np.ndarray] = []
        self.method_res:List[Result] = []
        # pf_true的结果
        self.pf_res:Result = None
        if pf is not None:
            self.pf_res:Tuple[Result, np.ndarray] = tool.algorithm.loadVariPickle(pf)
            if isinstance(self.pf_res, np.ndarray):
                self.pf:np.ndarray = self.pf_res
            else:
                self.pf:np.ndarray = self.pf_res.F
        if path is not None:
            for p in path:
                res:Tuple[Result, np.ndarray] = tool.algorithm.loadVariPickle(p)
                if isinstance(res, np.ndarray):
                    self.res.append(res)
                else:
                    self.res.append(res.F)
                    self.method_res.append(res)
        else:
            self.res = res
        self.pf_name = pf_name
        self.marker = tool.algorithm.getMarker()[:len(name)+1]
        self.color = tool.algorithm.getColor()[:len(name)+1]
        # 可微调颜色 图例

        self.name = name
        self.RES_DIR = os.getcwd()
        for dir in RES_DIR:
            self.RES_DIR = os.path.join(self.RES_DIR, dir)

    '''绘制单一scatter'''
    def scatter(self, res_idx=None, pdf=False, marksize=20, **kwargs):
        plot_type = 'scatter'
        sca = ScatterPlot(**kwargs)
        res_name = 'PSs'
        self.color[0] = '#63b2ee'
        if res_idx is not None:
            sca.add(self.res[res_idx], marker=self.marker[res_idx], color=self.color[res_idx], label=self.name[res_idx], s=marksize)
            res_name = res_name + self.name[res_idx]
        fname = '-'.join([tool.algorithm.getTime(), plot_type, res_name])
        sca.save(fname=os.path.join(self.RES_DIR, fname))
        if pdf:
            fname += '.pdf'
            sca.save(fname=os.path.join(self.RES_DIR, fname))


    '''负责绘制散点图, 同时绘制两个图, 用于比较
        res绘制出来，维度升高时建议最多绘制一个res'''
    def scatter_PF(self, res_idx=None, pdf=False, marksize=20, **kwargs):
        plot_type = 'scatter'
        sca = ScatterPlot(**kwargs)
        # NOTE 绘制PF可能意义不大，可以选择和对比算法比（除非和对比算法差异不明显）
        sca.add(self.pf, color='gray', s=15, marker=self.marker[-1], alpha=1)
        res_name = 'PSs'
        self.color[0] = '#63b2ee'	
        if res_idx is not None:
            sca.add(self.res[res_idx], marker=self.marker[res_idx], color=self.color[res_idx], label=self.name[res_idx], s=marksize)
            res_name = res_name + self.name[res_idx]
        fname = '-'.join([tool.algorithm.getTime(), plot_type, res_name])
        sca.save(fname=os.path.join(self.RES_DIR, fname))
        if pdf:
            fname += '.pdf'
            sca.save(fname=os.path.join(self.RES_DIR, fname))
            
    '''负责绘制散点图, 同时绘制多个散点'''
    def scatter_Algs(self, res_idx:List[int]=None, pdf=False, 
        marksize=20, para=None,
        **kwargs):
        plot_type = 'scatter'
        sca = ScatterPlot(**kwargs)
        self.color[0] = '#63b2ee'	
        res_name = ''
        # if res_idx is not None:
        for res_idx_i in res_idx:  # 把散点画上去
            sca.add(self.res[res_idx_i], marker=self.marker[res_idx_i], color=self.color[res_idx_i], label=self.name[res_idx_i], s=marksize)
            res_name = res_name + self.name[res_idx_i]
        fname = '-'.join([para, plot_type, res_name])
        sca.save(fname=os.path.join(self.RES_DIR, fname))
        if pdf:
            fname += '.pdf'
            sca.save(fname=os.path.join(self.RES_DIR, fname))

    '''包含多条线和图例的折线图，没用'''
    def ideal_nadir(self, **kwargs):
        plot_type = 'ideal_nadir'
        func = [1, 2, 3, 4, 5] # 横轴
        pf_ideal = np.min(self.pf, axis=0) # 准备数据
        pf_nadir = np.max(self.pf, axis=0)

        # 设置坐标轴
        plt.xlabel('$Attacker$')
        plt.ylabel("$U^d$")
        plt.xticks(ticks=func, labels=['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$'])
        
        # 绘图(pf) 打点
        plt.plot(func, pf_ideal, 'greenyellow', marker='*', markersize=10, label='PF(Ideal)', linestyle='-.')
        plt.plot(func, pf_nadir, 'greenyellow', marker='*', markersize=10, label='PF(Nadir)', linestyle='-')
        
        # 标注
        for i in range(len(func)):
            plt.text(func[i], pf_ideal[i]-0.5, round(pf_ideal[i], 2), ha='center', va='top', fontsize=10)
            plt.text(func[i], pf_nadir[i]+0.2, round(pf_nadir[i], 2), ha='center', va='bottom', fontsize=10)
        
        # 准备图例
        marker = ['P', 'd', '>', '3', 'x', 'X', ',']
        color =  ['b', 'g', 'c', 'y', 'k', 'm', 'w']
        
        # 绘图(res) 打点
        for i in range(len(self.name)):
            ideal = []
            nadir = []
            id = np.min(self.res[i], axis=0)
            na = np.max(self.res[i], axis=0)
            ideal.append(id)
            nadir.append(na)
            plt.plot(func, id, color[i], marker=marker[i], markersize=10, label=self.name[i], linestyle='-.')
            plt.plot(func, na, color[i], marker=marker[i], markersize=10, linestyle='-')
        
        for i in range(len(self.name)):
            pass
            
        plt.legend(loc='best', handlelength=4)
        
        fname = '-'.join([tool.algorithm.getTime(), plot_type])
        plt.savefig(fname=os.path.join(self.RES_DIR, fname))

    '''Indicator Curve of generation（负责统计指标，不负责绘图）
    there is a brief introduction for HISTORY number variable
    HISTORY:List[Generation] Convergence.Results.history
    Generation:HISTORY[i] its use-parttern is as same as Results, including .opt .pop etc.'''
    def IndicatorCurve(self, indicator='IGD+', interval=1, n_point:int=None, dump=False, name=None, code='real') ->List[float]:
        # indicator: the performance indicator we wanted
        # interval: the calculation frequence about iteration
        # n_point: a veriable about interval
        # dump: the flag about local persistence
        # name: file name
        # code: the flag about method (affect whether to use MIN-COV)
        # NOTE 除此之外，还用到了self的如下信息：
        #   self.method_res, self.pf, [self.pf_name, self.name,](需要但不重要)
        indicator_igdplus, indicator_hv = False, False
        if indicator == 'IGD+':
            indicator_igdplus = True
        elif indicator == 'HV':
            indicator_hv = True
        
        # 对比方法存在self.method_res内,IndicatorCurve属于特殊情况，
        # method_res虽为List但是长度为一，因此直接使用[0]取出
        gen_l = len(self.method_res[0].history)
        if n_point is None:
            n_point = int(gen_l / interval)
        point = np.linspace(0, gen_l-1, n_point).astype(np.int16)  # 需要显示结果的迭代轮数
        point_indicator:List[float] = []
        for g,gen in enumerate(self.method_res[0].history):
            if g not in point:
                continue
            # 精修算pf
            if code == 'integer' or 'heu':
                truing = Truing(res=gen)
                truing.mosgSearch_pop()
                pf:np.ndarray = truing.fit_pf
            elif code == 'real':
                pf:np.ndarray = gen.pop.get('F')
            else:
                print('undefined code')
            # pf得到以后和pf_true(self.pf)比，算Indicator
            perf = Performance(pf_total=[self.pf, pf], name_total=[self.pf_name, self.name[0]], 
                indicator_igdplus=indicator_igdplus, indicator_hv=indicator_hv)
            if indicator_hv:
                indicator_v = perf.hv[0]
            elif indicator_igdplus:
                indicator_v = perf.igd_plus[0]
            else:
                pass
            point_indicator.append(indicator_v)
        # tool.algorithm.dumpVariPickle(vari=point_indicator, name=self.name[0]+'IndWithGen')
        if len(point_indicator) != len(point):
            print('len(point_indicator) != len(point)')
        if dump:
            fname = name if name is not None else indicator
            # tool.algorithm.dumpVariPickle(vari=point_indicator, path=self.RES_DIR, name=fname)
            tool.algorithm.dumpVariJson(vari=point_indicator, path=self.RES_DIR, name=fname)
        return point_indicator
    def IndicatorCurvePlot(self, res:List[float], name:List[str], pdf=False, color=None, linestyle=None) ->None:
        plt.xlabel('Generation')
        plt.ylabel('IGD+')
        if color is None:
            color =  ['k', 'b', 'g', 'c', 'y', 'm', 'k']

        for i, r in enumerate(self.res):  # 画线段
            if linestyle is None:
                plt.plot(range(len(r)), r, color[i], linewidth=2, label=self.name[i])
            if linestyle is not None:
                plt.plot(range(len(r)), r, color[i], linewidth=2, label=self.name[i], linestyle=linestyle[i])
        for i, r in enumerate(res):  # 画曲线
            plt.plot([0, len(self.res[0])-1], [res[i]]*2, color[i+2], linewidth=2, label=name[i])

        for i, r in enumerate(self.res):  # 画Notation
            start_x = 0
            start_y = r[start_x]
            end_x = len(r)-1
            end_y = r[end_x]
            plt.text(start_x, start_y, s='$'+str(round(start_y, 2))+'$', ha='center', va='bottom', fontsize=10)
            plt.text(end_x, end_y, s='$'+str(round(end_y, 2))+'$', ha='center', va='bottom', fontsize=10)


        plt.legend(loc='best') # handlelength
        fname1 = 'IGD+-integerealcode-N5T50.pdf'
        fname2 = 'IGD+-integerealcode-N5T50'
        if pdf:
            plt.savefig(fname=os.path.join(self.RES_DIR, fname1))
        plt.savefig(fname=os.path.join(self.RES_DIR, fname2))

    # 绘制多条折线图，一条代表一个对比算法
    # plot_num和plot_part共同决定绘制提供的数据的前半部分还是后半部分
    def TimeIndicatorPlot(self, xaxis:str, xtick:Union[np.ndarray, List[int]], std:List[float]=None, plot_num=4, plot_part='first', 
        title=None, color:List[str]=None, marker:List[str]=None, loc='best', ncol=2, linewidth=3, markersize=12, elinewidth=2,
        pdf=False, bwith=2, grid=False, legend_title:str=None, legend_fontsize=12,
        fontsize=16, labelsize=15,
        # font1 = {'family' : 'Times New Roman', 'size':16}, 
        font1 = {'family' : 'serif', 'size':16}, 
        figsize=(7.5,5),
        ) ->None:
        plt.figure(figsize=figsize)
        # NOTE 需要自定义横轴
        # if xaxis == 'Attacker':
        #     plt.xticks(np.arange(0,len(xtick)), xtick)
        # elif xaxis == 'Target':
        #     plt.xticks()
        # 以下是等间距的情况， 如果不等间距则要换写法，具体换法是修改第一个参数np.arange(0,xtick[-1])
        plt.xticks(xtick, xtick)
        # plt.yticks(np.arange(0, 6000, 1000),np.arange(0, 6000, 1000))
        # NOTE 设定刻度和轴标签的大小
        # csfont = {'fontname':'Times New Roman'}
        plt.xlabel(xaxis, fontdict=font1, fontsize=fontsize)
        plt.ylabel('Runtime/s', fontdict=font1, fontsize=fontsize)
        # DEBUG 字体
        # plt.xlabel(xaxis, fontsize=fontsize)
        # plt.ylabel('Runtime/s', fontsize=fontsize)
        plt.tick_params(labelsize=labelsize)
        plt.grid(alpha=0.4)  # 网格

        # NOTE 默认开启grid需要精心设计X Y刻度
        # NOTE 同时刻度跨度需要重新设计(plt.xlim plt.xticks)
        if grid:
            # plt.xlim((0, 200))
            # plt.ylim((0, 300))
            # plt.xticks(np.arange(0, 201, 40))
            # plt.yticks(np.arange(0, 3501, 700))
            plt.grid() # 在不指定轴时默认xy，默认数量和刻度相同
        # NOTE 调整board width大小
        ax = plt.gca()
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)

        # 准备图例
        if marker is None:
            marker = ['P', 'd', '>', '|', 'x', 'X', ',', 'o', '*', 'D', 'h', 'H', 's', '1', '2']
        if color is None:
            color =  ['b', 'g', 'c', 'y', 'm', 'k', 'greenyellow', 'darkgray', 'deepskyblue', 'crimson', 'blueviolet']
        # std
        if std is None:
            for i, r in enumerate(self.res):
                if plot_part == 'first':
                    plt.plot(xtick[:plot_num], r[:plot_num], marker=marker[i], color=color[i], markersize=markersize, linewidth=linewidth, label=self.name[i])
                elif plot_part == 'second':
                    plt.plot(xtick[plot_num:len(r)], r[plot_num:], marker=marker[i], color=color[i], markersize=markersize, linewidth=linewidth, label=self.name[i])
                elif plot_part == 'both':
                    # NOTE 冗余代码，本来想处理特殊需求（方法1 2的线段颜色相同），后俩发现只需要在输入参数处做调整
                    # if i==0:
                    #     plt.plot(xtick[:len(r)], r, marker='H', color=color[0], markersize=10, linewidth=3, label=self.name[0])
                    # else:
                        plt.plot(xtick[:len(r)], r, marker=marker[i], color=color[i], markersize=markersize, linewidth=linewidth, label=self.name[i])
        else:
            # if self.name[0] == 'ORIGAMIGF':
            #     i, r = 0, self.res[0]
            #     plt.errorbar(xtick[:len(r)], r[:len(r)], yerr=std[i][:len(r)], elinewidth=2,  fmt='-'+marker[i], color=color[i], markersize=10, linewidth=3, label=self.name[i])
            #     for i, r in enumerate(self.res[1:]):
            #         if plot_part == 'first':
            #             plt.errorbar(xtick[:plot_num], r[:plot_num], yerr=std[i][:plot_num], elinewidth=2,  fmt='-'+marker[i], color=color[i], markersize=10, linewidth=3, label=self.name[i])
            #         elif plot_part == 'second':
            #             plt.errorbar(xtick[plot_num:len(r)], r[plot_num:], yerr=std[i][plot_num:], elinewidth=2, fmt='-'+marker[i], color=color[i], markersize=10, linewidth=3, label=self.name[i])
            #         elif plot_part == 'both':
            #             plot_num = len(r)
            #             plt.errorbar(xtick[:len(r)], r, fmt='-'+marker[i+1], yerr=std[i+1][:plot_num], color=color[i], elinewidth=0.1, markersize=10, linewidth=3, label=self.name[i+1])
            # else:
            for i, r in enumerate(self.res):
                if plot_part == 'first':
                    plt.errorbar(xtick[:plot_num], r[:plot_num], yerr=std[i][:plot_num], elinewidth=elinewidth,  fmt='-'+marker[i], color=color[i], markersize=markersize, linewidth=linewidth, label=self.name[i])
                elif plot_part == 'second':
                    plt.errorbar(xtick[plot_num:len(r)], r[plot_num:], yerr=std[i][plot_num:], elinewidth=elinewidth, fmt='-'+marker[i], color=color[i], markersize=markersize, linewidth=linewidth, label=self.name[i])
                elif plot_part == 'both':
                    plot_num = len(r)
                    plt.errorbar(xtick[:len(r)], r, fmt='-'+marker[i], yerr=std[i][:plot_num], color=color[i], markersize=markersize, linewidth=linewidth, label=self.name[i])

        if title is not None:
            plt.title(title, fontsize=fontsize)
        plt.legend(loc=loc, ncol=ncol, title=legend_title, fontsize=legend_fontsize, title_fontsize=legend_fontsize, framealpha=0.5) # handlelength
        time.sleep(1)
        fname1 = 'TimeIndicator-'+tool.algorithm.getTime()+'.pdf'
        fname2 = 'TimeIndicator-'+tool.algorithm.getTime()
        if pdf:
            print(os.path.join(self.RES_DIR, fname1))
            plt.savefig(fname=os.path.join(self.RES_DIR, fname1))
        plt.savefig(fname=os.path.join(self.RES_DIR, fname2))
        plt.close() 


    '''只画单一PF的pcp
    '''
    def pcp(self, res, **kwargs):
        plot_type = 'parallel_coordinate_plots'
        sca = PCPPlot(**kwargs)
        sca.set_axis_style(color='grey', alpha=1)
        sca.add(res, color='grey', alpha=0.3)
        ideal = np.min(res, axis=0)
        nadir = np.max(res, axis=0)
        sca.add(ideal, linewidth=5, color='grey', linestyle='-.', label=self.pf_name+'(Ideal)')
        sca.add(nadir, linewidth=5, color='grey', linestyle='-', label=self.pf_name+'(Nadir)')
        fname = '-'.join([tool.algorithm.getTime(), plot_type, self.name[0]])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

    '''画两个pcp的重叠，用于比较
    在默认的情况下，比较对象A为PF，但是可以传入自定义对象
    '''
    def pcp_PF(self, res_idx=None, pdf=False, **kwargs):
        res = self.res[res_idx]
        plot_type = 'parallel_coordinate_plots'
        sca = PCPPlot(**kwargs)
        # pf and another method's pcp_PF
        sca.set_axis_style(color='grey', alpha=1)
        sca.add(self.pf, color='grey', alpha=0.5)
        # sca.add(res, color=self.color[res_idx], alpha=0.5)
        sca.add(res, color='#63b2ee', alpha=0.5)  # 淡蓝
        # Ideal Nadir
        ideal_pf = np.min(self.pf, axis=0)
        nadir_pf = np.max(self.pf, axis=0)
        ideal_res = np.min(res, axis=0)
        nadir_res = np.max(res, axis=0)
        sca.add(ideal_pf, linewidth=4, color='#008080', linestyle=':', label=self.pf_name+'(Ideal)')  # 亮绿
        sca.add(nadir_pf, linewidth=4, color='#008080', linestyle='-', label=self.pf_name+'(Nadir)')
        sca.add(ideal_res, linewidth=4, color='#e30039', linestyle=':', label=self.name[res_idx]+'(Ideal)')  # 亮红
        sca.add(nadir_res, linewidth=4, color='#e30039', linestyle='-', label=self.name[res_idx]+'(Nadir)')
        fname = '-'.join([tool.algorithm.getTime(), plot_type, self.name[res_idx]])
        sca.save(fname=os.path.join(self.RES_DIR, fname))
        if pdf:
            fname += '.pdf'
            sca.save(fname=os.path.join(self.RES_DIR, fname))

    '''画不定数量的Algs的pcp的重叠, 用于比较
    '''
    def pcp_Algs(self, res_idx:List[int]=None, pdf=False, **kwargs):
        plot_type = 'parallel_coordinate_plots'
        methodname = ''
        sca = PCPPlot(**kwargs)
        sca.set_axis_style(color='grey', alpha=1)
        self.color[0] = '#63b2ee' # 淡蓝
        self.color[1] = '#e30039' # 亮红
        for res_idx_i in res_idx:
            res = self.res[res_idx_i]
            sca.add(res, color=self.color[res_idx_i], alpha=0.5) # 绘制数据
            # Ideal Nadir
            ideal_res = np.min(res, axis=0)
            nadir_res = np.max(res, axis=0)
            sca.add(ideal_res, linewidth=4, color=self.color[res_idx_i], linestyle=':', label=self.name[res_idx_i]+'(Ideal)')  # 上下界
            sca.add(nadir_res, linewidth=4, color=self.color[res_idx_i], linestyle='-', label=self.name[res_idx_i]+'(Nadir)')
            methodname = '-'.join([methodname, self.name[res_idx_i]])
        fname = '-'.join([plot_type, para, methodname])
        sca.save(fname=os.path.join(self.RES_DIR, fname))
        if pdf:
            fname += '.pdf'
            sca.save(fname=os.path.join(self.RES_DIR, fname))


    def heatmap(self, res, **kwargs):
        plot_type = 'heatmap'
        sca = HeatmapPlot(**kwargs)
        sca.add(res)
        fname = '-'.join([tool.algorithm.getTime(), plot_type, self.name[0]])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

    def petal_diagrame(self, **kwargs):
        pass
        plot_type = 'petal_diagram'
        sca = PetalPlot(**kwargs)
        sca.add(self.pf).show()
        for i in range(len(self.name)):
            sca.add(self.res[i], marker=self.marker, color=self.color[i], label=self.name[i])
        fname = '-'.join([tool.algorithm.getTime(), plot_type])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

    def radar_plot(self, **kwargs):
        pass
        plot_type = 'radar_plot'
        sca = RadarPlot(**kwargs)
        sca.add(self.pf).show()
        for i in range(len(self.name)):
            sca.add(self.res[i], marker=self.marker, color=self.color[i], label=self.name[i])
        fname = '-'.join([tool.algorithm.getTime(), plot_type])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

    def radviz(self, **kwargs):
        pass
        plot_type = 'radviz'
        sca = RadvizPlot(**kwargs)
        sca.set_axis_style(color="black", alpha=1)
        # sca.add(self.pf, color='gray', s=20, alpha=0.3)
        for i in range(len(self.name)):
            sca.add(self.res[i], color=self.color[i], s=70, label=self.name[i])
        fname = '-'.join([tool.algorithm.getTime(), plot_type])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

    def star_coordinate_plot_total(self, **kwargs):
        plot_type = 'star_coodinate_plot'
        sca = StarCoordinatePlot(**kwargs)
        # sca.add(self.pf, color='gray', s=20, alpha=0.3)
        for i in range(len(self.name)):
            sca.add(self.res[i], color=self.color[i], s=70, label=self.name[i])
        fname = '-'.join([tool.algorithm.getTime(), plot_type])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

    def star_coordinate_plot_PF(self, res_idx=None, pdf=False, **kwargs):
        plot_type = 'star_coodinate_plot'
        sca = StarCoordinatePlot(**kwargs)
        res_name = 'PF'
        if res_idx is not None:
            res_name += self.name[res_idx]
        sca.add(self.pf, color='gray', s=50, alpha=0.5)
        sca.add(self.res[res_idx], color=self.color[res_idx], s=70, label=self.name[res_idx])
        fname = '-'.join([tool.algorithm.getTime(), plot_type, res_name])
        sca.save(fname=os.path.join(self.RES_DIR, fname))
        if pdf:
            fname += '.pdf'
            sca.save(fname=os.path.join(self.RES_DIR, fname))


# name = []
# path = []
# res = []
# std = []


'''预定义线段、颜色、标记样式，目前部分没用上大，但是以后用的上'''
linestyle_str = [
    #  ('solid', 'solid'),      # Same as (0, ()) or '-'
    #  ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
    #  ('dashed', 'dashed'),    # Same as '--'
    #  ('dashdot', 'dashdot')  # Same as '-.'
    ('solid', '-'),      # Same as (0, ()) or '-'
     ('dotted', ':'),    # Same as (0, (1, 1)) or '.'
     ('dashed', '--'),    # Same as '--'
     ('dashdot', '-.')  # Same as '-.'
     ]
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
bwith = 2

# color_2 = ['#194f97', '#00686b', '#bd6b08', '#625ba1', '#898989', '#555555', '#898989', '#a195c5', '#103667']

RES_DIR = ['Results','20221017']
# '''TimeIndicator N=3, T=25-1000'''
# name = []
# path = []
# res = []
# std = []
# name.append('SDES')
# name.append('ORIGAMI-M')
# name.append('ORIGAMI-A')
# name.append('ORIGAMI-M-BS')
# name.append('DIRECT-MIN-COV')
# # time 
# ORIGAMIG = [ 26, 33, 27, 42, 64, 150, 222, 284, 354]
# std_g = '0.3	0.09	0.07	0.17	0.5	0.5	1	3	3'
# std_g = tool.algorithm.excelsplit(std_g)
# ORIGAMIM = [0.3551,
# 1.5358,
# 3.7784,
# 5.9667,
# 30.786,
# 147.414,
# 348.593,
# 769.2457,
# 1320.58]
# std_m = '0.03	0.01	0.04	0.06	0.28	0.63	1.2	2	5'
# std_m = tool.algorithm.excelsplit(std_m)
# ORIGAMIA = [7.407,
# 33.0084,
# 78.7273,
# 193.63363,
# 915.512,
# 4090.8]
# std_a = '0.37	0.09	0.07	0.17	0.56	0.46'
# std_a = tool.algorithm.excelsplit(std_a)
# ORIGAMIMBS = [0.0739,
# 0.239,
# 0.68,
# 1.272,
# 8.3963,
# 34.086,
# 185.67,
# 375.488,
# 606.474]
# std_mbs = '0	0	0.03	0.06	0.7	2.1	5	15	23'
# std_mbs = tool.algorithm.excelsplit(std_mbs)
# DIRECTMINCOV = [1.425,
# 6.225,
# 16.324,
# 26.236,
# 117.095,
# 445.06,
# 1457.7,
# 2228.8,
# 3872.265]
# res.append(ORIGAMIG)
# res.append(ORIGAMIM)
# res.append(ORIGAMIA)
# res.append(ORIGAMIMBS)
# res.append(DIRECTMINCOV)
# # NOTE std太小, 用不到
# std_dir = '0.09	0.1	5.6	0.7	2.1	10	20	30	50'
# std_dir = tool.algorithm.excelsplit(std_dir)
# std.append(std_g)
# std.append(std_m)
# std.append(std_a)
# std.append(std_mbs)
# std.append(std_dir)
# conv = Convergence(name=name, res=res, RES_DIR=RES_DIR)
# # NOTE TimeIndicatorPlot
# conv.TimeIndicatorPlot(xaxis='Target', xtick=[25,50,75,100, 200,400,600,800,1000], title='N=3, T=25-100', pdf=True, std=None, color=color_2, legend_title='Methods',
#     markersize=11) # N=3 T=25-100
# conv.TimeIndicatorPlot(xaxis='Target', xtick=[25,50,75,100, 200,400,600,800,1000], title='N=3 T=200-1000', plot_part='second', pdf=True, std=None, color=color_2, legend_title='Methods',
#     markersize=11)


# '''Time Indicator T=25 N=3-11'''
# name = []
# path = []
# res = []
# std = []
# ORIGAMIGF = [26, 136, 230, 293,325,337,366]
# std_gf = '0.3	4.3	3.7	5	8	9	9'
# std_gf = tool.algorithm.excelsplit(std_gf)
# # ORIGAMIG = [1.6,56,230,536,924,1372,1930]
# # std_g = '0.3	4.3	3.7	5.5	23.8	58.8	50'
# # std_g = tool.algorithm.excelsplit(std_g)
# ORIGAMIM = [0.3551,5.6955,148.322,2508.868]
# std_m = '0.03	0.57	13.5	26.6'
# std_m = tool.algorithm.excelsplit(std_m)
# ORIGAMIA = [7.407,50.894,636.555,407.6731,2213.7009,308.205]
# std_a = '0.37	4.6	3.7	3	20	5'
# std_a = tool.algorithm.excelsplit(std_a)
# ORIGAMIMBS = '0.0739	0.82	7.291	44.201	376.204'
# ORIGAMIMBS = tool.algorithm.excelsplit(ORIGAMIMBS)
# std_mbs = '0	0.01	0.08	2	10'
# std_mbs = tool.algorithm.excelsplit(std_mbs)
# DIRECTMINCOV = '1.425	10.9944	166.115	60.581	995.666	605.20215	3436.602'
# DIRECTMINCOV = tool.algorithm.excelsplit(DIRECTMINCOV)
# std_dir = '0.09	0.13	15.6	5	23	16	50'
# std_dir = tool.algorithm.excelsplit(std_dir)
# res.append(ORIGAMIGF)
# # res.append(ORIGAMIG)
# res.append(ORIGAMIM)
# res.append(ORIGAMIA)
# res.append(ORIGAMIMBS)
# res.append(DIRECTMINCOV)
# std.append(std_gf)
# # std.append(std_g)
# std.append(std_m)
# std.append(std_a)
# std.append(std_mbs)
# std.append(std_dir)
# name.append('SDES')
# # name.append('ORIGAMIG')
# name.append('ORIGAMI-M')
# name.append('ORIGAMI-A')
# name.append('ORIGAMI-M-BS')
# name.append('DIRECT-MIN-COV')
# conv = Convergence(name=name, res=res, RES_DIR=RES_DIR)
# # NOTE 展示对比算法随obj的变化
# conv.TimeIndicatorPlot(xaxis='Attacker', xtick=np.arange(3,10), plot_part='both', title='N=3-9, T=25', pdf=True, std=None, color=color_2, legend_title='Methods',
#     markersize=11)


# '''Time Indicator T=5-20 ORIGAMIM-GF'''
# name = []
# path = []
# res = []
# std = []
# name.append('25')
# name.append('50')
# name.append('75')
# name.append('100')
# name.append('200')
# name.append('400')
# name.append('600')
# name.append('800')
# name.append('1000')
# res25 = tool.algorithm.excelsplit('230	293	325	337	366	372	372	404	429	429	429	445	458	483	495 509')
# res50 = tool.algorithm.excelsplit('354	396	414	445	468	502	524	541	548	549	576	561	605	611	635.417 650')
# res75 = tool.algorithm.excelsplit('461	518	551	587	635	665	705	712	744	725	746	726	799	786	860.637 940')
# res100 = tool.algorithm.excelsplit('592	651	672	720	793	819	869	867	916	898	901	910	967	962	1063.266 1113')
# res200 = tool.algorithm.excelsplit('1022	1103	1207	1272	1367	1438.192	1521.798	1558	1687	1711	1728	1739	1839	1814	1852.281 1882')
# res400 = tool.algorithm.excelsplit('1860	2072	2315	2466	2491.356	2688.831	2869.316	2923	3066	3139	3148	3212	3369.459	3450	3516.165 3592')
# res600 = tool.algorithm.excelsplit('2837	3019	3505	3726.613	3819.673	3956.513	4000	4172.546	4418.4217	4505.479	4737	4802.969')
# res800 = tool.algorithm.excelsplit('3729 4034	4006	4379.799	4633.104	4978.931')
# res1000 = tool.algorithm.excelsplit('4452 4501 4799	4919.826')
# res.append(res25)
# res.append(res50)
# res.append(res75)
# res.append(res100)
# res.append(res200)
# res.append(res400)
# res.append(res600)
# res.append(res800)
# res.append(res1000)
# conv = Convergence(name=name, res=res, RES_DIR=RES_DIR)
# # NOTE 展示自己的算法随obj target的变化
# marker= ''
# conv.TimeIndicatorPlot(xaxis='Attacker', xtick=np.arange(5,21), plot_part='both', title='SDES, N=5-20, T=25-1000', std=None, color=color_2,
# pdf=True, loc='upper right', legend_title='Target Number', markersize=9)

# '''Time Indicator T=3-12 SDES (pop_size not fixed)'''
# name = [] 
# path = []
# res = []
# std = []
# name.append('25')
# name.append('50')
# name.append('75')
# name.append('100')
# name.append('200')
# name.append('400')
# name.append('600')
# name.append('800')
# name.append('1000')
# res25 = tool.algorithm.excelsplit('1.6	56	230	536	924	1372	1930	2649	3367	4210')
# res50 = tool.algorithm.excelsplit('3.6	105	352	665	1030	1509	2022	2704	3407	4138')
# res75 = tool.algorithm.excelsplit('5.7	166	463	838	1296	1838	2460	2977	4047	4910') # ?
# res100 = tool.algorithm.excelsplit('6.9	210	586	1033	1555	2192	2756	3368	4711')
# res200 = tool.algorithm.excelsplit('12.8	400	998	1679	2539	3845	4710')
# res400 = tool.algorithm.excelsplit('28.4	749	1813	3408	4985')
# res600 = tool.algorithm.excelsplit('46.3	1181	2787	4578 4933')
# res800 = tool.algorithm.excelsplit('59.1	1532	4034')
# res1000 = tool.algorithm.excelsplit('75.5	1849	4799')
# res.append(res25)
# res.append(res50)
# res.append(res75)
# res.append(res100)
# res.append(res200)
# res.append(res400)
# res.append(res600)
# res.append(res800)
# res.append(res1000)
# conv = Convergence(name=name, res=res, RES_DIR=RES_DIR)
# marker= ''
# conv.TimeIndicatorPlot(xaxis='Attacker', xtick=np.arange(3,13), plot_part='both', title='SDES with increasing pop$\_{}$size, N=3-12, T=25-1000', std=None, color=color_2,
# pdf=True)


'''可视化'''
# NOTE 不画pf 
# name = []
# path = []
# res = []
# std = []
# # method_path = 'Results/obj5target50/GeneticMOSGinteger-Popsize400Coderealgen100-2022_05_16_23_34-28084.txt'
# # pf_path = 'Results/pf/obj5target50/GMOSG7794MOSG960'
# # pf_name = ''
# # pf_path = 'Results/pf/obj5target25'
# # name.append('pf')
# # # method_pf
# # path.append(method_path)

# # NOTE 读SOTA Comparison Algs
# path_comparison = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/PF-Comparison.json')
# # NOTE 读ORIGAMIG(默认读SEED0)
# path_ORIGAMIG = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-N=4GFIndependentRepeatExperiments_SEED0-30.json')
# # NOTE After MINCOV
# path_MINCOV = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json')

# # NOTE Scatter Plot
# # NOTE 
# N=3
# T=50
# SEED = 0
# scale = 'obj'+str(N)+'target'+str(T)
# res1 = tool.algorithm.loadVariPickle(path_MINCOV[path_ORIGAMIG[scale][SEED]])
# # NOTE 由于已经经过MINCOV处理，不需要在做第二阶段
# # model1 = Truing(res=res1)
# # model1.mosgSearch_pop()
# # res1 = model.fit_pf
# res.append(res1)
# name.append('SDES')
# res2 = tool.algorithm.loadVariPickle(path_comparison['ORIGAMIA'][scale]).F
# res.append(res2)
# name.append('ORIGAMI-A')
# # res3 = tool.algorithm.loadVariPickle(path_comparison['ORIGAMIM'][scale]).F
# # res.append(res3)
# # name.append('ORIGAMI-M')
# # NOTE res和name传入conv，画图
# conv = Convergence(name=name, res=res, RES_DIR=RES_DIR)


# fontsize = 15 # label_fontsize, title_fontsize, tick_fontsize
# markersize = 10
# # marker = ['^', 'P', 'O', 'D', '*', 'x']
# label_fontsize = fontsize
# title_fontsize = fontsize
# legend_fontsize = fontsize
# tick_fontsize = fontsize

# NOTE Scatter  
# for表示要绘制的散点图的数量，一般把对比算法视作pf传入，以灰色背景画出，然后将自己的方法以彩色绘出
# for i in range(len(conv.res)):
#     conv.scatter_PF(figsize = (5, 6),
#         res_idx=i,
#         pdf=True,
#         marksize = 60,
#         # ,suptitle="PFs" 
#         # ,legend=(True, {'loc':'upper left'})
#         )
'''要用'''
# conv.scatter_Algs(
#         # figsize = (5, 6),
#         res_idx=list(range(len(name))),
#         pdf=True,
#         marksize = 20,
#         title={'fontsize':title_fontsize},
#         # ,suptitle="PFs" ,
#         legend=(True, {'framealpha':True, 'fontsize':legend_fontsize, 'handlelength':4}),
#         markersize=markersize,
#         )


# NOTE PCP 
# 三段代码对应三段不同的效果，第一段是绘制PF的，第二段是绘制除PF以外的1个res，第三段同时绘制pf（灰色）和当前方法（彩色）
# conv.pcp(res=conv.pf, figsize = (7, 7), 
#         n_ticks=10,
#         title="PFs", 
#         legend=(True, {'loc':'upper left'})
#         ,labels=['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$']
#         )
# for i in range(len(conv.res)):
#     conv.pcp(res=conv.res[i], figsize = (7, 7), 
#         n_ticks=10,
#         title=conv.name[i], 
#         legend=(True, {'loc':'upper left', 'handlelength':4})
#         ,labels=['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$']
#         )
# for i in range(len(conv.res)):
#     conv.pcp_PF(figsize = (7, 7), 
#         res_idx=i,
#         n_ticks=10,
#         # title=conv.name[i], 
#         pdf=True,
#         legend=(True, {'loc':'best', 'handlelength':4}),
#         labels=['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$'],
#         # labels=['$F_1$', '$F_2$', '$F_3$'],
#         )
'''要用'''
# conv.pcp_Algs(
#         # figsize = (7, 7), 
#         res_idx=list(range(len(name))),
#         n_ticks=10,
#         # title=conv.name[i], 
#         pdf=True,
#         legend=(True, {'loc':'best', 'handlelength':4}),
#         # labels=['$F_1$', '$F_2$', '$F_3$'],
#         # labels=['$F_1$', '$F_2$', '$F_3$'],
#         )

# NOTE Radviz
# pass
'''需要花时间调试，就算了'''
# conv.radviz(  title="Optimization",
#               legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
#               labels = ['$F_1$', '$F_2$', '$F_3$'],
#               endpoint_style={"s": 70, "color": "green"})

# NOTE Star Coordinate Plot
# 下面两端代码大致是一致的，第一段是将所有res和pf绘制到一张图，第二段是一图绘制一个res一个pf，绘制res数量次
'''要用'''
# conv.star_coordinate_plot_total(figsize = (7, 7)
#     ,legend=(True, {'loc':'upper left', 'bbox_to_anchor':(-0.1, 1.08, 0, 0)})
#     ,title="Optimization"
#     ,labels = ['$F_1$', '$F_2$', '$F_3$']
#     ,axis_style = {'color':'blue', 'alpha':0.7}
#     ,arrow_style={"head_length": 0.015, "head_width": 0.03})
# for i in range(len(conv.res)):
#     conv.star_coordinate_plot_PF(figsize = (7, 7)
#         ,legend=(True, {'loc':'upper left', 'bbox_to_anchor':(-0.1, 1.08, 0, 0)})
#         ,res_idx=i
#         # ,title= 'PF-' + conv.name[i]
#         ,pdf=False
#         ,labels = ['$F_1$', '$F_2$', '$F_3$']
#         ,axis_style = {'color':'blue', 'alpha':0.7}
#         ,arrow_style={"head_length": 0.015, "head_width": 0.03})


'''下面的图都由于部分原因不可用'''
# # NOTE 热力图变化太剧烈
# # Heatmap
# # pass
# # conv.heatmap(res=conv.pf, figsize = (7, 7), 
# #     title="PSs" 
# #     # ,cmap='Oranges_r'
# #     ,labels=['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$']
# #     )
# # for i in range(len(conv.res)):
# #     conv.heatmap(res=conv.res[i], figsize = (7, 7), 
# #     title=conv.name[i], 
# #     cmap='Oranges_r'
# #     ,labels=['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$']
# #     )

# # petal_diagrame
# # pass
# # conv.petal_diagrame(figsize = (7, 7), 
# #     title="PFs", 
# #     legend=(True, {'loc':'upper left'}))

# # radar_plot
# # pass
# # conv.radar_plot(figsize = (7, 7), 
# #     title="PFs", 
# #     legend=(True, {'loc':'upper left'}))


print(__name__, 'finish')

'''在main程序中执行同时绘制所有图'''
# NOTE PSP Scatter
if __name__ == '__main__':
    RES_DIR = ['Results','visualization', 'ScatterPlot_2']
    # NOTE 读SOTA Comparison Algs
    path_comparison = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/PF-Comparison.json')
    # NOTE 读ORIGAMIG(默认读SEED0)
    path_ORIGAMIG = tool.algorithm.loadVariJson('Results_Dir/GeneticMOSGinteger-res_dir-N=4GFIndependentRepeatExperiments_SEED0-30.json')
    # NOTE After MINCOV
    path_mincov = 'Results_Dir/GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json'
    path_MINCOV = tool.algorithm.loadVariJson(path_mincov)
    # N_TODO = [3, 4, 5, 6, 7, 8]
    N_TODO = [4,5]
    # T_TODO = [25, 50, 75, 100, 200, 400, 600, 800, 1000]
    T_TODO = [100, 400]
    rank_key = 'hv'
    # rank_key = 'igd+'
    rank_value = []
    # SEED_TODO = range(30)
    SEED_TODO = [7, 15]
    
    fontsize = 16 # label_fontsize, title_fontsize, tick_fontsize
    markesize = [60, 30, 30, 30, 30, 30, 30]
    # marker = ['^', 'P', 'O', 'D', '*', 'x']
    title_fontsize = fontsize +14
    legend_fontsize = 12+8
    label_fontsize = [fontsize-4, fontsize+10, fontsize+10, fontsize, fontsize, fontsize, fontsize]
    tick_fontsize = [fontsize-2, None, None, None, None, None, None]
    tick_fontsize = [None]
    legend = [True, True, True, True, True, True]
    figsize = [(5, 5), (10, 10), (12.5, 12.5), (15, 15), (17.5, 17.5), (20, 20)] # (length, width)
    fontstyle1 = [None, None, {'family':'serif', 'size':30}, ]
    legend_loc = [0]  # List[int] 存储需要现实legend的子图（N*N个子图计数方式是从0开始到N*N-1 -- flatten）
    for Ni, N in enumerate(N_TODO):
        # if N != 5:
        #     continue
        for T in T_TODO:
            Algs = []
            scale = 'obj'+str(N)+'target'+str(T)
            path_indicator_Algs = os.path.join('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg', scale)
            if not os.path.exists(path_indicator_Algs):
                continue
            indicator:Dict[str, Dict] = tool.algorithm.loadVariJson(path_indicator_Algs)
            for dict1 in indicator.values():
                for alg, dict2 in dict1.items():
                    rank_value.append(dict2[rank_key])
                    Algs.append(alg)
            if rank_key == 'igd+':
                best = int(np.argsort(rank_value)[0])
            else:
                best = int(np.argsort(rank_value)[-1])
            Alg = Algs[best] # SOTA comparison alg
            res_Alg = tool.algorithm.loadVariPickle(path_comparison[Alg][scale]).F
            for SEED in SEED_TODO:
                name = []
                res = []
                path_res = path_ORIGAMIG[scale][SEED]
                if path_res in path_MINCOV.keys():
                    # NOTE 若已经经过MINCOV处理，不需要在做第二阶段
                    res1 = tool.algorithm.loadVariPickle(path_MINCOV[path_res])
                else:
                    pass
                    # res1 = tool.algorithm.loadVariPickle(path_res)
                    # model1 = Truing(res=res1)
                    # model1.mosgSearch_pop()
                    # res1 = model1.fit_pf
                    # # 算出来不忘保存
                    # path_key = path_res
                    # fname = path_res.split('/')[-1]
                    # res_dir = 'Results/optimalAfterMINCOV'
                    # path_value = os.path.join(res_dir, fname)
                    # path_MINCOV[path_key] = path_value
                    # tool.algorithm.dumpVariJson(vari=path_MINCOV, name=path_mincov)
                    # tool.algorithm.dumpVariPickle(vari=res1, name=path_value)
                res.append(res1)
                name.append('SDES')
                res.append(res_Alg)
                if Alg == 'ORIGAMIA':
                    name.append('ORIGAMI-A')
                if Alg == 'ORIGAMIM':
                    name.append('ORIGAMI-M')
                if Alg == 'ORIGAMIMBS':
                    name.append('ORIGAMI-M-BS')
                if Alg == 'DIRECTMINCOV':
                    name.append('DIRECT-MIN-COV')
                conv = Convergence(name=name, res=res, RES_DIR=RES_DIR)
                
                para = tool.algorithm.paras2Str({'SEED':SEED, 'N':N, 'T':T, 'Indi':rank_key})
                conv.scatter_Algs(
                    figsize = figsize[N-3],
                    res_idx=list(range(len(name))[::-1]),
                    pdf=True,
                    marksize = markesize[N-3],
                    # title=('Attacker:'+str(N)+', Target:'+str(T), {'fontsize':title_fontsize}), # 小图标题，用的少
                    # 大图统一标题
                    fig_title={'Attacker':N, 'Target':T}, title_fontsize=title_fontsize, title_latex=False, # 给大标题用的字体，由于大标题不能直接传入fontdict
                    legend_loc=legend_loc, legend=(legend[N-3], {'framealpha':True, 'fontsize':legend_fontsize, 'title':'Methods', 'title_fontsize':legend_fontsize}), # NOTE 通知legend与文字的距离'handlelength':4
                    para = para, # 给文件命名的Dict，key包括：N、T、SEED、Indicator
                    label_fontsize=label_fontsize[N-3], # 这两个参数在scatter的_do()中调
                    fontstyle1 = fontstyle1[N-3],  # 控制对角线空图fi分图字体样式
                    # tick_fontsize=tick_fontsize[N-3], # 暂时用默认的就好
                    # suptitle='N=, ' + str(N) + 'T=' +str(T), # 未知作用
                    # sharex=True,sharey=True,
                    )
                

'''在main程序中执行同时绘制所有图'''
# # NOTE PCP
# if __name__ == '__main__':
#     RES_DIR = ['Results','visualization', 'PCP_2']
#     # NOTE 读SOTA Comparison Algs
#     path_comparison = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/PF-Comparison.json')
#     # NOTE 读ORIGAMIG(默认读SEED0)
#     path_ORIGAMIG = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-N=4GFIndependentRepeatExperiments_SEED0-30.json')
#     # NOTE After MINCOV
#     path_mincov = 'GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json'
#     path_MINCOV = tool.algorithm.loadVariJson(path_mincov)
#     # N_TODO = [3, 4, 5, 6, 7, 8]
#     N_TODO = [6]
#     # T_TODO = [25, 50, 75, 100, 200, 400, 600, 800, 1000]
#     T_TODO = [100]
#     rank_key = 'hv'
#     # rank_key = 'igd+'
#     rank_value = []
#     SEED_TODO = range(30)
    
#     fontsize = 16 # label_fontsize, title_fontsize, tick_fontsize
#     markesize = [60, 30, 30, 30, 30, 30, 30]
#     # marker = ['^', 'P', 'O', 'D', '*', 'x']
#     title_fontsize = fontsize -2 
#     legend_fontsize = 12
#     label_fontsize = [fontsize-4, fontsize, fontsize, fontsize, fontsize, fontsize, fontsize]
#     tick_fontsize = [fontsize-2, None, None, None, None, None, None]
#     legend = [True, True, True, True, True, True]
#     figsize = [(5, 5), (10, 10), (12.5, 12.5), (15, 15), (17.5, 17.5), (20, 20)] # (length, width)
#     legend_loc = [0]  # List[int] 存储需要现实legend的子图（N*N个子图计数方式是从0开始到N*N-1 -- flatten）
#     for Ni, N in enumerate(N_TODO):
#         # if N != 5:
#         #     continue
#         for T in T_TODO:
#             Algs = []
#             scale = 'obj'+str(N)+'target'+str(T)
#             path_indicator_Algs = os.path.join('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg', scale)
#             if not os.path.exists(path_indicator_Algs):
#                 continue
#             indicator:Dict[str, Dict] = tool.algorithm.loadVariJson(path_indicator_Algs)
#             for dict1 in indicator.values():
#                 for alg, dict2 in dict1.items():
#                     rank_value.append(dict2[rank_key])
#                     Algs.append(alg)
#             if rank_key == 'igd+':
#                 best = int(np.argsort(rank_value)[0])
#             else:
#                 best = int(np.argsort(rank_value)[-1])
#             Alg = Algs[best] # SOTA comparison alg
#             res_Alg = tool.algorithm.loadVariPickle(path_comparison[Alg][scale]).F
#             for SEED in SEED_TODO:
#                 name = []
#                 res = []
#                 path_res = path_ORIGAMIG[scale][SEED]
#                 if path_res in path_MINCOV.keys():
#                     # NOTE 若已经经过MINCOV处理，不需要在做第二阶段
#                     res1 = tool.algorithm.loadVariPickle(path_MINCOV[path_res])
#                 else:
#                     res1 = tool.algorithm.loadVariPickle(path_res)
#                     model1 = Truing(res=res1)
#                     model1.mosgSearch_pop()
#                     res1 = model1.fit_pf
#                     # 算出来不忘保存
#                     path_key = path_res
#                     fname = path_res.split('/')[-1]
#                     res_dir = 'Results/optimalAfterMINCOV'
#                     path_value = os.path.join(res_dir, fname)
#                     path_MINCOV[path_key] = path_value
#                     tool.algorithm.dumpVariJson(vari=path_MINCOV, name=path_mincov)
#                     tool.algorithm.dumpVariPickle(vari=res1, name=path_value)
#                 res.append(res1)
#                 name.append('SDES')
#                 res.append(res_Alg)
#                 if Alg == 'ORIGAMIA':
#                     name.append('ORIGAMI-A')
#                 if Alg == 'ORIGAMIM':
#                     name.append('ORIGAMI-M')
#                 if Alg == 'ORIGAMIMBS':
#                     name.append('ORIGAMI-M-BS')
#                 if Alg == 'DIRECTMINCOV':
#                     name.append('DIRECT-MIN-COV')
#                 conv = Convergence(name=name, res=res, RES_DIR=RES_DIR)
                
#                 para = tool.algorithm.paras2Str({'N':N, 'T':T, 'Indi':rank_key, 'SEED':SEED})
#                 conv.pcp_Algs(
#                         # figsize = (7, 7), 
#                         res_idx=list(range(len(name))[::-1]),
#                         n_ticks=10,
#                         # title=conv.name[i], # 给ax加title，效果同下（因为图中只有一个ax）
#                         # 给fig加title，大图统一标题
#                         fig_title={'Attacker':N, 'Target':T}, title_fontsize=title_fontsize, title_latex=False,
#                         pdf=True,
#                         legend=(True, {'loc':'best', 'fontsize':legend_fontsize, 'title':'Methods', 'title_fontsize':legend_fontsize, 'framealpha':0.5}),
#                         para=para,
#                         # labels=['$F_1$', '$F_2$', '$F_3$'], # 默认的label，若需自定义则传入labels
#                         label_fontsize=label_fontsize[Ni], # 这两个参数在scatter的_do()中调
#                         # tick_fontsize=tick_fontsize[Ni], # 暂时用默认的就好
#                     )