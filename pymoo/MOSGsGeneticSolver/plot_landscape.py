import sys
sys.path.append('./')
import json
import os

from pymoo.visualization.petal import Petal
from pymoo.visualization.radviz import Radviz
from pymoo.visualization.pcp import PCP
from pymoo.visualization.radar import Radar
from pymoo.visualization.star_coordinate import StarCoordinate
import matplotlib as mpl

from multiprocessing.pool import TERMINATE
from re import T
from select import select
from tkinter import N
from tracemalloc import stop
from turtle import Turtle
from typing import Dict
from matplotlib import projections
# from pymoo.MOSGsGeneticSolver.convergence import HeatmapPlot,HeatmapAnnotation
from pymoo.visualization.heatmap import Heatmap
import matplotlib

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

from pymoo.util.termination import max_gen
from typing import *

import tool.algorithm
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions, get_sampling, get_crossover, get_mutation, get_termination
from pymoo.model.result import Result
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.MOSGsGeneticSolver.visualization import Visualization
from pymoo.MOSGsGeneticSolver.resSave import resSave

from pymoo.algorithms.mosg_genetic_nsga3 import MOSGG
from securitygame_core.MO_security_game import MOSG
from pymoo.MOSGsGeneticSolver.truing import Truing

RES_DIR = os.path.join(os.getcwd(), 'Results', 'figures')


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

class Landscape():
    def __init__(self, RES_DIR=['Results', 'landscape_demo']):
        self.RES_DIR = os.getcwd()
        for i in range(len(RES_DIR)):
            self.RES_DIR = os.path.join(self.RES_DIR, RES_DIR[i])

    # U-ct
    def U_ct_3d(self):
        I = 0
        problem1 = MOSG(player_num=2, target_num=1)
        x = np.linspace(start=0, stop=1, num=10)
        payoff_attacker = []
        payoff_defender = []
        for xi in x:
            problem1.set_ct(xi)
            problem1.cal_payoff()
            problem1.cal_payoff_attacker()
            problem1.cal_payoff_defender()
            payoff_attacker.append(problem1.get_payoff_attacker()[0])
            payoff_defender.append(problem1.get_payoff_defender()[0])
        plt.xlabel('$c_t$')
        plt.ylabel('$U^{a/d}$')
        plt.plot(x, payoff_attacker, label='$U^a(c_t)$')
        plt.plot(x, payoff_defender, label='$U^d(c_t)$')
        plt.legend(loc='best')
        fname = 'U-ct'
        plt.savefig(os.path.join(self.RES_DIR, fname))

    # 绘制一张U-c的3d，c变量，3d中变量c的维度为2
    def U_c_3d_rainbow(self, rstride=1):
        problem2 = MOSG(player_num=2, target_num=2)
        x1 = np.linspace(0, 1, 100)
        x2 = np.linspace(0, 1, 100)
        # 网格化操作，将原有1维的x1,x2变成2维
        X1, X2 = np.meshgrid(x1, x2)
        payoff_attacker = np.empty_like(X1)
        payoff_defender = np.empty_like(X1)
        for i in range(len(x1)):
            for j in range(len(X2)):
                problem2.set_ct(np.array([x1[i], x2[j]]))
                problem2.cal_payoff()
                problem2.cal_payoff_attacker()
                problem2.cal_payoff_defender()
                payoff_attacker[i, j] = problem2.get_payoff_attacker()
                payoff_defender[i, j] = problem2.get_payoff_defender()
        # 多子图
        fname='U-c-rainbow'
        fig = plt.figure()
        fig.suptitle(fname)
        fname = tool.algorithm.getTime() + fname
        ax1 = plt.subplot2grid((1,2), (0,0), projection='3d')
        ax2 = plt.subplot2grid((1,2), (0,1), projection='3d')
        # 基本设定
        ax1.set_xlabel('$c_1$')
        ax1.set_ylabel('$c_2$')
        ax1.set_zlabel('$U^{a}$')
        ax2.set_xlabel('$c_1$')
        ax2.set_ylabel('$c_2$')
        ax2.set_zlabel('$U^{d}$')
        ax1.set_title('$U^a(c)$')
        ax2.set_title('$U^d(c)$')
        # 绘图
        ax1.plot_surface(X1, X2, payoff_attacker, rstride=rstride, cmap='rainbow', label='$U^a(c)$')
        ax2.plot_surface(X1, X2, payoff_defender, rstride=rstride, cmap='rainbow', label='$U^d(c)$')
        # plt.legend(loc='best')
        plt.savefig(os.path.join(self.RES_DIR, fname))

    def U_c_3d_wireframe(self, rstride=5,  cstride=5):
        problem2 = MOSG(player_num=2, target_num=2)
        x1 = np.linspace(0, 1, 10)
        x2 = np.linspace(0, 1, 10)
        # 网格化操作，将原有1维的x1,x2变成2维
        X1, X2 = np.meshgrid(x1, x2)
        payoff_attacker = np.empty_like(X1)
        payoff_defender = np.empty_like(X1)
        for i in range(len(x1)):
            for j in range(len(X2)):
                problem2.set_ct(np.array([x1[i], x2[j]]))
                problem2.cal_payoff()
                problem2.cal_payoff_attacker()
                problem2.cal_payoff_defender()
                payoff_attacker[i, j] = problem2.get_payoff_attacker()
                payoff_defender[i, j] = problem2.get_payoff_defender()
        for i in range(len(x1)):
            print(payoff_defender[i,:])
        # # 多子图
        # fname='U-c-wireframe'
        # fname = tool.algorithm.getTime() + fname
        # fig = plt.figure()
        # fig.suptitle(fname)

        # ax1 = plt.subplot2grid((1,2), (0,0), projection='3d')
        # ax2 = plt.subplot2grid((1,2), (0,1), projection='3d')
        # # 基本设定
        # ax1.set_xlabel('$c_1$')
        # ax1.set_ylabel('$c_2$')
        # ax1.set_zlabel('$U^{a}$')
        # ax2.set_xlabel('$c_1$')
        # ax2.set_ylabel('$c_2$')
        # ax2.set_zlabel('$U^{d}$')
        # ax1.set_title('$U^a(c)$')
        # ax2.set_title('$U^d(c)$')
        # # 绘图
        # ax1.plot_wireframe(X1, X2, payoff_attacker, rstride=rstride, cstride=cstride, label='$U^a(c)$')
        # ax2.plot_wireframe(X1, X2, payoff_defender, rstride=rstride, cstride=cstride, label='$U^d(c)$')
        # # plt.legend(loc='best')
        # plt.savefig(os.path.join(self.RES_DIR, fname))

    def U_c_heatmap(self, **kwargs):
        # 求Z值
        problem2 = MOSG(player_num=2, target_num=2)
        x1 = np.linspace(0, 1, 10)
        x2 = np.linspace(0, 1, 10)
        # 网格化操作，将原有1维的x1,x2变成2维
        X1, X2 = np.meshgrid(x1, x2)
        payoff_attacker = np.empty_like(X1)
        payoff_defender = np.empty_like(X1)
        for i in range(len(x1)):
            for j in range(len(X2)):
                problem2.set_ct(np.array([x1[i], x2[j]]))
                problem2.cal_payoff()
                problem2.cal_payoff_attacker()
                problem2.cal_payoff_defender()
                payoff_attacker[i, j] = problem2.get_payoff_attacker()
                payoff_defender[i, j] = problem2.get_payoff_defender()
        
        # Heatmap
        plot_type = 'heatmap'
        sca = HeatmapPlot(**kwargs)
        sca.add(payoff_defender)
        fname = '-'.join([tool.algorithm.getTime(), plot_type, kwargs['title']])
        sca.save(fname=os.path.join(self.RES_DIR, fname))
        

    # 绘制N张旋转的U-c的3d，c变量，3d中变量c的维度为2
    def U_c_3d_ratote(self):
        # 存在问题
        pass
        # 声明问题，定义c[c1, c2]，求U
        problem2 = MOSG(player_num=2, target_num=2)
        x1 = np.linspace(0, 1, 100)
        x2 = np.linspace(0, 1, 100)
        # 网格化操作，将原有1维的x1,x2变成2维
        X1, X2 = np.meshgrid(x1, x2)
        payoff_attacker = np.empty_like(X1)
        payoff_defender = np.empty_like(X1)
        for i in range(len(x1)):
            for j in range(len(X2)):
                problem2.set_ct(np.array([x1[i], x2[j]]))
                problem2.cal_payoff()
                problem2.cal_payoff_attacker()
                problem2.cal_payoff_defender()
                payoff_attacker[i, j] = problem2.get_payoff_attacker()
                payoff_defender[i, j] = problem2.get_payoff_defender()
        
        # 绘图基本信息  # 多子图
        fname='U-c'
        fig = plt.figure()
        fig.suptitle(fname)
        ax1 = plt.subplot2grid((1,2), (0,0), projection='3d')
        ax2 = plt.subplot2grid((1,2), (0,1), projection='3d')
        ax1.set_xlabel('$c_1$')
        ax1.set_ylabel('$c_2$')
        ax1.set_zlabel('$U^{a}$')
        ax2.set_xlabel('$c_1$')
        ax2.set_ylabel('$c_2$')
        ax2.set_zlabel('$U^{d}$')
        ax1.set_title('$U^a(c)$')
        ax2.set_title('$U^d(c)$')

        # 旋转不同角度绘制
        plt.ion()
        for i in range(30000):
            plt.clf()  # 清除之前画的图
            fig = plt.gcf()  # 获取当前图
            # ax1 = fig.gca(projection='3d')  # 获取当前轴
            # ax2 = fig.gca(projection='3d')  # 获取当前轴
            # 绘图
            ax1.plot_surface(X1, X2, payoff_attacker, rstride=1, cmap='rainbow', label='$U^a(c)$')
            ax2.plot_surface(X1, X2, payoff_defender, rstride=1, cmap='rainbow', label='$U^d(c)$')
            # plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
            plt.savefig(self.RES_DIR)
            print(i)
            plt.ioff()  # 关闭画图窗口

            # Z = Z - X1 + 2 * X2  # 变换Z值

        # 加这个的目的是绘制完后不让窗口关闭

# lands = Landscape()
lands = Landscape(RES_DIR=['Results', '20220818'])
# for i in range(1, 10, 1):
#     lands.U_c_3d_rainbow(rstride=i)
#     lands.U_c_3d_wireframe(rstride=i, cstride=i)

# lands.heatmap(figsize = (7, 7), 
#     title="PSs" 
#     # ,cmap='Oranges_r'
#     ,labels=['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$']
#     )

lands.U_c_heatmap(figsize = (7, 7), 
    title='$U^d(c)$', 
    cmap='Oranges_r'
    # ,labels=['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$']
    )

# lands.U_c_3d_wireframe()

# 声明问题，定义c[c1, c2]，求U
# problem2 = MOSG(player_num=2, target_num=2)
# x1 = np.linspace(0, 1, 11)  # np.linspace包含头也包含尾, len(x1)=len(x2)=11
# x2 = np.linspace(0, 1, 11)
# x1_str = [str(i) for i in x1]
# x2_str = [str(i) for i in x2]
# # 网格化操作，将原有1维的x1,x2变成2维
# X1, X2 = np.meshgrid(x1, x2)
# payoff_attacker = np.empty_like(X1)  # size(payoff_attacker)=(11,11)
# payoff_defender = np.empty_like(X1)
# for i in range(len(x1)):
#     for j in range(len(x2)):
#         if x1[i]+x2[i]>problem2.resource:
#             payoff_attacker[i, j] = 10
#             payoff_defender[i, j] = -10
#             continue

#         problem2.set_ct(np.array([x1[i], x2[j]]))
#         problem2.cal_payoff()
#         problem2.cal_payoff_attacker()
#         problem2.cal_payoff_defender()
#         payoff_attacker[i, j] = problem2.get_payoff_attacker()  # size(payoff_attacker)=(11,11)
#         payoff_defender[i, j] = problem2.get_payoff_defender()
# heatmap = HeatmapAnnotation(fname='Ud')
# heatmap.heatmapAnnotation(z=payoff_defender, x=x1_str, y=x2_str)



print(__name__, ' finish')
