import sys
import matplotlib
import numpy as np
import os

from sklearn.tree import plot_tree
sys.path.append('./')
from tkinter import N
from tkinter.messagebox import NO
from typing import *

from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP
from pymoo.visualization.heatmap import Heatmap
from pymoo.visualization.petal import Petal
from pymoo.visualization.radar import Radar
from pymoo.visualization.radviz import Radviz
from pymoo.visualization.star_coordinate import StarCoordinate
from pymoo.model.result import Result
from pymoo.MOSGsGeneticSolver.performance import Performance
import tool.algorithm


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
        self.RES_DIR = RES_DIR
        self.fname = fname

    def heatmap(data, row_labels, col_labels, ax=None,
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


    def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
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
        plt.savefig(figsize=figsize, path=fname)



class Convergence():
    def __init__(self, pf:str=None ,name:List[str]=None, path:List[str]=None, pf_name='PF', RES_DIR:List[str]=['Results','figures']):
        self.res:List[np.ndarray] = []
        if pf is not None:
            pf_res:Tuple[Result, np.ndarray] = tool.algorithm.loadVariPickle(pf)
            if isinstance(pf_res, np.ndarray):
                self.pf = pf_res
            else:
                self.pf = pf.F
        if path is not None:
            for p in path:
                res:Tuple[Result, np.ndarray] = tool.algorithm.loadVariPickle(p)
                if isinstance(res, np.ndarray):
                    self.res.append(res)
                else:
                    self.res.append(res.F)
        self.pf_name = pf_name
        self.marker = tool.algorithm.getMarker()[:len(name)+1]
        self.color = tool.algorithm.getColor()[:len(name)+1]
        # 可微调颜色 图例

        self.name = name
        self.RES_DIR = os.getcwd()
        for dir in RES_DIR:
            self.RES_DIR = os.path.join(self.RES_DIR, dir)

    '''将resA和resB一同打在同一幅图中
    这里称resA=pf'''
    def scatter_PF(self, res_idx=None, **kwargs):
        plot_type = 'scatter'
        sca = ScatterPlot(**kwargs)
        sca.add(self.pf, color='gray', marker=self.marker[-1], alpha=0.3)
        res_name = 'PSs'
        if res_idx is not None:
            sca.add(self.res[res_idx], marker=self.marker[res_idx], color=self.color[res_idx], label=self.name[res_idx])
            res_name = res_name + self.name[res_idx]
        fname = '-'.join([tool.algorithm.getTime(), plot_type, res_name])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

    '''作图意义不大
    ideal可在展示pcp时画出，nadir没有实际含义'''
    def ideal_nadir(self, **kwargs):
        plot_type = 'ideal_nadir'
        func = [1, 2, 3, 4, 5]
        pf_ideal = np.min(self.pf, axis=0)
        pf_nadir = np.max(self.pf, axis=0)
        plt.xlabel('$Objectives$')
        plt.ylabel("$U^d$")
        plt.xticks(ticks=func, labels=['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$'])
        plt.plot(func, pf_ideal, 'greenyellow', marker='*', markersize=10, label='PF(Ideal)', linestyle='-.')
        plt.plot(func, pf_nadir, 'greenyellow', marker='*', markersize=10, label='PF(Nadir)', linestyle='-')
        for i in range(len(func)):
            plt.text(func[i], pf_ideal[i]-0.5, round(pf_ideal[i], 2), ha='center', va='top', fontsize=10)
            plt.text(func[i], pf_nadir[i]+0.2, round(pf_nadir[i], 2), ha='center', va='bottom', fontsize=10)
        marker = ['P', 'd', '>', '3', 'x', 'X', ',']
        color =  ['b', 'g', 'c', 'y', 'k', 'm', 'w']
        for i in range(len(self.name)):
            ideal = []
            nadir = []
            id = np.min(self.res[i], axis=0)
            na = np.max(self.res[i], axis=0)
            ideal.append(id)
            nadir.append(na)
            plt.plot(func, id, color[i], marker=marker[i], markersize=10, label=name[i], linestyle='-.')
            plt.plot(func, na, color[i], marker=marker[i], markersize=10, linestyle='-')
        
        for i in range(len(self.name)):
            pass
            
        plt.legend(loc='best', handlelength=4)
        
        fname = '-'.join([tool.algorithm.getTime(), plot_type])
        plt.savefig(fname=os.path.join(self.RES_DIR, fname))

    '''只画单一PF的pcp
    只需要提供res参数'''
    def pcp(self, res, **kwargs):
        plot_type = 'parallel_coordinate_plots'
        sca = PCPPlot(**kwargs)
        sca.set_axis_style(color='grey', alpha=1)
        sca.add(res, color='grey', alpha=0.3)
        ideal = np.min(res, axis=0)
        nadir = np.max(res, axis=0)
        sca.add(ideal, linewidth=5, color='grey', linestyle='-.', label=self.pf_name+'(Ideal)')
        sca.add(nadir, linewidth=5, color='grey', linestyle='-', label=self.pf_name+'(Nadir)')
        fname = '-'.join([tool.algorithm.getTime(), plot_type, kwargs['title']])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

    '''画两个pcp的重叠，用于比较
    在默认的情况下，比较对象A为PF，如果是该参数可省略，也可以传入自定义对象；B是当前讨论的对象(res[res_idx])
    '''
    def pcp_PF(self, resA_idx=None, resB_idx=None, **kwargs):
        resB = self.res[resB_idx]
        if resA_idx is not None:
            resA = self.res[resA_idx]
        plot_type = 'parallel_coordinate_plots'
        sca = PCPPlot(**kwargs)
        # pf and another method's pcp_PF
        sca.set_axis_style(color='grey', alpha=1)

        if resA_idx is None:
            sca.add(self.pf, color='grey', alpha=0.3)
        else:
            sca.add(resA, color=self.color[resA_idx], alpha=1)
        sca.add(resB, color=self.color[resB_idx], alpha=1)
        # Ideal Nadir
        if resA_idx is None:
            ideal_pf = np.min(self.pf, axis=0)
            nadir_pf = np.max(self.pf, axis=0)
        else:
            ideal_pf = np.min(resA, axis=0)
            nadir_pf = np.max(resA, axis=0)
        ideal_res = np.min(resB, axis=0)
        nadir_res = np.max(resB, axis=0)
        if resA_idx is None:
            sca.add(ideal_pf, linewidth=3, color='grey', linestyle='-.', label=self.pf_name+'(Ideal)')
            sca.add(nadir_pf, linewidth=3, color='grey', linestyle='-', label=self.pf_name+'(Nadir)')
        else:
            sca.add(ideal_res, linewidth=3, color=self.color[resA_idx], linestyle='-.', label=self.name[resA_idx])
            sca.add(nadir_res, linewidth=3, color=self.color[resA_idx], linestyle='-', label=self.name[resA_idx])
        sca.add(ideal_res, linewidth=3, color=self.color[resB_idx], linestyle='-.', label=self.name[resB_idx])
        sca.add(nadir_res, linewidth=3, color=self.color[resB_idx], linestyle='-', label=self.name[resB_idx])
        fname = '-'.join([tool.algorithm.getTime(), plot_type, kwargs['title']])
        sca.save(fname=os.path.join(self.RES_DIR, fname))


    def heatmap(self, res, **kwargs):
        plot_type = 'heatmap'
        sca = HeatmapPlot(**kwargs)
        sca.add(res)
        fname = '-'.join([tool.algorithm.getTime(), plot_type, kwargs['title']])
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

    '''绘制单一pf的radviz图像'''
    def radviz(self, **kwargs):
        plot_type = 'radviz'
        sca = RadvizPlot(**kwargs)
        sca.set_axis_style(color="black", alpha=1)
        # sca.add(self.pf, color='gray', s=20, alpha=0.3)
        for i in range(len(self.name)):
            sca.add(self.res[i], color=self.color[i], s=70, label=self.name[i])
        fname = '-'.join([tool.algorithm.getTime(), plot_type])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

    '''把list中所有的pf都画出来'''
    def star_coordinate_plot_total(self, **kwargs):
        plot_type = 'star_coodinate_plot'
        sca = StarCoordinatePlot(**kwargs)
        sca.add(self.pf, color='gray', s=20, alpha=0.3)
        for i in range(len(self.name)):
            sca.add(self.res[i], color=self.color[i], s=70, label=self.name[i])
        fname = '-'.join([tool.algorithm.getTime(), plot_type])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

    '''将resA和resB一同画出，其中resA为pf'''
    def star_coordinate_plot_PF(self, res_idx=None, **kwargs):
        plot_type = 'star_coodinate_plot'
        sca = StarCoordinatePlot(**kwargs)
        res_name = 'PF'
        if res_idx is not None:
            res_name += self.name[res_idx]
        sca.add(self.pf, color='gray', s=20, alpha=0.3)
        sca.add(self.res[res_idx], color=self.color[res_idx], s=70, label=self.name[res_idx])
        fname = '-'.join([tool.algorithm.getTime(), plot_type, res_name])
        sca.save(fname=os.path.join(self.RES_DIR, fname))

'''主程序分为以下几大块：
1.读入数据
2.作图scatter pcp radviz star_coordinate_plot
'''
# 读入数据
RES_DIR = ['Results','demo']
name = []
path = []
# mosg
path_mosg = 'Results/pf_gmosg/pf_mosg'
name_mosg = 'mosg'
# gmosg
path_gmosg = 'Results/gmosgInt/CodeintegerX0Ter2'
name_gmosg = 'CodeIntegerX0Ter2'

# 做图1：scatter
name.append(name_gmosg)
path.append(path_gmosg)
conv1 = Convergence(path_mosg, name=name, path=path, pf_name=name_mosg, RES_DIR=RES_DIR)
# Scatter
for i in range(len(conv1.res)):
    conv1.scatter_PF(figsize = (14, 14) 
        ,res_idx=i
        # ,suptitle="PFs" 
        # ,legend=(True, {'loc':'upper left'})
        )

# 作图2：pcp
# # PCP
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
for i in range(len(conv1.res)):
    conv1.pcp_PF(figsize = (7, 7),
        resB_idx=i,
        n_ticks=10,
        title=name_mosg + conv1.name[i], 
        legend=(True, {'loc':'upper left', 'handlelength':4})
        ,labels=['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$']
        )

# 作图3：radviz
# conv1.radviz(  title="Optimization",
#               legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
#               labels = ['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$'],
#               endpoint_style={"s": 70, "color": "green"})

# 作图4：Star Coordinate Plot
# conv.star_coordinate_plot_total(figsize = (7, 7)
#     ,legend=(True, {'loc':'upper left', 'bbox_to_anchor':(-0.1, 1.08, 0, 0)})
#     ,title="Optimization"
#     ,labels = ['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$']
#     ,axis_style = {'color':'blue', 'alpha':0.7}
#     ,arrow_style={"head_length": 0.015, "head_width": 0.03})
for i in range(len(conv1.res)):
    conv1.star_coordinate_plot_PF(figsize = (7, 7)
        ,legend=(True, {'loc':'upper left', 'bbox_to_anchor':(-0.1, 1.08, 0, 0)})
        ,res_idx=i
        ,title="$PF-" + conv1.name[i] + '$'
        ,labels = ['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$']
        ,axis_style = {'color':'blue', 'alpha':0.7}
        ,arrow_style={"head_length": 0.015, "head_width": 0.03})

# ideal nadir
# conv.ideal_nadir()

print(__name__, 'finish')

