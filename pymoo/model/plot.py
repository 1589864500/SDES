# TODO LIST FOT PLOTTING SUB_FIGURE
# DONE1,2,3
# TODO:1 共享坐标轴xy，只显示最左边的y轴，最底部的x轴。
# TODO:2 子图间间距调整： 在贡献坐标轴后理想的展现效果是子图间无空隙
# TODO:3 fig画布的总标题位置调整


import importlib
from functools import reduce
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import Figure
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
# plt.rc('font', family='Times New Roman')
plt.rc('font', family='serif') # NOTE serif字体看起来确实好看

import sys
sys.path.append('./')

from pymoo.util.misc import set_if_none
from pymoo.visualization.util import default_number_to_text, in_notebook

class Plot:

    def __init__(self,
                 fig=None,
                 ax=None,
                 figsize=(8, 6),
                 title=None,
                 legend=False,
                 tight_layout=False,
                 bounds=None,
                 reverse=False,
                 cmap="tab10",
                 axis_style=None,
                 axis_label_style=None,
                 func_number_to_text=default_number_to_text,
                 labels="f",
                 sharex=False,
                 sharey=False,
                 hspace=None,
                 wspace=None,
                 grid_space=None,
                 # NOTE 后来加入的
                 label_fontsize=None, tick_fontsize=None, title_fontsize=None, legend_loc=None, title_latex=False,
                 fontstyle1=None,
                 **kwargs):

        super().__init__()

        # change the font of plots to serif (looks better)
        plt.rc('font', family='serif') # NOTE serif字体看起来确实好看
        # plt.rc('font', family='Times New Roman')
    
        self.label_fontszie = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.title_fontsize = title_fontsize
        self.title_latex=False
        self.fontstyle1=fontstyle1
        if legend_loc is not None:
            self.legend_loc = legend_loc
        else:
            self.legend_loc = [0] # 默认将legend绘制位置放在第一个ax

        # the matplotlib classes
        self.fig:Figure = fig
        self.ax:plt.axis = ax
        self.figsize = figsize
        self.sharex = sharex
        self.sharey = sharey
        self.hspace = hspace
        self.wspace = wspace
        self.grid_sapce = grid_space

        # NOTE:title属于每个子图，fig_title属于画布
        # the title of the figure - can also be a list if subfigures
        self.title = title
        if title is not None:
            if isinstance(title, Dict): # 这里接受两个类型Dict和List，Dict表示名字，List则会调用set_title
                para:List[str] = [key+':'+str(value) for key, value in title.items()]
                self.title = reduce(lambda x, y: x + ' ' + y, para)
            else:
                self.title = title
        self.fig_title = None
        if 'fig_title' in kwargs.keys():
            para: List[str] = [key + ':' + str(value) for key, value in kwargs['fig_title'].items()]
            if self.title_latex:
                self.fig_title = '$' + reduce(lambda x, y: x + '\ \ ' + y, para) + '$'
            else:
                self.fig_title = reduce(lambda x, y: x + ' ' + y, para)

        # axis_style调整轴样式，axis_label_style调整轴标记样式
        # the style to be used for the axis
        if axis_style is None:
            self.axis_style = {}
        else:
            self.axis_style = axis_style.copy()

        # the style of the axes
        if axis_label_style is None:
            self.axis_label_style = {}
        else:
            self.axis_label_style = axis_label_style.copy()

        # how numbers are represented if plotted
        self.func_number_to_text = func_number_to_text

        # if data are normalized reverse can be applied
        self.reverse = reverse

        # NOTE:self.axis_labels有两个作用，一个是subfigure.axis_labels，一个是subfigure on the diagonal.
        # the labels for each axis
        self.axis_labels = labels

        # the data to plot
        self.to_plot = []

        # whether to plot a legend or apply tight layout
        self.legend = legend
        self.tight_layout = tight_layout

        # the colormap or the color lists to use
        if isinstance(cmap, str):
            self.cmap = matplotlib.cm.get_cmap(cmap)
        else:
            self.cmap = cmap
        if isinstance(self.cmap, ListedColormap):
            self.colors = self.cmap.colors

        # the dimensional of the data
        self.n_dim = None

        # the boundaries for normalization
        self.bounds = bounds

    def init_figure(self, n_rows=1, n_cols=1, plot_3D=False, force_axes_as_matrix=False, 
        title_fontsize=14, y=0.92):
        if self.ax is not None:
            return

        if not plot_3D:
            self.fig, self.ax= plt.subplots(nrows=n_rows, ncols=n_cols, figsize=self.figsize, sharex=self.sharex, sharey=self.sharey)
            if self.hspace is not None or self.wspace is not None:
                self.fig.subplots_adjust(hspace=self.hspace, wspace=self.wspace)
        else:
            importlib.import_module("mpl_toolkits.mplot3d")
            self.fig = plt.figure(figsize=self.figsize)
            self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        # self.fig.suptitle(t=self.fig_title, y=0.92, fontsize=title_fontsize, family='Times New Roman')
        self.fig.suptitle(t=self.fig_title, y=y, fontsize=title_fontsize, family='sans-serif')
        # if there is more than one figure we represent it as a 2D numpy array
        if (n_rows > 1 or n_cols > 1) or force_axes_as_matrix:
            self.ax = np.array(self.ax).reshape(n_rows, n_cols)

    def do(self):

        if len(self.to_plot) > 0:
            # 表现to_plot=[F, {}] = [[[F]], {}]  [e[0].shape[1]=len(F)
            # 效果就是计算F的长度，也就是obj_n
            unique_dim = np.unique(np.array([e[0].shape[1] for e in self.to_plot]))
            if len(unique_dim) > 1:
                raise Exception("Inputs with different dimensions were added: %s" % unique_dim)

            self.n_dim = unique_dim[0]

        # actually call the class
        self._do()

        # convert the axes to a list
        axes = np.array(self.ax).flatten()

        for i, ax in enumerate(axes):
            # 用legend_loc标记需要现实legend的ax（即并非所有ax都显示legend）
            if i in self.legend_loc:
                # 一般i=0
                # 但是当N>3时（即(axes>1)），则写死将ax1的legend给ax0用
                lines = []
                labels = []
                if len(axes) > 1:
                    axLine, axLabel = axes[i+1].get_legend_handles_labels() # 获取axes[i+1]上的所有legend
                    lines.extend(axLine)
                    labels.extend(axLabel)
                
                legend, kwargs = get_parameter_with_options(self.legend)
                if legend:
                    if len(lines) > 0:
                        ax.legend(lines, labels, **kwargs)
                    else:
                        ax.legend(**kwargs)

            title, kwargs = get_parameter_with_options(self.title) # 从元组中获取名字和参数
            if self.title:
                if isinstance(self.title, list):
                    ax.set_title(title[i], **kwargs)
                else:
                    ax.set_title(title, **kwargs)

        if self.tight_layout:
            self.fig.tight_layout()

        return self

    def get_tick_fontsize(self):
        return self.tick_fontsize

    def get_label_fontsize(self):
        return self.label_fontszie
    
    def get_title_fontsize(self):
        return self.title_fontsize

    def get_fontstyle1(self):
        return self.fontstyle1

    def apply(self, func):
        func(self.ax)
        return self

    def get_plot(self):
        return self.ax

    def set_axis_style(self, **kwargs):
        for key, val in kwargs.items():
            self.axis_style[key] = val
        return self

    def reset(self):
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = None
        return self

    # add第二个参数**kwargs再不带参数的情况下为空，也会放到to_plot中，
    # 表现to_plot=[[F, {}]] = [[[7,2,8]], {}]
    def add(self, F, **kwargs):

        if F is None:
            return self
        elif F.ndim == 1:
            self.to_plot.append([F[None, :], kwargs])
        elif F.ndim == 2:
            self.to_plot.append([F, kwargs])
        elif F.ndim == 3:
            [self.to_plot.append([_F, kwargs.copy()]) for _F in F]

        return self

    def plot_if_not_done_yet(self):
        if self.ax is None:
            self.do()

    def show(self, **kwargs):
        self.plot_if_not_done_yet()

        # in a notebook the plot method need not to be called explicitly
        if not in_notebook() and matplotlib.get_backend() != "agg":
            plt.show(**kwargs)
            plt.close()

        return self

    def save(self, fname, **kwargs):
        self.plot_if_not_done_yet()
        set_if_none(kwargs, "bbox_inches", "tight")
        self.fig.savefig(fname, **kwargs)
        return self

    def get_labels(self):
        # 这有两种选择：
        # a)自定义，提供List，同时要求len(List)=obj_n
        # b)默认，axis_labels='f'，但是经过处理实际上返回的是List[str]，其中len=obj_n，str是latex公式，如'$f_{1}$'
        # 因此，若自定义我们也要提供List[str:latex]
        if isinstance(self.axis_labels, list):
            if len(self.axis_labels) != self.n_dim:
                raise Exception("Number of axes labels not equal to the number of axes.")
            else:
                return self.axis_labels
        else:
            return [f"${self.axis_labels}_{{{i}}}$" for i in range(1, self.n_dim + 1)]

    def __del__(self):
        if self.fig is not None:
            plt.close(self.fig)


def get_parameter_with_options(param):
    if param is None:
        return None, None
    else:
        if isinstance(param, tuple):
            val, kwargs = param
        else:
            val, kwargs = param, {}

        return val, kwargs
