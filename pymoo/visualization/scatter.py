# TODO：研究下什么场景下需要绘制多个任务,即len(to_plot)>1
# FIXME
# 每个子图缺少标题，大图缺少标题
# FIXME

from tkinter import font
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import matplotlib.pyplot as plt
# plt.rc('font', family='Times New Roman')
plt.rc('font', family='serif') # NOTE serif字体看起来确实好看

from pymoo.docs import parse_doc_string
from pymoo.model.plot import Plot
from pymoo.util.misc import set_if_none




class Scatter(Plot):

    def __init__(self,
                 angle=(45, 45),
                 **kwargs):
        """

        Scatter Plot

        Parameters
        ----------------

        axis_style : {axis_style}
        endpoint_style : dict
            Endpoints are drawn at each extreme point of an objective. This style can be modified.
        labels : {labels}

        Other Parameters
        ----------------

        figsize : {figsize}
        title : {title}
        legend : {legend}
        tight_layout : {tight_layout}
        cmap : {cmap}

        """

        super().__init__(**kwargs)
        self.angle = angle

    def _do(self):
        fontstyle1 = self.get_fontstyle1()
        if fontstyle1 is None:
            fontstyle1 = {'family':'serif', 'size':30}
        tick_fontsize = self.get_tick_fontsize()
        label_fontsize = self.get_label_fontsize()
        title_fontsize = self.get_title_fontsize()

        is_1d = (self.n_dim == 1)
        is_2d = (self.n_dim == 2)
        is_3d = (self.n_dim == 3)
        more_than_3d = (self.n_dim > 3) # 大于3D

        # create the figure and axis objects
        if is_1d or is_2d:
            self.init_figure()
        elif is_3d:
            self.init_figure(plot_3D=True, title_fontsize=title_fontsize)
        elif more_than_3d:
            self.init_figure(n_rows=self.n_dim, n_cols=self.n_dim, title_fontsize=title_fontsize)

        # now plot data points for each entry
        # 一般而言，to_plot只包含一个任务，即展示F
        # 但若是包含多个任务，不同任务的绘图颜色不同，程序self.colors有10种颜色可供选择
        # 选完以后存在_kwargs中{"color":[...,...,...]}
        for k, (F, kwargs) in enumerate(self.to_plot):

            # copy the arguments and set the default color
            _kwargs = kwargs.copy()
            set_if_none(_kwargs, "color", self.colors[k % len(self.colors)])

            # determine the plotting type - scatter or line
            _type = _kwargs.get("plot_type")
            # FIXME暂时看不懂为什么要将plot_type从_kwargs中删掉
            if "plot_type" in _kwargs:
                del _kwargs["plot_type"]

            if is_1d:
                F = np.column_stack([F, np.zeros(len(F))])
                labels = self.get_labels() + [""]

                self.plot(self.ax, _type, F, **_kwargs)
                self.set_labels(self.ax, labels, False, label_fontsize)

            elif is_2d:
                self.plot(self.ax, _type, F, **_kwargs)
                self.set_labels(self.ax, self.get_labels(), False, label_fontsize)

            elif is_3d:
                set_if_none(_kwargs, "alpha", 1.0)

                self.plot(self.ax, _type, F, **_kwargs)
                self.ax.xaxis.pane.fill = False
                self.ax.yaxis.pane.fill = False
                self.ax.zaxis.pane.fill = False

                self.set_labels(self.ax, self.get_labels(), True, label_fontsize)
                if tick_fontsize is not None:
                    self.ax.tick_params(axis='both', labelsize=tick_fontsize)

                if self.angle is not None:
                    self.ax.view_init(*self.angle)

            # 高维图例绘制不属于1d or 2d or 3d，则单独设计绘制细节
            else:
                
                # plt.title('test', fontsize=20) # 不起作用

                labels = self.get_labels()

                # N>3后，开始对角线画图
                for i in range(self.n_dim): # 绘制N*N的子图
                    for j in range(self.n_dim):

                        ax = self.ax[i, j]

                        # 如果不位于对角线则作图
                        if i != j:
                            # 程序默认_type=None，而self.plot()默认_type='scatter'
                            self.plot(ax, _type, F[:, [i, j]], **_kwargs)
                            # NOTE 这里我嫌烦只保留左边、下边子图的y轴、x轴，同时xshare, yshare=True
                            # 设置ax的label NOTE 并不是特别好用
                            # if i == self.n_dim-1:
                            #     self.set_labels(ax, [labels[i], None], is_3d, label_fontsize)
                            # if j == 0:
                            #     self.set_labels(ax, [None, labels[j]], is_3d, label_fontsize)
                            # NOTE 原版
                            # self.set_labels(ax, [None, None], is_3d)

                            # 设置次刻度尺精度
                            def minor_tick(x, pos):
                                # if not x % 5.:
                                #     return ''
                                return x
                            
                            # if i == self.n_dim-1 or j == 0:  # 控制子图的xaxis yaxis grid tick_fontsize
                            ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))
                            ax.yaxis.set_minor_formatter(FuncFormatter(minor_tick))
                            ax.grid(linestyle='-.', linewidth=0.5, zorder=0, which='both')

                            # 本来打算用上self.grid_space参数，后来想想还是算了
                            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                            if tick_fontsize is not None:
                                ax.tick_params(axis='both', which='major', length=5, width=1., labelsize=tick_fontsize)
                                ax.tick_params(axis='both', which='minor', length=3, width=0.5, labelsize=tick_fontsize/2)
                            else:
                                ax.tick_params(axis='both', which='major', length=5, width=1.)
                                ax.tick_params(axis='both', which='minor', length=3, width=0.5, labelsize=10)

                        # 若位于对角线则展示labels[i]
                        # 对角线上的图像不展示坐标轴刻度等信息
                        else:
                            # NOTE style1
                            # ax.set_xticks([]) # 将对角线上多余的ax的tick设置为空，但是保留框（spine）
                            # ax.set_yticks([]) # 但是这样还是存在问题，因为spine会遮挡其他ax的tick
                            # NOTE style2
                            ax.set_axis_off() # 将tick和spine隐藏
                            ax.scatter(0, 0, s=1, color="white")
                            if self.legend[0] is True and i == 0:
                                pass
                                # ax.text(0, -0.04, labels[i], ha='center', va='center', fontdict=fontstyle1)
                            else:
                                ax.text(0, 0, labels[i], ha='center', va='center', fontdict=fontstyle1)

        return self

    def plot(self, ax, _type, F, **kwargs):
        # parameter F:ndarray(1,2), dim1意义未知，dim2存储了当前子图所在的位置

        is_3d = F.shape[1] == 3
        if _type is None:
            _type = "scatter"

        if _type == "scatter":
            if is_3d:
                ax.scatter(F[:, 0], F[:, 1], F[:, 2], **kwargs)
            # 若不为3D，
            else:
                ax.scatter(F[:, 0], F[:, 1], **kwargs)
        else:
            if is_3d:
                ax.plot_trisurf(F[:, 0], F[:, 1], F[:, 2], **kwargs)
            else:
                ax.plot(F[:, 0], F[:, 1], **kwargs)

    def set_labels(self, ax, labels, is_3d, fontsize=None):
        # set the labels for each axis
        if fontsize is not None: # 改版
            ax.set_xlabel(labels[0], fontsize=fontsize)
            ax.set_ylabel(labels[1], fontsize=fontsize)
        else: # 原版
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])


        if is_3d:
            if fontsize is not None:
                ax.set_zlabel(labels[2], fontsize=fontsize)
            else:
                ax.set_zlabel(labels[2])


parse_doc_string(Scatter.__init__)
