
from fileinput import filename
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
import os

import sys
sys.path.append('./')
from pymoo.visualization.scatter import Scatter


class Visualization(Scatter):


    def __init__(self, RES_DIR=None, n_rows=None, **kwargs):
        # SIZEPERCOL = 1.6
        # SIZEPEROW = 1.5
        # figsize = (n_rows*SIZEPEROW, n_rows*SIZEPERCOL)
        figsize = (7, 7)
        self.RES_DIR = RES_DIR
        super(Visualization, self).__init__(figsize=figsize, **kwargs)

    def landscape(self, x=None, y=None, xspace=0.01, yspace=0.01, fname='landscape'):
        fig =plt.figure()
        ax = Axes3D(fig, auto_to_figure=False)
        ax.set_title(fname)
        fig.add_axes(ax)
        if fname == 'landscape' is False:
            fname = '_'.join(['landscape', fname])
        if x is None:
            pass
        if y is None:
            pass
        z = 1
        ax.plot_surface(x, y, z, cstride=1, cmap='rainbow')
        plt.savefig(os.path.join(self.RES_DIR, fname))
