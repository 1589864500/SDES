from cProfile import label
from ctypes.wintypes import PLARGE_INTEGER
import math
import sys
sys.path.append('./')
import tool.algorithm
import os
from matplotlib import pyplot as plt  # 用来绘制图形
# plt.rc('font',  family='Times New Roman')
plt.rc('font', family='serif') # NOTE serif字体看起来确实好看
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
from securitygame_core.MO_security_game import MOSG

#
# X1 = X2 = np.arange(-5, 15, 1)
# X1, X2 = np.meshgrid(X1, X2)
#
# Z = 1 / 2 * X1 ** 2
#
# # 创建绘制实时损失的动态窗口
# plt.ion()
# for i in range(30000):
#     plt.clf()  # 清除之前画的图
#     fig = plt.gcf()  # 获取当前图
#     ax = fig.gca(projection='3d')  # 获取当前轴
#     ax.plot_surface(X1, X2, Z, cmap='rainbow')
#     plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
#     plt.ioff()  # 关闭画图窗口Z
#
#     Z = Z - X1 + 2 * X2  # 变换Z值
#
# # 加这个的目的是绘制完后不让窗口关闭
# plt.show()
#

# X1 = X2 = np.arange(-5, 15, 1)
# X1, X2 = np.meshgrid(X1, X2)
#
# Z = 1 / 2 * X1 ** 2
#
# # 绘制三维图初始角度
# azim = -60
# elev = 30
#
# # 创建绘制实时损失的动态窗口
# plt.ion()
# for i in range(30000):
#     plt.clf()  # 清除之前画的图
#     fig = plt.gcf()  # 获取当前图
#     ax = fig.gca(projection='3d')  # 获取当前轴
#
#     ax.view_init(elev, azim)  # 设定角度
#
#     ax.plot_surface(X1, X2, Z, cmap='rainbow')
#     plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
#
#     elev, azim = ax.elev, ax.azim  # 将当前绘制的三维图角度记录下来，用于下一次绘制（放在ioff()函数前后都可以，但放在其他地方就不行）
#     # elev, azim = ax.elev, ax.azim - 1 # 可自动旋转角度，不需要人去拖动
#
#     plt.ioff()  # 关闭画图窗口Z
#
#     Z = Z - X1 + 2 * X2  # 变换Z值
#
# # 加这个的目的是绘制完后不让窗口关闭
# plt.show()
#

RES_DIR = os.path.join(os.getcwd(), 'Results', 'demo')

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
def getXYZ(xstart=0, xend=1, ystart=0, yend=0, interval=10, bits=1, resource_ratio=0.5):
    problem2 = MOSG(player_num=2, target_num=2, resource_ratio=resource_ratio)
    x1 = np.linspace(xstart, xend, interval)
    x2 = np.linspace(ystart, yend, interval)
    x1_str = [str(round(i,bits)) for i in x1]
    x2_str = [str(round(i,bits)) for i in x2]
    # 网格化操作，将原有1维的x1,x2变成2维
    X1, X2 = np.meshgrid(x1, x2)
    payoff_attacker = np.empty_like(X1)
    payoff_defender = np.empty_like(X1)
    for i in range(len(x1)):
        for j in range(len(X2)):
            if x1[i]+x2[j]>problem2.resource:
                payoff_attacker[i, j] = 10
                payoff_defender[i, j] = 10
                continue

            problem2.set_ct(np.array([x1[i], x2[j]]))
            problem2.cal_payoff()
            problem2.cal_payoff_attacker()
            problem2.cal_payoff_defender()
            payoff_attacker[i, j] = problem2.get_payoff_attacker()
            payoff_defender[i, j] = problem2.get_payoff_defender()
    payoff_attacker[payoff_attacker==10] = math.floor(np.min(payoff_attacker))
    payoff_defender[payoff_defender==10] = math.floor(np.min(payoff_defender))
    return x1_str, x2_str, x1, x2, payoff_attacker, payoff_defender

'''原版'''
def heatmap(data, row_labels=None, col_labels=None, ax=None,
            cbar_kw={}, cbarlabel="", 
            label_fontsize=None,
            **kwargs):
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

    # plt.rc('font',  family='Times New Roman')
    plt.rc('font', family='serif') # NOTE serif字体看起来确实好看
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    if label_fontsize is not None:
        ax.set_xticks(np.arange(data.shape[1]), labelsize=label_fontsize)
        ax.set_yticks(np.arange(data.shape[0]), labelsize=label_fontsize)
    else:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if col_labels is not None:
        ax.set_xticklabels(col_labels)
    if row_labels is not None:
        ax.set_yticklabels(row_labels)

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

'''改版'''
def heatmapSG(data, row_labels=None, col_labels=None, ax=None,
            cbar_kw={}, cbarlabel="", 
            label_fontsize = 12, legend_fontsize=12,
            **kwargs):
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
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom") # 将colorbar显示在右侧，暂时不会改

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1])) # NOTE set函数只负责设定tick，因此不能修改tick（要有param实现）
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if col_labels is not None:
        ax.set_xticklabels(col_labels, fontsize=legend_fontsize) # set同上
    if row_labels is not None:
        ax.set_yticklabels(row_labels, fontsize=legend_fontsize)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False, 
                   labelsize=label_fontsize)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False,
                labelsize=label_fontsize)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, 
                     table_fontsize = None,
                     **textkw):
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
            if table_fontsize is not None:
                text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize=table_fontsize, **kw)
            else:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

'''绘制T=2 N=2的payoff of attacker and defender的热力图'''
def payoff(player='defender', pdf=False, res_dir=None, interval=10, 
        table_fontsize=None, label_fontsize=None, legend_fontsize=None):
    if res_dir is not None:
        RES_DIR = res_dir
    # 有一定间隔的热力图
    x1_str, x2_str, x1, x2, payoff_attacker, payoff_defender = getXYZ(xend=1, yend=1, interval=interval+1, bits=2, resource_ratio=0.5)

    # fig, ax = plt.subplots(figsize=(7,8))
    fig, ax = plt.subplots()
    if player=='defender':
        im, cbar = heatmapSG(payoff_defender, x1_str, x2_str, ax=ax,
                        cmap="YlGn", cbarlabel="$defender\ payoff$",
                        label_fontsize=label_fontsize, legend_fontsize=legend_fontsize)
    else:
        im, cbar = heatmapSG(payoff_attacker, x1_str, x2_str, ax=ax,
                        cmap="YlGn", cbarlabel="$attacker\ payoff$",
                        label_fontsize=label_fontsize, legend_fontsize=legend_fontsize)
    texts = annotate_heatmap(im, valfmt="{x:.1f}", table_fontsize=table_fontsize)  # NOTE 给heatmap添加文字说明
    # texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    fig.tight_layout()
    fname = os.path.join(RES_DIR, tool.algorithm.getTime() + player+'_payoff')
    fname_pdf = os.path.join(RES_DIR, tool.algorithm.getTime() + player+'_payoff.pdf')
    plt.savefig(fname=fname)
    if pdf:
        plt.savefig(fname=fname_pdf)

# NOTE: 由于MOSG问题维度太高，导致这个图看不出东西
'''gametable [target, 2, (player-1)*2]
用于分析MOSG问题初始化分布  
'''
def gametable(player=None, player_num=6, target_num=25):
    problem = MOSG(player_num=player_num, target_num=target_num)
    # if player=='attacker':
    #     sg_idx += player-1
    X = ['defener', 'attacker']
    fig, ax = plt.subplots(player_num, 1)
    for sg_idx in range(player_num-1):
        def_suc = problem.gametable[:, 0, sg_idx].astype(int)
        att_suc = problem.gametable[:, 1, sg_idx + player_num-1].astype(int)
        suc_pay = np.vstack([def_suc, att_suc])
        im, cbar = heatmap(suc_pay, row_labels=X, ax=ax[sg_idx],
            cmap="YlGn", cbarlabel="$F_"+str(sg_idx+1)+"$")
        texts = annotate_heatmap(im, valfmt="{x:}")
    plt.savefig(os.path.join(RES_DIR, tool.algorithm.getTime() + 'gametable'))
def gametable_m_v(player=None, player_num=6, target_num=25):
    problem = MOSG(player_num=player_num, target_num=target_num)
    def_m_v = {}
    att_m_v = {}
    print("{sg_idx:[mean, var]}")
    for sg_idx in range(player_num-1):
        def_suc = problem.gametable[:, 0, sg_idx].astype(int)
        att_suc = problem.gametable[:, 1, sg_idx + player_num-1].astype(int)
        def_m_v[sg_idx+1] = [np.mean(def_suc), np.var(def_suc)]
        att_m_v[sg_idx+1] = [np.mean(att_suc), np.var(att_suc)]
    print('defender,', def_m_v)
    print('attacker,', att_m_v)


# 简单的分类热力图
# fig, ax = plt.subplots()
# im = ax.imshow(harvest)
#
# # We want to show all ticks...
# ax.set_xticks(np.arange(len(farmers)))
# ax.set_yticks(np.arange(len(vegetables)))
# # ... and label them with the respective list entries
# ax.set_xticklabels(farmers)
# ax.set_yticklabels(vegetables)
#
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
#
# # Loop over data dimensions and create text annotations.
# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         text = ax.text(j, i, harvest[i, j],
#                        ha="center", va="center", color="w")
#
# ax.set_title("Harvest of local farmers (in tons/year)")
# fig.tight_layout()
# plt.show()

# # 更复杂的热力图
# np.random.seed(19680801)
# fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 7))
# # Replicate the above example with a different font size and colormap.
# im, _ = heatmap(harvest, vegetables, farmers, ax=ax,
#                 cmap="Wistia", cbarlabel="harvest [t/year]")
# annotate_heatmap(im, valfmt="{x:.1f}", size=7)
# # Create some new data, give further arguments to imshow (vmin),
# # use an integer format on the annotations and provide some colors.
# data = np.random.randint(2, 100, size=(7, 7))
# y = ["Book {}".format(i) for i in range(1, 8)]
# x = ["Store {}".format(i) for i in list("ABCDEFG")]
# im, _ = heatmap(data, y, x, ax=ax2, vmin=0,
#                 cmap="magma_r", cbarlabel="weekly sold copies")
# annotate_heatmap(im, valfmt="{x:d}", size=7, threshold=20,
#                  textcolors=["red", "white"])
# # Sometimes even the data itself is categorical. Here we use a
# # :class:`matplotlib.colors.BoundaryNorm` to get the data into classes
# # and use this to colorize the plot, but also to obtain the class
# # labels from an array of classes.
# data = np.random.randn(6, 6)
# y = ["Prod. {}".format(i) for i in range(10, 70, 10)]
# x = ["Cycle {}".format(i) for i in range(1, 7)]
# qrates = np.array(list("ABCDEFG"))
# norm = matplotlib.colors.BoundaryNorm(np.linspace(-3.5, 3.5, 8), 7)
# fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])
# im, _ = heatmap(data, y, x, ax=ax3,
#                 cmap=plt.get_cmap("PiYG", 7), norm=norm,
#                 cbar_kw=dict(ticks=np.arange(-3, 4), format=fmt),
#                 cbarlabel="Quality Rating")
# annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
#                  textcolors=["red", "black"])
# # We can nicely plot a correlation matrix. Since this is bound by -1 and 1,
# # we use those as vmin and vmax. We may also remove leading zeros and hide
# # the diagonal elements (which are all 1) by using a
# # :class:`matplotlib.ticker.FuncFormatter`.
# corr_matrix = np.corrcoef(np.random.rand(6, 5))
# im, _ = heatmap(corr_matrix, vegetables, vegetables, ax=ax4,
#                 cmap="PuOr", vmin=-1, vmax=1,
#                 cbarlabel="correlation coeff.")
# def func(x, pos):
#     return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")
# annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)
# plt.tight_layout()
# plt.show()

# NOTE 这是个什么东西
# gametable_m_v()                

# NOTE 试试def payoff和def gametable
RES_DIR = os.path.join(os.getcwd(), 'Results', '20220823')
payoff(player='defender', pdf=True, res_dir=RES_DIR, table_fontsize=13, label_fontsize=11, legend_fontsize=13)
payoff(player='attacker', pdf=True, res_dir=RES_DIR, table_fontsize=13, label_fontsize=11, legend_fontsize=13)
# gametable()        

print(__name__ + ' finish!')