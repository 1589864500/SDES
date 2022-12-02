import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./')


import tool.algorithm


from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
# class Scatter用于画图
from pymoo.visualization.scatter import Scatter

# NOTE NSGA-III
# # ref_dirs代表的目标数和pop_size相关，正比（组合级别增长）
# # create the reference directions to be used for the optimization
# ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# # create the algorithm object
# algorithm = NSGA3(pop_size=92,
#                   ref_dirs=ref_dirs, display=True, save_history=False, verbose=False)

# # execute the optimization
# res = minimize(get_problem("dtlz1"),
#                algorithm,
#                seed=1,
#                termination=('n_gen', 600))

# Scatter().add(res.F).show()

# # res = minimize(get_problem("dtlz1^-1"),
# #                algorithm,
# #                seed=1,
# #                termination=('n_gen', 600))
# #
# # Scatter().add(res.F).show()

 
linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'
 
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
 
 
def plot_linestyles(ax, linestyles, title):
    X, Y = np.linspace(0, 100, 10), np.zeros(10)
    yticklabels = []
 
    for i, (name, linestyle) in enumerate(linestyles):
        ax.plot(X, Y+i, linestyle=linestyle, linewidth=1.5, color='black')
        yticklabels.append(name)
 
    ax.set_title(title)
    ax.set(ylim=(-0.5, len(linestyles)-0.5),
           yticks=np.arange(len(linestyles)),
           yticklabels=yticklabels)
    ax.tick_params(left=False, bottom=False, labelbottom=False)
    ax.spines[:].set_visible(False)
 
    # For each line style, add a text annotation with a small offset from
    # the reference point (0 in Axes coords, y tick value in Data coords).
    for i, (name, linestyle) in enumerate(linestyles):
        ax.annotate(repr(linestyle),
                    xy=(0.0, i), xycoords=ax.get_yaxis_transform(),
                    xytext=(-6, -12), textcoords='offset points',
                    color="blue", fontsize=8, ha="right", family="monospace")
 
 
ax0, ax1 = (plt.figure(figsize=(10, 8))
            .add_gridspec(2, 1, height_ratios=[1, 3])
            .subplots())
 
plot_linestyles(ax0, linestyle_str[::-1], title='Named linestyles')
plot_linestyles(ax1, linestyle_tuple[::-1], title='Parametrized linestyles')
 
plt.tight_layout()
plt.savefig('./Results/demo/figureFromTest')

print('__main__ finish~!')