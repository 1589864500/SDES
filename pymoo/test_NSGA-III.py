from crypt import methods
from multiprocessing.pool import TERMINATE
from re import T
from tkinter import N
from turtle import Turtle
from typing import Dict

import numpy as np

import json
import os
import sys
sys.path.append('./')

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
from pymoo.problems.securitygame.MOSG import SGs1
from pymoo.MOSGsGeneticSolver.truing import Truing


# TODO: 将程序做问题规模上的拓展
# DONE
# TODO: pymoo工具包中nsga3自带一个筛选帕累托最优解的动作，我需要尝试把这个去掉
# TODO
# TODO: 修改Scatter().add(res.F).show()，把问题参数展示出来
#
# TODO: 和演化算法相关的很多工作没展开：
#  a)比如方法最后的帕累托解帅选是怎么样的;
# DONE
#  b)比如如何判断演化收敛（最优解不再更新），前提是观察演化算法的收敛曲线;
# FIXME: igd曲线
# TODO: 查看演化算法是否没找到边界解
obj_n = 5
pop_size = 500
target_n = 25
gen_n = 500
time = '00:50:00'
# pop_size设计有两种思路：
# a)和推荐的参考向量数量相等，参考向量数量与目标数呈指数级的增长关系；
# b)认为自己定义线性的增长关系，好处是时间复杂度下降，坏处是种群数量帕累托解的质量不高
# 注：据实验观察，以5目标为例，reference direction num=1820，pop_size小到50找到的最优解和1820相同
pop_size_np = np.linspace(start=50, stop=500, num=10, dtype=np.int32)
# obj_np = np.arange(3,13)
obj_np = np.arange(3,8)
target_np0 = np.array([10])
target_np1 = np.linspace(25, 100, 4, dtype=np.int32)
target_np2 = np.linspace(200, 1000, 5, dtype=np.int32)
target_np = np.hstack((target_np0, target_np1, target_np2))
SAVE_PATH:Dict[str,str] = {}
RES_DIR:str = './Results'
RES_DIR_NAME:str = 'Results'
Solver = 'GeneticMOSG1'
NUM = '2_4_e=1'  # 本次运行的序列号
SHOW = True
SAVE_HISTORY = False
MINCOV = 0

time_expensive = np.full(shape=[obj_np.size, target_np.size], fill_value=True)

# 演化算法能接受的问题规模
# time_expensive[:] = False
# 对比算法MOSG能接受的问题规模
time_expensive[0, :] = False
time_expensive[1, :7] = False
time_expensive[2, :5] = False
time_expensive[3, :2] = False
time_expensive[4, 0] = False

# TERMINATION
TERMINATION1 = get_termination('n_gen', gen_n)  # maxgen
TERMINATION2 = get_termination('time', '00:50:00')  # time
TERMINATION3 = get_termination('x_tol', tol=0.0025, n_last=30, n_max_gen=1000, nth_gen=5)  # x_tol
TERMINATION4 = get_termination('f_tol', tol=0.0025, n_last=30, n_max_gen=1000, nth_gen=5)  # f_tol
TERMINATION = [TERMINATION4]

for i in range(len(TERMINATION)):
    ct_star_total = None
    # for pop_idx, pop_size in enumerate(pop_size_np):
    # for pop_size in [pop_size_np[2]]:
    for pop_size in [pop_size]:
    #     for obj_idx, obj_n in enumerate(obj_np):
        for obj_n in [obj_np[2]]:
            # for target_idx, target_n in enumerate(target_np):
            for target_n in [target_np[1]]:
                # Step1: 把Security Game包装成get_problem形式，获取名字为"SGs1"
                # Step2: 继承某个演化算法，然后修改配置参数，包括交叉变异的算子、问题的规模参数target, obj等
                # Step3： res = minimize(get_problem()，展示
                # create the reference directions to be used for the optimization

                # if time_expensive[obj_idx, target_idx]:
                #     continue
                    
                ref_dirs = get_reference_directions("das-dennis", obj_n, n_partitions=12)

                # create the algorithm objecthttps://c.y.qq.com/base/fcgi-bin/u?__=DaRhdn
                algorithm = MOSGG(pop_size=pop_size, player_num=obj_n+1, target_num=target_n, ct_star_total=ct_star_total,
                                ref_dirs=ref_dirs, display=True, verbose=True,
                                sampling=get_sampling('bin_random'),
                                crossover=get_crossover('bin_hux'),
                                mutation=get_mutation('bin_bitflip'))
                problem = SGs1(player_num=obj_n+1, target_num=target_n, mincov=MINCOV)

                # execute the optimization
                res:Result = minimize(problem=problem,
                            algorithm=algorithm,
                            seed=1,
                            termination=TERMINATION[i], 
                            save_history=SAVE_HISTORY
                            )
                ct_star_total = problem.ct_star_total

                print('obj_n={}, target_n={}, gen_n={}, pop_size={}, 参数下，程序运行时间:{}秒'
                    .format(obj_n, target_n, gen_n, pop_size, res.exec_time))

                if SHOW:

                    # 保存res的目录文件、数据文件
                    model_save = resSave(res, Solver, res.exec_time, RES_DIR_NAME)
                    para1 = {'obj': obj_n, 'target': target_n, 'gen':gen_n, 'pop_size':pop_size, 'mincov':MINCOV}
                    para_filename:str = tool.algorithm.paras2Str(para1)
                    para2 = {'obj': obj_n, 'target': target_n}
                    para_dir = tool.algorithm.paras2Str(para2)
                    # model_save.saveResult保存数据文件，并返回保存的地址save_path
                    save_path = model_save.saveResult(para_dir, para_filename)
                    SAVE_PATH[para_filename] = save_path
                    filename = '-'.join([Solver, 'res_dir', str(NUM)]) + '.json'
                    # 保存目录文件
                    with open(filename, 'w') as json_file:
                        json.dump(obj=SAVE_PATH, fp=json_file, indent=4)

                    # 保存res可视化结果
                    para_temp = {"obj":obj_n, "target":target_n, "gen":gen_n, "pop_{size}":pop_size, 'mincov':MINCOV}
                    # Visualization(n_rows=obj_n, title=para_temp).add(res.F).show()
                    filename = '-'.join([Solver, para_filename+'.png'])
                    file_dir = os.path.join(RES_DIR_NAME, para_dir, filename)
                    Visualization(n_rows=obj_n, fig_title=para_temp,
                                sharex=True, sharey=True,
                                hspace=0,wspace=0).add(res.F).save(fname=file_dir)
                    # Scatter(figsize=(9.1,7)).add(res.F).show()