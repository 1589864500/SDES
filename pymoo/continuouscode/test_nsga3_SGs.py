'''Continuous code'''

from crypt import methods
from multiprocessing.pool import TERMINATE
from re import T
import select
from tkinter import N
from turtle import Turtle

import numpy as np

import json
import os
import sys
# sys.path.append('/home/wuyp/Projects/pymoo/')
sys.path.append('/home/haohao/Project_wuyp/MOSG/pymoo/')

from pymoo.util.termination import max_gen
from typing import *

import tool.algorithm
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.model.mutation import Mutation
from pymoo.factory import get_crossover_options, get_problem, get_reference_directions, get_sampling, get_crossover, get_mutation, get_selection, get_termination
from pymoo.model.result import Result
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.MOSGsGeneticSolver.visualization import Visualization
from pymoo.MOSGsGeneticSolver.resSave import resSave

from pymoo.algorithms.mosg_genetic_nsga3 import MOSGG
from pymoo.continuouscode.MOSG import SGs1
# from pymoo.integercode.truing import Truing


# obj_n = 5
pop_size = 400
ref_num = pop_size
# target_n = 25
gen_n = 300
time = '00:50:00'
# pop_size_np = np.linspace(start=50, stop=500, num=10, dtype=np.int32)
# obj_np = np.arange(3,13)
obj_np = np.arange(3,8)
target_np0 = np.array([10])
target_np1 = np.linspace(25, 100, 4, dtype=np.int32)
target_np2 = np.linspace(200, 1000, 5, dtype=np.int32)
target_np = np.hstack((target_np0, target_np1, target_np2))
SAVE_PATH:Dict[str,str] = {}
RES_DIR:str = './Results/integercode'
Solver = 'GeneticMOSGinteger'
NUM = '_e=1'  # 本次运行的序列号
SHOW = True
SAVE_HISTORY = True
MINCOV = 0
CODE = 'real'

time_expensive = np.full(shape=[obj_np.size, target_np.size], fill_value=True)

# 演化算法能接受的问题规模
# time_expensive[:] = False
# 对比算法MOSG能接受的问题规模
# time_expensive[0, 1:7] = False  # N=3
# time_expensive[1, 1:5] = False  # N=4
time_expensive[2, 2] = False  # N=5
# time_expensive[3, :2] = False
# time_expensive[4, 0] = False

# TERMINATION  
#   1.最大轮数 2.最长时间 3.x变化幅度 4.fitness变化幅度
TERMINATION1 = get_termination('n_gen', gen_n)  # maxgen
TERMINATION2 = get_termination('time', '00:50:00')  # time
TERMINATION3 = get_termination('x_tol', tol=0.0025, n_last=30, n_max_gen=1500, nth_gen=5)  # x_tol
TERMINATION4 = get_termination('f_tol', tol=0.0025, n_last=30, n_max_gen=1500, nth_gen=5)  # f_tol
TERMINATION = [TERMINATION1, TERMINATION2, TERMINATION3, TERMINATION4]
# idx_ter = 3

# SAMPLING
SAMPLING1 = get_sampling('real_random')
# SAMPLING2 = tool.algorithm.loadVariPickle(path='/home/wuyp/Projects/pymoo/Results/obj5target50/GeneticMOSGinteger-Popsize400Coderealgen100-2022_05_17_09_19-29633.txt').pop
# SAMPLING = [SAMPLING1, SAMPLING2]
SAMPLING = [SAMPLING1]
idx_sam = 0

# CROSSOVER
CROSSOVER1 = get_crossover('real_sbx')
# CROSSOVER2 = get_crossover('int_one_point')
# CROSSOVER3 = get_crossover('int_two_point')
# CROSSOVER4 = get_crossover('int_ux')
# CROSSOVER5 = get_crossover('int_hux')
# CROSSOVER6 = get_crossover('int_exp')
# CROSSOVER = [CROSSOVER1, CROSSOVER2, CROSSOVER3, CROSSOVER4, CROSSOVER5, CROSSOVER6]
CROSSOVER = [CROSSOVER1]
# idx_cro = 4

# MUTATION
MUTATION1 = get_mutation('real_pm')
MUTATION = [MUTATION1]
idx_mut = 0

# # SELECTION
# SELECTION1 = get_selection('random')
# SELECTION2 = get_selection('tournament')
# SELECTION = [SELECTION1, SELECTION2]
# idx_sel = 1

# ALGOTHRIM

# EXPERIMENTTYPE = 'AblationNaiveEA'
EXPERIMENTTYPE = 'ComparisonNaiveEAandORIGAMIG_NaiveEA'
EXPERIMENTTYPE = 'T=50ComparisonNaiveEAandORIGAMIG_NaiveEA'
DESCRIPTION:str = EXPERIMENTTYPE + "_SEED" # 本次运行的序列号
SAVE_PATH:Dict[str,str] = {}  # 存到json中的文件目录列表
for SEED in range(30):
    DESCRIPTION += str(SEED)  
    # for i in range(len(TERMINATION)):
    # for idx_cro in range(len(CROSSOVER)):
    RES_DIR_NAME:str = os.path.join('Results', EXPERIMENTTYPE, 'SEED' + str(SEED))  # 数据存储的起始路径/根路径
    for idx_cro in [0]:
        ct_star_total = None
        # for pop_idx, pop_size in enumerate(pop_size_np):
        # for pop_size in [pop_size_np[2]]:
        # for pop_size in [pop_size]:
        for idx_ter in [0]:
            for obj_idx, obj_n in enumerate(obj_np):
            # for obj_n in [obj_np[2]]:
                for target_idx, target_n in enumerate(target_np):
                # for target_n in [target_np[2]]:
                    # Step1: 把Security Game包装成get_problem形式，获取名字为"SGs1"
                    # Step2: 继承某个演化算法，然后修改配置参数，包括交叉变异的算子、问题的规模参数target, obj等
                    # Step3： res = minimize(get_problem()，展示
                    # create the reference directions to be used for the optimization

                    if time_expensive[obj_idx, target_idx]:
                        continue
                    print('obj:{}, tar:{}'.format(obj_n, target_n))

                    # REFERENCE DIRECTION
                    ref_dirs = get_reference_directions("energy", obj_n, ref_num, seed=SEED)

                    # create the algorithm objecthttps://c.y.qq.com/base/fcgi-bin/u?__=DaRhdn
                    algorithm = MOSGG(pop_size=pop_size, player_num=obj_n+1, target_num=target_n, ct_star_total=ct_star_total,
                                    ref_dirs=ref_dirs, display=False, verbose=False,
                                    sampling=SAMPLING[idx_sam],
                                    crossover=CROSSOVER[idx_cro],
                                    mutation=MUTATION[idx_mut],
                                    # selection=SELECTION[idx_sel]
                                    )
                    problem = SGs1(player_num=obj_n+1, target_num=target_n, mincov=MINCOV)

                    # execute the optimization
                    res:Result = minimize(problem=problem,
                                algorithm=algorithm,
                                seed=1,
                                termination=TERMINATION[idx_ter], 
                                save_history=SAVE_HISTORY
                                )
                    ct_star_total = problem.ct_star_total

                    print('obj_n={}, target_n={}, gen_n={}, pop_size={}, 参数下，程序运行时间:{}秒'
                        .format(obj_n, target_n, gen_n, pop_size, res.exec_time))

                    if SHOW:

                        # 保存res的目录文件、数据文件
                        model_save = resSave(res, Solver, res.exec_time, RES_DIR_NAME)
                        para1 = para1 = {'Popsize':pop_size, 'Code':CODE, 'gen':gen_n}
                        para_filename:str = tool.algorithm.paras2Str(para1)
                        para2 = {'obj': obj_n, 'target': target_n}
                        para_dir = tool.algorithm.paras2Str(para2)
                        # model_save.saveResult保存数据文件，并返回保存的地址save_path
                        save_path = model_save.saveResult(para_dir, para_filename)
                        SAVE_PATH[save_path] = res.exec_time

                        # 保存res可视化结果
                        para_temp = para1
                        # Visualization(n_rows=obj_n, title=para_temp).add(res.F).show()
                        filename = '-'.join([Solver, para_filename+'.png'])
                        file_dir = os.path.join(RES_DIR_NAME, para_dir, filename)
                        Visualization(n_rows=obj_n, fig_title=para_temp,
                                    sharex=True, sharey=True,
                                    hspace=0,wspace=0).add(res.F).save(fname=file_dir)
                        # Scatter(figsize=(9.1,7)).add(res.F).show()
filename = '-'.join([Solver, 'res_dir', str(DESCRIPTION)]) + '.json'
# 保存目录json文件
with open(filename, 'w') as json_file:
    json.dump(obj=SAVE_PATH, fp=json_file, indent=4)