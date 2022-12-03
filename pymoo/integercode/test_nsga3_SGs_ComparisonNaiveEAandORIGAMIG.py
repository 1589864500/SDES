'''Integer code'''

from crypt import methods
from multiprocessing.pool import TERMINATE
from re import T
import select
from tkinter import N
from tracemalloc import start
from turtle import Turtle

import numpy as np

import json
import os
import gc
import sys
sys.path.append('')
print(sys.path)

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
from pymoo.integercode.MOSG import SGs1
from pymoo.integercode.truing import Truing

# ref_num = pop_size
# target_n = 25
# gen_n = 1000
# gen_n = pop_size
# time = '00:50:00'
# pop_size_np = np.linspace(start=50, stop=500, num=10, dtype=np.int32)
obj_np = np.arange(3,20)
target_np0 = np.array([10])  # 0
target_np1 = np.linspace(25, 100, 4, dtype=np.int32)  # 1-4
target_np2 = np.linspace(200, 1000, 5, dtype=np.int32)  # 5-9
target_np = np.hstack((target_np0, target_np1, target_np2))
RES_DIR:str = './Results/integercode' # ????
Solver = 'GeneticMOSGinteger'
SEED = 2
SHOW = True
SAVE_HISTORY = True
MINCOV = 0
CODE = 'integer'
# CODE = 'withoutOpt'
# CODE = 'continuous'
# CODE = 'time'

time_expensive = np.full(shape=[obj_np.size, target_np.size], fill_value=True)

# NOTE 问题规模
# NOTE FIX Objecitve
# time_expensive[0, 6] = False # 3
# time_expensive[1, 4] = False # 4
time_expensive[2, 2] = False # 5
# time_expensive[3, 6:] = False # 6
# time_expensive[4, 1:] = False # 7
# time_expensive[5, 5:7] = False # 8
# time_expensive[6, 4:6] = False # 9
# time_expensive[7, 3:5] = False # 10
# time_expensive[8, 4] = False # 11
# time_expensive[9, 3] = False # 12
# time_expensive[10, 6] = False # 13
# time_expensive[11, 6] = False # 14
# time_expensive[12, 6] = False # 15
# time_expensive[13, 6] = False # 16
# time_expensive[14, 6] = False # 17
# time_expensive[15, 6] = False # 18
# time_expensive[16, 6] = False # 19
# NOTE FIX Target
# time_expensive[:, 1] = False # 25
# time_expensive[1:10, 2] = False # 50
# time_expensive[1:7, 3] = False # 75
# time_expensive[1:6, 4] = False # 100
# time_expensive[1:5, 5] = False # 200
# time_expensive[1:3, 6] = False # 400
# time_expensive[3, 7] = False # 600
# time_expensive[2:4, 8] = False # 800
# time_expensive[2:4, 9] = False # 1000

# TERMINATION  
#   1.最大轮数 2.最长时间 3.x变化幅度 4.fitness变化幅度
max_gen = np.linspace(0,900,10,dtype=np.int32)
# TERMINATION1 = get_termination('n_gen', gen_n)  # maxgen
# TERMINATION2 = get_termination('time', '00:50:00')  # time
# TERMINATION3 = get_termination('x_tol', tol=0.0025, n_last=30, n_max_gen=600, nth_gen=5)  # x_tol
# TERMINATION4 = get_termination('f_tol', tol=0.0025, n_last=30, n_max_gen=600, nth_gen=5)  # f_tol
# TERMINATION = [TERMINATION2, TERMINATION1, TERMINATION3, TERMINATION4]
# TERMINATION = [TERMINATION1]
# idx_ter = 0

# SAMPLING
SAMPLING1 = get_sampling('int_random')
SAMPLING = [SAMPLING1]
idx_sam = 0
# CROSSOVER
CROSSOVER1 = get_crossover('int_sbx')
CROSSOVER2 = get_crossover('int_one_point')
CROSSOVER3 = get_crossover('int_two_point')
CROSSOVER4 = get_crossover('int_ux')
CROSSOVER5 = get_crossover('int_hux')
CROSSOVER6 = get_crossover('int_exp')
# CROSSOVER = [CROSSOVER1, CROSSOVER2, CROSSOVER3, CROSSOVER4, CROSSOVER5, CROSSOVER6]
# CROSSOVER = [CROSSOVER1, CROSSOVER5, CROSSOVER1, CROSSOVER5]
CROSSOVER = [CROSSOVER1, CROSSOVER5]
# CROSSOVER = [CROSSOVER5, CROSSOVER1, CROSSOVER5]
CROSSOVER_1 = [CROSSOVER1]
CROSSOVER_2 = [CROSSOVER5]
idx_cro = 0

# MUTATION
MUTATION1 = get_mutation('int_pm')
MUTATION = [MUTATION1]
idx_mut = 0

# # SELECTION
# SELECTION1 = get_selection('random')
# SELECTION2 = get_selection('tournament')
# SELECTION = [SELECTION1, SELECTION2]
# idx_sel = 1

# # NOTE ALGOTHRIM 第一次运行
# EXPERIMENTTYPE = 'T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG-heu'
# # DESCRIPTION:str = 'IndependentRepeatExperiments' + "_SEED" # 本次运行的序列号
# # DESCRIPTION:str = 'SensitivityStudy' + "_SEED" # 本次运行的序列号
# DESCRIPTION:str = EXPERIMENTTYPE + "_SEED" # 本次运行的序列号
# SAVE_PATH:Dict[str,str] = {}  # 存到json中的文件目录列表
# # for SEED in range(5):
# # for SEED in range(5, 10):
# # for SEED in range(10, 15):
# # for SEED in range(15, 20):
# # for SEED in range(20, 25):
# # for SEED in range(25, 30):
# for SEED in range(30):
#     DESCRIPTION += str(SEED)
#     # NOTE ...
#     TIME = []
#     # RES_DIR_NAME:str = os.path.join('Results', 'IndependentRepeat', 'SEED' + str(SEED))  # 数据存储的起始路径/根路径
#     RES_DIR_NAME:str = os.path.join('Results', EXPERIMENTTYPE, 'SEED' + str(SEED))  # 数据存储的起始路径/根路径
#     # for i in range(len(TERMINATION)):
#     for idx_cro in range(len(CROSSOVER_1)):
#     # for k in range(1):
#         ct_star_total = None
#         # for pop_idx, pop_size in enumerate(pop_size_np):
#         # for pop_size in [pop_size_np[2]]:
#         # for pop_size in [pop_size]:
#         # for idx_ter in range(1,len(TERMINATION)):
#         # for obj_n in [obj_np[2]]:
#         for obj_idx, obj_n in enumerate(obj_np):
#             for target_idx, target_n in enumerate(target_np):
#             # for target_n in target_np[4]:
#                 # TERMINATION1 = get_termination('x_tol', tol=0.0025, n_last=20, n_max_gen=max_gen[obj_n], nth_gen=5)  # x_tol
#                 # TERMINATION2 = get_termination('f_tol', tol=0.0025, n_last=20, n_max_gen=max_gen[obj_n], nth_gen=5)  # f_tol
#                 # TERMINATION = [TERMINATION1, TERMINATION2]
#                 # for idx_ter in range(len(TERMINATION)):
#                 break_flag = False
#                 # for pop_size in [150, 200, 250, 300]:
#                 # for pop_size in [350, 400, 450, 500]:
#                 for pop_size in [400]:
#                 # for i in range(3):
#                     # pop_size=400
#                     # pop_size = 200
#                     # pop_size += 200 * (obj_n-4)
#                     ref_num = pop_size
#                     # gen_n = pop_size
#                     # gen_n = 300
#                     gen_n = 100 # for Comparision NaiveEA and ORIGAMIG
#                     if obj_n == 3:
#                         pop_size  = 50
#                         ref_num = pop_size
#                         gen_n = 50
#                     TERMINATION1 = get_termination('n_gen', gen_n)  # maxgen
#                     TERMINATION = [TERMINATION1]
#                     idx_ter = 0
                    
#                     # Step1: 把Security Game包装成get_problem形式，获取名字为"SGs1"
#                     # Step2: 继承某个演化算法，然后修改配置参数，包括交叉变异的算子、问题的规模参数target, obj等
#                     # Step3： res = minimize(get_problem()，展示
#                     # create the reference directions to be used for the optimization

#                     if time_expensive[obj_idx, target_idx]:
#                         continue
                        
#                     print('obj:{}, tar:{}'.format(obj_n, target_n))
#                     # REFERENCE DIRECTION   
#                     ref_dirs = get_reference_directions("energy", obj_n, ref_num, seed=SEED)

#                     # create the algorithm objecthttps://c.y.qq.com/base/fcgi-bin/u?__=DaRhdn
#                     algorithm = MOSGG(pop_size=pop_size, player_num=obj_n+1, target_num=target_n, ct_star_total=ct_star_total,
#                                     ref_dirs=ref_dirs, display=False, verbose=False,
#                                     sampling=SAMPLING[idx_sam],
#                                     crossover=CROSSOVER[idx_cro],
#                                     mutation=MUTATION[idx_mut],
#                                     save_history=SAVE_HISTORY
#                                     # selection=SELECTION[idx_sel]
#                                     )
#                     problem = SGs1(player_num=obj_n+1, target_num=target_n, mincov=MINCOV)

#                     # execute the optimization
#                     res:Result = minimize(problem=problem,
#                                 algorithm=algorithm,
#                                 seed=1,
#                                 termination=TERMINATION[idx_ter], 
#                                 # termination=TERMINATION1, 
#                                 save_history=SAVE_HISTORY
#                                 )
#                     ct_star_total = problem.ct_star_total

#                     print('obj_n={}, target_n={}, gen_n={}, pop_size={}, 参数下，程序运行时间:{}秒'
#                         .format(obj_n, target_n, gen_n, pop_size, res.exec_time))
#                     # Time Efficiency 实验用到，注释
#                     # TIME.append(res.exec_time)
#                     # if not time_expensive[obj_idx, target_idx] and res.exec_time > 5000:
#                     #     print('__main__ finish!')
#                     #     tool.algorithm.dumpVariPickle(vari=TIME, name='TIME_target'+str(target_n)+tool.algorithm.getTime())
#                     #     break_flag = True
#                     #     time_expensive[:, target_idx] = True
#                     #     exit()

#                     if SHOW:

#                         # 保存res的目录文件、数据文件
#                         #　NOTE DATE, 同时model_save中包装了存储路径
#                         model_save = resSave(res, Solver, res.exec_time, RES_DIR_NAME)
#                         # NOTE 文件名
#                         # para1 = {'Ablation':'Lcode', 'Popsize':pop_size, 'Code':CODE, 'gen':gen_n}  # AblationStudy
#                         para1 = {'Popsize':pop_size, 'Code':CODE, 'gen':gen_n}  # boolearn scoring mechanism by default
#                         para_filename:str = tool.algorithm.paras2Str(para1)
#                         # NOTE 上一级目录名
#                         para2 = {'obj': obj_n, 'target': target_n}
#                         para_dir = tool.algorithm.paras2Str(para2)
#                         # model_save.saveResult保存数据文件，并返回保存的地址save_path
#                         save_path = model_save.saveResult(para_dir, para_filename)
#                         SAVE_PATH[save_path] = res.exec_time

#                         # 保存res可视化结果
#                         para_temp = para1
#                         # Visualization(n_rows=obj_n, title=para_temp).add(res.F).show()
#                         filename = '-'.join([Solver, para_filename+'.png'])
#                         file_dir = os.path.join(RES_DIR_NAME, para_dir, filename)
#                         Visualization(n_rows=obj_n, fig_title=para_temp,
#                                     sharex=True, sharey=True,
#                                     hspace=0,wspace=0).add(res.F).save(fname=file_dir)
#                         # Scatter(figsize=(9.1,7)).add(res.F).show()
#                     gc.collect()
#                 if break_flag:
#                     # exit()
#                     break



# NOTE 初始化(随机初始化或者继承已有方法结果初始化)
path_sampling:List[str] = tool.algorithm.loadVariJson(path='GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG_SEED0-30_iteration100-200.json')['obj5target50']
path_sampling:List[str] = tool.algorithm.loadVariJson(path='GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG_SEED0-30_iteration0-100.json')['obj5target50']
path_sampling:List[str] = tool.algorithm.loadVariJson(path='GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG-heu_SEED0-30_iteration0-100.json')['obj5target50']
path_sampling:List[str] = tool.algorithm.loadVariJson(path='GeneticMOSGinteger-res_dir-T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG-heu_SEED0-30_iteration100-200.json')['obj5target50']
# NOTE ALGOTHRIM 重复运行
EXPERIMENTTYPE = 'T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG'
EXPERIMENTTYPE = 'T=50ComparisonNaiveEAandORIGAMIG_ORIGAMIG-heu'
# DESCRIPTION:str = 'IndependentRepeatExperiments' + "_SEED" # 本次运行的序列号
# DESCRIPTION:str = 'SensitivityStudy' + "_SEED" # 本次运行的序列号
DESCRIPTION:str = EXPERIMENTTYPE + "_SEED" # 本次运行的序列号
SAVE_PATH:Dict[str,str] = {}  # 存到json中的文件目录列表
for path_i in path_sampling:
    SEED = int(path_i.split('/')[-3].split('D')[-1])\
    # NOTE 断点续传
    # if SEED not in range(30):
        # continue
    SAMPLING[0] = tool.algorithm.loadVariPickle(path=path_i).pop
    DESCRIPTION += str(SEED)
    # NOTE ...
    TIME = []
    # RES_DIR_NAME:str = os.path.join('Results', 'IndependentRepeat', 'SEED' + str(SEED))  # 数据存储的起始路径/根路径
    RES_DIR_NAME:str = os.path.join('Results', EXPERIMENTTYPE, 'SEED' + str(SEED))  # 数据存储的起始路径/根路径
    # for i in range(len(TERMINATION)):
    for idx_cro in range(len(CROSSOVER_1)):
    # for k in range(1):
        ct_star_total = None
        # for pop_idx, pop_size in enumerate(pop_size_np):
        # for pop_size in [pop_size_np[2]]:
        # for pop_size in [pop_size]:
        # for idx_ter in range(1,len(TERMINATION)):
        # for obj_n in [obj_np[2]]:
        for obj_idx, obj_n in enumerate(obj_np):
            for target_idx, target_n in enumerate(target_np):
            # for target_n in target_np[4]:
                # TERMINATION1 = get_termination('x_tol', tol=0.0025, n_last=20, n_max_gen=max_gen[obj_n], nth_gen=5)  # x_tol
                # TERMINATION2 = get_termination('f_tol', tol=0.0025, n_last=20, n_max_gen=max_gen[obj_n], nth_gen=5)  # f_tol
                # TERMINATION = [TERMINATION1, TERMINATION2]
                # for idx_ter in range(len(TERMINATION)):
                break_flag = False
                # for pop_size in [150, 200, 250, 300]:
                # for pop_size in [350, 400, 450, 500]:
                for pop_size in [400]:
                # for i in range(3):
                    # pop_size=400
                    # pop_size = 200
                    # pop_size += 200 * (obj_n-4)
                    ref_num = pop_size
                    # gen_n = pop_size
                    # gen_n = 300
                    gen_n = 100 # for Comparision NaiveEA and ORIGAMIG
                    if obj_n == 3:
                        pop_size  = 50
                        ref_num = pop_size
                        gen_n = 50
                    TERMINATION1 = get_termination('n_gen', gen_n)  # maxgen
                    TERMINATION = [TERMINATION1]
                    idx_ter = 0
                    
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
                                    save_history=SAVE_HISTORY
                                    # selection=SELECTION[idx_sel]
                                    )
                    problem = SGs1(player_num=obj_n+1, target_num=target_n, mincov=MINCOV)

                    # execute the optimization
                    res:Result = minimize(problem=problem,
                                algorithm=algorithm,
                                seed=1,
                                termination=TERMINATION[idx_ter], 
                                # termination=TERMINATION1, 
                                save_history=SAVE_HISTORY
                                )
                    ct_star_total = problem.ct_star_total

                    print('obj_n={}, target_n={}, gen_n={}, pop_size={}, 参数下，程序运行时间:{}秒'
                        .format(obj_n, target_n, gen_n, pop_size, res.exec_time))
                    # Time Efficiency 实验用到，注释
                    # TIME.append(res.exec_time)
                    # if not time_expensive[obj_idx, target_idx] and res.exec_time > 5000:
                    #     print('__main__ finish!')
                    #     tool.algorithm.dumpVariPickle(vari=TIME, name='TIME_target'+str(target_n)+tool.algorithm.getTime())
                    #     break_flag = True
                    #     time_expensive[:, target_idx] = True
                    #     exit()

                    if SHOW:

                        # 保存res的目录文件、数据文件
                        #　NOTE DATE, 同时model_save中包装了存储路径
                        model_save = resSave(res, Solver, res.exec_time, RES_DIR_NAME)
                        # NOTE 文件名
                        # para1 = {'Ablation':'Lcode', 'Popsize':pop_size, 'Code':CODE, 'gen':gen_n}  # AblationStudy
                        para1 = {'Popsize':pop_size, 'Code':CODE, 'gen':gen_n}  # boolearn scoring mechanism by default
                        para_filename:str = tool.algorithm.paras2Str(para1)
                        # NOTE 上一级目录名
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
                    gc.collect()
                if break_flag:
                    # exit()
                    break

filename = '-'.join([Solver, 'res_dir', str(DESCRIPTION)]) + '.json'
# 保存目录json文件
with open(filename, 'w') as json_file:
    json.dump(obj=SAVE_PATH, fp=json_file, indent=4)