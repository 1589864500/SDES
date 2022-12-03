
from re import T
import time
import os
import datetime
import json
import sys
sys.path.append('./')


import numpy as np


import tool.algorithm
from MOSGs.IterativeConstraints import IterativeConstraints
from securitygame_core.MO_security_game import MOSG
from MOSGs.resultMOSGs import resultMOSGs
from pymoo.MOSGsGeneticSolver.performance import Performance
from pymoo.MOSGsGeneticSolver.visualization import Visualization

obj_np = np.arange(3,15)
target_np0 = np.array([10])  # idx=0
target_np1 = np.linspace(25, 100, 4, dtype=np.int32)  # idx=1-4
target_np2 = np.linspace(200, 1000, 5, dtype=np.int32)  # idx=5-9  -5 - -1
target_np3 = np.array([200,400])
target_np = np.hstack((target_np0, target_np1, target_np2))
time_record = np.full((obj_np.size, target_np.size), 0.)
SAVE_PATH = {}
RES_DIR = './Results/epsilon-constraint'
RES_DIR_NAME = 'Results'
NUM = ''  # 本次运行的序列号
SHOW = True
epsilon = 1
DEBUG = False

# NOTE ORIGAMIM
# n=3-5的实验做完了，n=5的target极限是200。
Solver = 'ORIGAMIM'
time_expensive = np.full(shape=[len(obj_np), target_np.size], fill_value=True)
time_expensive[0, 1:] = False  # obj=3
time_expensive[1, 1:7] = False  # 4
time_expensive[2, 1:5] = False  # 5

# # NOTE ORIGAMIA
# # n=3-5的实验做完了，n=5的target极限是200。
# Solver = 'ORIGAMIA'
# time_expensive = np.full(shape=[len(obj_np), target_np.size], fill_value=True)
# time_expensive[0, 1:6] = False  # obj=3
# time_expensive[1, 1:5] = False  # 4
# time_expensive[2, 1:4] = False  # 5
# time_expensive[3, 1] = False  # 6
# # time_expensive[4, 1] = False  # 7
# time_expensive[5, 1] = False  # 8

# # NOTE ORIGAMIMBS
# # n=3-5的实验做完了，n=5的target极限是200。
# Solver = 'ORIGAMIMBS'
# time_expensive = np.full(shape=[len(obj_np), target_np.size], fill_value=True)
# time_expensive[0, 1:] = False  # obj=3
# time_expensive[1, 1:6] = False  # 4
# time_expensive[2, 1:5] = False  # 5
# time_expensive[3, 1:4] = False  # 6
# time_expensive[4, 1:3] = False  # 7


# # NOTE DIRECTMINCOV
# # n=3-5的实验做完了，n=5的target极限是200。
# Solver = 'DIRECTMINCOV'
# time_expensive = np.full(shape=[len(obj_np), target_np.size], fill_value=True)
# time_expensive[0, 1:8] = False  # obj=3
# time_expensive[1, 1:6] = False  # 4
# time_expensive[2, 1:5] = False  # 5
# time_expensive[3, 1:5] = False  # 6
# time_expensive[4, 1:3] = False  # 7
# time_expensive[5, 1] = False  # 8



for object_idx in range(obj_np.size):
    for target_idx in range(target_np.size):
# for object_idx in range(2):
#     for target_idx in range(1):
# for object_idx in [0]:
#     for target_idx in [1]:

        if time_expensive[object_idx, target_idx]:
                continue

        obj_n = obj_np[object_idx]
        target_n = target_np[target_idx]
        print('solver:{}, obj:{}, tar:{}'.format(Solver, obj_n, target_n))
        problem = MOSG(player_num=obj_n+1, target_num=target_n)  # the information of MOSG will not change
        solver = IterativeConstraints(MOSG=problem, 
                Solver=Solver, 
                res_dir=RES_DIR, 
                res_dir_name=RES_DIR_NAME, 
                epsilon=epsilon,
                debug = DEBUG)
        T1 = time.perf_counter()
        solver.do(b=np.full((obj_n,), float('-inf')))
        T2 = time.perf_counter()
        time_record[object_idx, target_idx] = T2 - T1  # * 1000
        solver.result(running_time=time_record[object_idx, target_idx])
        print('obj_n={}, target_n={}参数下，程序运行时间:{}秒'.format(obj_n, target_n, time_record[object_idx, target_idx]))

        if time_record[object_idx, target_idx]/1000 > 5000:
                exit()

        if SHOW:
                # 本地持久化
                para = {'obj':obj_n, 'target':target_n}
                para_dir = tool.algorithm.paras2Str(para)
                save_path = solver.saveResult(para_dir)
                SAVE_PATH[para_dir] = save_path
                filename = '-'.join([Solver, 'res_dir', str(NUM)]) + '.json'
                file_dir = os.path.join(RES_DIR, filename)
                with open(file_dir, 'w') as json_file:
                        json.dump(obj=SAVE_PATH, fp=json_file, indent=4)

                # # fit可视化
                # # Visualization(n_rows=obj_n, title=para_temp).add(res.F).show()
                # filename = '-'.join([Solver, para_dir + '.png'])
                # file_dir = os.path.join(RES_DIR, para_dir, filename)
                # Visualization(n_rows=obj_n, fig_title=para,
                #               sharex=True, sharey=True,
                #               hspace=0, wspace=0).add(solver.res.F).save(fname=file_dir)

