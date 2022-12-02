from MOSGs.IterativeConstraints import IterativeConstrains
from securitygame_core.MO_security_game import MOSG

import time

import os
import numpy as np
object_np = np.arange(7,8)
obj_n = 8
target_np0 = np.array([10])
target_np1 = np.linspace(25, 100, 1, dtype=np.int32)
target_np2 = np.linspace(200, 1000, 5, dtype=np.int32)
target_np3 = np.array([200,400])
target_np = np.hstack((target_np0, target_np1))
target_n = 100
time_record = np.full((object_np.size, target_np.size), 0.)
Solver = 'ORIGAMIM'

for object_idx in range(object_np.size):
    for target_idx in range(target_np.size):
# for object_idx in range(1):
#     for target_idx in range(1):
        obj_n = object_np[object_idx]
        target_n = target_np[target_idx]
        problem = MOSG(player_num=obj_n+1, target_num=target_n)  # the information of MOSG will not change
        # TODO:为什么IterativeConstrains对想法在新的一轮迭代中不会销毁，而是沿用上一轮旧的对象
        solver = IterativeConstrains(MOSG=problem, Solver=Solver)
        # DONE:我觉得b最好写成和obj_n等长，毕竟后续代码优化要用
        T1 = time.perf_counter()
        solver.do(b=np.full((obj_n,), float('-inf')))
        T2 = time.perf_counter()
        time_record[object_idx, target_idx] = T2 - T1  # * 1000
        print('obj_n={}, target_n={}参数下，程序运行时间:{}秒'.format(obj_n, target_n, time_record[object_idx, target_idx]) )
res = solver.result(TIME_RECORD=time_record)
res1 = np.load('TIME_RECORD.npy')
print(res1)