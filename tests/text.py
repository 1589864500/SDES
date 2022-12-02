'''生成path_image'''

import numpy as np
from typing import *
import os
import sys
sys.path.append('./')
import tool.algorithm

'''暂时不知道功能'''
# def func(ct, attack_target, target):
#     # 根据ct相对大小重排序，并计算最终方案
#     idx:np.ndarray = np.argsort(ct)  # 从小到大排序
#     print(idx)
#     ct = ct[idx]
#     attack_target = attack_target[idx]  # 按方案相对大小从小到大排序
#     count = np.zeros(attack_target.shape)  # count记录违反次数，越小越好
#     count[0] = np.sum(attack_target[1:] != target)  # 默认的最优解是选择第一个方案的可行解：统计idx>0部分攻击集目标是t的数量（为t是不合理的）
#     for obj_idx in range(1, attack_target.size):
#             # 以obj_idx为界，根据左右变化更新count
#             # obj_target将从中间进入左边，而右边最左边的一位将进入中间
#             # 若进入左边的不为target则视为违反
#         count[obj_idx] = count[obj_idx - 1]
#         if attack_target[obj_idx-1] == target:
#             count[obj_idx] += 1
#             # 若进入中间的为target，相当于违反约束程度变小了
#         if attack_target[obj_idx] != target:
#             count[obj_idx] -= 1
#     return ct[np.argmin(count)],count
# ct = np.array([5,4,3,2,1,0])
# attack_target = np.array([0,1,0,1,1,1])
# target = 1
# print(func(ct,attack_target, target))

'''生成path_image'''
image_path = {}
RES_DIR = 'Results/floatscoringmechanism_ind/res/refined_numpy'
name_path = tool.algorithm.loadVariJson(name='temp/GeneticMOSGinteger-res_dir-FSM_SEED0-10.json')
for files in name_path.values():
    for file in files:
        filename = file.split('/')[-1]
        res = tool.algorithm.loadVariPickle(file).F
        tool.algorithm.dumpVariPickle(vari=res, name=os.path.join(RES_DIR, filename))
        image_path[file] = os.path.join(RES_DIR, filename)
tool.algorithm.dumpVariJson(vari=image_path, name='temp/FSM_image_path_new.json')
