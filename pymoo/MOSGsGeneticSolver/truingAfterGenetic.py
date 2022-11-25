
import os


import numpy as np
import math

import sys
sys.path.append('./')

from pymoo.MOSGsGeneticSolver.performance import Performance
from pymoo.MOSGsGeneticSolver.visualization import Visualization
from pymoo.model.result import Result
import pymoo.util.nds.non_dominated_sorting


import tool.algorithm
from pymoo.integercode.truing import Truing
from pymoo.problems.securitygame import SGs1

para_dir = {'obj':5, 'target':25}

RES_DIR:str = './Results'
Solver = 'GeneticMOSG1'
RES_DIR_GMOSG = os.path.join(os.getcwd(), 'Results', 'pfgmosg')

# 需要保存：pf  mosg  partmosg  gmosg  gmosg*N  gmosg_mincov_1  ggmosg  gmosg_total

# 读取已经处理好的数据
# fname1 = 'pf_gmosg_mincov_0'
# fname2 = 'pf_mosg'
# fname3 = 'pf_partmosg'
# fname5 = 'pf_gmosg*N'
# fname6 = 'pf_gmosg_mincov_1'
# fname7 = 'pf_ggmosg'
# fname_total = 'obj5target25'
fname1 = 'gmosg'
fname2 = 'ORIGAMIM'
fname3 = 'ORIGAMIA'
fname4 = 'ORIGAMIMBS'
fname5 = 'DIRECTMINCOV'
fname_pf = 'obj5target25_pf'
# pf2 = tool.algorithm.loadVariPickle('Results/obj5target25/ORIGAMIM-obj5target25-2022_04_11_07_15-96575.txt').F
# pf3 = tool.algorithm.loadVariPickle('Results/obj5target25/ORIGAMIA-obj5target25-2022_04_25_00_11-519603.txt').F
# pf4 = tool.algorithm.loadVariPickle('Results/obj5target25/ORIGAMIMBS-obj5target25-2022_04_25_00_00-5559.txt').F
# pf5 = tool.algorithm.loadVariPickle('Results/obj5target25/DIRECTMINCOV-obj5target25-2022_04_23_15_00-153081.txt').F
# pf = tool.algorithm.loadVariPickle('Results/pf/GMOSGX0T2P200074GMOSGX1T2P200053')

# # mosg
# p2 = 'Results/ORIGAMIM-obj_5_target_25-2022_02_20_19_20-135809'
# # PATH = os.path.abspath(p2)
# res2:Result = tool.algorithm.loadVariPickle(p2)
# old_F_mosg = res2.F

# # pf
# p3 = 'Results/pf/obj_5_target_25'
# # PATH = os.path.abspath(p3)
# pf: Performance = tool.algorithm.loadVariPickle(p3)
# flag = []
p3 = 'Results/pf/obj4target25/PF4030P2M154P2G270P3M204P3G324P4M252P4G385P5M301P5G446'
# PATH = os.path.abspath(p3)
pf: np.ndarray = tool.algorithm.loadVariPickle(p3)

# # gmosg*N # 没效果提升
# p5_1 = 'Results/obj5target25/GeneticMOSG1-obj5target25gen500pop_size500-2022_04_09_09_16-942652.txt'
# p5_2 = 'Results/obj5target25/GeneticMOSG1-obj5target25gen500pop_size500-2022_04_09_09_31-879027.txt'
# p5_3 = 'Results/obj5target25/GeneticMOSG1-obj5target25gen500pop_size500-2022_04_09_09_46-871727.txt'
# p5_4 = 'Results/obj5target25/GeneticMOSG1-obj5target25gen500pop_size500-2022_04_09_10_00-847909.txt'
# p5_5 = 'Results/obj5target25/GeneticMOSG1-obj5target25gen500pop_size500-2022_04_09_10_14-832781.txt'
# res5_1:Result = tool.algorithm.loadVariPickle(p5_1)
# res5_2:Result = tool.algorithm.loadVariPickle(p5_2)
# res5_3:Result = tool.algorithm.loadVariPickle(p5_3)
# res5_4:Result = tool.algorithm.loadVariPickle(p5_4)
# res5_5:Result = tool.algorithm.loadVariPickle(p5_5)
# res5 = [res5_1, res5_2, res5_3, res5_4, res5_4, res5_5]
# # print(part_F_gmosg_1.shape, part_F_gmosg.shape)

# gmosg_mincov_0
p1 = '/home/wuyp/Projects/pymoo/Results/obj4target25/GeneticMOSGinteger-Popsize200Codeintegergen300-2022_05_28_00_30-58916.txt'
res1:Result = tool.algorithm.loadVariPickle(p1)
# old_F_gmosg = res1.F

# gmosg_mincov_1
# p6 = '/home/wuyp/Projects/pymoo/Results/obj5target25/GeneticMOSG1-obj5target25gen500pop_size500mincov1-2022_04_11_22_49-3634942.txt'
# res6:Result = tool.algorithm.loadVariPickle(p6)

model1 = Truing(res=res1)
# model6 = Truing(res=res6)
# model7 = Truing(res=res6)
# model4 = Truing(res=res1)
# model5 = Truing(res=res1)

# # gmosg_mincov_0
# # 复杂度很低的精修方案
# model1.mosgSearch_pop()
# pf1 = model1.fit_pf

# # gmosg*N
# pf5_list = []
# for res5_i in res5:
#     model5_i = Truing(res=res5_i)
#     model5_i.mosgSearch_pop()
#     pf5_i = model5_i.fit_pf
#     pf5_list.append(pf5_i)
# pf5 = np.vstack(pf5_list)

# # partmosg
# # 在F_mosg中取出pop_size个nondominant solutions, 并展示pop_size大小
# sort_idx = Performance().getPF_idx(old_F_mosg)
# # print(sort_idx)  # 表明best popsize和top popsize不同，best popsize更均匀 更好
# # part1: best popsize == top popsize
# old_F_mosg_part1 = old_F_mosg[sort_idx[:len(pf1)]]

# ggmosg
# 再次利用演化算法搜优的改进，消耗大但效果提升不明显，考虑减少迭代次数
pf7_without_mincov = model1.geneticSearch()

'''存储'''
# _, fname1 = os.path.split(p1)
# fname2 = 'pf_mosg'
# fname3 = 'pf_partmosg'
# fname5 = 'pf_gmosg*N'
# fname6 = 'pf_gmosg_mincov_1'
# fname7 = 'pf_ggmosg'
# dir = os.path.join(RES_DIR_GMOSG, tool.algorithm.paras2Str(para_dir))
# tool.algorithm.creatDir(dir)
# tool.algorithm.dumpVariPickle(pf1, path=os.path.join(dir, fname1))
# tool.algorithm.dumpVariPickle(pf5, path=os.path.join(RES_DIR_GMOSG, fname5))
# tool.algorithm.dumpVariPickle(pf6, path=os.path.join(RES_DIR_GMOSG, fname6))
# tool.algorithm.dumpVariPickle(pf7, path=os.path.join(RES_DIR_GMOSG, fname7))
# tool.algorithm.dumpVariPickle(old_F_mosg, path=os.path.join(RES_DIR_GMOSG, fname2))
# tool.algorithm.dumpVariPickle(old_F_mosg_part1, path=os.path.join(RES_DIR_GMOSG, fname3))




# 把目前试过的GMOSG的最优的实验结果存下来
# pf_gmosg = np.vstack([pf1, pf5, pf6, pf7])
# pf_gmosg = Performance().getPF(pf_total=pf_gmosg)
# fname = tool.algorithm.paras2Str(para=para_dir)
# tool.algorithm.dumpVariPickle(pf_gmosg, path=os.path.join(RES_DIR_GMOSG, fname))

# show pf
Truing(res=res1, para_dir=para_dir).mulResCompare(pf_total=[pf, pf7_without_mincov],
name_total=['pf', fname1],
indicator_hv=True, indicator_igdplus=True
)
# Truing(res=res1, para_dir=para_dir).mulResCompare(pf_total=[pf, pf1, pf2, pf3, pf4, pf5],
# name_total=['pf', fname1, fname2, fname3, fname4, fname5],
# indicator_hv=True, indicator_igdplus=True
# )
# Truing(res=res1, para_dir=para_dir).mulResCompare(pf_total=[pf, old_F_mosg, old_F_mosg_part1, pf1, pf6, pf7],
#                              name_total=['pf', 'pf_mosg', 'pf_partmosg', 'pf_gmosg_mincov_0', 'pf_gmosg_mincov_1', 'pf_ggmosg'])