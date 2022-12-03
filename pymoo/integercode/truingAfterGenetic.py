
import os
from re import T
import gc


import numpy as np
import math

import sys
# sys.path.append('/home/wuyp/Projects/pymoo/')
sys.path.append('')

from pymoo.MOSGsGeneticSolver.performance import Performance
from pymoo.MOSGsGeneticSolver.visualization import Visualization
from pymoo.model.result import Result
import pymoo.util.nds.non_dominated_sorting
from pymoo.MOSGsGeneticSolver.performance import Performance


import tool.algorithm
from pymoo.integercode.truing import Truing
# from pymoo.floatscoringmechanism.truing import Truing
# from pymoo.MOSGsGeneticSolver.truing import Truing

para_dir = {'obj':5, 'target':75}

RES_DIR:str = './Results'
RES_DIR_W = os.path.join(os.getcwd(), 'Results', tool.algorithm.paras2Str(para_dir))
RES_DIR_R = os.path.join(os.getcwd(), 'Results', 'gmosgInt')

# 需要保存：pf  mosg  partmosg  gmosg  gmosg*N  gmosg_mincov_1  ggmosg  gmosg_total


'''读入演化算法数据，过mincov函数提升一次效果'''
# # 读取已经处理好的数据

# gen=300固定
# fname1 = ''
# fname1 = 'Results/obj4target50/GeneticMOSGinteger-Popsize200Codeintegergen300-2022_06_12_11_20-94843.txt'
# fname1 = 'Results/obj4target100/GeneticMOSGinteger-Popsize200Codeintegergen300-2022_06_21_09_17-209166.txt'
# fname1 = 'Results/obj5target75/GeneticMOSGinteger-Popsize400Codeintegergen300-2022_06_21_09_26-467197.txt'

# NOTE Sensitivity (ORIGAMIG)
# Pop=150
# N=4 T=25
# fname1 = 'Results/obj4target25/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_26_12_18-71994.txt'
# fname2 = 'Results/obj4target25/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_13_55-73686.txt'
# fname3 = 'Results/obj4target25/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_14_06-76134.txt'
# fname4 = 'Results/obj4target25/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_14_27-76549.txt'
# # N=5 T=25
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=50
# fname1 = 'Results/obj4target50/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_14_39-167351.txt'
# fname2 = 'Results/obj4target50/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_14_58-164501.txt'
# fname3 = 'Results/obj4target50/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_15_36-163310.txt'
# fname4 = 'Results/obj4target50/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_15_17-163812.txt'
# # N=5 T=50
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=75
# fname1 = 'Results/obj4target75/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_15_56-264414.txt'
# fname2 = 'Results/obj4target75/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_16_28-278364.txt'
# fname3 = 'Results/obj4target75/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_16_58-268810.txt'
# fname4 = 'Results/obj4target75/GeneticMOSGinteger-Popsize150Codeintegergen600-2022_05_27_17_29-268154.txt'
# # N=5 T=75
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=100
# fname1 = ''
# fname2 = ''
# fname3 = ''
# fname4 = ''
# NOTE Pop=200
# N=4 T=25
# fname1 = 'Results/obj4target25/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_13_57-112760.txt'
# fname2 = 'Results/obj4target25/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_14_08-107268.txt'
# fname3 = 'Results/obj4target25/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_14_18-118669.txt'
# fname4 = 'Results/obj4target25/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_14_29-113305.txt'
# # N=5 T=25
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=50
# fname1 = 'Results/obj4target50/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_15_40-221410.txt'
# fname2 = 'Results/obj4target50/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_15_21-221208.txt'
# fname3 = 'Results/obj4target50/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_14_43-220807.txt'
# fname4 = 'Results/obj4target50/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_26_12_35-285915.txt'
# # N=5 T=50
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=75
# fname1 = 'Results/obj4target75/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_17_36-367367.txt'
# fname2 = 'Results/obj4target75/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_17_05-370713.txt'
# fname3 = 'Results/obj4target75/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_16_34-364774.txt'
# fname4 = 'Results/obj4target75/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_16_03-366624.txt'
# # N=5 T=75
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=100
# fname1 = ''
# fname2 = ''
# fname3 = ''
# fname4 = ''
# NOTE Pop=250
# NOTE 重复实验算方差
# N=4 T=25
# fname1 = 'Results/obj4target25/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_14_32-159667.txt'
# fname2 = 'Results/obj4target25/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_14_00-154213.txt'
# fname3 = 'Results/obj4target25/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_14_21-165786.txt'
# fname4 = 'Results/obj4target25/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_14_10-149407.txt'
# # N=5 T=25
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=50
# fname1 = 'Results/obj4target50/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_15_45-278453.txt'
# fname2 = 'Results/obj4target50/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_15_26-287319.txt'
# fname3 = 'Results/obj4target50/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_15_07-280203.txt'
# fname4 = 'Results/obj4target50/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_14_48-296889.txt'
# # N=5 T=50
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=75
# fname1 = 'Results/obj4target75/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_17_44-451699.txt'
# fname2 = 'Results/obj4target75/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_17_13-461746.txt'
# fname3 = 'Results/obj4target75/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_16_42-447280.txt'
# fname4 = 'Results/obj4target75/GeneticMOSGinteger-Popsize250Codeintegergen600-2022_05_27_16_12-469763.txt'
# # N=5 T=75
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=100
# fname1 = ''
# fname2 = ''
# fname3 = ''
# fname4 = ''
# NOTE Pop=300
# N=4 T=25
# fname1 = 'Results/obj4target25/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_14_04-220979.txt'
# fname2 = 'Results/obj4target25/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_14_14-217263.txt'
# fname3 = 'Results/obj4target25/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_14_25-233056.txt'
# fname4 = 'Results/obj4target25/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_14_36-230816.txt'
# # N=5 T=25
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=50
# fname1 = 'Results/obj4target50/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_14_55-377424.txt'
# fname2 = 'Results/obj4target50/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_15_14-367783.txt'
# fname3 = 'Results/obj4target50/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_15_33-367101.txt'
# fname4 = 'Results/obj4target50/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_15_52-363678.txt'
# # N=5 T=50
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=75
# fname1 = 'Results/obj4target75/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_16_22-583396.txt'
# fname2 = 'Results/obj4target75/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_16_53-563481.txt'
# fname3 = 'Results/obj4target75/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_17_24-585459.txt'
# fname4 = 'Results/obj4target75/GeneticMOSGinteger-Popsize300Codeintegergen600-2022_05_27_17_54-555053.txt'
# # N=5 T=75
# fname1 = ''
# fname2 = ''
# fname3 = ''
# # N=4 T=100 P=200
# fname1 = 'Results/obj4target100/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_20_14-536665.txt'
# fname2 = 'Results/obj4target100/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_19_31-484511.txt'
# fname3 = 'Results/obj4target100/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_18_50-487855.txt'
# fname4 = 'Results/obj4target100/GeneticMOSGinteger-Popsize200Codeintegergen600-2022_05_27_18_09-481916.txt'
# NOTE 算均值
# N=4 T=25
# fname1 = '/home/wuyp/Projects/pymoo/Results/obj4target25/GeneticMOSGinteger-Popsize150Codeintegergen300-2022_05_28_15_04-43882.txt'
# fname2 = '/home/wuyp/Projects/pymoo/Results/obj4target25/GeneticMOSGinteger-Popsize200Codeintegergen300-2022_05_28_15_05-56776.txt'
# fname3 = '/home/wuyp/Projects/pymoo/Results/obj4target25/GeneticMOSGinteger-Popsize250Codeintegergen300-2022_05_28_15_07-79460.txt'
# fname4 = '/home/wuyp/Projects/pymoo/Results/obj4target25/GeneticMOSGinteger-Popsize300Codeintegergen300-2022_05_28_15_09-112048.txt'
# N=4 T=50
# fname1 = '/home/wuyp/Projects/pymoo/Results/obj4target50/GeneticMOSGinteger-Popsize150Codeintegergen300-2022_05_28_15_10-85754.txt'
# fname2 = '/home/wuyp/Projects/pymoo/Results/obj4target50/GeneticMOSGinteger-Popsize200Codeintegergen300-2022_05_28_15_12-117548.txt'
# fname3 = '/home/wuyp/Projects/pymoo/Results/obj4target50/GeneticMOSGinteger-Popsize250Codeintegergen300-2022_05_28_15_15-155200.txt'
# fname4 = '/home/wuyp/Projects/pymoo/Results/obj4target50/GeneticMOSGinteger-Popsize300Codeintegergen300-2022_05_28_15_19-197586.txt'
# # N=4 T=75
# fname1 = '/home/wuyp/Projects/pymoo/Results/obj4target75/GeneticMOSGinteger-Popsize150Codeintegergen300-2022_05_28_15_21-136359.txt'
# fname2 = '/home/wuyp/Projects/pymoo/Results/obj4target75/GeneticMOSGinteger-Popsize200Codeintegergen300-2022_05_28_15_25-183638.txt'
# fname3 = '/home/wuyp/Projects/pymoo/Results/obj4target75/GeneticMOSGinteger-Popsize250Codeintegergen300-2022_05_28_15_29-238533.txt'
# fname4 = '/home/wuyp/Projects/pymoo/Results/obj4target75/GeneticMOSGinteger-Popsize300Codeintegergen300-2022_05_28_15_35-300526.txt'
# # N=4 T=100
# fname1 = '/home/wuyp/Projects/pymoo/Results/obj4target100/GeneticMOSGinteger-Popsize150Codeintegergen300-2022_05_28_15_38-179827.txt'
# fname2 = '/home/wuyp/Projects/pymoo/Results/obj4target100/GeneticMOSGinteger-Popsize200Codeintegergen300-2022_05_28_15_42-234323.txt'
# fname3 = '/home/wuyp/Projects/pymoo/Results/obj4target100/GeneticMOSGinteger-Popsize250Codeintegergen300-2022_05_28_15_48-306930.txt'
# fname4 = '/home/wuyp/Projects/pymoo/Results/obj4target100/GeneticMOSGinteger-Popsize300Codeintegergen300-2022_05_28_15_55-381819.txt'
# # N=5 T=25
# fname1 = '/home/wuyp/Projects/pymoo/Results/obj5target25/GeneticMOSGinteger-Popsize350Codeintegergen300-2022_05_28_15_07-182095.txt'
# fname2 = '/home/wuyp/Projects/pymoo/Results/obj5target25/GeneticMOSGinteger-Popsize400Codeintegergen300-2022_05_28_15_12-230886.txt'
# fname3 = '/home/wuyp/Projects/pymoo/Results/obj5target25/GeneticMOSGinteger-Popsize450Codeintegergen300-2022_05_28_15_17-265322.txt'
# fname4 = '/home/wuyp/Projects/pymoo/Results/obj5target25/GeneticMOSGinteger-Popsize500Codeintegergen300-2022_05_28_15_22-307812.txt'
# # N=5 T=50
# fname1 = '/home/wuyp/Projects/pymoo/Results/obj5target50/GeneticMOSGinteger-Popsize350Codeintegergen300-2022_05_28_15_28-328744.txt'
# fname2 = '/home/wuyp/Projects/pymoo/Results/obj5target50/GeneticMOSGinteger-Popsize400Codeintegergen300-2022_05_28_15_36-407038.txt'
# fname3 = '/home/wuyp/Projects/pymoo/Results/obj5target50/GeneticMOSGinteger-Popsize450Codeintegergen300-2022_05_28_15_44-462963.txt'
# fname4 = '/home/wuyp/Projects/pymoo/Results/obj5target50/GeneticMOSGinteger-Popsize500Codeintegergen300-2022_05_28_15_54-538231.txt'
# # N=5 T=75
# fname1 = '/home/wuyp/Projects/pymoo/Results/obj5target75/GeneticMOSGinteger-Popsize350Codeintegergen300-2022_05_28_16_02-434852.txt'
# fname2 = '/home/wuyp/Projects/pymoo/Results/obj5target75/GeneticMOSGinteger-Popsize400Codeintegergen300-2022_05_28_16_11-503601.txt'
# fname3 = '/home/wuyp/Projects/pymoo/Results/obj5target75/GeneticMOSGinteger-Popsize450Codeintegergen300-2022_05_28_16_22-595335.txt'
# fname4 = '/home/wuyp/Projects/pymoo/Results/obj5target75/GeneticMOSGinteger-Popsize500Codeintegergen300-2022_05_28_16_35-687493.txt'

# NOTE load pf for ORIGAMIG (also named BSMMINOCV)
# fname1 = 'Results/obj4target50/typeBSMMINCOV2022_06_12_21_10_01'
# fname1 = 'Results/obj4target100/typeBSMMINCOV2022_06_21_16_19_52'
fname1 = 'Results/obj5target75/typeBSMMINCOV2022_06_21_16_24_12'
pf1 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname1))
fname1 = 'BSMMINCOV'

# NOTE load ORIGAMIDG (Sensitivity ORIGAMIDG or one ORIGAMIDG)
# NOTE Sensitivity ORIGAMIDG
# N=4 T=50
# fname0_1 = 'Results/obj4target50/typeDGpopsize50gen102022_06_20_21_59_11'
# fname0_2 = 'Results/obj4target50/typeDGpopsize50gen202022_06_20_22_51_28'
# fname0_3 = 'Results/obj4target50/typeDGpopsize50gen302022_06_20_23_49_36'
# fname0_4 = 'Results/obj4target50/typeDGpopsize50gen402022_06_21_00_45_58'
# fname0_5 = 'Results/obj4target50/typeDGpopsize50gen502022_06_21_13_30_16'
# fname0_6 = 'Results/obj4target50/typeDGpopsize50gen602022_06_21_14_25_24'
# fname0_7 = 'Results/obj4target50/typeDGpopsize100gen502022_06_12_22_23_54'
# N=4 T=100
# FIX gen  Adjust pop
# fname0_1 = 'Results/obj4target100/typeDGpopsize10gen102022_07_01_14_03_36'
# fname0_2 = 'Results/obj4target100/typeDGpopsize20gen102022_07_01_14_14_18'
# fname0_3 = 'Results/obj4target100/typeDGpopsize30gen102022_07_01_14_24_56'
# fname0_4 = 'Results/obj4target100/typeDGpopsize40gen102022_07_01_14_36_23'
# fname0_5 = 'Results/obj4target100/typeDGpopsize50gen102022_07_01_14_45_46'
# Adjust gen  FIX pop
# fname0_1 = 'Results/obj4target100/typeDGpopsize50gen102022_07_01_14_45_46'
# fname0_2 = 'Results/obj4target100/typeDGpopsize50gen202022_07_01_16_06_21'
# fname0_3 = 'Results/obj4target100/typeDGpopsize50gen302022_07_01_17_42_02'
# fname0_4 = 'Results/obj4target100/typeDGpopsize50gen402022_07_01_19_35_56'
# fname0_5 = 'Results/obj4target100/typeDGpopsize50gen502022_07_01_21_38_15'
# N=5 T=75
# FIX gen  Adjust pop
fname0_1 = 'Results/obj5target75/typeDGpopsize10gen102022_07_01_14_02_37'
fname0_2 = 'Results/obj5target75/typeDGpopsize20gen102022_07_01_14_18_44'
fname0_3 = 'Results/obj5target75/typeDGpopsize30gen102022_07_01_14_40_27'
fname0_4 = 'Results/obj5target75/typeDGpopsize40gen102022_07_01_14_59_38'
fname0_5 = 'Results/obj5target75/typeDGpopsize50gen102022_07_01_15_24_01'
# Adjust gen  FIX pop
# fname0_1 = 'Results/obj5target75/typeDGpopsize50gen102022_07_01_15_24_01'
# fname0_2 = 'Results/obj5target75/typeDGpopsize50gen202022_07_01_17_56_36'
# fname0_3 = 'Results/obj5target75/typeDGpopsize50gen302022_07_01_21_31_06'
# fname0_4 = 'Results/obj5target75/typeDGpopsize50gen402022_07_02_01_03_55'
# fname0_5 = 'Results/obj5target75/typeDGpopsize50gen502022_07_02_04_52_01'

pf0_1 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname0_1))
pf0_2 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname0_2))
pf0_3 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname0_3))
pf0_4 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname0_4))
pf0_5 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname0_5))
# pf0_6 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname0_6))
# pf0_7 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname0_7))
fname0_1 = 'pop10gen10'
fname0_2 = 'pop20gen10'
fname0_3 = 'pop30gen10'
fname0_4 = 'pop40gen10'
fname0_5 = 'pop50gen10'
# fname0_1 = 'pop50gen10'
# fname0_2 = 'pop50gen20'
# fname0_3 = 'pop50gen30'
# fname0_4 = 'pop50gen40'
# fname0_5 = 'pop50gen50'
# fname0_6 = ''
# fname0_7 = ''
pf0_1 = np.vstack([pf0_1, pf1])
pf0_2 = np.vstack([pf0_2, pf1])
pf0_3 = np.vstack([pf0_3, pf1])
pf0_4 = np.vstack([pf0_4, pf1])
pf0_5 = np.vstack([pf0_5, pf1])
# pf0_6 = np.vstack([pf0_6, pf1])
# pf0_7 = np.vstack([pf0_7, pf1])

# one ORIGAMIDG
# N=4, T=100
# fname0 = 'Results/obj4target100/typeDGpopsize10gen102022_07_01_10_27_39'
# pf0 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname0))
# N=5, T=75
# fname0 = 'Results/obj5target75/typeDGpopsize10gen102022_07_01_10_36_23'
# pf0 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname0))
# fname0 = 'pop10gen10'
# pf0 = np.vstack([pf0, pf1])


# NOTE Ablation
# N=5, T=50
# fname1 = '/home/wuyp/Projects/pymoo/Results/obj5target50/GeneticMOSGinteger-Popsize400Coderealgen300-2022_05_28_15_55-110970.txt'
# fname2 = '/home/wuyp/Projects/pymoo/Results/obj5target50/GeneticMOSGinteger-AblationLcodePopsize400Codeintegergen300-2022_05_28_15_36-318389.txt'
# fname3 = 'Results/obj5target50/GeneticMOSGinteger-Popsize400Codeintegergen300-2022_05_28_15_36-407038.txt'
# fname4 = ''
# N=5, T=75
# fname1 = '/home/wuyp/Projects/pymoo/Results/obj5target75/GeneticMOSGinteger-Popsize400Coderealgen300-2022_05_28_15_52-111491.txt'
# fname2 = '/home/wuyp/Projects/pymoo/Results/obj5target75/GeneticMOSGinteger-AblationLcodePopsize400Codeintegergen300-2022_05_28_15_37-378245.txt'
# fname3 = 'Results/obj5target75/GeneticMOSGinteger-Popsize400Codeintegergen300-2022_05_28_01_02-514654.txt'
# fname4 = ''

# NOTE Ablation
# res1 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname1)) # ccode
# pf1 = res1.F
# res2 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname2)) # Lcode
# pf2 = res2.F
# res3 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname3)) # Lcode with heuopt
# pf3 = res3.F
# res4 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname2)) # Lcode with MINCOV
# model4 = Truing(res=res4)
# model4.mosgSearch_pop()
# pf4 = model4.fit_pf


# NOTE load fname for ORIGAMIG to do MINCOV
# NOTE or load for the input of ORIGAMIDG
# fname1 = 'Results/obj4target100/GeneticMOSGinteger-Popsize200Codeintegergen300-2022_06_21_09_17-209166.txt'
# fname1 = 'Results/obj5target75/GeneticMOSGinteger-Popsize400Codeintegergen300-2022_06_21_09_26-467197.txt'
# res1 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname1))
# res2 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname2))
# res3 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname3))
# res4 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname4))
# res5 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname5))
# res6 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname6))
# res7 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname7))
# res8 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname8))
# res9 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname9))
# res10 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname10))
# res11 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname11))
# res12 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), fname12))

# model1 = Truing(res=res1)
# model2 = Truing(res=res1)
# model3 = Truing(res=res1)
# model4 = Truing(res=res1)
# model5 = Truing(res=res1)
# model6 = Truing(res=res1)
# model7 = Truing(res=res1)
# model8 = Truing(res=res1)
# model9 = Truing(res=res1)
# model10 = Truing(res=res1)
# model7 = Truing(res=res7)
# model8 = Truing(res=res8)
# model9 = Truing(res=res9)
# model10 = Truing(res=res10)
# model11 = Truing(res=res11)
# model12 = Truing(res=res12)

# NOTE mosgSearch_pop using MIN-COV
# model1.mosgSearch_pop()
# pf1 = model1.fit_pf
# model2.mosgSearch_pop()
# pf2 = model2.fit_pf
# model3.mosgSearch_pop()
# pf3 = model3.fit_pf
# model4.mosgSearch_pop()
# pf4 = model4.fit_pf
# model5.mosgSearch_pop()
# pf5 = model5.fit_pf
# model6.mosgSearch_pop()
# pf6 = model6.fit_pf
# model7.mosgSearch_pop()
# pf7 = model7.fit_pf
# model8.mosgSearch_pop()
# pf8 = model8.fit_pf
# model9.mosgSearch_pop()
# pf9 = model9.fit_pf
# model10.mosgSearch_pop()
# pf10 = model10.fit_pf
# model11.mosgSearch_pop()
# pf11 = model11.fit_pf
# model12.mosgSearch_pop()
# pf12 = model12.fit_pf

# NOTE geneticSearch using double-genetic\
# NOTE fixing pop_size=50, adjusting gen_n=10-50
# pop_size, gen_n = 50, 50 
# pop_size, gen_n = 50, 40 
# pop_size, gen_n = 50, 30 
# pop_size, gen_n = 50, 20 
# NOTE fixing gen_n=10, adjusting pop_size=10-50
# pop_size, gen_n = 50, 10  
# pop_size, gen_n = 40, 10 
# pop_size, gen_n = 30, 10 
# pop_size, gen_n = 20, 10 
# pop_size, gen_n = 10, 10 
# model1.geneticSearch(pop_size, gen_n)
# model1.geneticSearch(pop_size=50, gen_n=40)
# pf1 = model1.fit_pf
# model2.geneticSearch(pop_size=20, gen_n=40)
# pf2 = model2.fit_pf
# model3.geneticSearch(pop_size=30, gen_n=40)
# pf3 = model3.fit_pf
# model4.geneticSearch(pop_size=40, gen_n=40)
# pf4 = model4.fit_pf
# model5.geneticSearch(pop_size=50, gen_n=40)
# pf5 = model5.fit_pf
# model6.geneticSearch(pop_size=10, gen_n=40)
# pf6 = model1.fit_pf
# model7.geneticSearch(pop_size=20, gen_n=40)
# pf7 = model2.fit_pf
# model8.geneticSearch(pop_size=30, gen_n=40)
# pf8 = model3.fit_pf
# model9.geneticSearch(pop_size=40, gen_n=40)
# pf9 = model4.fit_pf
# model10.geneticSearch(pop_size=50, gen_n=40)
# pf10 = model5.fit_pf

# # NOTE 将精修后的pf:numpy本地持久化
# fname1 = tool.algorithm.paras2Str({"type":"DG", 'popsize':pop_size, 'gen':gen_n})
# fname1 = 'P400Accode'
# fname1 = 'P400ALcode'
# fname1 = 'P400AheuristicOpt'
# fname1 = 'P400Amincov'
# fname2 = 'popsize20gen40'
# fname3 = 'popsize30gen40'
# fname4 = 'popsize40gen40'
# fname5 = 'popsize50gen40'
# fname6 = 'popsize10gen50'
# fname7 = 'popsize20gen50'
# fname8 = 'popsize30gen50'
# fname9 = 'popsize40gen50'
# fname10 = 'popsize50gen50'
# fname11 = 'CodeintegerX5Ter2'
# fname12 = 'CodeintegerXT5er3'
# tool.algorithm.dumpVariPickle(pf1, path=os.path.join(RES_DIR_W, fname1+tool.algorithm.getTime()))
# tool.algorithm.dumpVariPickle(pf2, path=os.path.join(RES_DIR_W, fname2+tool.algorithm.getTime()))
# tool.algorithm.dumpVariPickle(pf3, path=os.path.join(RES_DIR_W, fname3+tool.algorithm.getTime()))
# tool.algorithm.dumpVariPickle(pf4, path=os.path.join(RES_DIR_W, fname4+tool.algorithm.getTime()))
# tool.algorithm.dumpVariPickle(pf5, path=os.path.join(RES_DIR_W, fname5+tool.algorithm.getTime()))
# tool.algorithm.dumpVariPickle(pf6, path=os.path.join(RES_DIR_W, fname6+tool.algorithm.getTime()))
# tool.algorithm.dumpVariPickle(pf7, path=os.path.join(RES_DIR_W, fname7+tool.algorithm.getTime()))
# tool.algorithm.dumpVariPickle(pf8, path=os.path.join(RES_DIR_W, fname8+tool.algorithm.getTime()))
# tool.algorithm.dumpVariPickle(pf9, path=os.path.join(RES_DIR_W, fname9+tool.algorithm.getTime()))
# tool.algorithm.dumpVariPickle(pf10, path=os.path.join(RES_DIR_W, fname10+tool.algorithm.getTime()))
# tool.algorithm.dumpVariPickle(pf11, path=os.path.join(RES_DIR_W, fname11))
# tool.algorithm.dumpVariPickle(pf12, path=os.path.join(RES_DIR_W, fname12))

'''
同时要把mosg读进来比较
'''
# fname1 = 'CodeintegerX0Ter2'
# fname2 = 'CodeintegerX0Ter3'
# fname3 = 'CodeintegerX1Ter2'
# fname4 = 'CodeintegerXT1er3'
# fname5 = 'CodeintegerX2Ter2'
# fname6 = 'CodeintegerX2Ter3'
# fname7 = 'CodeintegerX3Ter2'
# fname8 = 'CodeintegerX3Ter3'
# fname9 = 'CodeintegerX4Ter2'
# fname10 = 'CodeintegerX4Ter3'
# fname11 = 'CodeintegerX5Ter2'
# fname12 = 'CodeintegerXT5er3'
# NOTE pf for ORIGAMIG
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target25/GMOSG290MOSG10')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target50/GMOSG3112MOSG17')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target75/GMOSG3380MOSG21')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target100/GMOSG10460MOSG16')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target200/GMOSG5570MOSG18')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target400/GMOSG3996MOSG19')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj4target25/PF4030P2M154P2G270P3M204P3G324P4M252P4G385P5M301P5G446')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj4target50/PF9370P2M154P2G255P3M200P3G287P4M253P4G359P5M303P5G393')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj4target75/PF12933P2M154P2G188P3M204P3G265P4M251P4G300P5M304P5G343')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj4target100/PF12455P2M154P2G201P3M204P3G251P4M254P4G299P5M304P5G344')
# pf = tool.algorithm.loadVariPickle(path=os.path.join(RES_DIR_R, fname12))
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj5target25/GMOSG5312MOSG1248')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj5target50/GMOSG7794MOSG960')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj5target75/PF10332P2M405P2G462P3M405P3G473')
# NOTE pf for ORIGAMIDG
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target25/GMOSG290MOSG10')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target50/GMOSG3112MOSG17')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target75/GMOSG3380MOSG21')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target100/GMOSG10460MOSG16')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target200/GMOSG5570MOSG18')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj3target400/GMOSG3996MOSG19')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj4target25/PF4030P2M154P2G270P3M204P3G324P4M252P4G385P5M301P5G446')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj4target50/booleanscoringmechanism11574ORIGAMIDGP1160P2357P3716P4419P5400P6386P72564')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj4target75/PF12933P2M154P2G188P3M204P3G265P4M251P4G300P5M304P5G343')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj4target100/PF12455P2M154P2G201P3M204P3G251P4M254P4G299P5M304P5G344')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj4target100/booleanscoringmechanism14466ORIGAMIDGP1120')
# pf = tool.algorithm.loadVariPickle(path=os.path.join(RES_DIR_R, fname12))
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj5target25/GMOSG5312MOSG1248')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj5target50/GMOSG7794MOSG960')
# pf = tool.algorithm.loadVariPickle(path='Results/pf/obj5target75/PF10332P2M405P2G462P3M405P3G473')
pf = tool.algorithm.loadVariPickle(path='Results/pf/obj5target75/booleanscoringmechanism12077ORIGAMIDGP1253')
# mosg 
# ORIGAMIM
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj3target25/ORIGAMIM-obj3target25-2022_04_11_00_20-232.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj3target50/ORIGAMIM-obj3target50-2022_04_11_00_20-978.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj3target75/ORIGAMIM-obj3target75-2022_04_11_00_20-2380.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj3target100/ORIGAMIM-obj3target100-2022_04_11_00_20-3717.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj3target200/ORIGAMIM-obj3target200-2022_04_11_00_21-18829.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj3target400/ORIGAMIM-obj3target400-2022_04_11_00_22-84894.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj4target25/ORIGAMIM-obj4target25-2022_04_11_00_48-3842.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj4target50/ORIGAMIM-obj4target50-2022_06_12_20_29-18055.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj4target75/ORIGAMIM-obj4target75-2022_04_11_00_48-35772.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj4target100/ORIGAMIM-obj4target100-2022_06_12_20_33-123478.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj5target25/ORIGAMIM-obj5target25-2022_04_11_06_34-98317.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj5target50/ORIGAMIM-obj5target50-2022_04_11_07_18-191793.txt').F
pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj5target75/ORIGAMIM-obj5target75-2022_04_11_07_22-252346.txt').F
# pf_origamim = tool.algorithm.loadVariPickle(path='Results/obj5target100/ORIGAMIM-obj5target100-2022_04_11_07_39-994018.txt').F
name_m = 'ORIGAMIM'
name_a = 'ORIGAMIA'
name_mbs = 'ORIGAMIMBS'
name_d = 'DIRECTMINCOV'
# ORIGAMIA
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj3target25/ORIGAMIA-obj3target25-2022_04_20_09_12-6593.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj3target50/ORIGAMIA-obj3target50-2022_04_20_09_12-27277.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj3target75/ORIGAMIA-obj3target75-2022_04_20_09_14-65129.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj3target100/ORIGAMIA-obj3target100-2022_04_30_22_22-148579.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj3target200/ORIGAMIA-obj3target200-2022_04_20_09_29-760542.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj3target400/ORIGAMIA-obj3target400-2022_04_20_10_22-3171904.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj4target25/ORIGAMIA-obj4target25-2022_04_24_23_59-39837.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj4target50/ORIGAMIA-obj4target50-2022_06_12_20_32-205515.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj4target75/ORIGAMIA-obj4target75-2022_04_26_09_36-981906.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj4target100/ORIGAMIA-obj4target100-2022_06_12_21_20-1864655.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj5target25/ORIGAMIA-obj5target25-2022_04_25_00_11-519603.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj5target50/ORIGAMIA-obj5target50-2022_04_25_00_35-1458974.txt').F
pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj5target75/ORIGAMIA-obj5target75-2022_04_26_16_29-2423346.txt').F
# pf_origamia = tool.algorithm.loadVariPickle(path='Results/obj5target100/ORIGAMIA-obj5target100-2022_04_26_17_54-5130899.txt').F
# ORIGAMIMBS
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj3target25/ORIGAMIMBS-obj3target25-2022_04_20_17_49-73.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj3target50/ORIGAMIMBS-obj3target50-2022_04_20_17_49-214.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj3target75/ORIGAMIMBS-obj3target75-2022_04_20_17_49-600.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj3target100/ORIGAMIMBS-obj3target100-2022_04_20_17_49-1133.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj3target200/ORIGAMIMBS-obj3target200-2022_04_20_17_49-7835.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj3target400/ORIGAMIMBS-obj3target400-2022_04_20_17_49-29813.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj4target25/ORIGAMIMBS-obj4target25-2022_04_25_00_00-624.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj4target50/ORIGAMIMBS-obj4target50-2022_06_12_20_29-2231.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj4target75/ORIGAMIMBS-obj4target75-2022_04_26_16_08-9784.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj4target100/ORIGAMIMBS-obj4target100-2022_06_12_20_31-20504.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj5target25/ORIGAMIMBS-obj5target25-2022_04_24_23_55-5394.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj5target50/ORIGAMIMBS-obj5target50-2022_04_24_23_55-8523.txt').F
pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj5target75/ORIGAMIMBS-obj5target75-2022_04_26_16_09-59899.txt').F
# pf_origamimbs = tool.algorithm.loadVariPickle(path='Results/obj5target100/ORIGAMIMBS-obj5target100-2022_04_26_16_11-115067.txt').F
# DIRECTMINCOV
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj3target25/DIRECTMINCOV-obj3target25-2022_04_21_11_13-1345.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj3target50/DIRECTMINCOV-obj3target50-2022_04_21_11_13-9726.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj3target75/DIRECTMINCOV-obj3target75-2022_04_21_11_14-20884.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj3target100/DIRECTMINCOV-obj3target100-2022_04_30_22_18-21019.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj3target200/DIRECTMINCOV-obj3target200-2022_04_21_11_16-95061.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj3target400/DIRECTMINCOV-obj3target400-2022_04_21_11_25-573312.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj4target25/DIRECTMINCOV-obj4target25-2022_04_30_23_21-8717.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj4target50/DIRECTMINCOV-obj4target50-2022_06_12_20_30-49183.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj4target75/DIRECTMINCOV-obj4target75-2022_04_21_13_11-101403.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj4target100/DIRECTMINCOV-obj4target100-2022_06_12_20_35-135858.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj5target25/DIRECTMINCOV-obj5target25-2022_04_30_23_23-124429.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj5target50/DIRECTMINCOV-obj5target50-2022_04_24_23_56-304308.txt').F
pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj5target75/DIRECTMINCOV-obj5target75-2022_04_23_15_12-344375.txt').F
# pf_directmincov = tool.algorithm.loadVariPickle(path='Results/obj5target100/DIRECTMINCOV-obj5target100-2022_04_23_15_21-543119.txt').F

# # show pf
res1 = tool.algorithm.loadVariPickle(path=os.path.join(os.getcwd(), 'Results/obj3target75/GeneticMOSGinteger-Popsize50CodeintegerX0T0tol0.0025-2022_05_06_17_04-4612.txt'))
# NOTE 用于和对比方法作比较 (for ORIGAMIG)
# Truing(res=res1, para_dir=para_dir).mulResCompare(pf_total=[pf, pf1, pf_origamim, pf_origamia, pf_origamimbs, pf_directmincov],
# name_total=['pf', fname1, name_m, name_a, name_mbs, name_d],
# indicator_igdplus=True, indicator_hv=True
# )
# NOTE 评估多个方法的变体（ORIGANMIG Ablation Sensitivity Study）
# Truing(res=res1, para_dir=para_dir).mulResCompare(pf_total=[pf, pf1, pf2, pf3, pf4, pf5], #pf_origamim, pf_origamia, pf_origamimbs, pf_directmincov],
# name_total=['pf', fname1, fname2, fname3, fname4, fname5], #name_m, name_a, name_mbs, name_d],
# indicator_igdplus=True, indicator_hv=True
# )
# NOTE mulResCompare for ORIGAMIDG Sensitivity
Truing(res=res1, para_dir=para_dir).mulResCompare(pf_total=[pf, pf0_1, pf0_2, pf0_3, pf0_4, pf0_5, pf1, pf_origamim, pf_origamia, pf_origamimbs, pf_directmincov],
name_total=['pf', fname0_1, fname0_2, fname0_3, fname0_4, fname0_5, fname1, name_m, name_a, name_mbs, name_d],
indicator_igdplus=True, indicator_hv=False
)
# NOTE mulResCompare for ORIGAMIDG
# Truing(res=res1, para_dir=para_dir).mulResCompare(pf_total=[pf, pf0, pf1, pf_origamim, pf_origamia, pf_origamimbs, pf_directmincov],
# name_total=['pf', fname0, fname1, name_m, name_a, name_mbs, name_d],
# indicator_igdplus=True, indicator_hv=False
# )
# NOTE 只评估一个方法 (for PF fixed scenario)
# Truing(res=res1, para_dir=para_dir).mulResCompare(pf_total=[pf, pf1], #pf_origamim, pf_origamim, pf_origamia, pf_origamimbs, pf_directmincov],
# name_total=['pf', fname1], #name_m, name_a, name_mbs, name_d],
# indicator_igdplus=True, indicator_hv=True
# )

# 把目前试过的GMOSG的最优的实验结果存下来
# pf_gmosg = np.vstack([pf1, pf5, pf6, pf7])
# pf_gmosg = Performance().getPF(pf_total=pf_gmosg)
# fname = tool.algorithm.paras2Str(para=para_dir)
# tool.algorithm.dumpVariPickle(pf_gmosg, path=os.path.join(RES_DIR_GMOSG, fname))

# show pf
# Truing(res=res1, para_dir=para_dir).mulResCompare(pf_total=[pf, old_F_mosg, old_F_mosg_part1, pf1,  pf5, pf6, pf7, pf_gmosg],
# name_total=['pf', 'pf_mosg', 'pf_partmosg', 'pf_gmosg', 'pf_gmosg*N', 'pf_gmosg_mincov_1', 'pf_ggmosg', 'pf_gmosg_total'],
# )
# Truing(res=res1, para_dir=para_dir).mulResCompare(pf_total=[pf, old_F_mosg, old_F_mosg_part1, pf1, pf6, pf7],
#                              name_total=['pf', 'pf_mosg', 'pf_partmosg', 'pf_gmosg_mincov_0', 'pf_gmosg_mincov_1', 'pf_ggmosg'])