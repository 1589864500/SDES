'''实验内容包括
PART1: 数据预处理, 汇总json文件
PART2: optimalAfterMINCOV.json修复程序 (程序断断续续运行导致部分没有写入地址记录文件)
PART3: 生成PF-ORIGAMIG(重复实验)
PART4:生成PF-ORIGAMI-M-A(基线)
PART5:合并PF(total)
'''

import os
from pickle import TRUE
from re import T
from tkinter import NONE


import numpy as np
import math
from typing import *
import json


import sys
sys.path.append('./')
from pymoo.MOSGsGeneticSolver.generatePF import generatePF
import tool.algorithm


import sys
from yaml import dump

DONE = []
# ['obj3target25', 'obj3target50', 'obj3target75', 'obj3target100', 'obj3target200', 'obj3target400', 'obj3target600', 'obj3target800', 'obj3target1000',
#      'obj4target25', 'obj4target50', 'obj4target75', 'obj4target100', 'obj4target200',
#      'obj7target25', 'obj7target50', 'obj8target25']

'''PART1:数据预处理, 汇总json文件'''
# NOTE 将_SEED012345678910类型的文件处理成_SEED0-30
PATH1 = 'GeneticMOSGinteger-res_dir-N=4GFIndependentRepeatExperiments_SEED1.json'
PATH1 = 'temp/GeneticMOSGinteger-res_dir-_e=1.json'
# PATH2 = 'GeneticMOSGinteger-res_dir-GFIndependentRepeatExperiments_SEED0123456789.json'
# PATH3 = 'GeneticMOSGinteger-res_dir-GFIndependentRepeatExperiments_SEED10111213141516171819.json'
# PATH4 = 'GeneticMOSGinteger-res_dir-N=4GFIndependentRepeatExperiments_SEED151617181920212223242526272829.json'
# PATH5 = 'GeneticMOSGinteger-res_dir-GFIndependentRepeatExperiments_SEED20212223242526272829.json'
# PATH6 = 'GeneticMOSGinteger-res_dir-ComparisonNaiveEAandORIGAMIG_ORIGAMIG_SEED.json'
path:List[Dict[str, str]] = [tool.algorithm.loadVariJson(PATH1)]
# path.append(tool.algorithm.loadVariJson(PATH2))
# path.append(tool.algorithm.loadVariJson(PATH3))
# path.append(tool.algorithm.loadVariJson(PATH4))
# path.append(tool.algorithm.loadVariJson(PATH5))
# path.append(tool.algorithm.loadVariJson(PATH6))
# # NOTE 从空文件开始
path_res = 'temp/GeneticMOSGinteger-res_dir-FSM_SEED0-10.json'
# path_res = 'GeneticMOSGinteger-res_dir-N=4GFIndependentRepeatExperiments_SEED0-30.json'
if not os.path.exists(path_res):
    res:Dict[str,List[str]] = {}
# NOTE 从已有文件后面继续写
else:
    res:Dict[str,List[str]] = tool.algorithm.loadVariJson(path=path_res)
for path_i in path:
    for key in path_i.keys():
        para:List[str] = key.split('/')
        if para[-2] in DONE:
            continue
        if para[-2] in res:
            res[para[-2]].append(key)
        else:
            res[para[-2]] = [key]
tool.algorithm.dumpVariJson(vari=res,name=path_res)

# NOTE 将SensitivityN=5/obj5target100类型的文件合并到SensitivityN=5
# PATH1 = 'Results/IndependentRepeat/res/SensitivityN=5/obj5target100'
# path:List[Dict[str, str]] = [tool.algorithm.loadVariJson(PATH1)]
# # # NOTE 从空文件开始
# path_res = 'Results/IndependentRepeat/res/SensitivityN=5.json'
# if not os.path.exists(path_res):
#     res:Dict[str,List[str]] = {}
# # NOTE 从已有文件后面继续写
# else:
#     res:Dict[str,List[str]] = tool.algorithm.loadVariJson(path=path_res)
# para = PATH1.split('/')[-1]
# res[para] = path[0]
# tool.algorithm.dumpVariJson(vari=res,name=path_res)


'''PART2:optimalAfterMINCOV.json修复程序（程序断断续续运行导致部分没有写入地址记录文件）'''
exit()
p = 'GeneticMOSGinteger-res_dir-N=4GFIndependentRepeatExperiments_SEED0-30.json'
p = tool.algorithm.loadVariJson(p)
res = {}
for para, path_list in p.items():
        # if para == 'obj3target200' or para == 'obj4target400':
        if para == 'obj3target1000':
                seed = 0
                for path_i in path_list:
                        fname = path_i.split('/')[-1]  # 文件名
                        res_dir = 'Results/optimalAfterMINCOV'
                        fname = os.path.join(res_dir, fname)
                        if not os.path.exists(fname):
                                print(seed, ' NO')
                        else:
                                res[path_i] = fname
                                print(seed, ' YES')
                        seed += 1
        res_dir = 'GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json'
        RES = tool.algorithm.loadVariJson(res_dir)
        for k,v in res.items():
                RES[k] = v
        tool.algorithm.dumpVariJson(vari=RES, name=res_dir)


'''PART3:生成PF-ORIGAMIG(重复实验)'''
# NOTE 读取30轮-G重复实验数据,计算-G的PF
path:Dict[str, List[str]] = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-IndependentRepeatExperiments_SEED0-30.json')
path:Dict[str, List[str]] = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-N=4GFIndependentRepeatExperiments_SEED0-30.json')
path_image: Dict[str, str] = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json')
DONE = []
# TODO = ['obj4target400', 'obj3target400', 'obj3target600']
TODO = ['obj3target1000']
# TODO = ['obj5target100', 'obj6target25', 'obj6target50', 'obj6target75', 'obj6target100']
# TODO = ['obj3target100', 'obj3target200', 'obj4target25', 'obj4target50', 'obj4target75', 'obj4target100', 'obj4target200']
# TODO = ['obj7target100', 'obj7target200', 'obj8target25', 'obj7target25', 'obj7target50', 'obj8target25']
# NOTE res存储初始PF，处理完后覆盖到原文件
res_path = os.path.join('Results', 'IndependentRepeat', 'res', 'PF-G.json')
if os.path.exists(res_path):
    res = tool.algorithm.loadVariJson(path=res_path)
else:
    res = {}
for para, path_list in path.items():
    # DEBUG
    # if para == 'obj4target50':
    if para not in DONE and para in TODO:
        path_res = []
        name = []
        # NOTE 若要将PF-G中的信息作为基础数据更新
        # if res[para] == '':
        #     path_res = []
        # else:
        #     path_res = [res[para]]
        # name = ['PFIndependenteRepeat']
        for path_i in path_list:
            path_res.append(path_image[path_i])
            name.append('')
        RES_DIR = os.path.join('Results', 'pf', para)
        gen = generatePF(path=path_res, name=name, mosg=False, RES_DIR=RES_DIR)
        res_i = gen.do()
        res[para] = res_i
tool.algorithm.dumpVariJson(vari=res, path=['Results', 'IndependentRepeat', 'res'], name='PF-G.json')


'''PART4:生成PF-ORIGAMI-M-A(基线)'''
# # NOTE pf for ORIGAMIG
path_ORIGAMIG = tool.algorithm.loadVariJson(path=['Results', 'IndependentRepeat', 'res', 'PF-G.json'])
# # mosg 
# # ORIGAMIM
# path_ORIGAMIM:Dict[str, str] = {}
# path_ORIGAMIM['obj3target25'] = 'Results/IndependentRepeat/seed1/obj3target25/ORIGAMIM-obj3target25-2022_05_20_20_10-295.txt'
# path_ORIGAMIM['obj3target50'] = 'Results/IndependentRepeat/seed1/obj3target50/ORIGAMIM-obj3target50-2022_04_11_00_20-978.txt'
# path_ORIGAMIM['obj3target75'] = 'Results/IndependentRepeat/seed1/obj3target75/ORIGAMIM-obj3target75-2022_04_11_00_20-2380.txt'
# path_ORIGAMIM['obj3target100'] = 'Results/IndependentRepeat/seed1/obj3target100/ORIGAMIM-obj3target100-2022_04_11_00_20-3717.txt'
# path_ORIGAMIM['obj3target200'] = 'Results/IndependentRepeat/seed1/obj3target200/ORIGAMIM-obj3target200-2022_04_11_00_21-18829.txt'
# path_ORIGAMIM['obj3target400'] = 'Results/IndependentRepeat/seed1/obj3target400/ORIGAMIM-obj3target400-2022_04_11_00_22-84894.txt'
# path_ORIGAMIM['obj3target600'] = 'Results/IndependentRepeat/seed1/obj3target600/ORIGAMIM-obj3target600-2022_04_10_23_36-227487.txt'
# path_ORIGAMIM['obj3target800'] = 'Results/IndependentRepeat/seed1/obj3target800/ORIGAMIM-obj3target800-2022_04_11_00_34-479577.txt'
# path_ORIGAMIM['obj4target25'] = 'Results/IndependentRepeat/seed1/obj4target25/ORIGAMIM-obj4target25-2022_04_11_00_48-3842.txt'
# path_ORIGAMIM['obj4target50'] = 'Results/IndependentRepeat/seed1/obj4target50/ORIGAMIM-obj4target50-2022_06_12_20_29-18055.txt'
# path_ORIGAMIM['obj4target75'] = 'Results/IndependentRepeat/seed1/obj4target75/ORIGAMIM-obj4target75-2022_06_12_20_31-51234.txt'
# path_ORIGAMIM['obj4target100'] = 'Results/IndependentRepeat/seed1/obj4target100/ORIGAMIM-obj4target100-2022_06_12_20_33-123478.txt'
# path_ORIGAMIM['obj4target200'] = 'Results/IndependentRepeat/seed1/obj4target200/ORIGAMIM-obj4target200-2022_04_11_00_53-188465.txt'
# path_ORIGAMIM['obj4target400'] = 'Results/IndependentRepeat/seed1/obj4target400/ORIGAMIM-obj4target400-2022_04_11_01_10-1022798.txt'
# path_ORIGAMIM['obj5target25'] = 'Results/IndependentRepeat/seed1/obj5target25/ORIGAMIM-obj5target25-2022_07_11_16_30-101490.txt'
# path_ORIGAMIM['obj5target50'] = 'Results/IndependentRepeat/seed1/obj5target50/ORIGAMIM-obj5target50-2022_07_11_17_05-204142.txt'
# path_ORIGAMIM['obj5target75'] = 'Results/IndependentRepeat/seed1/obj5target75/ORIGAMIM-obj5target75-2022_04_11_07_22-252346.txt'
# path_ORIGAMIM['obj5target100'] = 'Results/IndependentRepeat/seed1/obj5target100/ORIGAMIM-obj5target100-2022_04_11_07_39-994018.txt'
# # name_m = 'ORIGAMIM'
# # name_a = 'ORIGAMIA'
# # name_mbs = 'ORIGAMIMBS'
# # name_d = 'DIRECTMINCOV'
# # ORIGAMIA
# path_ORIGAMIA:Dict[str, str] = {}
# path_ORIGAMIA['obj3target25'] = 'Results/IndependentRepeat/seed1/obj3target25/ORIGAMIA-obj3target25-2022_04_20_09_12-6593.txt'
# path_ORIGAMIA['obj3target50'] = 'Results/IndependentRepeat/seed1/obj3target50/ORIGAMIA-obj3target50-2022_04_20_09_12-27277.txt'
# path_ORIGAMIA['obj3target75'] = 'Results/IndependentRepeat/seed1/obj3target75/ORIGAMIA-obj3target75-2022_04_20_09_14-65129.txt'
# path_ORIGAMIA['obj3target100'] = 'Results/IndependentRepeat/seed1/obj3target100/ORIGAMIA-obj3target100-2022_04_30_22_22-148579.txt'
# path_ORIGAMIA['obj3target200'] = 'Results/IndependentRepeat/seed1/obj3target200/ORIGAMIA-obj3target200-2022_04_20_09_29-760542.txt'
# # path_ORIGAMIA['obj3target400'] = 'Results/IndependentRepeat/seed1/obj3target400/ORIGAMIA-obj3target400-2022_04_20_10_22-3171904.txt'
# path_ORIGAMIA['obj4target25'] = 'Results/IndependentRepeat/seed1/obj4target25/ORIGAMIA-obj4target25-2022_04_24_23_59-39837.txt'
# path_ORIGAMIA['obj4target50'] = 'Results/IndependentRepeat/seed1/obj4target50/ORIGAMIA-obj4target50-2022_06_12_20_32-205515.txt'
# path_ORIGAMIA['obj4target75'] = 'Results/IndependentRepeat/seed1/obj4target75/ORIGAMIA-obj4target75-2022_06_12_20_49-1103830.txt'
# path_ORIGAMIA['obj4target100'] = 'Results/IndependentRepeat/seed1/obj4target100/ORIGAMIA-obj4target100-2022_06_12_21_20-1864655.txt'
# path_ORIGAMIA['obj5target25'] = 'Results/IndependentRepeat/seed1/obj5target25/ORIGAMIA-obj5target25-2022_07_11_16_35-449671.txt'
# path_ORIGAMIA['obj5target50'] = 'Results/IndependentRepeat/seed1/obj5target50/ORIGAMIA-obj5target50-2022_07_11_17_22-1245971.txt'
# # path_ORIGAMIA['obj5target75'] = 'Results/IndependentRepeat/seed1/obj5target75/ORIGAMIA-obj5target75-2022_04_26_16_29-2423346.txt'
# path_ORIGAMIA['obj6target25'] = 'Results/IndependentRepeat/seed1/obj6target25/ORIGAMIA-obj6target25-2022_05_20_20_26-331955.txt'
# path_ORIGAMIA['obj7target25'] = 'Results/IndependentRepeat/seed1/obj7target25/ORIGAMIA-obj7target25-2022_05_20_20_53-1650216.txt'
# path_ORIGAMIA['obj8target25'] = 'Results/IndependentRepeat/seed1/obj8target25/ORIGAMIA-obj8target25-2022_05_20_20_57-233252.txt'
# # ORIGAMIMBS
# path_ORIGAMIMBS:Dict[str, str] = {}
# path_ORIGAMIMBS['obj3target25'] = 'Results/IndependentRepeat/seed1/obj3target25/ORIGAMIMBS-obj3target25-2022_04_20_17_49-73.txt'
# path_ORIGAMIMBS['obj3target50'] = 'Results/IndependentRepeat/seed1/obj3target50/ORIGAMIMBS-obj3target50-2022_04_20_17_49-214.txt'
# path_ORIGAMIMBS['obj3target75'] = 'Results/IndependentRepeat/seed1/obj3target75/ORIGAMIMBS-obj3target75-2022_04_20_17_49-600.txt'
# path_ORIGAMIMBS['obj3target100'] = 'Results/IndependentRepeat/seed1/obj3target100/ORIGAMIMBS-obj3target100-2022_04_20_17_49-1133.txt'
# path_ORIGAMIMBS['obj3target200'] = 'Results/IndependentRepeat/seed1/obj3target200/ORIGAMIMBS-obj3target200-2022_04_20_17_49-7835.txt'
# path_ORIGAMIMBS['obj3target400'] = 'Results/IndependentRepeat/seed1/obj3target400/ORIGAMIMBS-obj3target400-2022_04_20_17_49-29813.txt'
# path_ORIGAMIMBS['obj3target600'] = 'Results/IndependentRepeat/seed1/obj3target600/ORIGAMIMBS-obj3target600-2022_04_21_09_25-140806.txt'
# path_ORIGAMIMBS['obj3target800'] = 'Results/IndependentRepeat/seed1/obj3target800/ORIGAMIMBS-obj3target800-2022_04_21_09_30-273993.txt'
# path_ORIGAMIMBS['obj3target1000'] = 'Results/IndependentRepeat/seed1/obj3target1000/ORIGAMIMBS-obj3target1000-2022_04_21_09_37-440675.txt'
# path_ORIGAMIMBS['obj4target25'] = 'Results/IndependentRepeat/seed1/obj4target25/ORIGAMIMBS-obj4target25-2022_04_25_00_00-624.txt'
# path_ORIGAMIMBS['obj4target50'] = 'Results/IndependentRepeat/seed1/obj4target50/ORIGAMIMBS-obj4target50-2022_06_12_20_29-2231.txt'
# path_ORIGAMIMBS['obj4target75'] = 'Results/IndependentRepeat/seed1/obj4target75/ORIGAMIMBS-obj4target75-2022_06_12_20_30-11314.txt'
# path_ORIGAMIMBS['obj4target100'] = 'Results/IndependentRepeat/seed1/obj4target100/ORIGAMIMBS-obj4target100-2022_06_12_20_31-20504.txt'
# path_ORIGAMIMBS['obj4target200'] = 'Results/IndependentRepeat/seed1/obj4target100/ORIGAMIMBS-obj4target100-2022_06_12_20_31-20504.txt'
# path_ORIGAMIMBS['obj5target25'] = 'Results/IndependentRepeat/seed1/obj5target25/ORIGAMIMBS-obj5target25-2022_06_12_20_27-6114.txt'
# path_ORIGAMIMBS['obj5target50'] = 'Results/IndependentRepeat/seed1/obj5target50/ORIGAMIMBS-obj5target50-2022_07_11_17_02-7183.txt'
# path_ORIGAMIMBS['obj5target75'] = 'Results/IndependentRepeat/seed1/obj5target75/ORIGAMIMBS-obj5target75-2022_04_26_16_09-59899.txt'
# path_ORIGAMIMBS['obj5target100'] = 'Results/IndependentRepeat/seed1/obj5target100/ORIGAMIMBS-obj5target100-2022_04_26_16_11-115067.txt'
# path_ORIGAMIMBS['obj6target25'] = 'Results/IndependentRepeat/seed1/obj6target25/ORIGAMIMBS-obj6target25-2022_05_20_20_11-31238.txt'
# path_ORIGAMIMBS['obj6target50'] = 'Results/IndependentRepeat/seed1/obj6target50/ORIGAMIMBS-obj6target50-2022_04_25_00_04-222549.txt'
# # path_ORIGAMIMBS['obj6target75'] = '' #? 当时ORIGAMIMBS程序跑不出N=6T=75,但是能跑出N=7T=75
# path_ORIGAMIMBS['obj7target25'] = 'Results/IndependentRepeat/seed1/obj7target25/ORIGAMIMBS-obj7target25-2022_05_20_20_15-250804.txt'
# path_ORIGAMIMBS['obj7target50'] = 'Results/IndependentRepeat/seed1/obj7target50/ORIGAMIMBS-obj7target50-2022_04_25_00_27-1085372.txt'
# # DIRECTMINCOV
# path_DIRECTMINCOV:Dict[str, str] = {}
# path_DIRECTMINCOV['obj3target25'] = 'Results/IndependentRepeat/seed1/obj3target25/DIRECTMINCOV-obj3target25-2022_04_21_11_13-1345.txt'
# path_DIRECTMINCOV['obj3target50'] = 'Results/IndependentRepeat/seed1/obj3target50/DIRECTMINCOV-obj3target50-2022_04_21_11_13-9726.txt'
# path_DIRECTMINCOV['obj3target75'] = 'Results/IndependentRepeat/seed1/obj3target75/DIRECTMINCOV-obj3target75-2022_04_21_11_14-20884.txt'
# path_DIRECTMINCOV['obj3target100'] = 'Results/IndependentRepeat/seed1/obj3target100/DIRECTMINCOV-obj3target100-2022_04_30_22_18-21019.txt'
# path_DIRECTMINCOV['obj3target200'] = 'Results/IndependentRepeat/seed1/obj3target200/DIRECTMINCOV-obj3target200-2022_04_21_11_16-95061.txt'
# path_DIRECTMINCOV['obj3target400'] = 'Results/IndependentRepeat/seed1/obj3target400/DIRECTMINCOV-obj3target400-2022_04_21_11_25-573312.txt'
# path_DIRECTMINCOV['obj3target600'] = 'Results/IndependentRepeat/seed1/obj3target600/DIRECTMINCOV-obj3target600-2022_04_21_11_50-1475655.txt'
# path_DIRECTMINCOV['obj4target25'] = 'Results/IndependentRepeat/seed1/obj4target25/DIRECTMINCOV-obj4target25-2022_04_30_23_21-8717.txt'
# path_DIRECTMINCOV['obj4target50'] = 'Results/IndependentRepeat/seed1/obj4target50/DIRECTMINCOV-obj4target50-2022_06_12_20_30-49183.txt'
# path_DIRECTMINCOV['obj4target75'] = 'Results/IndependentRepeat/seed1/obj4target75/DIRECTMINCOV-obj4target75-2022_06_12_20_32-115289.txt'
# path_DIRECTMINCOV['obj4target100'] = 'Results/IndependentRepeat/seed1/obj4target100/DIRECTMINCOV-obj4target100-2022_06_12_20_35-135858.txt'
# path_DIRECTMINCOV['obj4target200'] = 'Results/IndependentRepeat/seed1/obj4target200/DIRECTMINCOV-obj4target200-2022_04_21_13_26-743876.txt'
# path_DIRECTMINCOV['obj5target25'] = 'Results/IndependentRepeat/seed1/obj5target25/DIRECTMINCOV-obj5target25-2022_07_11_16_30-103491.txt'
# path_DIRECTMINCOV['obj5target50'] = 'Results/IndependentRepeat/seed1/obj5target50/DIRECTMINCOV-obj5target50-2022_07_11_17_07-256254.txt'
# path_DIRECTMINCOV['obj5target75'] = 'Results/IndependentRepeat/seed1/obj5target75/DIRECTMINCOV-obj5target75-2022_04_23_15_12-344375.txt'
# path_DIRECTMINCOV['obj5target100'] = 'Results/IndependentRepeat/seed1/obj5target100/DIRECTMINCOV-obj5target100-2022_04_23_15_21-543119.txt'
# path_DIRECTMINCOV['obj6target25'] = 'Results/IndependentRepeat/seed1/obj6target25/DIRECTMINCOV-obj6target25-2022_04_23_15_23-89499.txt'
# path_DIRECTMINCOV['obj6target50'] = 'Results/IndependentRepeat/seed1/obj6target50/DIRECTMINCOV-obj6target50-2022_04_23_15_28-240896.txt'
# path_DIRECTMINCOV['obj6target75'] = 'Results/IndependentRepeat/seed1/obj6target75/DIRECTMINCOV-obj6target75-2022_04_23_15_53-1527022.txt'
# path_DIRECTMINCOV['obj6target100'] = 'Results/IndependentRepeat/seed1/obj6target100/DIRECTMINCOV-obj6target100-2022_04_28_10_02-1319168.txt'
# path_DIRECTMINCOV['obj7target25'] = 'Results/IndependentRepeat/seed1/obj7target25/DIRECTMINCOV-obj7target25-2022_04_23_16_10-963049.txt'
# path_DIRECTMINCOV['obj7target50'] = 'Results/IndependentRepeat/seed1/obj7target50/DIRECTMINCOV-obj7target50-2022_04_23_16_29-1126089.txt'
# path_DIRECTMINCOV['obj8target25'] = 'Results/IndependentRepeat/seed1/obj8target25/DIRECTMINCOV-obj8target25-2022_04_23_16_38-459795.txt'
# res = {'ORIGAMIM':path_ORIGAMIM, 'ORIGAMIA':path_ORIGAMIA, 'ORIGAMIMBS':path_ORIGAMIMBS, 'DIRECTMINCOV':path_DIRECTMINCOV}
# tool.algorithm.dumpVariJson(vari=res, path=['Results', 'IndependentRepeat', 'res'], name='PF-Comparison.json')


'''PART5:合并PF(total)'''
path_all = tool.algorithm.loadVariJson(path=['Results', 'IndependentRepeat', 'res', 'PF-Comparison.json'])
path_ORIGAMIM = path_all['ORIGAMIM']
path_ORIGAMIA = path_all['ORIGAMIA']
path_ORIGAMIMBS = path_all['ORIGAMIMBS']
path_DIRECTMINCOV = path_all['DIRECTMINCOV']
res_path = os.path.join('Results', 'IndependentRepeat', 'res', 'PF-final.json')
if os.path.exists(res_path):
    res = tool.algorithm.loadVariJson(res_path)
else:
    res = {}
for para, path in path_ORIGAMIG.items():
    # DEBUG
    if para == 'obj4target25':
        pass
    if para in DONE or para not in TODO:
        continue
    name = ['PFIndependenteRepeat']
    r = [tool.algorithm.loadVariPickle(path)]
    if para in path_ORIGAMIM.keys():
        r.append(tool.algorithm.loadVariPickle(path_ORIGAMIM[para]).F)
        name.append('')
    if para in path_ORIGAMIA.keys():
        r.append(tool.algorithm.loadVariPickle(path_ORIGAMIA[para]).F)
        name.append('')
    if para in path_ORIGAMIMBS.keys():
        r.append(tool.algorithm.loadVariPickle(path_ORIGAMIMBS[para]).F)
        name.append('')
    if para in path_DIRECTMINCOV.keys():
        r.append(tool.algorithm.loadVariPickle(path_DIRECTMINCOV[para]).F)
        name.append('')
    RES_DIR = os.path.join('Results', 'pf', para)
    gen = generatePF(res=r, name=name, mosg=False, RES_DIR=RES_DIR)
    res_i = gen.do()
    res[para] = res_i
tool.algorithm.dumpVariJson(vari=res, name=res_path)


print('__main__finish~!')