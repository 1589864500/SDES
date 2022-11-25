
import os
from re import T
from tkinter import NONE


import numpy as np
import math
from typing import *

import sys

from yaml import dump
sys.path.append('./')

from pymoo.MOSGsGeneticSolver.performance import Performance
from pymoo.MOSGsGeneticSolver.visualization import Visualization
from pymoo.model.result import Result
import pymoo.util.nds.non_dominated_sorting


import tool.algorithm
from pymoo.integercode.truing import Truing
from pymoo.problems.securitygame import SGs1

'''将需要比较的pf name存入List，由do存储每个pf的贡献度'''
class generatePF():
    def __init__(self, path:List[str]=None, res:List[np.ndarray]=None, name:List[str]=None, 
    mosg=False, RES_DIR='./Results', BSM=True, DG=False, DGpopsize=10):
        self.RES_DIR = RES_DIR
        self.name = []
        
        # mosg对比算法和gmosg的处理方式是分开的
        if not mosg:
            if res is None:
                self.res = []
                for i, p in enumerate(path):
                    r:Tuple[Result, np.ndarray] = tool.algorithm.loadVariPickle(p)
                    if not isinstance(r, np.ndarray):
                        truing1 = Truing(res=r)
                        truing2 = Truing(res=r)
                        truing1.mosgSearch_pop()
                        if BSM:
                            self.res.append(truing1.fit_pf)
                            self.name.append(name[i]+'M')
                        if DG:
                            truing2.geneticSearch(pop_size=DGpopsize)
                            self.res.append(truing2.fit_pf)
                            self.name.append(name[i]+'G')
                    else:
                        self.res.append(r)
                        self.name.append(name[i])
            else:
                self.res = res
                self.name = name
        else:
            if res is None:
                self.res = []
                for i, p in enumerate(path):
                    r:Tuple[Result, np.ndarray] = tool.algorithm.loadVariPickle(p)
                    if not isinstance(r, np.ndarray):
                        self.res.append(r.F)
                        self.name.append(name[i])
                    else:
                        self.res.append(r)
                        self.name.append(name[i])
            else:
                self.res = res
                self.name = name

        self.pf = None

    def do(self, para=None):
        # para_dir = tool.algorithm.paras2Str(para)
        # path = os.path.join(self.RES_DIR, 'pf', para_dir)

        # # 这段是再Performance pf是pf_total现求得情况下的代码
        # pf_len = [len(r) for r in self.res]
        # print('pflen:{} pf_name{}'.format(pf_len, self.name))
        # start_idx = [sum(pf_len[:idx]) for idx in range(len(pf_len)+1)]  # start_idx长度N+1，最后一位不是start而是end
        # perf = Performance(pf_total=self.res, name_total=self.name, pf_dump=True)
        # pf_sur_len = [len(np.where((perf.pf_idx>=start_idx[i]) & (perf.pf_idx<start_idx[i+1]))[0]) for i in range(len(pf_len))]
        # para_dir = {self.name[i]:pf_sur_len[i] for i in range(len(self.name))}
        # perf.dump(para_fname=para_dir, para_dir=path)

        # 如果pf不再更新，代码如下：
        pf_len = [len(r) for r in self.res]
        res = [np.vstack(self.res)]
        for r in self.res: res.append(r)
        name = ['pf']
        for n in self.name: name.append(n)
        perf = Performance(pf_total=res, name_total=name, pf_dump=True)
        para_dir = {self.name[i]:pf_len[i] for i in range(len(self.name))}
        return perf.dump(para_fname=para_dir, para_dir=self.RES_DIR)

def check(path):
    for p in path:
        res = tool.algorithm.loadVariPickle(p)
        print(res.F.shape, end='')

if __name__ == '__main__':
    # para_dir = {'obj':3, 'target':25}
    # para_dir = {'obj':3, 'target':50}
    # para_dir = {'obj':3, 'target':75}
    # para_dir = {'obj':3, 'target':100}
    # para_dir = {'obj':3, 'target':200}
    # para_dir = {'obj':3, 'target':400}
    # para_dir = {'obj':4, 'target':25}
    # para_dir = {'obj':4, 'target':50}
    # para_dir = {'obj':4, 'target':75}
    # para_dir = {'obj':4, 'target':100}
    # para_dir = {'obj':5, 'target':25}
    # para_dir = {'obj':5, 'target':50}
    para_dir = {'obj':5, 'target':75}
    path = []
    name = []
    print(para_dir)

    # GMOSG 
    path.append('Results/pf/obj5target75/PF10332P2M405P2G462P3M405P3G473')
    # name.append('P1')
    # name.append('GMOSG')
    name.append('booleanscoringmechanism')
    # name.append('PF')
    path.append('Results/obj5target75/typeDGpopsize10gen102022_07_01_10_36_23')
    # name.append('floatscoringmechanismP1')
    name.append('ORIGAMIDGP1')
    # name.append('ORIGAMIMBS')
    # name.append('MOSG')
    # path.append('Results/obj5target75/typeFSMMINCOV2022_06_21_17_49_48')
    # name.append('P2')
    # name.append('ORIGAMIM')
    # path.append('Results/obj4target50/typeDGpopsize50gen302022_06_20_23_49_36')
    # name.append('P3')
    # name.append('ORIGAMIA')
    # path.append('Results/obj4target50/typeDGpopsize50gen402022_06_21_00_45_58')
    # name.append('P4')
    # name.append('DIRECTMINCOV')
    # path.append('Results/obj4target50/typeDGpopsize50gen502022_06_21_13_30_16')
    # name.append('P5')
    # path.append('Results/obj4target50/typeDGpopsize50gen602022_06_21_14_25_24')
    # name.append('P6')
    # path.append('Results/obj4target50/typeDGpopsize100gen502022_06_12_22_23_54')
    # name.append('P7')
    # check(path)
    gen = generatePF(path=path, name=name, mosg=False)
    gen.do(para=para_dir)


    #total PF



    # popsize
    # path.append('Results/obj3target25/GeneticMOSGinteger-Popsize2000Codeinteger-2022_04_29_20_07-20455726.txt')
    # name.append('obj3target25')
    # path.append('Results/obj3target50/GeneticMOSGinteger-Popsize2000Codeinteger-2022_04_29_19_22-17790035.txt')
    # name.append('obj3target50')
    # path.append('Results/obj3target75/GeneticMOSGinteger-Popsize2000Codeinteger-2022_04_29_19_10-16936152.txt')
    # name.append('obj3target75')
    # path.append('Results/obj3target100/GeneticMOSGinteger-Popsize2000Codeinteger-2022_04_30_05_54-16602283.txt')
    # name.append('obj3target100')
    # path.append('Results/obj3target200/GeneticMOSGinteger-Popsize2000Codeinteger-2022_04_30_16_13-19142570.txt')
    # name.append('obj3target200')
    # path.append('Results/obj3target400/GeneticMOSGinteger-Popsize2000Codeinteger-2022_04_30_22_54-43196768.txt')
    # name.append('obj3target400')
    # path.append('Results/obj3target600/GeneticMOSGinteger-Popsize1000Codeinteger-2022_05_03_01_47-16894023.txt')
    # name.append('obj3target600')
    # path.append('Results/obj3target600/GeneticMOSGinteger-Popsize1000Codeinteger-2022_05_02_20_38-19072265.txt')
    # name.append('obj3target600')
    # path.append('Results/obj3target800/GeneticMOSGinteger-Popsize1000Codeinteger-2022_05_02_14_38-20403643.txt')
    # name.append('obj3target800')
    # path.append('Results/obj3target1000/GeneticMOSGinteger-Popsize1000Codeinteger-2022_05_02_08_52-29167524.txt')
    # name.append('obj3target1000')
    # path.append('Results/obj4target25/GeneticMOSGinteger-Popsize2000Codeinteger-2022_04_28_14_33-16208868.txt')
    # name.append('obj4target25')
    # path.append('Results/obj4target50/GeneticMOSGinteger-Popsize2000Codeinteger-2022_05_03_11_24-8626910.txt')
    # name.append('obj4target50')
    # path.append('Results/obj4target75/GeneticMOSGinteger-Popsize2000Codeinteger-2022_05_03_11_55-10058355.txt')
    # name.append('obj4target75')
    # path.append('Results/obj4target100/GeneticMOSGinteger-Popsize2000Codeinteger-2022_04_29_15_43-26508158.txt')
    # name.append('obj4target100')
    # path.append('Results/obj5target25/GeneticMOSGinteger-Popsize2000CodeintegerHistoryFalse-2022_04_25_20_43-12618952.txt')
    # name.append('obj5target25')
    # path.append('Results/obj5target50/GeneticMOSGinteger-Popsize2000CodeintegerHistoryFalse-2022_04_26_15_53-4016703.txt')
    # name.append('obj5target50')
    # path.append('Results/obj5target75/GeneticMOSGinteger-Popsize2000CodeintegerHistoryFalse-2022_04_26_22_07-1803739.txt')
    # name.append('obj5target75')
    # check(path)


