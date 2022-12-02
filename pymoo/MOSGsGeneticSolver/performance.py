# from fcntl import F_GET_SEALS
from enum import unique
import functools
from re import T
from tkinter import N
from typing import List, Union, Dict
from functools import reduce
import os
from matplotlib.pyplot import axis

import numpy as np
from pymoo.factory import get_problem
from pymoo.performance_indicator.hv import Hypervolume
from pymoo.visualization.scatter import Scatter


import pymoo.util.nds.non_dominated_sorting


# TODO LIST FOR PERFORMANCE INDICATOR
# TODO 1: 完成已经复现的算法的效果计算，并将数据保存下来
# TODO 2: 数据读入这里，归一化处理（具体的归一化方法需要询问）
    # 比如混合了多个算法的帕累托前沿解，是不是要放一起做归一化？
# TODO 3: 统计所有的指标IGD IGD+ HV

from pymoo.factory import get_performance_indicator
import tool.algorithm

# IGD
# Input: pf:np.ndarray, A:np.ndarray.
# igd = get_performance_indicator("igd", pf)
# print("IGD", igd.calc(A))

# HV
# Input: A:np.ndarray;
# hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
# print("hv", hv.calc(A))

# IGD+
# igd_plus = get_performance_indicator("igd+", pf)
# print("IGD+", igd_plus.calc(A))

BATCH = 400

RES_DIR_NAME = 'Results'

class Performance():

    '''# pf_total分成n+1部分，pf_total[0]为pf_total[1]-[n]的拼接
    '''
    def __init__(self, pf_total:List[np.ndarray] = None,  # 所有算法结果的和
                 name_total:List[str]=None,  # 每个算法的名称（按固定顺序）
                 normalize=False,
                 pf_dump=False, pf_dir='pf',
                 indicator_hv=False, indicator_igdplus=False,indicator_igd=False,indicator_gd=False,indicator_gdplus=False,
                 para_dir=None,):
        '''para_dir:持久化PF时传入，表示文件名称，Dict形式
                pf_dir:pf持久化时传入，标记pf存放的文件夹名称，默认Results/pf
                pf_dump:选择是否将PF持久化
                indicator:选择是否计算性能指标'''
        
        self.para_dir = para_dir
        
        # 为防止一次性传入所有数据导致内存爆炸，分批次传入，一次传入BATCH条数据
        if pf_total is not None :
            self.len_total = [len(res) for res in pf_total[1:]]
            self.pf_num = len(name_total[1:])  # 统计提供的不同算法（求得的）结果数量
            self.pf_name = name_total
            self.pf_len = [len(res) for res in pf_total[1:]]
            self.pf = None
            self.normalize = normalize
            self.pf_dump = pf_dump
            self.pf_dir = pf_dir

            # self.pf_total = np.vstack(pf_total[1:])
            # pf不再是pf_total和合并，而是直接用传入的pf
            # self.pf_idx = self.getPF_idx(pf_total=self.pf_total)
            # self.pf = self.pf_total[self.pf_idx]
            self.pf = pf_total[0]

            # if normalize:
            #     pf_total_normalized = tool.algorithm.normalization(pf_total)
            #     pf = tool.algorithm.normalization(pf)
            # else:
            #     pf_total_normalized = pf_total

            # 预处理res_pf_total，将每个算法的结果单独列出来
            pf_split:List[np.ndarray] = pf_total[1:]

            # Indicator
            self.hv:List[float] = []
            self.gd:List[float] = []
            self.gd_plus: List[float] = []
            self.igd:List[float] = []
            self.igd_plus: List[float] = []
            
            # 下面指标中，List[float]结构的第一个是pf_true，而后len(res)对应的是method_pf
            if indicator_hv:
                # HV indicator
                self.HV = self.HV()
                self.hv_pf = self.getHV(self.pf)
                for pf in pf_split:
                    # pf_idx = self.getPF(pf)
                    # self.hv.append(self.getHV(pf[pf_idx]))
                    self.hv.append(self.getHV(pf))
            if indicator_gd:
                # GD indicator
                self.GD = self.GD()
                self.gd_pf = self.getGD(self.pf)
                for pf in pf_split:
                    # pf_idx = self.getPF(pf)
                    # self.igd.append(self.getIGD(pf[pf_idx]))
                    self.gd.append(self.getGD(pf))
            if indicator_gdplus:
                # GD_PLUS indicator
                self.GDPlus = self.GDPlus()
                self.gd_plus_pf = self.getGDPlus(self.pf)
                for pf in pf_split:
                    # pf_idx = self.getPF(pf)
                    # self.igd.append(self.getIGD(pf[pf_idx]))
                    self.gd_plus.append(self.getGDPlus(pf))
            if indicator_igd:
                # IGD indicator
                self.IGD = self.IGD()
                self.igd_pf = self.getIGD(self.pf)
                for pf in pf_split:
                    # pf_idx = self.getPF(pf)
                    # self.igd.append(self.getIGD(pf[pf_idx]))
                    self.igd.append(self.getIGD(pf))
            if indicator_igdplus:
                # IGD_PLUS indicator
                self.IGDPlus = self.IGDPlus()
                self.igd_plus_pf = self.getIGDPlus(self.pf)
                for pf in pf_split:
                    # pf_idx = self.getPF(pf)
                    # self.igd_plus.append(self.getIGDPlus(pf[pf_idx]))
                    self.igd_plus.append(self.getIGDPlus(pf))


    def HV(self):
        if self.normalize:
            ref_point = np.array([1.2, 1.2])
        else:
            min = np.amin(a=self.pf, axis=0)
            max = np.amax(a=self.pf, axis=0)
            ref_point = max + (max-min)*0.2
        return get_performance_indicator("hv", ref_point=ref_point)
    def getHV(self, res):
        return self.HV.calc(res)

    def GD(self):
        return get_performance_indicator("gd", self.pf)
    def getGD(self, res):
        return self.GD.calc(res)

    def GDPlus(self):
        return get_performance_indicator("gd+", self.pf)
    def getGDPlus(self, res):
        return self.GDPlus.calc(res)

    def IGD(self):
        return get_performance_indicator("igd", self.pf)
    def getIGD(self, res):
        return self.IGD.calc(res)

    def IGDPlus(self):
        return get_performance_indicator("igd+", self.pf)
    def getIGDPlus(self, res):
        return self.IGDPlus.calc(res)


    # getPF和getPF_idx效果是相同的，只是返回的pf顺序不同

    '''getPF接受经过vstack后的pf_total
    getPF返回pf'''
    def getPF(self, pf_total:np.ndarray) ->np.ndarray:
        point = 0  # 标记将要遍历的data idx
        pf_total_len = pf_total.shape[0]  # 传入的非常大的pf的长度
        pf = None
        while point < pf_total_len:  # 数据没读完
            batch = min(pf_total_len-point, BATCH)  # 本次要读取的数据的长度
            pf_batch = pf_total[point:point + batch]  # 本次读出来的数据
            point += batch  # 标记前移
            if pf is not None:
                pf = np.vstack([pf, pf_batch])
            else:
                pf = pf_batch
            idx = pymoo.util.nds.non_dominated_sorting.find_non_dominated(F=pf)
            pf = pf[idx]
        return pf

    ''' # 和getPF不同，该函数返回pf的idx，试用性更强
    # 算法逻辑上是将pf_total分成多块，一块最大BATCH个，算每块的idx然后合并（为防止fp_total太大内存爆炸）
    # 在返回下标前，会作unique操作删除重复项（会影响效率，可选参数，未实现）'''
    def getPF_idx(self, pf_total:np.ndarray) ->np.ndarray:
        point = 0  # 标记将要遍历的data idx
        pf_total_len = pf_total.shape[0]  # 排序前提供的pf point number
        part_idx:List[np.ndarray] = []  # 将pf_total分成多个pf_parts装入List
        part_len:List[int] = []  # 每个pf_part的长度
        while point < pf_total_len:  # 数据没读完
            batch = min(pf_total_len-point, BATCH)  # 本次要读取的数据的长度
            pf_batch = pf_total[point:point + batch]  # 本次读出来的数据
            point += batch  # 标记前移
            # 第一次计算pf，每个part单独算，然后合并再算
            first_idx = pymoo.util.nds.non_dominated_sorting.find_non_dominated(F=pf_batch)
            part_len.append(batch)  # 记录本次读出部分的长度
            part_idx.append(first_idx)

        # 第二次计算pf，需要注意的是返回的下标是作用在前一次结果pf_idx上的，不可直接返回
        part_start = [np.sum(part_len[:i]) for i in range(len(part_len))]
        pf_idx = [part_idx[idx] + int(start) for idx,start in enumerate(part_start)]  # 将多个pf_parts合并成一个，然后算pf
        first_idx_total:np.ndarray = functools.reduce(lambda x,y: np.hstack([x,y]), pf_idx)
        second_idx = pymoo.util.nds.non_dominated_sorting.find_non_dominated(F=pf_total[first_idx_total])
        final_idx = first_idx_total[second_idx]
        return final_idx

    # NOTE dumpPF,但实际上用不到，用于PF更新的。
    def dump(self, para_fname=None, para_dir=None)->str:
        '''dump接受的第一个参数十分特殊，是一个{name:survive num}的字典
        第二个参数则为文件路径'''
        # （选择性）将pf存下来
        if self.pf_dump:
            filename = tool.algorithm.paras2Str(para_fname)
            tool.algorithm.creatDir(para_dir)
            path = os.path.join(para_dir, filename)
            self.pf = np.unique(self.pf, axis=0)
            tool.algorithm.dumpVariPickle(self.pf, path=path)
            return path
        else:
            print('ERROR!!! self.pf_dump is False')
            exit()

    # NOTE dump the Performance Indicator Values (e.g., HV, IGD+) 
    def dumpJson(self, fname, file_exist=False, repeat:str=None):
        if file_exist:
            if os.path.exists(path=os.path.join(self.para_dir, fname)):
                res = tool.algorithm.loadVariJson(path=os.path.join(self.para_dir, fname))
            else:
                res:Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        else:
            res:Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        for name_i in self.pf_name[1:]:
            res[fname+'_'+repeat] = {name_i: {'hv': self.hv}}
            res[fname+'_'+repeat][name_i]['gd'] = self.gd
            res[fname+'_'+repeat][name_i]['gd+'] = self.gd_plus
            res[fname+'_'+repeat][name_i]['igd'] = self.igd
            res[fname+'_'+repeat][name_i]['igd+'] = self.igd_plus
        # if not os.path.exists(os.path.join(self.para_dir, fname)):
        #     file = open(os.path.join(self.para_dir, fname), 'w')
        #     file.close()
        tool.algorithm.dumpVariJson(vari=res, path=self.para_dir, name=fname)

    def __repr__(self):
        return '多目标指标计算结果：' \
               + '\n   hv: ' + str(self.hv) + '\n   ' \
                       'gd: ' + str(np.round(self.gd,2)) + '\n   ' \
                       'gd+: ' + str(np.round(self.gd_plus,2)) + '\n   ' \
                       'igd: ' + str(np.round(self.igd,2)) + '\n   ' \
                       'igd+: ' + str(np.round(self.igd_plus,2)) \
               + '\nAlgorithom name is: ' + str(self.pf_name[1:]) \
               + '\npf length: ' + str(self.pf_len)

