# 该程序读取实验结果，并绘图
# TODO LIST FOR PLOT:
# TODO 1: 所有问题规模下的pf
# TODO 2: 所有算法在所有问题规模下指标（T, igd, igd_plus, hv）
# TODO 3: GMOSG各指标随各个维度的变化情况
# FIXME: 其他实验
#

import numpy as np


import json
import pickle
import os
import operator
from typing import Dict, List, Union


import matplotlib.pyplot as plt

from MOSGs.resultMOSGs import resultMOSGs
from pymoo.MOSGsGeneticSolver.performance import Performance
from MOSGs.read_result_dir import ReadResultDir
import tool.algorithm


class PlotRes():
    def __init__(self):
        # 表示迭代参数分别为50 100 150时GMOSG返回的结果，此外每个都会返回相同的MOSG结果
        self.gen50 = ReadResultDir(MOSG='ORIGAMIM-res_dir-0.json', GeneticMOSG='GeneticMOSG1-res_dir-0.json')
        self.gen100 = ReadResultDir(MOSG='ORIGAMIM-res_dir-0.json', GeneticMOSG='GeneticMOSG1-res_dir-1.json')
        self.gen150 = ReadResultDir(MOSG='ORIGAMIM-res_dir-0.json', GeneticMOSG='GeneticMOSG1-res_dir-2.json')
        self.gen300 = ReadResultDir(MOSG='ORIGAMIM-res_dir-0.json', GeneticMOSG='GeneticMOSG1-res_dir-3.json')
        # 获取指标
        perf_path:Dict[str,str] = {}
        for para_dir in self.gen50.para_dir_gmosg:
            # 若该问题规模MOSG和GMOSG都能处理
            if para_dir in self.gen50.para_dir_mosg:
                mosg_path = self.gen50.res_path_mosg_all[para_dir]
                gmosg_gen50_dict = self.gen50.res_path_gmosg_dir[para_dir]
                gmosg_gen100_dict = self.gen100.res_path_gmosg_dir[para_dir]
                gmosg_gen150_dict = self.gen150.res_path_gmosg_dir[para_dir]
                gmosg_gen300_dict = self.gen300.res_path_gmosg_dir[para_dir]
                gmosg_gen100_path = []
                gmosg_gen50_path = []
                gmosg_gen150_path = []
                gmosg_gen300_path = []
                for key in gmosg_gen50_dict.keys():
                    gmosg_gen50_path.append(gmosg_gen50_dict[key])
                for key in gmosg_gen100_dict.keys():
                    gmosg_gen100_path.append(gmosg_gen100_dict[key])
                for key in gmosg_gen150_dict.keys():
                    gmosg_gen150_path.append(gmosg_gen150_dict[key])
                for key in gmosg_gen300_dict.keys():
                    gmosg_gen150_path.append(gmosg_gen300_dict[key])
                perf_path[para_dir] = self.get_perf_path(para_dir)
                perf:Performance = self.get_perf(mosg_path,
                    gmosg_gen50_path, gmosg_gen100_path, gmosg_gen150_path, gmosg_gen300_path)
                tool.algorithm.dumpVariPickle(vari=perf, path=perf_path[para_dir])

            # 若只有GMOSG能处理，暂时不考虑
            else:
                pass

    def get_perf(self, *args:Union[str,List[str]]) ->Performance:
        # 提供某个问题规模下所有解的结果的路径
        res:Union[np.ndarray, None] = None
        res_len:List[int] = []
        res_path:List[str] = []
        res_name:List[str] = []

        # 输入属于预处理
        for path in args:
            if isinstance(path, List):
                for p in path:
                    res_path.append(p)
                    _, basename = os.path.split(p)
                    res_name.append(basename)
            elif isinstance(path, str):
                res_path.append(path)
                _, basename = os.path.split(path)
                res_name.append(basename)
            else:
                print('error')
                exit()

        for path in res_path:
            data:resultMOSGs = tool.algorithm.loadVariPickle(path)
            if res is None:
                res = data.F
            else:
                res = np.vstack((res, data.F))
            res_len.append(data.F.shape[0])
        return Performance(pf_total=res, len_total=res_len, name_total=res_name)
    def get_perf_path(self, para_dir) ->str:
        pf_dir = './results/pf'
        pf_para_dir = os.path.join(pf_dir)
        if not os.path.exists(pf_dir):
            os.makedirs(pf_dir)
        return os.path.join(pf_para_dir, para_dir)


if __name__ == '__main__':
    demo = PlotRes()