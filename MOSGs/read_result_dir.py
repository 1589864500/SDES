# 该程序读取实验结果存储的路径

import numpy as np


import json
import pickle
import os
import operator
from typing import Dict, List, Union

from pymoo.MOSGsGeneticSolver.performance import Performance
import tool.algorithm


class ReadResultDir():

    def __init__(self, path_dir='./Results', **kwargs):
        # **kwargs在当前具体问题中默认接收两个对象
        # 这样的设计是失误的，早知道一个一个文件读了，写的还快点

        # path_dir表示存放结果的根目录
        # kwargs表示结果根目录下的一级目录需要读取出来的json文件名称
        os.chdir(path_dir)
        self.method_para:Dict[str, Dict[str,str]] = {}
        for method_name, filename in kwargs.items():
            file_dir = os.path.join(os.getcwd(), filename)
            with open(file_dir, 'r') as json_f:
                self.method_para[method_name] = json.load(json_f)

        self.para_dir_mosg:List[str] = self.get_para(solver='MOSG')  # 记录所有的问题规模 如obj3_target25
        self.para_dir_gmosg: List[str] = self.get_para(solver='GeneticMOSG')  # 不同solber能处理的问题规模的范围不同

        # case1 MOSG的全部结果
        self.res_path_mosg_all:Dict[str,str] = self.read_res_dir(solver='MOSG', read_all=True)
        self.res_path_gmosg_all:Dict[str,str] = self.read_res_dir(solver='GeneticMOSG', read_all=True)

        # case2 按具体问题规模读取MOSG/GeneticMOSG结果
        # self.res_path_MOSG_dir:Dict[str,str] = {}  # 用不上
        self.res_path_gmosg_dir:Dict[str,Dict[str,str]] = {}
        # for para in self.para_dir_MOSG:
        #     self.res_path_MOSG_dir = demo.read_res_dir(solver='MOSG', read_part_by_dir=para)
        for para in self.para_dir_gmosg:
            self.res_path_gmosg_dir[para] = self.read_res_dir(solver='GeneticMOSG', read_part_by_dir=para)

        # case3 按某一维度GeneticMOSG结果
        self.res_path_gmosg_obj:Dict[str,Dict[str,str]] = self.read_res_dir(solver='GeneticMOSG', read_part_by_dim='obj')
        self.res_path_gmosg_target:Dict[str,Dict[str,str]] = self.read_res_dir(solver='GeneticMOSG', read_part_by_dim='target')
        self.res_path_gmosg_gen:Dict[str,Dict[str,str]] = self.read_res_dir(solver='GeneticMOSG', read_part_by_dim='gen')
        self.res_path_gmosg_pop:Dict[str,Dict[str,str]] = self.read_res_dir(solver='GeneticMOSG', read_part_by_dim='pop')

    def read_res_dir(self, solver:str,
                    read_part_by_dim:str=None,  # 提供某个维度，都适合
                    read_part_by_dir:str=None,  # 提供具体问题参数，都适合
                    read_all:bool=False) ->Union[Dict[str,str], Dict[str,Dict[str,str]]]:  # 读取所有，适合MOSG
        res_path:Union[Dict[str,str], Dict[str,Dict[str,str]]] = {}
        if read_all:
            res_path = self.method_para[solver]
        if read_part_by_dir is not None:
            for res_dir, path in self.method_para[solver].items():
                if self.dirMatch1(res_dir=res_dir, require=read_part_by_dir):
                    res_path[res_dir] = path
        if read_part_by_dim is not None:
            if read_part_by_dim == 'gen':
                res_path = self.dirMatch2(solver, require=2)
            elif read_part_by_dim == 'obj':
                res_path = self.dirMatch2(solver, require=0)
            elif read_part_by_dim == 'target':
                res_path = self.dirMatch2(solver, require=1)
            elif read_part_by_dim == 'pop':
                res_path = self.dirMatch2(solver, require=3)
        return res_path

    def dirMatch1(self, res_dir, require):
        dir = res_dir.split('_')
        dir_require = require.split('_')
        return operator.eq(dir[:4], dir_require[:4])
    def dirMatch2(self, solver, require:int) ->Dict[str,Dict[str,str]]:
        # require标记dim idx, 方便分片
        # solver方便区分读取哪个求解器
        res_path:Dict[str,Dict[str,str]] = {}
        for res_dir, path in self.method_para[solver].items():
            dir = res_dir.split('_')
            # 命名的时候考虑不当导致需要额外处理pop字段
            dir = dir[:-2] + [dir[-1]]
            key_first = '_'.join(dir[require*2:require*2+2])
            key_second = '_'.join(dir[:require*2]+dir[require*2+2:])
            if key_first not in res_path.keys():
                res_path[key_first] = {key_second:path}
            else:
                res_path[key_first][key_second] = path
        return res_path

    # 获取所有的问题尺寸
    def get_para(self, solver):
        para:List[str] = []
        for res_dir in self.method_para[solver].keys():
            dir = res_dir.split('_')
            para_dir = ''.join(dir[:4])
            if para_dir not in para:
                para.append(para_dir)
        return para

    # 根据提供的地址目录读取结果
    def read_res(self):
        pass

    # def _get_method_para(self):
    #     return self.method_para


if __name__ == '__main__':
    demo = ReadResultDir(GeneticMOSG='GeneticMOSG1-res_dir-2.json', MOSG='ORIGAMIM-res_dir-0.json')
    # _method_para = demo._get_method_para()

    # 3种使用方法

    # method1
    res_path1 = demo.read_res_dir(solver='MOSG', read_all=True)

    # method2
    res_path2_1 = demo.read_res_dir(solver='MOSG', read_part_by_dir='obj_5_target_25')
    res_path2_2 = demo.read_res_dir(solver='GeneticMOSG', read_part_by_dir='obj_5_target_25')

    # method3
    res_path3_1 = demo.read_res_dir(solver='GeneticMOSG', read_part_by_dim='obj')
    res_path3_2 = demo.read_res_dir(solver='GeneticMOSG', read_part_by_dim='target')
    res_path3_3 = demo.read_res_dir(solver='GeneticMOSG', read_part_by_dim='gen')
    res_path3_4 = demo.read_res_dir(solver='GeneticMOSG', read_part_by_dim='pop')
