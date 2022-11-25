
import pickle
import os
import datetime
import math


from pymoo.model.result import Result
from pymoo.MOSGsGeneticSolver.performance import Performance


import tool.algorithm


class resSave():


    def __init__(self, res:Result, Solver, running_time=0, res_dir=None):
        self.res:Result = res
        self.running_time = running_time
        self.res_dir = res_dir
        self.Solver = Solver

    '''
    分别提供一级目录和二级目录，将结果存在二级目录下。（两级目录相同时只需要提供一级目录）
    '''
    def saveResult(self, para_dir, para_filename=None):
        # Input:
        # para_dir表示根目录之上存储,文件之上的参数目录,用于不同规模实验结果的分类
        # para_filename表示文件名称用到的参数,用于文件命名

        # para_dir与para_filename相同时, para_filename可省略
        if para_filename is None:
            para_filename = para_dir
        dir = os.path.join(os.getcwd(), self.res_dir, para_dir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename1 = [self.Solver, para_filename, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'), str(math.floor(self.running_time*1000))]
        filename = '-'.join(filename1) + '.txt'
        save_path = os.path.join(dir, filename)
        # 保存结果
        print(save_path)
        tool.algorithm.dumpVariPickle(self.res.F, save_path)
        # 返回（绝对）存储路径
        return save_path

    def loadResult(self, para_dir, filename):
        os.chdir(self.res_dir)
        path = os.path.join(os.getcwd(), para_dir, filename)
        return tool.algorithm.loadVariPickle(path)