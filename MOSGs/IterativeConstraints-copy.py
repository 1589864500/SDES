# from securitygame_core.MO_security_game import MOSG

import datetime
import math


import copy
import numpy as np
import os


import tool.algorithm
from MOSGs.ORIGAMIM import ORIGAMIM
from MOSGs.ORIGAMIA import ORIGAMIA
# from MOSGs.MILPs import MILPs
from MOSGs.resultMOSGs import resultMOSGs
from pymoo.MOSGsGeneticSolver.performance import Performance
from pymoo.MOSGsGeneticSolver.resSave import resSave





class IterativeConstrains():
    def __init__(self, MOSG=None, previousBoundsList=None, infeasibleBoundsList=None, epsilon=1, Solver='ORIGAMIM', res_dir=None):
        # self.b = b  # initial previousBoundsList and b
        if infeasibleBoundsList is None:
            infeasibleBoundsList = {}
        if previousBoundsList is None:
            previousBoundsList = {}
        self.previousBoundsList = previousBoundsList
        self.infeasibleBoundsList = infeasibleBoundsList
        self.MOSG = MOSG
        self.epsilon = epsilon
        self.Solver = Solver
        self.res_ct = []
        self.res_fit = []
        self.deep = 0
        self.count = 0
        self.errorCount = 0
        self.hv:float = 0
        if res_dir is not None:
            self.res_dir = res_dir
        else:
            self.res_dir = os.getcwd()

    def deepUpdate(self, increment):
        self.deep += increment
        if increment > 0:
            print("deep ++:{}".format(self.deep))
            print('第{}次调用do程序'.format(self.count))
        else:
            print("deep --:{}".format(self.deep))

    # for convenience, do() function only considers the change of ct(resource allocation) and b(bound), but not payoff or other details
    def do(self, b):
        # b.size = objective
        # debug
        chechpointB = None
        # self.count += 1
        # self.deepUpdate(1)
        # print("当前讨论的基线，b：{}".format(np.round(b,3)))
        if ~tool.algorithm.dicQuery(b[1:], self.previousBoundsList):  # if b not in previousBoundsList, goto explore
            self.previousBoundsList = tool.algorithm.dicAppendNumpy(b[1:], self.previousBoundsList)  # add b into previousBoundsList to show we will explore b just now
            if self.Solver == 'MILPs':
                # use MILPs/ORIGAMIM solver to solve a resources allocation (c_t) solution (maybe the solution will be infeasible is b is too tight)
                SGSolver = MILPs(MOSG=self.MOSG, b=b)
            elif self.Solver == 'ORIGAMIM':
                SGSolver = ORIGAMIM(MOSG=self.MOSG)
            elif self.Solver == 'ORIGAMIA':
                origamim = ORIGAMIM(MOSG=self.MOSG)
                SGSolver = ORIGAMIA(ORIGAMIM=origamim)
            else:
                print('Undefined SGSolver, exit()')
                exit()
            if self.Solver == 'MILPs':
                c = SGSolver.do()
            elif self.Solver == 'ORIGAMIM':
                c, count = SGSolver.do(b)
                if count is not None:
                    self.errorCount += count
            elif self.Solver == 'ORIGAMIA':
                c = SGSolver.do(b)
            else:
                print('Undefined SGSolver')
            if c is not None:
                # print("{}函数求得新ct：{}".format(self.Solver, np.round(c,3)))
                # SGSolver解出来的c是obj2-objn需要的最低需求，我们把left都分配给obj1，理论上一次do()能得到一个可行解。
                self.MOSG.set_ct(c)
                self.MOSG.cal_payoff()
                self.MOSG.cal_payoff_defender()
                v = self.MOSG.get_payoff_defender()
                self.res_fit.append(v * -1)  # fit乘-1转化为最小化问题
                self.res_ct.append(c)

                # debug
                # if np.any(v < b):
                #     print('ERROR')
                # else:
                #     print('本轮迭代ct结果满足b约束')

                # print("c=SGSolver.do(b)  sum(c):", np.sum(c))
                # print("ct对应的v：{}".format(np.round(v, 3)))
                for i in range(1, b.size):
                    bPrime = copy.copy(b)
                    bPrime[i] = v[i] + self.epsilon
                    # print("新bPrime：{}, 在第{}个指标上更新".format(np.round(bPrime, 3), i+1))
                    if tool.algorithm.vLessD(bPrime[1:], self.infeasibleBoundsList): # if bPrime not greater than anyone in infeasibleBoundsList, it is worth exploring
                        # print("递归下降，说明bPrime还能作为更严格的约束求得可行解。")
                        # self.debugDo(RESULTs = self.res_ct)
                        self.do(bPrime)
            else:
                # print('当前b探索分支到头/不可行，b:{}'.format(b))
                self.infeasibleBoundsList = tool.algorithm.dicAppendNumpy(b[1:], self.infeasibleBoundsList)
        # else:
        #     print('b is in previousBoundsList, do not goto explore')
        # print("========================================\n")
        # self.deepUpdate(-1)

    def result(self, running_time):
        self.res:resultMOSGs = resultMOSGs(np.array(self.res_ct), np.array(self.res_fit), running_time)
        self.res_save = resSave(res=self.res, Solver=self.Solver, running_time=self.res.exec_time, res_dir=self.res_dir)

    def saveResult(self, para_dir):
        return self.res_save.saveResult(para_dir=para_dir)

    def loadResult(self, para_dir, filename):
        return self.res_save.loadResult(para_dir=para_dir, filename=filename)

    def debugDo(self, **kwargs):
        for arg, value in kwargs.items():
            if isinstance(value, (np.ndarray)):
                print("{}({}):\n{}".format(arg, value.size, np.round(value)))
            elif isinstance(value, (list, map, tuple)):
                print("{}({}):\n{}".format(arg, len(value), np.round(value, 2)))
            else:
                print("{}:\n{}".format(arg, np.round(value)))
