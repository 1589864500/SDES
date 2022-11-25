import functools
import random
from typing import *


import numpy as np
import math


from pymoo.model.problem import Problem
from pymoo.util.nds.non_dominated_sorting import find_non_dominated


from MOSGs.ORIGAMIM import ORIGAMIM
from pymoo.problems.securitygame import SGs1
from securitygame_core.MOSGs_genetic import MOSG
from pymoo.MOSGsGeneticSolver.performance import Performance


# 精修演化算法的对冲突的选择进行编码
# 特点是：冲突的数量为止，每个冲突的规模未知
#   冲突数量未知导致我们无法确定输入的数量，用List结构
#   冲突规模未知导致程序每个部分不等长（总长度固定）


class TruingGP(Problem):

    def __init__(self, problem:SGs1, conflict:Dict[int,List[float]]=None, ct:np.ndarray=None, ct_star_total=None):
        # conflict表示发生冲突的具体取值，Dict长度为M，List内包含具体的冲突方案ct，按递增排序

        self.problem:SGs1 = problem
        self.ct_original = ct  # 备用方案，若解不可行则沿用该解
        self.conflict_target:List[int] = [target for target in conflict.keys()]  # 只取出发生冲突的target
        self.conflict_ct:List[List[float]] = [ct for ct in conflict.values()]  # 发生冲突的target对应的冲突方案
        self.part_len:List[int] = [math.ceil(math.log(len(ct_set), 2)) for ct_set in self.conflict_ct]
        self.part_max:List[int] = [len(ct_set) for ct_set in self.conflict_ct]
        self.n_var = sum(self.part_len)  # 每个部分的编码长度
        self.len:int = len(self.part_len)  # len表示编码长度，即只包含冲突的target的长度
        self.part_start:List[int] = [sum(self.part_len[:idx]) for idx in range(len(self.part_len))]  # 由于冲突规模未知，需计算具体问题的具体编码长度
        self.ct_star_total = ct_star_total
        super(TruingGP, self).__init__(n_var=self.n_var, n_obj=problem.n_obj, n_constr=0, xl=0, xu=1)

    # NOTE: truing_by_mincov是在SG框架中的程序，所以处理的是maximise task
    # 该函数并非不需要ct，而是提供idx下标，访问ct；相反不需要fit
    def truing_by_mincov(self, ct_i:np.ndarray, b:np.ndarray, pf_total=None, ct_star_total:[np.ndarray, None]=None)\
            ->[np.ndarray, np.ndarray]:

        K = 3
        count = 0  # 记录精修的次数
        # 每次搜索到新方案ct后，结合game table算fitness
        problem_mosg = MOSG(player_num=self.problem.player, target_num=self.problem.target)
        model = ORIGAMIM(MOSG=problem_mosg)

        for gameidx in range(self.problem.player-1):  # 第二层for遍历obj
            # 利用ct尝试精修，用到MINCOV
            model.c = ct_i
            model.updateC()
            model.updateU(i=gameidx)
            # 需要注意的是，由于gmosg编码的破坏性，next和idx对应不一定是相同的，但是影响并不是很大只要next>1
            idx = np.argsort(-model.U_ia)
            # NOTE: 由于不好确定next的具体取值，将next设为T一定没错，只是时间会增长
            # next = model.getNextLen(gap)
            next = min(model.getNextLen(epsilon=0.1)*K, self.problem.target)
            ct_star = model.MINCOV(gameIdx=gameidx, b=b, next=next, idx=idx)
            if ct_star is None:
                continue
            if ct_star_total is not None and ct_star in ct_star_total:
                continue
            if ct_star_total is not None:
                ct_star_total = np.vstack([ct_star, ct_star_total])
            else:
                ct_star_total = ct_star
            model.c = ct_star
            model.updateC()
            model.updateU(i=gameidx)
            # self.problem.cal_payoff(ct_star)
            # fit_star = self.problem.cal_payoff_defender()
            for obj_idx in range(self.problem.player-1):  # 第二层for遍历obj
                model.leftAllocation(b=b, obj_idx=obj_idx)
                ct_final = model.c
                self.problem.cal_payoff(ct_final)
                fit_final = self.problem.cal_payoff_defender()
                if pf_total is None:
                    pf_total = fit_final
                else:
                    pf_total = np.vstack([pf_total, fit_final])
                # ct_total = np.vstack([ct_total, ct_final])
                count += 1
        # print('找到更优解{}处'.format(count))
        return pf_total, ct_star_total

    def calCt(self, idx:List[int]) ->List[float]:
        # idx_i表示冲突target_i最终选择的方案的下标，长度与self.conflict_ct相同

        # 处理异常值
        idx:List[int] = [idx[i] if idx[i] < self.part_max[i] else self.part_max[i] - 1 for i in range(self.len)]

        return [self.conflict_ct[i][idx[i]] for i in range(self.len)]


    def calFit(self, x_pop:np.ndarray) ->Union[np.ndarray, np.ndarray]:
        fit_pop = np.empty(shape=(x_pop.shape[0], self.problem.player-1))
        ct_pop = np.empty(shape=(x_pop.shape[0], self.problem.target))
        for idx, ind in enumerate(x_pop):

            # decode
            # 对演化算法的ind做解码，先按parts分块,每块再从01串转为real
            # str(int(ind)) means turn bool to int to str
            conflict_idx_real:List = []
            for part_idx in range(self.len): 
                x_part:np.ndarray = ind[self.part_start[part_idx]:self.part_start[part_idx] + self.part_len[part_idx]]
                if x_part.size == 1:
                    part_binary:str = str(int(x_part[0]))
                else:
                    part_binary: str = functools.reduce(lambda x, y: str(int(x)) + str(int(y)), x_part)
                conflict_idx_real.append(int(part_binary, 2))

            # 求ct
            self.ct_original[np.array(self.conflict_target)] = self.calCt(conflict_idx_real)

            # 求fit
            self.problem.cal_payoff(ct=self.ct_original)
            # TODO LIST
            # b = self.problem.cal_payoff_defender()
            # fit, self.ct_star_total = self.truing_by_mincov(ct_i=self.ct_original, b=b, pf_total=b,
            #                                                 ct_star_total=self.ct_star_total)
            # if fit.ndim == 1:
            #     fit_pop[idx] = fit
            # else:
            #     rank = list(find_non_dominated(F=fit))
            #     pf_i = random.sample(rank, 1)
            #     fit_pop[idx] = fit[pf_i]
            fit_pop[idx] = self.problem.cal_payoff_defender()
            ct_pop[idx] = self.ct_original

            # # restore ct
            # self.ct_original = ct_temp

        return fit_pop, ct_pop


    def _evaluate(self, x, out, *args, **kwargs):

        fit, ct = self.calFit(x_pop=x)

        out['F'] = -fit
        out['CT'] = ct

        return out

