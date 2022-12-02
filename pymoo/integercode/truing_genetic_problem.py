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
        self.conflict_target:List[int] = [target for target in conflict.keys()]  # 只取出发生冲突的target,记住下标
        self.conflict_ct:List[List[float]] = [ct for ct in conflict.values()]  # 发生冲突的target对应的冲突方案
        # 算二进制编码长度,如果采用integer编码则不需要计算
        # self.part_len:List[int] = [math.ceil(math.log(len(ct_set), 2)) for ct_set in self.conflict_ct]  
        # 算冲突的方案数量
        # 上下界, 如果是binary code上下界为[0,1], 如果是integer code下界为0,上界为part_max-1
        self.part_max:List[int] = [len(ct_set) for ct_set in self.conflict_ct]
        # self.n_var = sum(self.part_len)  # 每个部分的编码长度
        self.n_var = len(self.conflict_ct) # 为存在冲突的target的len,并非整个T
        # self.len:int = len(self.part_len)  # 语义上,采用inteager code后, self.len=self.n_var
        # 这个也是binary code特有的变量,inteager不再需要, 表示每个part开始的下标
        # self.part_start:List[int] = [sum(self.part_len[:idx]) for idx in range(len(self.part_len))]  # 由于冲突规模未知，需计算具体问题的具体编码长度
        self.ct_star_total = ct_star_total
        super(TruingGP, self).__init__(n_var=self.n_var, n_obj=problem.n_obj, n_constr=0, xl=np.zeros(shape=(len(self.part_max),)), xu=np.array(self.part_max)-1)  # TODO 上下界

    # NOTE: truing_by_mincov是在SG框架中的程序，所以处理的是maximise task
    # 该函数并非不需要ct，而是提供idx[下标]，访问ct；相反不需要fit
    def truing_by_mincov(self, ct_i:np.ndarray, b:np.ndarray, 
        pf_total=None, ct_star_total:np.ndarray=None)\
            ->Union[np.ndarray, np.ndarray]:

        K = 3 # ?
        count = 0  # 记录精修的次数
        # 每次搜索到新方案ct后，结合game table算fitness
        problem_mosg = MOSG(player_num=self.problem.player, target_num=self.problem.target)
        model = ORIGAMIM(MOSG=problem_mosg)

        for gameidx in range(self.problem.player-1):  # 第一层for遍历obj
            # 利用ct尝试精修，用到MINCOV
            model.c = ct_i # ct_i表示第一层演化的种群的第i个个体
            model.updateC()
            model.updateU(i=gameidx)
            # 需要注意的是，由于gmosg编码的破坏性，next和idx对应不一定是相同的，但是影响并不是很大只要next>1
            # （简单了就是gmosg编码得到的next不是真的next，为了不错过最优解需要扩K倍，而扩大的代价是复杂度上升）
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
            # MIN-COV程序到此结束，后一段for是为leftAllocation设计的，因此后两行cal_payoff_defender被注释掉，挪到leftAllocation中
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

    def calFit(self, x_pop:np.ndarray) ->Union[np.ndarray, np.ndarray, np.ndarray]:
        fit_pop = np.empty(shape=(x_pop.shape[0], self.problem.player-1))
        ct_pop = np.empty(shape=(x_pop.shape[0], self.problem.target))
        fit_total:np.ndarray=None
        for idx, ind in enumerate(x_pop):

            '''decode部分 负责将ind转化为冲突部分的ct (conflict_idx_real)
            求ct部分 负责将冲突的ct (conflict_idx_real) 和不冲突的ct合并,以求得完整的ct (self.ct_original)'''

            
            # decode 
            # for binary code
            # 对演化算法的ind做解码，先按parts分块,每块再从01串转为real
            # str(int(ind)) means turn bool to int to str
            # conflict_idx_real:List = []
            # for part_idx in range(self.n_var):  # inteager code中, 1 part==1 bit
            #     x_part:np.ndarray = ind[self.part_start[part_idx]:self.part_start[part_idx] + self.part_len[part_idx]]
            #     if x_part.size == 1:
            #         part_binary:str = str(int(x_part[0]))
            #     else:
            #         part_binary: str = functools.reduce(lambda x, y: str(int(x)) + str(int(y)), x_part)
            #     conflict_idx_real.append(int(part_binary, 2))
            # for inteager code
            ct_conflict = [self.conflict_ct[target_idx][ct_idx] for target_idx, ct_idx in enumerate(ind)] # List[float]


            # 求ct
            # for binary code
            # self.ct_original[np.array(self.conflict_target)] = self.calCt(conflict_idx_real)
            # for inteager code
            self.ct_original[np.array(self.conflict_target)] = ct_conflict

            # 求fit
            self.problem.cal_payoff(ct=self.ct_original)

            # NOTE METHOD2 随意返回一个ct_pop占位置
            # fit_pop[idx] = self.problem.cal_payoff_defender()
            # ct_pop[idx] = self.ct_original

            # NOTE METHOD1
            fit_original = self.problem.cal_payoff_defender()
            if fit_total is None:
                fit_total = fit_original
            else:
                fit_total = np.vstack([fit_total, fit_original])

            b = self.problem.cal_payoff_defender()  # cal_payoff_defender计算第二层演化得到的个体的收益，并将cal_payoff_defender作为下界传入MIN-COV
            fit, self.ct_star_total = self.truing_by_mincov(ct_i=self.ct_original, b=b, pf_total=b,
                                                            ct_star_total=self.ct_star_total)
            # 一个个体/integer code有能力解码得多个
            if fit.ndim == 1: 
                fit = fit[np.newaxis, :]
                fit_pop[idx] = fit
            else:
                rank = list(find_non_dominated(F=fit))
                pf_i = random.sample(rank, 1) # 这里rank=1是非常不合理的，应该变成都堆在一起，然后排序
                fit_pop[idx] = fit[pf_i]
            fit_total = np.vstack([fit_total, fit])
            fit_pop[idx] = self.problem.cal_payoff_defender()
            ct_pop[idx] = self.ct_original

            # # restore ct
            # self.ct_original = ct_temp

        return fit_pop, ct_pop, fit_total


    def _evaluate(self, x, out, *args, **kwargs):

        fit, ct, fit_total = self.calFit(x_pop=x)

        out['F'] = -fit
        out['CT'] = ct
        # 记录全局fit_total
        out['FIT_TOTAL'] = -fit_total

        return out

