# 演化算法得到粗略解后，将演化算法作为先验知识做进一步优化

import numpy as np
from sklearn import preprocessing


from typing import *
import operator
import time
import copy

import pkg_resources

import sys
sys.path.append('./')

from pymoo.factory import get_reference_directions, get_sampling, get_crossover, get_mutation
from pymoo.model.individual import Individual
from pymoo.model.population import Population


from pymoo.model.result import Result
from pymoo.optimize import minimize
from pymoo.problems.securitygame.MOSG import SGs1
from MOSGs.ORIGAMIM import ORIGAMIM
from pymoo.MOSGsGeneticSolver.performance import Performance


from securitygame_core.MO_security_game import MOSG
from pymoo.MOSGsGeneticSolver.truing_genetic_problem import TruingGP
from pymoo.MOSGsGeneticSolver.genetic_turing import GeneticTruing


import tool.algorithm



class Truing():


    def __init__(self, res:Result=None, para_dir=None):

        self.para_dir = para_dir

        # 读入数据
        self.res:Result = res
        self.problem:SGs1 = res.problem

        # 来自种群，规模远大于opt
        self.x:np.ndarray = res.pop.get('X')
        self.fit:np.ndarray = res.pop.get('F') * -1  # maximize fit
        feasible_bool:np.ndarray = res.pop.get('feasible')
        self.feasible = np.array(list(map(lambda x: int(x),feasible_bool)))
        self.ct:np.ndarray = res.pop.get('CT')

        # 来自opt，复杂度较高的方法推荐opt，规模小
        self.opt = res.opt
        self.opt_x = res.opt.get('X')
        self.opt_fit = res.opt.get('F') * -1
        feasible_bool:np.ndarray = res.opt.get('feasible')
        self.opt_feasible = np.array(list(map(lambda x: int(x),feasible_bool)))
        # self.opt_ct:np.ndarray = res.opt.get('CT')

        # 最终保存的pf信息
        self.fit_pf:Union[None,np.ndarray] = None  # minimize Fitness
        self.ct_pf:List[np.ndarray] = []  # 一般算法算完就返回，不需要存ct；只有分阶段的算法需要保存


    def check(self):
        a = self.res.pop.get('F')
        for best_fit in self.res.F:
            for idx in range(a.shape[0]):
                if operator.eq(best_fit, a[idx]).all():
                    break
                if idx == a.shape[0]-1:
                    print('opt_fit dont exist in pop_fit, error!!!')


    # NOTE: 提供的pf_total必须是minimize task
    def mulResCompare(self, pf_total:List[np.ndarray], name_total:List[str]):
        pf:np.ndarray = np.vstack(pf_total)
        len_total = [len(pf) for pf in pf_total]
        print(Performance(pf, len_total, name_total, para_dir=self.para_dir))


    def update_by_idx(self, idx:Union[List[int], np.ndarray]):
        self.ct = self.ct[idx]
        self.fit = self.fit[idx]
        self.feasible = self.feasible[idx]
        self.x = self.x[idx]


    # 将编码X转化为ct
    def x2CtGamma(self, x:np.ndarray, fit_true:np.ndarray=None, model:SGs1=None)\
          ->Tuple[Union[np.ndarray, None], Dict[int,List[int]], Dict[int,List[float]], Dict[int, int]]:
        if model is None:
            model = SGs1(player_num=self.problem.player, target_num=self.problem.target)
        # decode DNA into strategy
        strategy = model.GetQuery(x)
        return model.Strategy2Ct(strategy)
        # DEBUG
        # if ct is not None:  # feasible
        #     model.cal_payoff(ct)
        #     fit = model.cal_payoff_defender()
        #     if not operator.eq(fit, fit_true).all():
        #         print('error1')
        # else:  # not feasible  # 一般情况下不应该出现该情况
        #     return None

    # NOTE: truing_by_mincov是在SG框架中的程序，所以处理的是maximise task
    # 该函数并非不需要ct，而是提供idx下标，访问ct；相反不需要fit
    def truing_by_mincov(self, ct_i:np.ndarray, b:np.ndarray, pf_total=None, ct_star_total:np.ndarray=None)\
            ->Tuple[np.ndarray, np.ndarray]:
        K = 3
        count = 0  # 记录精修的次数
        # 每次搜索到新方案ct后，结合game table算fitness
        problem_mosg = MOSG(player_num=self.problem.player, target_num=self.problem.target)
        model = ORIGAMIM(MOSG=problem_mosg)

        for gameidx in range(self.fit.shape[1]):  # 第二层for遍历obj
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
            for obj_idx in range(self.fit.shape[1]):
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

    # 下面mosgSearch_pop和mosgSearch_opt分别代表从全体种群中开始搜索还是从最优解中开始搜索
    #   注：两者复杂度相差不大，因此直接使用mosgSearch_pop，因为它效果优于_opt
    #   _opt与_pop的区别：种群数量远大于最优解，最优解是种群的子集

    '''# 遍历求解
    # 重新算当前Ct的Fit，作为b，传给Direct Search解
    # 每个目标依次单独优化，优化目标为单目标上提升最大化并且所需资源最小化，留一个目标，吧剩余资源分给他。'''
    def mosgSearch_pop(self):  # 在整个种群中找精修的方法

        # 先从res把整数编码转成实数编码ct
        pf_total:Union[None,np.ndarray] = None
        # ct_total:Union[None,np.ndarray] = None
        ct_star_total = None
        for idx in range(self.x.shape[0]):
            ct, _, _, _ = self.x2CtGamma(x=self.x[idx], model=self.problem, fit_true=self.fit[idx])
            if ct is not None:
                self.ct[idx] = ct
                self.feasible[idx] = 1
            else:
                self.feasible[idx] = 0
        self.update_by_idx(idx=self.feasible == 1)

        timer_start = time.perf_counter()
        for i in range(self.ct.shape[0]):  # 第一层for遍历pop
            # 想法是每个ct用MINCOV算一次ctPrime，然后循环check是否违反b
            # 若死锁型违反则无视（TODO），或者返回一个相对好的
            # 得到cPrime后调用leftAllocation循环N次
            b = self.fit[i]
            # 把已知的可行解放入pf集，保证解不退化
            if pf_total is None:
                pf_total = b
                # ct_total
            else:
                pf_total = np.vstack([pf_total, b])
                # ct_total
            pf_total, ct_star_total = self.truing_by_mincov(self.ct[i], b, pf_total, ct_star_total)
        timer_end = time.perf_counter()

        print('mosgSearch_pop:{}'.format(timer_end - timer_start))
        # 暂时忽略ct_pf
        # self.ct_pf = ct_star_total
        self.fit_pf = -pf_total

    def initialization(self, ub:np.ndarray, lb:np.ndarray, pop_size:int, sample_real:np.ndarray) ->np.ndarray:
        # 分别计算distance from sample to up-bound+1 and to low-bound,
        # 然后取axis_normalised*dis2lb if axis_normalised<0 otherwise axis_normalised*dis2ub
        # 最后加上best_sample
        sample_len = len(sample_real)
        conv = np.eye(N=sample_len)
        axis = np.random.multivariate_normal(mean=sample_real, cov=conv, size=pop_size)
        axis[0] = 0  # 第一条数据保存为best_sample，不发生偏移
        axis_normalised = preprocessing.MaxAbsScaler().fit_transform(axis)  # [-1, 1]
        dis2lb = sample_real - lb
        dis2ub = ub-0.01 - sample_real
        sampling = np.repeat(a=sample_real[np.newaxis,:], repeats=pop_size, axis=0)
        loc_neg = axis_normalised < 0
        loc_pos = axis_normalised > 0
        sampling_neg = sampling + dis2lb * axis_normalised
        sampling_pos = sampling + dis2ub * axis_normalised
        sampling[loc_pos] = sampling_pos[loc_pos]
        sampling[loc_neg] = sampling_neg[loc_neg]
        return np.floor(sampling).astype(int)

    def real2binary(self, real:np.ndarray, problem:TruingGP) ->np.ndarray:
        part_start = [sum(problem.part_len[:i]) for i in range(len(problem.part_len))]
        binary_len = sum(problem.part_len)
        binary = np.empty(shape=[binary_len,])
        for idx, target in enumerate(problem.conflict_target):
            r = real[idx]
            b = bin(r)[2:]
            b = '0' * (problem.part_len[idx] - len(b)) + b
            for i, b_i in enumerate(b):
                binary[part_start[idx] + i] = int(b_i)
        return binary

    # def calGamma(self):
    #     self.Gamma:List[Dict[int, List[int]]] = []
    #     for idx in range(self.x.shape[0]):  # 第一个循环pop
    #         x =

    # TODO LIST
    # TODO 1: 初始化
    # TODO 2: 把mincov加到每轮迭代中

    # 让函数直接返回发生冲突的ct集，然后对着ct集编码
    def geneticSearch(self, pop_size:int=100, gen_n:int=100) ->np.ndarray:
        # 由于Gamma数据结构复杂持久化，因此根据读入数据重新计算Gamma
        ref_dirs = get_reference_directions("das-dennis", self.problem.n_obj, n_partitions=12)
        ct_star_total = None
        for i in range(self.opt_fit.shape[0]):  # for用于遍历pop
            # NOTE: 这里求idx=i时的self.opt_x的ct，讲道理应该要存到self.opt_ct，没存，以后万一要用可能照成一定麻烦
            ct, _, conflict_ct, ct_idx = self.x2CtGamma(x=self.opt_x[i], model=self.problem)
            if ct is None:  # not feasible
                continue
            problem = TruingGP(conflict=conflict_ct, ct=ct, problem=self.problem, ct_star_total=ct_star_total)
            ub = np.array(problem.part_max)
            lb = np.zeros(ub.shape)
            # ct_idx_binary = self.real2binary(ct_idx_real, problem)
            ct_idx_real:np.ndarray = np.array([ct_idx[target] for target in problem.conflict_target])
            sampling_real = self.initialization(ub=ub, lb=lb, pop_size=pop_size, sample_real=ct_idx_real)
            sampling_binary = np.ndarray(shape=[pop_size, problem.n_var])
            for j in range(pop_size):
                sampling_binary[j] = self.real2binary(sampling_real[j], problem)
            algorithm = GeneticTruing(pop_size=pop_size,
                              ref_dirs=ref_dirs, display=False, save_history=False, verbose=False,
                              # sampling=get_sampling('bin_random'),
                              sampling=sampling_binary,
                              crossover=get_crossover('bin_hux'),
                              mutation=get_mutation('bin_bitflip'),
                              eliminate_duplicates=True)
            # NOTE: res: minimize task
            res = minimize(problem=problem,
                           algorithm=algorithm,
                           seed=1,
                           termination=('n_gen', gen_n))
            ct_star_total = problem.ct_star_total
            # print('gen_n={}, pop_size={}, 参数下，程序运行时间:{}秒'
            #       .format(gen_n, pop_size, res.exec_time))
            if self.fit_pf is None:  # NOTE: minimise task
                self.fit_pf = res.opt.get('F')
                self.ct_pf = res.opt.get('CT')
            else:
                self.fit_pf = np.vstack([self.fit_pf, res.opt.get('F')])
                self.ct_pf = np.vstack([self.ct_pf, res.opt.get('CT')])
        pf_total_temp = self.fit_pf

        # mincov 精修
        # NOTE: maximise task
        self.fit_pf = -self.fit_pf
        ct_star_total = None
        pf_total = None
        for idx, b in enumerate(self.fit_pf):
            if np.isnan(self.ct_pf[idx]).any():
                continue
            if pf_total is None:
                pf_total = b
            else:
                pf_total = np.vstack([pf_total, b])
            pf_total, ct_star_total = self.truing_by_mincov(self.ct_pf[idx], b, pf_total, ct_star_total)
        self.fit_pf = -pf_total
        return pf_total_temp






