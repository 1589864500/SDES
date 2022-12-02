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

from pymoo.factory import get_reference_directions, get_sampling, get_crossover, get_mutation, get_termination
from pymoo.model.individual import Individual
from pymoo.model.population import Population


from pymoo.model.result import Result
from pymoo.optimize import minimize
from MOSGs.ORIGAMIM import ORIGAMIM
from pymoo.MOSGsGeneticSolver.performance import Performance


from securitygame_core.MO_security_game import MOSG
from pymoo.integercode.MOSG import SGs1
# from pymoo.MOSGsGeneticSolver.truing_genetic_problem import TruingGP
from pymoo.integercode.truing_genetic_problem import TruingGP
# from pymoo.floatscoringmechanism.truing_genetic_problem import TruingGP
from pymoo.MOSGsGeneticSolver.genetic_turing import GeneticTruing


import tool.algorithm



class Truing():


    def __init__(self, res:Result=None, para_dir=None):

        self.para_dir = para_dir

        if res is not None:
            # 读入数据
            self.res:Result = res
            self.problem:SGs1 = res.problem

            # 来自种群，规模远大于opt
            self.x:np.ndarray = res.pop.get('X') # 在integer code中表示攻击集的长度，因此len(X)=N
            self.fit:np.ndarray = res.pop.get('F') * -1  # maximize fit
            feasible_bool:np.ndarray = res.pop.get('feasible')
            self.feasible = np.array(list(map(lambda x: int(x),feasible_bool)))
            self.ct:np.ndarray = res.pop.get('CT') # 和self.x一一对应，len(ct)=T

            # 来自opt，复杂度较高的方法推荐opt，规模小
            self.opt = res.opt
            self.opt_x = res.opt.get('X')
            self.opt_fit = res.opt.get('F') * -1
            feasible_bool:np.ndarray = res.opt.get('feasible')
            self.opt_feasible = np.array(list(map(lambda x: int(x),feasible_bool)))
            self.opt_ct:np.ndarray = res.opt.get('CT')

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
    def mulResCompare(self, pf_total:List[np.ndarray], name_total:List[str], 
        indicator_hv=False, indicator_igdplus=False,indicator_igd=False,indicator_gd=False,indicator_gdplus=False,
        dump=False, fname=None, file_exist=False, repeat=None, para_dir=None): 
        perm = Performance(pf_total=pf_total, name_total=name_total, 
            indicator_hv=indicator_hv, indicator_igdplus=indicator_igdplus,indicator_igd=indicator_igd,indicator_gd=indicator_gd,indicator_gdplus=indicator_gdplus,
            para_dir=para_dir)
        print(perm)
        if dump:
            perm.dumpJson(fname=fname, file_exist=file_exist, repeat=str(repeat))



    def update_by_idx(self, idx:Union[List[int], np.ndarray]):
        self.ct = self.ct[idx]
        self.fit = self.fit[idx]
        self.feasible = self.feasible[idx]
        self.x = self.x[idx]


    # 将编码X转化为ct
    def x2CtGamma(self, strategy:np.ndarray, fit_true:np.ndarray=None, model:SGs1=None)\
          ->Tuple[Union[np.ndarray, None], Dict[int,List[int]], Dict[int,List[float]], Dict[int, int]]:
        if model is None:
            model = SGs1(player_num=self.problem.player, target_num=self.problem.target)
        # decode DNA(strategy) into ct 
        return model.strategy2CtInt(strategy)

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
            model.c = ct_i  # 个体i的向量c
            model.updateC()
            model.updateU(i=gameidx)
            # 需要注意的是，由于gmosg编码的破坏性，next和idx对应不一定是相同的，但是影响并不是很大只要next>1
            idx = np.argsort(-model.U_ia)
            # NOTE: 由于不好确定next的具体取值，将next设为T一定没错，只是时间会增长
            # next = model.getNextLen(gap)
            next = min(model.getNextLen(epsilon=0.1)*K, self.problem.target)
            ct_star = model.MINCOV(gameIdx=gameidx, b=b, next=next, idx=idx)  # 虽然传入的是向量b，实际上用到的只有b[gameidx]
            if ct_star is None:  # 在求opt的，mincov时不会出现这种情况，除非个体不可行
                continue
            if ct_star_total is not None and ct_star in ct_star_total:
                continue
            if ct_star_total is not None:
                ct_star_total = np.vstack([ct_star, ct_star_total])
            else:
                ct_star_total = ct_star
            # model.c = ct_star  # 这里另model.c=ct_star等于用star替换了原ct TODO 把star保留会怎么样？
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
                    pf_total = fit_final    # 只有fit_final才会影响pf_total的逻辑是有问题的，那些ct_star=None也应该能够影响。
                else:
                    pf_total = np.vstack([pf_total, fit_final])
                # ct_total = np.vstack([ct_total, ct_final])
                count += 1
        # print('找到更优解{}处'.format(count))
        return pf_total, ct_star_total

    ''' # 下面mosgSearch_pop和mosgSearch_opt分别代表从全体种群中开始搜索还是从最优解中开始搜索
    #   注：两者复杂度相差不大，因此直接使用mosgSearch_pop，因为它效果优于_opt
    #   _opt与_pop的区别：种群数量远大于最优解，最优解是种群的子集

    # 先将integer code转为continuous code(ct)
    # 重新算当前Ct的Fit，作为b，传给Direct Search解
    # 每个目标依次单独优化，优化目标为单目标上提升最大化并且所需资源最小化，留一个目标，吧剩余资源分给他。'''
    def mosgSearch_pop(self):  # 在整个种群中找精修的方法
        # 先从res把整数编码转成实数编码ct
        pf_total:Union[None,np.ndarray] = None
        # ct_total:Union[None,np.ndarray] = None
        ct_star_total = None
        
        # NOTE for Integer  先将integer code转为continuous code(ct)和Gamma
        for idx in range(self.x.shape[0]):
            ct, _, _, _ = self.x2CtGamma(strategy=self.x[idx], model=self.problem, fit_true=self.fit[idx])
            if ct is not None:
                self.ct[idx] = ct
                self.feasible[idx] = 1
            else:
                self.feasible[idx] = 0
        self.update_by_idx(idx=self.feasible == 1)

        # 重新算当前Ct的Fit，作为b，传给Direct Search解 
        # 每个目标依次单独优化，优化目标为单目标上提升最大化并且所需资源最小化，留一个目标，吧剩余资源分给他。
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
        sample_len = len(sample_real)  # 希望生成和参考解长度相同的解
        conv = np.eye(N=sample_len)  # conv表示(协)方差，用于控制采样，这里选择最经典的eye矩阵
        axis = np.random.multivariate_normal(mean=sample_real, cov=conv, size=pop_size)
        axis[0] = 0  # 第一条数据保存为best_sample，不发生偏移
        axis_normalised = preprocessing.MaxAbsScaler().fit_transform(axis)  # [-1, 1], 第一行全为0，保证参考解不发生变化
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

    '''# 让函数直接返回发生冲突的ct集，然后对着ct集编码
    # 由于Gamma数据结构复杂持久化，因此根据读入数据重新计算Gamma'''
    def geneticSearch(self, pop_size:int=100, gen_n:int=50) ->np.ndarray:
        
        TERMINATION1 = get_termination('n_gen', gen_n)  # maxgen
        TERMINATION2 = get_termination('time', '00:50:00')  # time
        TERMINATION3 = get_termination('x_tol', tol=0.0025, n_last=20, n_max_gen=150, nth_gen=5)  # x_tol
        TERMINATION4 = get_termination('f_tol', tol=0.0025, n_last=20, n_max_gen=150, nth_gen=5)  # f_tol
        # TERMINATION = [TERMINATION3, TERMINATION4]
        idx_ter = 3
        ref_num=pop_size
        # ref_dirs = get_reference_directions("das-dennis", self.problem.n_obj, n_partitions=12)
        ref_dirs = get_reference_directions("energy", self.problem.n_obj, ref_num, seed=1)
        ct_star_total = None
        
        
        # NOTE: minimise task 首先保留原结果的pf
        self.fit_pf = self.opt_fit
        self.ct_pf = self.opt_ct
        for i in range(self.fit.shape[0]):  # for遍历pop，复杂度高
        # for i in range(self.opt_fit.shape[0]):  # for用于遍历opt, 复杂度低
            # NOTE: 
            # ct:尺寸为T的float vector，表示选择的方案
            # _:表示攻击集的Dict，key-target, value-发生冲突的目标/攻击者
            # conflict_ct: integer:[T, M]的float Dict，integer表示发生冲突的target，并不是所有的target都存在多个冲突方法
            # ct_idx:尺寸为T的integer vector，表示选择的方案的下标，但是下标是从1开始的？
            ct, _, conflict_ct, ct_idx =self.x2CtGamma(strategy=self.x[i], model=self.problem)
            # NOTE: 这里求idx=i时的self.opt_x的ct，讲道理应该要存到self.opt_ct，没存，以后万一要用可能照成一定麻烦
            # ct, _, conflict_ct, ct_idx = self.x2CtGamma(strategy=self.opt_x[i], model=self.problem)
            if ct is None:  # not feasible
                continue
            # 提供给TruingGP的信息是长度为T完全版,但是TruingGP.__init__只取出存在冲突的内容,长度为T'
            problem = TruingGP(conflict=conflict_ct, ct=ct, problem=self.problem, ct_star_total=ct_star_total)
            # ub = np.array(problem.part_max)  # 表示方案的长度
            # lb = np.zeros(ub.shape)  # 表示下标的下界，即0
            # ct_idx_binary = self.real2binary(ct_idx_real, problem)
            # 虽然ct_idx_real名字包含real，但是实际上指只冲突的target最终选择的方案的idx（ct_idx和ct_idx_real最大的区别是尺寸(T和T'<T)）
            ct_idx_real:np.ndarray = np.array([ct_idx[target] for target in problem.conflict_target])  # conflict_target表示存在冲突的target
            if ct_idx_real.size == 0:
                continue
            # sampling_real = self.initialization(ub=ub, lb=lb, pop_size=pop_size, sample_real=ct_idx_real)
            # sampling_real拿到的是备选方案的下表，是integer型，若采用binary code则还需要如下加工
            #   若采用inteager code则可直接使用
            # sampling_binary = np.ndarray(shape=[pop_size, problem.n_var])
            # for j in range(pop_size):
            #     sampling_binary[j] = self.real2binary(sampling_real[j], problem)
            algorithm = GeneticTruing(pop_size=pop_size,
                              ref_dirs=ref_dirs, display=False, save_history=False, verbose=False,
                              # sampling=get_sampling('bin_random'),
                              sampling = get_sampling('int_random'),
                            #   sampling=sampling_binary,
                            # sampling=sampling_real,
                            #   crossover=get_crossover('bin_hux'),
                              crossover=get_crossover('int_sbx'),
                            #   mutation=get_mutation('bin_bitflip'),
                              mutation=get_mutation('int_pm'),
                              eliminate_duplicates=True)
            # NOTE: res: minimize task
            res = minimize(problem=problem,
                           algorithm=algorithm,
                           seed=gen_n,
                           termination=('n_gen', gen_n)
                            # termination=TERMINATION[i]
                           )
            # DEBUG 查看演化一个个体需要的时间，总时间为该事件乘popsize og the first-level EA
            # print('gen_n={}, pop_size={}, 参数下，程序运行时间:{}秒'
            #       .format(gen_n, pop_size, res.exec_time))
            if res.F is None:
                print('res.F is None')
                continue
            # METHOD1
            # self.fit_pf = np.vstack([self.fit_pf, res.opt.get('F')])
            # METHOD2  # NOTE 实验表明res.opt.get('F')提供的PF数量远不如FIT_TOTAL  那么，为何res.opt.get('F')没把这些都记住？
            self.fit_pf = np.vstack([self.fit_pf, res.opt.get('F'), res.algorithm.evaluator.vair_global['FIT_TOTAL']])

            # self.ct_pf = np.vstack([self.ct_pf, res.opt.get('CT')])
            if i % 1 == 0:  # 得到fit_pf以后需要立刻处理，否则容易内存溢出
                idx = Performance().getPF_idx(self.fit_pf)
                self.fit_pf = self.fit_pf[idx]
                # self.ct_pf = self.ct_pf[idx]
        # minimise task
        pf_total_temp = self.fit_pf

        # NOTE  mincov 精修动作本来在这，但是挪到演化个体中了
        # NOTE: maximise task
        # self.fit_pf = -self.fit_pf
        # ct_star_total = None
        # pf_total = None
        # for idx, b in enumerate(self.fit_pf):
        #     if np.isnan(self.ct_pf[idx]).any():
        #         continue
        #     if pf_total is None:
        #         pf_total = b
        #     else:
        #         pf_total = np.vstack([pf_total, b])
        #     pf_total, ct_star_total = self.truing_by_mincov(self.ct_pf[idx], b, pf_total, ct_star_total)
        # self.fit_pf = -pf_total

        return pf_total_temp






