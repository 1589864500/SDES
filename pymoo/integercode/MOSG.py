'''Integer code'''

# 原本期望把方法全都放到MOSGs_genetic程序中，后面发现要把class SecurityGame类构建成问题集，因此把主要代码都搬到这里来了
# from securitygame_core.MO_security_game import MOSG
# TODO LIST FOR PYMOO AND MOSGs MODELING
# TODO: 在pymoo代码中，我将资源的合理化约束放在了GeneticAlgorithm(Algorithm)类中，应该移出来，具体做法是在外面包一层class mosg_genetic_nsga3
# DONE: 见mosg_genetic_nsga3重写_next()
# TODO: 在最最简单的ct实数编码以后，需要修改编码方式（和class Individual有关），继续沿用原来的交叉变异
# DONE: 检查发现class Population, class Individual只负责属性层面定义编码形式，不定义编码的物理含义，
#       编码的交叉变异主要见genetic_algorithm._next，
#       编码的初始化（物理含义）见genetic_algorithm._initialize.
#       注：._initialize与._next是平行级别的，种群第一次处理使用_initialize，不是第一次则用_next
# TODO: 有点担心原方法被改出问题了，结束以后吧nsga3在经典问题上面跑跑看一下结果
# DONE: 目前看到的结果比较差说明直接用就是很差
# TODO: 需要研究3个内容：1、算法迭代结束result的各个属性；2、算法每轮迭代_each_iteration()要处理的事；3、class Population相关的内容.
# FIXME:1、？；2、？；3、DONE 见class Population
# TODO: 改为binary编码后，n_var发生改变:
    # 具体表现为原来编码是长度为T的ct，而现在编码是长度更短的binary，以及解释binary编码和计算fit的方法。
    # binary code与real code详细的物理联系：real code即strategy set idx，例如realcode=3说明\Gamma={idx1,idx2,idx3}
    # binary编码后还有更多需要注意的:
        # 存在self中的变量：gametable的U_i^u,a的排序，执行N次  # DONE
        # __init__()要确定n_var，要用更快的方法算，即利用ratio  # DONE
            # 确定N parts的编码长度和n_var后，我们还要保留每个part的子长度，方便未来做分割  # DONE
        # 虽然策略空间是离散的，还是穷举开销仍然很大。方法是遇到一个记一个，记为strategy_dic={idx:(ct,payoff,feasible)}  # DONE
            # strategy_dic的key不能简单的用idx，因为idx实际上是N个目标的idx的拼接，可以考虑用字符串：'idx1_idx2_..._idxn'  # DONE
            # FIXME:目前不可行解的ct没算，其次payoff没算  # DONE
    # 以上都是准备阶段要做的，还有一些运行时做要的，在交叉变异得到off以后，要做解释
    # 解释阶段要做的是：
        # FIXME：交叉变异可能需要重新设计，目前的也够用
        # 评估最开始的一步是decode，需要将binary code->real_code->str(strategy)  # DONE
        # 评估时，先查Dict，再计算。有查必有存，遇到新解存起来  # DONE
        # 原方法ct和payoff计算位置不同，为了将两个对象一同处理，代码部分一起计算(见CalFIt)  # DONE
# TODO: 编码合理性机制需要完善
# FIXME:这个可以往后放，因为binary encode用不到，其次其正确性还有待验证
# TODO: 如何更快的计算资源消耗？？？
# DONE：见LenFeasible（）
# TODO：payoff等变量带pop_size，这是不合理的（整个class MOSG都不应该考虑pop_size，而在class SecurityGame1中考虑pop_size）
# DONE：目前class MOSG中只有cal_payoff是带pop_size计算的
# TODO：CalFit若计算了fitness，甚至不需要保存ct_pop
# FIXME
# DEBUG：发现一个惊天大bug，因为payoff_a,_d两部分是合在一起存储的，导致每次使用要分片，分片的下标范围有待再次验证
# DONE
# DEBUG: 出现了不合范围（绝对值不能大于10）的结果，参数为gen_n=10 obj_n=5 pop_size=50 target_n=10
# DONE: 讲MAXVAL从-10改成10后就好了
# TODO: 计算时间时可以传回缓存表，用于下一轮搜优加速
# FIXME: 但是需要注意的是当问题某些维度发生变化时缓存表失效，如target,obj，而resource_ratio则没有关系


# TODO LIST FOR CONFLICT SOLUTION
# TODO: 统计不同尺度下问题出现矛盾的分布情况（大小2 3 4 ...的冲突的频率）
# DONE: 粗略的看了一下问题规模从obj_n:[3,12], pop_size:50, target_n:[10,1000], gen_n:10对应的冲突分布：
    # 最大规模问题需要面临conflict_size=5,6的情况，虽然数量不多（分别为20,5个以内），但是大小为2,3的冲突非常常见，300左右
    # 较小规模问题一般冲突为2,3
# TODO: 研究决策树的写法
# TODO: 研究树相关的表示方法
# TODO: 回看问题，尝试剖析本质

# TODO LIST FOR ALGORITHM IMPLEMENTATION
# TODO: 所有算法复现
# TODO: 搞清楚几种算法之间的本质区别
# TODO: 搞清楚算法比较，帕累托前质量评分相关的内容
# TODO: 线性规划建模的算法复现
# TODO: 比较数所算法的优劣情况（运行时间与解的质量）


from ast import Lambda
from xmlrpc.client import boolean
import autograd.numpy as anp
import numpy as np
import os
from typing import *
import math
import functools
import random


from MOSGs.ORIGAMIM import ORIGAMIM
from pymoo.model.problem import Problem
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.util.normalization import normalize
import securitygame_core.MOSGs_genetic

from tool.algorithm import *

MAXVAL = 10  # MOSGs认为payoff越
# 默认数据集参数为 n_var个变量，3个obj，0个约束，变量最大值为1，最小值为0，变量类型为double
class MOSG(Problem):
    # NOTE: 问题编码的x是Gamma.size，真实在用作下标时应-1，否则有越界错误

    # player_num refers as the total number of attackers and defenders
    # target_num refers as the number of items witch need to be protected
    def __init__(self, player_num=6, target_num=25, resource_ratio=0.2,
                 sampling='uniform', ct_star_total=None,
                 conflict_solution:Union['binary', 'float', 'random'] = 'binary',
                 mincov:int=0):

        # 博弈相关
        self.player = player_num
        self.target = target_num
        self.resource = resource_ratio * target_num

        # MOSG相关
        self.conflict_solution = conflict_solution  # ??????????
        self.ct_star_total = ct_star_total

        # TODO: 变量待删减
        self.ct = None
        self.fit = None  # 用于最后存储，最小化问题村的时候需要*-1
        self.feasible = None
        self.x = None
        self.Gamma = None
        self.n_eval = 0
        self.mincov = mincov

        # sampling gametable
        self.sampling = sampling.lower()
        if self.sampling == "uniform":
            np.random.seed(12345)  # 42
            self.gametable = np.random.randint(low=1, high=11, size=(self.target, 2, 2 * (self.player - 1)))
            self.gametable = self.gametable.astype(np.float64)
        elif self.sampling == "col_normal":
            pass
        elif self.sampling == "row_normal":
            pass
        elif self.sampling == "dep_normal":
            pass
        else:
            print("The distribution uesd by now has not defined!!!!\nexit(code)")
            exit()
        # 前半部分players:defender 后半部分players:attacker
        # gametable value越大越好
        self.gametable[..., 1, 0:(self.player - 1)] *= -1
        self.gametable[..., 0, (self.player - 1):] *= -1
        # 0:cover 1:uncover
        # the attacker payoff for cover or uncover [target, attacker]
        self.U_ua = self.gametable[:, 1, self.player-1:]  # cal U^u,a_idx and U^u,d_idx
        self.U_ca = self.gametable[:, 0, self.player-1:]  # cal U^c,a_idx and U^c,d_idx
        self.payoff = np.zeros((self.gametable.shape[0], self.gametable.shape[-1]))

        # U_ua_argsort记录每列排序(列按target排序) [target, attacker]
        # 唯一需要注意的是由于::-1操作，U_ua值相同时idx大的排在前面
        self.U_ua_argsort = np.argsort(self.U_ua, axis=0)[::-1,:]  # ::-1 turn ascending order to descending order on axis=0
        
        # self.U_ca_argsort = np.argsort(self.U_ca, axis=1)[::-1,:]
        # ratio表示U^a_i(t)下降一个单位需要的资源数量，计算个策略需要消耗的资源时将用到
        
        # ratio[i,j]表示target_i在obj_j下 下降一个高度单位需要的资源（cover与uncover差距越大，需要的资源越少）
        self.ratio = 1 / (self.gametable[:,1,self.player-1:] - self.gametable[:,0,self.player-1:])  
        
        # self.strategy_dict: item = (ct, fit, feasible, Gamma)
        self.strategy_dict: Dict[str, Tuple[Tuple[np.ndarray, None], Tuple[np.ndarray, None], bool, Union[None, Dict[int,List[int]]]]] = {}
        self.part_len = self.CalDNAMaxLen()  # binary code length
        self.n_var = player_num-1
        super().__init__(n_var=self.n_var, n_obj=self.player-1, n_constr=0, xl=np.ones_like(self.part_len), xu=self.part_len)


    '''        
    # 融合简化的SGs特性，计算最大的编码长度
    # 编码长度分成N个parts，part_i对应obj_i最大的编码长度，不同obj的编码长度不同
    # Input：信息是资源总量resources、博弈表gametable（U^a,u_i(t)）
    # 算法思路(见PartDNAMaxLen)：假设resources只为obj_i服务，那么最多能保护的target_num即part_i的DNAMaxLen'''
    def CalDNAMaxLen(self):

        '''IMPLEMENTATION: 一种较快的求具体策略的资源消耗的方法
        求\Gamma_idx的maxlen
        逻辑分为三步：
            1.若大小小于等于1，feasible
            2.若排在前面的对象完全保护也无法达到后面对象完全不保护的收益水平，unfeasible
            3.若不是1 2，但是需要的资源大于拥有的资源，unfeasible
            4.若不是1 2 3，feasible
        '''
        def LenFeasible(self, len, idx)->boolean:
            # len表示攻击集的大小
            FEASIBLE = False
            feasible = True
            if len <= 1:
                return feasible
            U_i_ua_idx = self.U_ua_argsort[len - 1, idx]  # 第len个需要被看齐的target的下标
            U_i_ua_idx_left = self.U_ua_argsort[:len - 1, idx]  # 前len-1个需要看齐的target的下标
            
            # 首先查看前len-1个target中是否存在即使分配满资源(ct=1)也无法和第len个target看齐的情况：
            if np.max(self.U_ca[U_i_ua_idx_left, idx]) > self.U_ua[U_i_ua_idx, idx]:
                # 若存在，则认为feasible=False
                return FEASIBLE

            
            # ratio表示\Gamma中的对象下降一个单位高度需要的资源
            ratio = self.ratio[U_i_ua_idx_left, idx]

            # \Gamma中除最后一个的targets的U_i_ua到最后一个target的U_i_ua的gap
            gap2U_ua_idx = self.U_ua[U_i_ua_idx_left, idx] - self.U_ua[U_i_ua_idx, idx]  

            # resource_needed = len_np * ratio_np <= resource
            resource_needed = np.sum(ratio * gap2U_ua_idx)
            if resource_needed > self.resource:
                # 若存在（资源超出），feasible=False
                return FEASIBLE
            else:
                return feasible

        # 计算每个part需要的资源，idx为part的下标
        def PartDNAMaxLen(self, idx):
            # 算出每个part实数版的最大长度
            maxlen_lb = maxlen_feasible = 1  #???????????????????
            maxlen_ub = self.target
            while maxlen_lb <= maxlen_ub:
                maxlen_mid = (maxlen_lb + maxlen_ub) // 2  # 待查询的攻击集的大小
                # 判断当前长度是否可行，对应收缩上界或下界
                if LenFeasible(self, maxlen_mid, idx):
                    maxlen_feasible = maxlen_mid
                    maxlen_lb = maxlen_mid + 1
                else:
                    maxlen_ub = maxlen_mid - 1
            # return real code directly
            return maxlen_feasible
            # turn real encode length to binary encode length
            # return math.ceil(math.log(maxlen_feasible, 2))

        part_len = np.zeros((self.player - 1,))
        for i in range(self.player - 1):
            # every part's length, i表示第i个目标
            part_len[i] = PartDNAMaxLen(self, i)
        return part_len

    # 闭式解 closed-form solution, 只要满足U_ica<U_iad<U_iua,ct解必定可行（0<ct<1）
    # U_iad refer to U_ia or U_id  player:{0,1} 0:defender 1:attacker
    def CFSolution(self, U_iad:Union[float, np.ndarray], U_iua:Union[float, np.ndarray],
                   U_ica:Union[float, np.ndarray]) ->Union[float, np.ndarray]:
        ct = (U_iad - U_iua) / (U_ica - U_iua)
        return ct

    '''ConflictSolution接受：target idx, conflicting obj list, the size of Gamma_i, method types, ideal(unreachable) attacked target
    ConflictSolution返回：最终确定的方案ct_i, 最终确定的方案的下标i, 经过大小排序的ct(为增强ct的规律性)'''
    def ConflictSolution(self, target:int, obj:List[int], real_code:np.ndarray, type:str='heuristic', at_ideal=None) ->Tuple[float, int, np.ndarray]:
        # Input:
            # target 表示当前讨论的对象
            # obj 存储造成冲突的对象
            # real_code 用于计算obj_i对应的Gamma_i的size
            # 利用CFSolution计算每个obj提供的可选方案ct
        # Output:
            # 所有可选方案ct中最优的方案

        '''将所有方案根据ct的相对大小重排序，最终方案为count违反次数最小的
        具体返回：selcted ct, selected obj idx after ascending sort, all ct after ascending sort'''
        def MaxStrMatching(ct:np.ndarray, attack_target:np.ndarray, target:int) ->Tuple[float, int, np.ndarray]:
            # ct 发生冲突的方案, 长度M
            # attack_target 攻击集真实攻击的对象, 长度M
            # target 发生冲突的对象

            # NOTE: 第一步修改了ct的原始排序，因此后续的ct和传入的ct不是同一个
            idx:np.ndarray = np.argsort(ct)  # 从小到大排序
            ct:np.ndarray = ct[idx]  # 从小到大排序
            attack_target = attack_target[idx]  # 按方案相对大小从小到大排序
            count = np.zeros(attack_target.shape)  # count记录违反次数，越小越好
            count[0] = np.sum(attack_target[1:] != target)  # 默认的最优解是选择第一个方案的可行解：统计idx>0部分攻击集目标是t的数量（为t是不合理的）
            for obj_idx in range(1, attack_target.size):
                # 以obj_idx为界，根据左右变化更新count
                # obj_target将从中间进入左边，而右边最左边的一位将进入中间
                # 若进入左边的不为target则视为违反
                count[obj_idx] = count[obj_idx - 1]
                if attack_target[obj_idx - 1] == target:
                    count[obj_idx] += 1
                    # 若进入中间的为target，相当于违反约束程度变小了
                if attack_target[obj_idx] != target:
                    count[obj_idx] -= 1
            min_idx = np.argmin(count)
            return ct[min_idx], min_idx, ct

        # 本来是一个精修的方案，现在考虑移到外面去计算
        # 将所有方案根据总收益重排序，最终方案为收益最大的
        # def MaxPayoffSum(ct:np.ndarray):
        #     payf_sum = np.empty(shape=ct.shape)
        #     for idx, cti in enumerate(ct):

        # 统计攻击集Gamma信息，并计算冲突方案的优劣

        # 得到每个目标提供的方案ct （冲突方案）

        # TODO: DEBUG
        # 分别需要下降的M个目标，conflict set，长度M
        U_iad_idx:np.ndarray = self.U_ua_argsort[real_code-1, obj]  # U_iad_idx标记攻击集中最最后一个target，同时也是调整的目标
        U_iad:np.ndarray = self.U_ua[U_iad_idx, obj]

        # 提供需要下降的t的U_iua, U_ica
        # 注：分别表示不同obj对应的同一个target的U，记作U_iua(t), U_ica(t)
        U_iua:np.ndarray = self.U_ua[target, obj]
        U_ica:np.ndarray = self.U_ca[target, obj]

        # ct表示conflict set提供的所有方案，长度M
        ct:np.ndarray = self.CFSolution(U_iad, U_iua, U_ica)  # 长度为M 发生冲突的obj数量

        if type == 'heuristic':
            # 统计Gamma信息: 发生冲突的攻击者真实攻击对象, 长度M
            # NOTE 错误的计算方式
            # payoff_d = self.MaskUnselectedPart()
            # attack_target:np.ndarray = np.argmax(payoff_d, axis=-2)[obj]  # argmax搜索最大值出现的下标表示最终选定的target
            # 计算冲突方案优劣
            return MaxStrMatching(ct, at_ideal, target)
        elif type == 'random' or type == 'float':
            idx:np.ndarray = np.argsort(ct)  # 从小到大排序
            ct:np.ndarray = ct[idx]  # 从小到大排序
            r = np.random.randint(low=0, high=len(ct))
            return ct[r], idx[r], ct

    # def ConflictSolution2(self, target:int, obj:List[int], real_code:np.ndarray) -> float:
    #     U_iad_idx:np.ndarray = self.U_ua_argsort[real_code, obj]  # U_iad_idx标记攻击集中最最后一个target，同时也是调整的目标
    #     U_iad:np.ndarray = self.U_ua[U_iad_idx, obj]
    #     U_iua:np.ndarray = self.U_ua[target, obj]
    #     U_ica:np.ndarray = self.U_ca[target, obj]
    #     ct:np.ndarray = self.CFSolution(U_iad, U_iua, U_ica)
    #     rand_idx = np.random.randint(low=0, high=ct.size, size=1)
    #     return ct[rand_idx]
    #
    # def ConflictSolution3(self, target:int, obj:List[int], real_code:np.ndarray) -> float:
    #     return 0

    # NOTE: truing_by_mincov是在SG框架中的程序，所以处理的是maximise task
    # 该函数并非不需要ct，而是提供idx下标，访问ct；相反不需要fit
    def truing_by_mincov(self, ct_i:np.ndarray, b:np.ndarray, pf_total=None, ct_star_total:Tuple[np.ndarray, None]=None)\
            ->Tuple[np.ndarray, np.ndarray]:

        K = 3
        count = 0  # 记录精修的次数
        # 每次搜索到新方案ct后，结合game table算fitness
        problem_mosg = securitygame_core.MOSGs_genetic.MOSG(player_num=self.player, target_num=self.target)
        model = ORIGAMIM(MOSG=problem_mosg)

        for gameidx in range(self.player-1):  # 第二层for遍历obj
            # 利用ct尝试精修，用到MINCOV
            model.c = ct_i
            model.updateC()
            model.updateU(i=gameidx)
            # 需要注意的是，由于gmosg编码的破坏性，next和idx对应不一定是相同的，但是影响并不是很大只要next>1
            idx = np.argsort(-model.U_ia)
            # NOTE: 由于不好确定next的具体取值，将next设为T一定没错，只是时间会增长
            # next = model.getNextLen(gap)
            next = min(model.getNextLen(epsilon=0.1)*K, self.target)
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
            for obj_idx in range(self.player-1):  # 第二层for遍历obj
                model.leftAllocation(b=b, obj_idx=obj_idx)
                ct_final = model.c
                self.cal_payoff(ct_final)
                fit_final = self.cal_payoff_defender()
                if pf_total is None:
                    pf_total = fit_final
                else:
                    pf_total = np.vstack([pf_total, fit_final])
                # ct_total = np.vstack([ct_total, ct_final])
                count += 1
        # print('找到更优解{}处'.format(count))
        return pf_total, ct_star_total

    '''
    方法接受Population，处理时拆分成N个Individual，然后组合Population返回
        1、拆分ind，每个ind对应\Gamma_ideal
    '''
    def calFitInt(self, pop:np.ndarray, method:int=2):
        # TODO: 很多变量以及用不到了，需要删掉，比如ct_pop和Gamma_pop
        ct_pop:np.ndarray = np.empty(shape=(pop.shape[0], self.target))
        fit_pop:np.ndarray = np.empty((pop.shape[0], self.player-1))
        Gamma_pop:List[Dict[int,List[int]]] = []  # ?????????
        feasible_pop = np.empty((pop.shape[0],))
        for idx, strategy in enumerate(pop):
            CT = np.zeros(shape=(self.target,))
            FIT = np.full(shape=(self.player - 1,), fill_value=-MAXVAL)
            FEASIBLE = False
            GAMMA = None

            # 查询Dict，若query存在则直接用，若不存在则算并存
            # 存储以下信息：ct, fit, feasible, Gamma
            query = functools.reduce(lambda x,y: str(x)+str(y), strategy)
            # 存在
            if query in self.strategy_dict.keys():
                ct, fit, feasible, Gamma = self.strategy_dict[query]
                if not feasible:
                    # 对不可行解，ct_pop记录，fit_pop记录
                    fit_pop[idx] = FIT
                    ct_pop[idx] = CT
                    Gamma_pop.append(GAMMA)
                    feasible_pop[idx] = FEASIBLE
                else:
                    ct_pop[idx] = ct
                    fit_pop[idx] = fit
                    feasible_pop[idx] = feasible
                    Gamma_pop.append(Gamma)
            # 不存在
            else:
                # Strategy2Ct分为以下几类：
                #   1：一个Strategy对应一个Ct
                #   2：一个Strategy对应一组Ct（int型）

                # decode DNA into strategy
                ct, Gamma, _, _ = self.strategy2CtInt(strategy)
                if ct is not None:  # feasible
                    feasible = True
                    feasible_pop[idx] = feasible
                    ct_pop[idx] = ct
                    Gamma_pop.append(Gamma)
                    self.cal_payoff(ct)
                    # 具体计算编码时，有两种方法，在算法的最后几轮，可以使用该方法精修
                    # NOTE METHOD 1
                    if method == 1:
                        b = self.cal_payoff_defender()
                        fit, self.ct_star_total = self.truing_by_mincov(ct_i=ct, b=b, pf_total=b,
                                                                        ct_star_total=self.ct_star_total)
                        if fit.ndim == 1:
                            fit_pop[idx] = fit
                        else:
                            rank = list(find_non_dominated(F=fit))
                            pf_i = random.sample(rank, 1)
                            fit_pop[idx] = fit[pf_i]
                    # NOTE: METHOD2
                    elif method == 2:
                        fit_pop[idx] = self.cal_payoff_defender()
                    # 将ind记录到查询字典中
                    self.strategy_dict[query] = (ct, fit_pop[idx], feasible, Gamma)
                else:  # not feasible
                    # 对不可行解，ct_pop不记录，fit_pop记录
                    fit_pop[idx] = FIT
                    feasible_pop[idx] = FEASIBLE
                    # NOTE DEBUG
                    Gamma_pop.append(GAMMA)
                    ct_pop[idx] = CT
                    self.strategy_dict[query] = (CT, FIT, FEASIBLE, GAMMA)

        self.ct = ct_pop
        # self.x = pop
        # self.feasible = feasible_pop
        self.fit = fit_pop * -1  # multiply -1 to turn maximize fit problem to minimize problem
        self.Gamma = Gamma_pop


    '''
    Strategy2Ct将int型/字符串型的dna编码转化为实数型的ct资源分配策略 NOTE 输入real_code表示Lcode/integer code
    1、Strategy2Ct根据数字编号有以下几个不同版本：
        一个\Gamma_ideal只返回一个ct
    2、Strategy2Ct步骤：
        将real转为ct，涉及到冲突合并
    返回值包括：
    1、c: the coverage vector c all all target
    2、P: all conflict obj  happended on each target
    3、conflict_ct: all conflict ct provided by all conflict obj
    4、selected_ct_idx: the idx of final selected ct
    '''
    def strategy2CtInt(self, real_code:np.ndarray, type:str='heuristic') -> \
            Tuple[Union[np.ndarray, None], Dict[int,List[int]], Dict[int,List[float]], Dict[int, int]]:
        # Input: real_code N个\Gamma大小
        # Output: 实数型编码ct、冲突集信息P。注：P要经过一定的转化才能变成Gamma

        # DONE：后续十分的复杂，大概是有N个子集，求每个子集的并集S1，然后分为以下三种情况：
            # 元素在S1外，ct=0
            # 元素在S1内但是只被某个子集独有，ct=closed-form solution
            # 元素在S2内但是被多个子集共有，ct=conflict solution
        # S记录全部targets拥有次数，记录每个target在U1中出现的次数（不出现：0，独有：1，共有：>1）
        # P记录全部target(key)对应的拥有该target的\Gamma_i(values)

        ct = np.zeros((self.target,))
        at_ideal = np.zeros((self.player-1,))
        S = np.zeros((self.target,)) # 每个target上发生冲突的数量
        P: Dict[int,List[int]] = {}  # 冲突集详情 {target_i : [obj_i,...]}, 下标从0开始
        conflict_ct:Dict[int, List[float]] = {}
        selected_ct_idx:List[int, int] = {}
        
        # NOTE 输入只提供Lcode，所有信息都从Lcode中得到，如Gamma_i
        
        # 填充S, P 计算at^ideal
            # 具体地，for做两件事：对obj_i，1、找出\Gamma_i每个元素target，S[target] += 1；2、P[target].append(obj_i)
        for obj_idx in range(self.player-1):
            Gamma_i = self.U_ua_argsort[:real_code[obj_idx], obj_idx]  # \Gamma_i中所有target的idx  
            # update S: 所有\Gamma中的target的idx计数+1
            S[Gamma_i] += 1
            # update P: P一开始为{}，
            for gamma_i_idx in Gamma_i:  # gamma_i_idx is the index of targets in \Gamma_obj_idx
                # 若dict中没有gamma_i_idx，则新建List，若有则扩大List
                P[gamma_i_idx] = P[gamma_i_idx] + [obj_idx] if gamma_i_idx in P else [obj_idx]
            # calcalate at^ideal
            # 算达成Gamma_i条件的c
            c_temp = np.zeros((self.target,))  # c_ideal
            # idx 与 Gamma_i相同
            U_iad_idx:int = Gamma_i[-1]
            U_iad:float = self.U_ua[U_iad_idx, obj_idx]
            # 然后提供每个需要下降的t的U_iua, U_ica
            U_iua:List[float] = self.U_ua[Gamma_i[:-1], obj_idx]
            U_ica:List[float] = self.U_ca[Gamma_i[:-1], obj_idx]
            c_temp[Gamma_i[:-1]] = self.CFSolution(U_iad,U_iua,U_ica)
            # 更新payoff
            self.cal_payoff(c_temp)
            # 取payoff^a最大的idx作为attacked target(at)（idx要在conflict中）
            at_ideal[obj_idx] = self.cal_at_ideal(obj_idx, Gamma_i)  # 只取Gamma_i中的元素


        # 统计冲突分布情况
        conflict_t = {}  # 记录count>1的target作为key，ct alternatives作为value
        # 求得并集后，\Gamma=Union(\Gamma_i). 根据\Gamma算ct
        for target_idx, count in enumerate(S):
            # self.cal_payoff(ct=ct)
            if count == 1:  # 先把能确定的确定下来
                # 闭式解
                # obj_idx表示关注当前target的某个攻击者，不同obj_idx对应的闭式解不同
                obj_idx:int = P[target_idx][0]
                # 体现在\Gamma_idx不同，对应U_iad_idx不同
                # 向U_iad_idx看齐，代表Gamma最后一位，idx=real_code[obj_idx]-1

                # 首先确定要看齐（下降）的目标
                U_iad_idx:int = self.U_ua_argsort[real_code[obj_idx]-1, obj_idx]  # U_iad_idx标记攻击集中最最后一个target，下标刚好是real_code
                U_iad:float = self.U_ua[U_iad_idx, obj_idx]

                # 然后提供每个需要下降的t的U_iua, U_ica
                U_iua = self.U_ua[target_idx, obj_idx]
                U_ica = self.U_ca[target_idx, obj_idx]

                ct[target_idx] = self.CFSolution(U_iad,U_iua,U_ica)
            elif count > 1:  # 那些确定不了的用随机来确定
                #   首先只保留参与冲突的目标的信息  conflict set, and the size of Gamma_i in conflict set
                conflict_obj:List[int] = P[target_idx]
                conflict_real_code:np.ndarray = real_code[conflict_obj]  # real_code_i means the size of Gamma of obj_i

                # ORIGAMI-G
                # 提供target where the conflict happened,  conflict set, the size of Gamma_i in conflict set
                # 返回the selected ct, selected ct idx, ct ordered by ** / all conflict ct
                ct[target_idx], selected_ct_idx[target_idx], conflict_ct[target_idx]\
                    = self.ConflictSolution(target_idx, conflict_obj, conflict_real_code, type='heuristic', at_ideal=at_ideal)
                # ORIGAMI-G的退化版本
                # ct[target_idx], selected_ct_idx[target_idx], conflict_ct[target_idx]\
                #     = self.ConflictSolution(target_idx, conflict_obj, conflict_real_code, type='random')
        if np.sum(ct) > self.resource:
            return None, P, conflict_ct, selected_ct_idx

        # 除了ct，还存储演化算法完成后会用到的结果
        return ct, P, conflict_ct, selected_ct_idx


    def MaskUnselectedPart(self, epsilon=0.1):
        # epsilon 对最优解的容忍度
        payoff_d = self.payoff[..., :self.player - 1]
        payoff_a = self.payoff[..., self.player - 1:]
        col_max_a = np.max(payoff_a, axis=-2)
        unselected_mask = np.where(col_max_a[..., np.newaxis, :] - payoff_a > epsilon)
        payoff_d[unselected_mask] = -MAXVAL  # 调整后payoff_d selected part的value必定大于unselected part
        return payoff_d

    def cal_payoff(self, ct: np.ndarray) -> None:  # payoff: [T, 2*(player-1)]
        # calculate payoff of attacker and defender for all targets
        # 原版
        self.payoff = self.gametable[:,0,:] * ct[..., np.newaxis] + self.gametable[:,1,:] * (1 - ct[..., np.newaxis])
        # 考虑种群大小的版本
        # self.payoff = self.gametable[np.newaxis, :, 0, :] * ct[..., np.newaxis] + self.gametable[np.newaxis, :, 1,
        #         :] * (1 - ct[..., np.newaxis])

    # NOTE: 暂时用不到因此未完成
    def cal_payoff_attacker(self) ->None:
        # attacker will select the target with the greatest payoff to attack
        # 原版
        # self.payoff为二维表[T,(self.player-1)*2]，分别对应每个target被攻击时防御者、攻击者的综合收益。思路为攻击者取payofff后半部分每列最大值
        self.payoff_attacker = np.amax(self.payoff, axis=0)[-(self.player-1):]
        # 考虑种群大小的版本
        # payoff_attacker = np.amax(self.payoff, axis=1)[:, -(self.player - 1):]
        # return payoff_attacker

    def cal_payoff_defender(self) ->np.ndarray:
        # when we figure out which target would be attack by attacker, then we goto quire its utility for defender
        # return U_i^d(c)

        # 思路为找出每列所有和最大值相近的值的位置，然后在从每列选出的值选出对防御者最大的值
        # 该部分代码是维度无关的，带pop_size和不带pop_size版都能用，只需要调整axis
        payoff_d = self.MaskUnselectedPart()
        col_max_d = np.max(payoff_d, axis=-2)
        return col_max_d

    def get_gametable(self):
        return self.gametable
    def get_payoff(self):
        return self.payoff
    
    def cal_at_ideal(self, obj:int, Gamma:List[int]):
        payoff_Gamma = self.payoff[Gamma, obj]  # the defender payoff
        return Gamma[np.argmax(payoff_Gamma)]



# 将SGs形式化建模以后，剩下就是规模变化的工作：class SGs1, SGs2, ...
class SGs1(MOSG):


    # def __init__(self, player_num=5, target_num=25, resource_ratio=0.2, sampling='uniform'):
    #     super(SGs1, self).__init__(player_num=player_num, target_num=target_num, resource_ratio=resource_ratio,
    #                                sampling=sampling)
    def __init__(self, **kwargs):
        super(SGs1, self).__init__(**kwargs)



    # 相比pymoo problem标准结构，这里少了obj_func, _calc_pareto_front函数
    # 其中obj_func和_evaluate合并了，而_calc_pareto_front则在这里不适用
    # out:{'F':None, 'CV':None, 'G':None}，'F'即fitness，暂不清楚'CV', 'G'具体什么含义
    # x.size == [pop_size, dna]

    # IMPLEMENTATION: ct實數編碼
    # def _evaluate(self, x, out, *args, **kwargs):
    #     # 在这里x即防御者策略ct，并計算fit
    #     self.cal_payoff(x)
    #     out["F"] = -self.cal_payoff_defender()
    #     return out

    '''
    API接口
    IMPLEMENTATION: binary encode
    '''
    def _evaluate(self, x, out, *args, **kwargs):

        # 每轮检验前都确保self.resGamma, self.resCt都是从[]开始的

        # self.res_Gamma, self.res_Ct, fit_pop = self.CalFit(x)
        self.n_eval += 1
        if self.mincov == 0:
            flag = 2
        elif self.n_eval % self.mincov == 0:
            flag = 1
        else:
            flag = 2
        self.calFitInt(x, method=flag)
        # 本程序演化算法是朝min演化
        out["F"] = self.fit  # minimize fit
        out["CT"] = self.ct
        # out['Gamma'] = self.Gamma  # pymoo框架存不了复杂的数据结构
        return out


def evaluation(model, x, out):
    model._evaluate(x, out)
    print(out)

if __name__ == '__main__':
    model = SGs1()
    # x = np.random.random((2, model.target))
    x = np.random.random((2, model.n_var))
    x = (x < 0.5).astype(np.bool)
    out = {'F':None, 'CV':None, 'G':None}
    evaluation(model, x, out)
