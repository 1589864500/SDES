from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.model.initialization import Initialization
from pymoo.model.mating import Mating
from pymoo.model.population import Population
from pymoo.model.repair import NoRepair

import numpy as np


# class GeneticAlgorithm继承class Algorithm，功能上具体实现了_next()和_initialize()
class GeneticAlgorithm(Algorithm):

    def __init__(self,
                 pop_size=None,
                 sampling=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 survival=None,
                 n_offsprings=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=None,
                 mating=None,
                 min_infeas_pop_size=0,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        # the population size used
        self.pop_size = pop_size

        # minimum number of individuals surviving despite being infeasible - by default disabled
        self.min_infeas_pop_size = min_infeas_pop_size

        # the survival for the genetic algorithm
        self.survival = survival

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # simply set the no repair object if it is None
        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        if mating is None:
            mating = Mating(selection,
                            crossover,
                            mutation,
                            repair=self.repair,
                            eliminate_duplicates=self.eliminate_duplicates,
                            n_max_iterations=100)
        self.mating = mating

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize(self):

        # create the initial population
        # _initialize函数最有意义的部分即self.initialization.do()，
        # 使用不同的initialization实例即可得到物理含义不同的编码(编码的物理含义还体现在class Problem ._evaluate()算fit上）
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        pop.set("n_gen", self.n_gen)

        # ORIGAMI-DG的特殊之处，需要预先定义pop的"FIT_TOTAL"属性
        pop.set("FIT_TOTAL", None)
        # then evaluate using the objective function
        self.evaluator.eval(self.problem, pop, algorithm=self)

        # NOTE 将全局信息存入ｓｕｒｖｉｖａｌ也是非常不错的选择
        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self,
                                   n_min_infeas_survive=self.min_infeas_pop_size)

        self.pop, self.off = pop, pop

    # genetic_algorithm程序记录了交叉变异的过程
    # 问题一：如何将约束内容放进去？
    def _next(self):

        # 交叉变异相关全在.mating.do方法中  实际上调用的是class InfillCriterion.do()，里面包含class Mating._do()
        # do the mating using the current population
        self.off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        self.off.set("n_gen", self.n_gen)

        # 一种应对找不到新解的机制（复制），一般任务用不到；
        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return

        # mating得到的off数量可能不够offspring数量，一般任务用不到；
        # if not the desired number of offspring could be created
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        # # 在计算fitness前，需要将编码合法化
        # # self.off = self.off.set('X', CheckDNA().checkdna(self.off.get('X'),self.off.get('resource_ratio')) )
        # self.off = self.off.set('X', CheckDNA().checkdna(self.off.get('X')))

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        self.pop = Population.merge(self.pop, self.off)

        # the do survival selection
        # 從子代+父代中選出優勝者，稱爲survival selection
        if self.survival:
            self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self,
                                        n_min_infeas_survive=self.min_infeas_pop_size)

    def _finalize(self):
        pass

# class CheckDNA():
#
#     # 在GeneticAlgorithm中，利用get方法从objective list中将需要的narray抽取出来，命名为pop，可直接按照numpy操作
#     def checkdna(self, pop, resource_ratio=0.2):
#         resource = resource_ratio * pop.shape[1]
#         pop = pop / np.sum(pop, axis=1)[:,np.newaxis] * resource
#         # 下面的代码没想好怎么写，主要用于处理resource_ratio比较大的时候
#         # pos = pop >= 1
#         # while np.any(pos):
#         #     redundant = np.sum(self.ct[pos] - 1)
#         #     self.ct[pos] = 1
#         #     temp = np.random.choice(100, size=self.target)
#         #     temp[pos] = 0
#         #     temp = temp / np.sum(temp) * redundant
#         #     self.ct += temp
#         #     pos = self.ct >= 1
#         #     if np.all(pos):
#         #         self.ct = np.full(self.ct.shape, 1)
#         #         break
#         return pop