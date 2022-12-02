from pymoo.algorithms.nsga3 import NSGA3
from pymoo.model.population import Population
import numpy as np


# class MOSGG主要工作是重写了_next（原方法属于class GeneticAlgorithm）
class MOSGG(NSGA3):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 n_offsprings=None,
                 **kwargs):
        super(MOSGG, self).__init__(ref_dirs=ref_dirs, pop_size=pop_size, n_offsprings=n_offsprings, **kwargs)

    # self.off的结构为：
    # 简单的说，off是个class Individual list，每个Individual都包含如下属性：attr:{'F', 'CV', 'feasible', 'data', 'X', 'G'}
    # 以上属性都可以调用.set .get方法设置
    def _next(self):

        # 交叉变异相关全在.mating.do方法中
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

        # NOTE: 该部分在binary code版本用不到
        # 在计算fitness前，需要将编码合法化
        # self.off = self.off.set('X', CheckDNA().checkdna(self.off.get('X'),self.off.get('resource_ratio')) )
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


class CheckDNA():

    # 在GeneticAlgorithm中，利用get方法从objective list中将需要的narray抽取出来，命名为pop，可直接按照numpy操作
    def checkdna(self, pop, resource_ratio=0.2):
        resource = resource_ratio * pop.shape[1]
        pop = pop / np.sum(pop, axis=1)[:,np.newaxis] * resource
        # 下面的代码没想好怎么写，主要用于处理resource_ratio比较大的时候
        # pos = pop >= 1
        # while np.any(pos):
        #     redundant = np.sum(self.ct[pos] - 1)
        #     self.ct[pos] = 1
        #     temp = np.random.choice(100, size=self.target)
        #     temp[pos] = 0
        #     temp = temp / np.sum(temp) * redundant
        #     self.ct += temp
        #     pos = self.ct >= 1
        #     if np.all(pos):
        #         self.ct = np.full(self.ct.shape, 1)
        #         break
        return pop