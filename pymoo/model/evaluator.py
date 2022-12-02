import numpy as np
from pymoo.MOSGsGeneticSolver.performance import Performance

from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.model.problem import Problem
from pymoo.util.misc import at_least_2d_array
from typing import *


def set_feasibility(pop):
    for ind in pop:
        cv = ind.get("CV")
        if cv is not None:
            ind.feasible = cv <= 0


def set_cv(pop, feasbility=True):
    for ind in pop:
        if ind.G is None:
            ind.CV = np.zeros(1)
        else:
            ind.CV = Problem.calc_constraint_violation(at_least_2d_array(ind.G))[0]

    if feasbility:
        set_feasibility(pop)


class Evaluator:
    """

    The evaluator class which is used during the algorithm execution to limit the number of evaluations.
    This can be based on convergence, maximum number of evaluations, or other criteria.

    """

    def __init__(self,
                 skip_already_evaluated=True,
                 evaluate_values_of=["F", "CV", "G"]):
        self.n_eval = 0
        self.evaluate_values_of = evaluate_values_of
        self.skip_already_evaluated = skip_already_evaluated
        self.vair_global:Dict[str, Union[List, np.ndarray]] = {}

    def eval(self,
             problem,
             pop,
             **kwargs):
        """

        This function is used to return the result of one valid evaluation.

        Parameters
        ----------
        problem : class
            The problem which is used to be evaluated
        pop : np.array or Population object
        kwargs : dict
            Additional arguments which might be necessary for the problem to evaluate.

        """

        # 下5行代码表示若接收非Population对象，则封装成Population
        is_individual = isinstance(pop, Individual)
        is_numpy_array = isinstance(pop, np.ndarray) and not isinstance(pop, Population)

        # make sure the object is a population
        if is_individual or is_numpy_array:
            pop = Population().create(pop)

        # 好处是不必计算上次保留的个体的fitness，加快计算
        # find indices to be evaluated
        if self.skip_already_evaluated:
            I = [k for k in range(len(pop)) if pop[k].F is None]
        else:
            I = np.arange(len(pop))

        # update the function evaluation counter
        self.n_eval += len(I)

        # actually evaluate all solutions using the function that can be overwritten
        if len(I) > 0:
            self._eval(problem, pop[I], **kwargs)

            # set the feasibility attribute if cv exists
            set_feasibility(pop[I])

        if is_individual:
            return pop[0]
        elif is_numpy_array:
            if len(pop) == 1:
                pop = pop[0]
            return tuple([pop.get(e) for e in self.evaluate_values_of])
        else:
            return pop

    def _eval(self, problem, pop, isDominatedSort=False, **kwargs):
        '''虽然这段内容是pymoo源码，但是为了方便，做出如下修改（去重与求pf）'''
        # NOTE: Problem._evaluate的出口
        out = problem.evaluate(pop.get("X"),
                               return_values_of=self.evaluate_values_of,
                               return_as_dictionary=True,
                               **kwargs)

        # NOTE 允许存放全局信息，尤其是长度不等于ｐｏｐｓｉｚｅ的全局信息（self.evaluator）
        # NOTE　魔改了这段代码，实现能够处理长度不为popsize的out信息
        for key, val in out.items():
            if val is None:
                continue
            elif key != 'FIT_TOTAL' and key != 'CT_TOTAL':
                pop.set(key, val)
            elif key == 'FIT_TOTAL':
                if key not in self.vair_global:
                    self.vair_global[key] = val
                    if 'CT_TOTAL' in out:
                        self.vair_global['CT_TOTAL'] = out['CT_TOTAL']
                elif isDominatedSort:
                    #! 非支配排序
                    val = np.vstack([self.vair_global[key], val])
                    idx = Performance().getPF_idx(val)
                    self.vair_global[key] = val[idx]
                    if 'CT_TOTAL' in out:
                        self.vair_global['CT_TOTAL'] = np.vstack([self.vair_global['CT_TOTAL'], out['CT_TOTAL']])[idx]
                else: 
                    #! 不排序
                    self.vair_global[key] = np.vstack([self.vair_global[key], val])
                    if 'CT_TOTAL' in out:
                        self.vair_global['CT_TOTAL'] = np.vstack([self.vair_global['CT_TOTAL'], out['CT_TOTAL']])
        # DEBUG
        # if (len(self.vair_global['CT_TOTAL']) != len(self.vair_global['FIT_TOTAL'])):
        #     print('ERROR! len(self.vair_global[CT_TOTAL]) != len(self.vair_global[FIT_TOTAL])')
        #! 去重
        if kwargs['algorithm'].n_gen % 10 == 0:
            if 'FIT_TOTAL' in out: self.vair_global['FIT_TOTAL'], idx = np.unique(self.vair_global['FIT_TOTAL'], return_index=True, axis=0)
            if 'CT_TOTAL' in out: self.vair_global['CT_TOTAL'] = self.vair_global['CT_TOTAL'][idx]
                # DEBUG Memory error
                # print(len(self.vair_global['CT_TOTAL']), end=' ')
        # DEBUG 测试evaluator功能的完备性
        # if 'test' not in self.vair_global:
        #     self.vair_global['test'] = [1]
        # else:
        #     self.vair_global['test'].append(2)
            
