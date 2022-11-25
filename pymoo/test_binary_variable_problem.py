import numpy as np

# TODO: class Population涉及编码的物理含义层面，在genetic_algorithm.py文件中有提及
# DONE: 用通用的genetic algorithm框架之所以能在多种编码间切换，
#       比如这里物理含义由sampling=get_sampling("bin_random")定
#       二项式编码的合理化操作由crossover=get_crossover("bin_hux"),mutation=get_mutation("bin_bitflip")定

# from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.so_genetic_algorithm import GA
# pymoo.factory.get_mutation, .get_sampling一般算法的__init__部分会默认声明，这里手动给定了
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
# 效果同pymoo.factory.get_problem，声明问题
from pymoo.problems.single.knapsack import create_random_knapsack_problem

problem = create_random_knapsack_problem(30)

algorithm = GA(
    pop_size=200,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_hux"),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               verbose=False)

print("Best solution found: %s" % res.X.astype(int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)
