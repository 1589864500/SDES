# 目的：为展示每种采样对应的效果
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.operators.sampling.random_permutation_sampling import PermutationRandomSampling
from pymoo.operators.sampling.random_sampling import BinaryRandomSampling
from pymoo.problems.securitygame import SecurityGame1

n_sampling = 1
problem = SecurityGame1()
sampling1 = PermutationRandomSampling()
x1 = sampling1._do(problem=problem, n_samples=n_sampling)

sampling2 = LatinHypercubeSampling()
x2 = sampling2._do(problem=problem, n_samples=n_sampling)

sampling3 = BinaryRandomSampling()
x3 = sampling3._do(problem=problem, n_samples=n_sampling)
print('PermutationRandomSampling: ', x1, '\nLatinHypercubeSampling: ', x2, '\nBinaryRandomSampling: ', x3)
