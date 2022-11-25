import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem


class SimulatedBinaryCrossover(Crossover):
    def __init__(self, eta, n_offsprings=2, prob_per_variable=0.5, **kwargs):
        super().__init__(2, n_offsprings, **kwargs)
        self.eta = float(eta)
        self.prob_per_variable = prob_per_variable

    def _do(self, problem, X, **kwargs):
        _, n_half, dim = X.shape
        low_bound, up_bound = problem.xl, problem.xu
        data1 = X[0, :, :].copy()
        data2 = X[1, :, :].copy()
        do_crossover2 = np.random.random((n_half, dim)) < self.prob_per_variable
        do_crossover2[np.abs(data1 - data2) < 1.0e-14] = False
        y1 = np.minimum(data1, data2)
        y2 = np.maximum(data1, data2)
        delta = y2 - y1
        # r 為SBX算子多項式分佈模擬特定概率分佈 用到的
        r1 = np.random.random((n_half, dim))

        def cal_betaq(beta):
            alpha = 2.0 - np.power(beta, -(self.eta + 1.0))

            mask, mask_not = (r1 <= (1.0 / alpha)), (r1 > (1.0 / alpha))

            betaq = np.zeros(mask.shape)
            betaq[mask] = np.power((r1 * alpha), (1.0 / (self.eta + 1.0)))[mask]
            betaq[mask_not] = np.power((1.0 / (2.0 - r1 * alpha)), (1.0 / (self.eta + 1.0)))[mask_not]

            return betaq

        # ??????????????????????(關於判斷距離上下限的方法)
        delta[delta < 1.0e-10] = 1.0e-10
        beta = 1 + (2.0 * (y1 - low_bound) / delta)
        betaq = cal_betaq(beta)
        c1 = 0.5 * ((y1 + y2) - betaq * delta)
        beta = 1 + (2.0 * (up_bound - y2) / delta)
        betaq = cal_betaq(beta)
        c2 = 0.5 * ((y1 + y2) + betaq * delta)

        # # # ????????????????交换
        r2 = np.random.random((n_half, dim)) < 0.5
        temp = c1[r2]
        c1[r2] = c2[r2]
        c2[r2] = temp

        up_bound = up_bound[np.newaxis, :]
        low_bound = low_bound[np.newaxis, :]
        up_bound = np.repeat(up_bound, c1.shape[0], axis=0)
        low_bound = np.repeat(low_bound, c1.shape[0], axis=0)
        idx1 = c1 > up_bound
        idx2 = c1 < low_bound
        idx3 = c2 > up_bound
        idx4 = c2 < low_bound
        c1[idx1] = up_bound[idx1].copy()
        c1[idx2] = low_bound[idx2].copy()
        c2[idx4] = low_bound[idx4].copy()
        c2[idx3] = up_bound[idx3].copy()
        data1[do_crossover2] = c1[do_crossover2]
        data2[do_crossover2] = c2[do_crossover2]
        return np.stack((data1,data2), axis=0)

class SBX(SimulatedBinaryCrossover):
    pass
