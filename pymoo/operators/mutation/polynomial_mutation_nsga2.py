import numpy as np

from pymoo.model.mutation import Mutation
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem


class PolynomialMutation(Mutation):
    def __init__(self, eta, prob=None):
        super().__init__()
        self.eta = float(eta)

        if prob is not None:
            self.prob = float(prob)
        else:
            self.prob = None

    def _do(self, problem, X, **kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var
        low_bound, up_bound = problem.xl, problem.xu
        N, dim = X.shape
        result = np.empty_like(X)
        c_pop = np.empty_like(X)
        r = np.random.random((N, dim))
        do_mutate = (r <= self.prob)
        not_mutate = (r > self.prob)
        c_pop[not_mutate] = X[not_mutate]
        X = X[do_mutate]

        low_bound = np.repeat(low_bound[np.newaxis, :], N, axis=0)[do_mutate]
        up_bound = np.repeat(up_bound[np.newaxis, :], N, axis=0)[do_mutate]

        r = np.random.random(X.shape)
        deltaq = np.empty_like(X)
        r1 = r < 0.5
        r2 = r >= 0.5
        delta1 = (X - low_bound) / (up_bound - low_bound)
        delta2 = (up_bound - X) / (up_bound - low_bound)
        mutate_pow = 1.0 / (self.eta + 1.0)

        xy = 1.0 - delta1
        val = 2.0 * r + (1.0 - 2.0 * r) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mutate_pow) - 1.0
        deltaq[r1] = d[r1]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mutate_pow))
        deltaq[r2] = d[r2]

        _c_pop = X + deltaq * (up_bound - low_bound)
        _c_pop[_c_pop < low_bound] = low_bound[_c_pop < low_bound]
        _c_pop[_c_pop > up_bound] = up_bound[_c_pop > up_bound]

        c_pop[do_mutate] = _c_pop

        return c_pop

class PM(PolynomialMutation):
    pass