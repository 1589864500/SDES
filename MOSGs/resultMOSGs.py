import numpy as np

from pymoo.model.result import Result
from pymoo.MOSGsGeneticSolver.performance import Performance


class resultMOSGs(Result):

    # 需要注意的是res提供的fit必须是minimize task
    def __init__(self, res_ct:np.ndarray, res_fit:np.ndarray, running_time:float):
        super().__init__()
        pf_idx = Performance().getPF_idx(pf_total=res_fit)
        self.X = res_ct[pf_idx]
        self.F = res_fit[pf_idx]
        self.exec_time = running_time

if __name__ is '__main__':
    test = resultMOSGs()