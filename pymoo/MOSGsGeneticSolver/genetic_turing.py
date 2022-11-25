from pymoo.algorithms.nsga3 import NSGA3


class GeneticTruing(NSGA3):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
