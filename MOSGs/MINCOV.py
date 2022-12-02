# from securitygame_core.MO_security_game import MOSG

import os
import numpy as np


class MINCOV():
    # input: game index i, initial coverage c, lower bound b;
    def __init__(self, MOSG, Gamma, m):
        self.cStar = None
        self.minResources = m
        self.Gamma = Gamma # Gamma is a list of set
        self.MOSG = MOSG

    def do(self, i, c, b):
        for tPrime in self.Gamma[i]:
            cPrime = c
            # cPrime_tPrime = (b[i]-) / ()
        pass