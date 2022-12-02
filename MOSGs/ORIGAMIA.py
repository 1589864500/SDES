import copy

import numpy as np

import tool.algorithm
from MOSGs.ORIGAMIAM import ORIGAMIM

class ORIGAMIA():
    # b给的是完整长度
    def __init__(self, MOSG, alpha=0.001):
        # ORIGAMIM作为求解器，不应该声明在init中，反复调用一个求解器会导致参数变化出现失误。只需在需要的地方调用
        self.MOSG = MOSG
        self.alpha = alpha
        self.N = self.MOSG.player - 1
        self.iteration = 1
        self.c = None
        # self.U_id = None
        # self.U_ia = None

    # def updateU_id(self, i, ORIGAMIM, display=False):
    #     if display and self.U_ia is not None and self.U_id is not None:
    #         print('     原：\nU_id:{}\nU_ia:{}'.format(self.U_id, self.U_ia))
    #     # self.MOSG.set_ct(self.c)
    #     # self.MOSG.cal_payoff()
    #     ORIGAMIM.updateC(self.c)
    #     ORIGAMIM.updateU(i=i)
    #     # debug
    #     self.U_ia = self.MOSG.get_U_ik(i=i, k=1)
    #     self.U_id = self.MOSG.get_U_ik(i=i, k=0)
    #     if display:
    #         print('     更新后：\nU_id:{}\nU_ia:{}'.format(self.U_id, self.U_ia))

    '''输入：b向量是N维，但是第一维需要更换成算法要求的min(Uiud)'''
    def do(self, b):
        # print('     进入ORIGAMIA+++++++++++++++++++++++++++++++++++++')
        # TODO:伪代码下标从1开始，这里从0开始，需要留意``````````````
        # 取U_1ud作为上界，记为b1plus
        b[0] = np.min(self.MOSG.get_U_ijk(i=0, j=1, k=0))  # min(Uiud)
        b[np.where(b == -float("inf"))[0]] = -10.
        # N次外循环表示每次求一个目标的最优，并将其置为约束
        for i in range(self.N):
            gen = 0
            lower = b[i]
            upper = np.max(self.MOSG.get_U_ijk(i=i, j=0, k=0))  # max(Uicd)
            # print('     当前讨论下标{}，下界b{}'.format(i, b))
            # 内循环迭代寻解的边界，过程中，若找到可行解，下界收缩，U_id,U_ia更新
            while upper - lower > self.alpha:
                gen += 1
                self.iteration += 1
                b[i] = (upper + lower) / 2
                # TODO 声明ORIGAMIM
                cPrime, count = ORIGAMIM(self.MOSG).do(b)
                if cPrime is None:
                    # print('     ORIGAMIM无解，降低upper')
                    upper = b[i]
                # 如果cPrime是可行解，赋予self.c的同时，更新MOSG参数，以便计算Uid
                else:
                    self.MOSG.set_ct(cPrime)
                    self.MOSG.cal_payoff()
                    self.MOSG.cal_payoff_defender()
                    paypff_attacker = self.MOSG.get_payoff_defender()
                    # print('         b要求{},cPrime达成收益{}'.format(b, paypff_attacker))
                    if (paypff_attacker >= b).all():
                        self.c = cPrime
                        lower = b[i]
                    else:
                        upper = b[i]
                    # debug
                    # TODO: show U_id
                    # self.updateU_id(i)
                    # print('     ORIGAMIM有解，提高lower')
                    # print('     找到可行解sum(c):{}, b:{}, U_id{}'.format(np.sum(self.c), b, self.U_id))
            # print('     找到可行解sum(c):{}, b:{}, U_id{}'.format(np.sum(self.c), b, self.U_id))
            #
            # print('     upper-lower收敛消耗{}轮)'.format(gen))
            # if self.U_id is None or self.U_ia is None:
            #     print('error')
            if self.c is None:
                return self.c
            self.MOSG.set_ct(self.c)
            self.MOSG.cal_payoff()
            self.MOSG.cal_payoff_defender()
            # print('     目标{}找到达标收益{}>b{}'.format(i, self.MOSG.get_payoff_defender()[i], b[i]))
            b[i] = self.MOSG.get_payoff_defender()[i]
            # b[i] = self.MOSG.get_U_id(self.U_id, self.U_ia)
        # print('     退出ORIGAMIA+++++++++++++++++++++++++++++++++++++')
        # if self.c is not None:
            # print('ORIGAMIA找到可行解')
        # else:
            # print('ORIGAMIA找不到可行解')
        return self.c



