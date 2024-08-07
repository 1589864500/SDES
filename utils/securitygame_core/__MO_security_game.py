import numpy as np


class MOSG(object):
    ''' gametable [target, 2, (player-1)*2]
    2: 0:防住了 1:没防住
    1-N player:defender N+1-2N:attacker
    player_num refers as the total number of attackers and defenders
    target_num refers as the number of items witch need to be protected'''
    def __init__(self, player_num=None, target_num=None, resource_ratio=0.2, ai=None,
                 sampling="uniform"):
        self.player = player_num
        self.target = target_num
        self.resource = resource_ratio * target_num
        # get ct
        self.ct = np.random.choice(100, size=self.target)
        self.ct = self.ct / np.sum(self.ct) * self.resource
        pos = self.ct >= 1
        while np.any(pos):
            redundant = np.sum(self.ct[pos] - 1)
            self.ct[pos] = 1
            temp = np.random.choice(100, size=self.target)
            temp[pos] = 0
            temp = temp / np.sum(temp) * redundant
            self.ct += temp
            pos = self.ct >= 1
            if np.all(pos):
                self.ct = np.full(self.ct.shape, 1, dtype=np.float32)
                break
        #原版(不用做任何调整)

        # 本测试默认pop_size=1
        # self.ct = self.ct[np.newaxis,]
        self.ai = ai
        self.sampling =\
            sampling.lower()
        if self.sampling == "uniform":
            np.random.seed(12345)  # 42
            # [T, 2, player-1]
            self.gametable = np.random.randint(low=1, high=11, size=(self.target, 2, 2 * (self.player - 1)))
            self.gametable = self.gametable.astype(np.float64)
        elif self.sampling == "col_normal":
            pass
        elif self.sampling == "row_normal":
            pass
        elif self.sampling == "dep_normal":
            pass
        else:
            print("The distribution uesd by now has not defined!!!!\nexit(code)")
            exit()
        self.gametable[..., 1, 0:(self.player - 1)] *= -1
        self.gametable[..., 0, (self.player - 1):] *= -1
        self.payoff = np.zeros((self.gametable.shape[0], self.gametable.shape[-1]))
        self.payoff_attacker = np.zeros(player_num - 1)
        self.payoff_defender = np.zeros(player_num - 1)
        self.cal_payoff()

    def validation(self):
        print("ct:size={} []={}\n sum={}\n".format(self.ct.shape, self.ct, np.sum(self.ct)))

        print("gametable:size={} []=\n".format(self.gametable.shape))
        print('\n'.join(str(self.gametable[..., i]) for i in range(self.gametable.shape[-1])))

        print("payoff:size={} []={}\n".format(self.payoff.shape, self.payoff))
        print("payoff attacker:size={} []={}\npayoff defender:size={} []={}\n".format(self.payoff_attacker.shape,
                                                                                      self.payoff_attacker,
                                                                                      self.payoff_defender.shape,
                                                                                      self.payoff_defender))

    # 接受带种群大小的维度，后续处理多考虑一维
    def set_ct(self, x):
        self.ct = x

    def cal_payoff(self):
        # calculate payoff of attacker and defender for all targets

        # 原版
        self.payoff = np.round(self.gametable[:,0,:] * self.ct[..., np.newaxis] + self.gametable[:,1,:] * (1 - self.ct[..., np.newaxis])
                               , 5)
        # 考虑种群大小的版本
        # self.payoff = self.gametable[np.newaxis, :, 0, :] * self.ct[..., np.newaxis] + self.gametable[np.newaxis, :, 1,
        #                                                                                :] * (
        #                       1 - self.ct[..., np.newaxis])

    # 下面两个暂时用不到
    def cal_payoff_attacker(self):
        # attacker will select the target with the greatest payoff to attack
        # return U_i^a(c)
        # 原版
        self.payoff_attacker = np.amax(self.payoff, axis=0)[-(self.player-1):]
        # 考虑种群大小的版本
        # self.payoff_attacker = np.amax(self.payoff, axis=1)[:, -(self.player - 1):]

    def cal_payoff_defender(self):
        # when we figure out which target would be attack by attacker, then we goto quire its utility for defender
        # return U_i^d(c)
        # 原版
        for i in range(self.player-1):
            gamei_idx = np.where( self.payoff[:,i+self.player-1]==np.max(self.payoff[:,i+self.player-1]) )[0]
            self.payoff_defender[i] = np.max(self.payoff[gamei_idx,i])
        # 考虑种群大小的版本
        # self.payoff_defender = np.amax(self.payoff, axis=1)[:, :(self.player - 1)]

    # TODO:可用get_U_ik代替,但是注意，这不是求U_id(t) U_ia(t)的函数！
    # method: obji对应的defender and attacker's playtable，calculate the maximal payoff for defender
    def get_U_id(self, U_id, U_ia):
        a_idx = np.where(U_ia == np.max(U_ia))[0]
        # print("防御者可引导的下标集合为：{}\n防御者的收益表为：{}\n最终防御者选择下标：{}".format(
        #     aIdx, U_id, aIdx[np.argmax(U_id[aIdx])]
        # ))
        return np.max(U_id[a_idx])

    def get_payoff_defender(self):
        return self.payoff_defender

    def get_payoff_attacker(self):
        return self.payoff_attacker

    def get_U_ijk(self, i, j, k):
        # return U_i^c,a or U_i^u,a or ..., it is a vector [T,]
        # return gametable
        i += (self.player - 1) * k
        return self.gametable[:, j, i]

    def get_U_ik(self, i, k):
        # return U_i^d(c, for all t) or ..., it is a vector [T,]
        # return payoff
        i += (self.player - 1) * k
        return self.payoff[:, i]

    def get_gametable(self):
        return self.gametable

    def set_gametable(self, gametable):
        self.gametable = gametable

    def get_payoff(self):
        return self.payoff

def main():
    models = MOSG(player_num=4, target_num=25, resource_ratio=0.2)
    # models.cal_payoff()
    models.cal_payoff_defender()
    models.cal_payoff_attacker()
    models.validation()


if __name__ == "__main__":
    main()