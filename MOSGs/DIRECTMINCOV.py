import copy

import numpy as np

import tool.algorithm
from securitygame_core.MO_security_game import MOSG

class DIRECTMINCOV():
    def __init__(self, MOSG):
        self.MOSG = MOSG
        self.c = np.full((self.MOSG.target,), 0, dtype=np.float32)
        self.updateC()
        self.errorCount = 0

        # U_id, U_ia are vectors, U_ia(ct,t) is a scale.
        # the payoff of defender against attacker_i in resource ct (target which will be attack depends on U_ia)
        # U_id[j] means the payoff defender would get, if target_j is attacked.
        self.U_id = None
        # the payoff of attacker_i in resource ct
        # the property of U_ia is similar to U_id
        self.U_ia = None
        self.U_iua = None
        self.U_ica = None
        self.U_iud = None
        self.U_icd = None
        self.noninducibleNextTarget = None
        self.resourcesExceeded = None

    # MOSGs中，gametable不随问题变化而变化，比如U_ca,U_ua等；但是ct变化时playtable会发生变化，比如U_a,U_d
    # 因此updateU出现必定有updateC
    def updateU(self, i, display=False, message1="", mincov=False, message2=""):
        # i: the idx of the obj we discuss

        # if mincov:
        #     print('     原U_id:{}'.format(np.round(self.U_id, 3)))
        #     print('     ', end='')
        # if display:
        #     print('原U_ia：{}'.format(np.round(self.U_ia, 3)))
        self.U_id = self.MOSG.get_U_ik(i, 0)
        self.U_ia = self.MOSG.get_U_ik(i, 1)
        self.U_ica = self.MOSG.get_U_ijk(i, 0, 1)
        self.U_iua = self.MOSG.get_U_ijk(i, 1, 1)
        self.U_icd = self.MOSG.get_U_ijk(i, 0, 0)
        self.U_iud = self.MOSG.get_U_ijk(i, 1, 0)
        # if len(np.where(np.abs(self.U_ia) > 10)[0]) > 0 or len(np.where(np.abs(self.U_id) > 10)[0]) > 0:
        #     # tool.algorithm.saveVari(U_ia=self.U_ia, U_id=self.U_id, payoff=self.MOSG.get_payoff(), gametable=self.MOSG.get_gametable(), ct=self.c, i=np.array([i]))
        #     print('error1')
        # if mincov:
        #     print('     {}(U_id)：{}'.format(message2, np.round(self.U_id, 3)))
        #     print('     ', end='')
        # if display:
        #     print('{}t对攻击者吸引效果具体为(U_ia)：{}'.format(message1, np.round(self.U_ia, 3)))


    # updateC和updateU的作用是：由于ORIGAMIM没有继承MOSG，class MOSG和ORIGAMIM有各自的c
    # self.c只能修改到ORIGAMIM的c，因此额外写了updateC，同时ORIGAMIM中其他信息也要更新，因此要用到updateU
    # 总的来说只要对self.c赋值最好顺便再执行updateC updateU
    def updateC(self, addedCov=None):
        if addedCov is not None:
            self.c = self.c + addedCov
        self.c[np.where((self.c<0) & (self.c>-1e-5))[0]] = 0
        self.c[np.where((self.c>1) & (self.c-1<1e-5))[0]] = 1
        # lb_idx = np.where(self.c<0)[0]
        # ub_idx = np.where(self.c>1)[0]
        # if len(lb_idx) > 0 or len(ub_idx) > 0:  
        #     print('error2')
        self.MOSG.set_ct(self.c)
        self.MOSG.cal_payoff()

    # U_iad refer to U_ia or U_id  player:{0,1} 0:defender 1:attacker
    def calAddedCov(self, U_iad, t, player, flag=False):
        # 若calAddedCov返回负数，说明U_iad > self.U_iua[t]（因为分母必定为负数）
        if player == 1:
            ct = (U_iad - self.U_iua[t]) / (self.U_ica[t] - self.U_iua[t])
        else:
            ct = (U_iad - self.U_iud[t]) / (self.U_icd[t] - self.U_iud[t])
        if ct < 0:
            if flag:
                print('不可能发生')
            return 0
        elif ct > 1:
            if flag:
                print('不可能发生')
            return 1
        else:
            if ct < -1e-5 or ct > 1+1e-5:
                print('error2')
            return ct

    # U_id(ct, t)的导数，表示变化一个单位的U_id需要多少ct
    def calRatio(self):
        return 1 / (self.U_iua - self.U_ica)

    def getNextLen(self, epsilon=0):
        # TODO:gap
        U_a_max = np.max(self.U_ia)
        return len(np.where(U_a_max - self.U_ia <= epsilon)[0])

    # 在ORIGAMIA/M优化obj2-objn以后，会有一定资源剩余(left_resources)，如何分配这些资源优化obj1即该函数解决的
    # 值得注意的是：ct所有元素的取值情况在ORIGAMIA/M后达到了平衡，再分配left时需要额外注意
    # method: 优化obj1时需要留意obj2-objn，若不满足约束需放弃更改
    def leftAllocation(self, b, obj_idx=0) ->None:
        # 算法思路同do()中的攻击集扩张过程，扩张到第一次破快平衡即停止，并回溯到上次记录最优
        best_c = self.c
        # leftAllocation只优化obj1
        # obj_idx = 0
        # 统计攻击集情况，按U_a排序
        self.updateU(i=obj_idx)
        idx = np.argsort(-self.U_ia)
        # 最基本的循环判定条件为： 资源剩余量、攻击集大小要满足条件
        # 若满足while条件，攻击集扩大，next+=1
        noninducibleNextTarget = False
        resourcesExceeded = False
        safeimprovement = True  # 代表本次ct的改动是安全的，否则ct退回上一次改动
        while 1:
            # 计算next，getNextLen()返回Gamma size
            addedCov = np.zeros(shape=self.c.shape)
            next = self.getNextLen() + 1  # 当前Gamma大小+1

            # 若扩张到极限，则停止
            if next == len(idx) + 1:
                break

            # if:将Gamma扩大一格 else:Gamma无法扩张，做最后一次资源分配
            if np.max(self.U_ica[idx[0:(next - 1)]]) > self.U_ia[idx[next - 1]]:
                x = np.max(self.U_ica[idx[0:(next - 1)]])
                noninducibleNextTarget = True
            else:
                x = self.U_ia[idx[next - 1]]
            # 计算将所有对象和引导对象x看齐需要花费的addedCov
            for j in range(next - 1):
                target = idx[j]
                addedCov[target] = self.calAddedCov(x, target, 1, flag=True) - self.c[target]
                if addedCov[target] < 0:
                    print('error2')
            self.updateC(addedCov)
            left = self.MOSG.resource - np.sum(self.c)
            if np.sum(addedCov) > left:
                resourcesExceeded = True
                ratio = self.calRatio()[idx[:next - 1]]
                addedCov[idx[:next - 1]] = left * (ratio / np.sum(ratio))  # a vector /in [0, target)

            # 往下部分和do()有所不同，开始设计while出口

            # 破坏obj2-objn导致退出循环
            for i in range(1, self.MOSG.player-1):
                self.updateU(i=i, display=True, message1='do()中添加addedCov后，')
                if b[i] > self.MOSG.get_U_id(self.U_id, self.U_ia):
                    safeimprovement = False
                    break
            if safeimprovement:
                best_c = self.c
                self.updateU(i=obj_idx, display=True, message1='do()中添加addedCov后，')
            else:
                self.c = best_c
                self.updateC()
                self.updateU(i=obj_idx)
                break
            # left, next原因导致退出循环
            if resourcesExceeded or noninducibleNextTarget:
                break

    def MINCOV(self, gameIdx, b, next, idx):
        '''参数解释：
            gameIdx the obj idx
            b lower bound
            next Gammasize
            idx target order'''
        cStar = None  # cStar is expected to be as small as possible.
        # DONE:MINCOV属于精修阶段，找到的方案起码不会比原方案来的差，则minResources应该为np.sum(self.c)
        minResources = self.MOSG.resource  # minResources is a theory of upper bound
        tempC = copy.copy(self.c)
        # print('     MINCOV函数变量详情：b:{}, next={}, gameIdx={}, Gamma={}, 原方案的资源数c={}'.format(np.round(b, 3), next, gameIdx, idx[:next], np.round(np.sum(self.c), 3)))

        # 大循环将攻击集gameInd的每个对象一一讨论，讨论如果他被引导会发生什么，能否继续保证满足约束b[gameIdx]的同时最小化资源消耗
        for tPrime in idx[:next]:
            # print('     研究对象tPrime:{}'.format(tPrime))
            # if not the first iteration, reset ct by tempC
            if tPrime != idx[0]:
                self.c = copy.copy(tempC)
                self.updateC()
                self.updateU(gameIdx)
            self.c[tPrime] = self.calAddedCov(b[gameIdx], tPrime, 0)
            # update ct
            self.updateC()
            self.updateU(gameIdx)
            # self.updateU(gameIdx, display=True, message1="MINCOV将tPrime作为最终攻击对象后，", mincov=True, message2="MINCOV将tPrime作为最终攻击对象后,")
            # 若T中有对象产生的收益比tPrime来得高，则通过撤销资源抑制其引导效果
            for t in idx:
                if t == tPrime:
                    continue
                # 若存在t的吸引力比tPrime来得大，
                if self.U_ia[t] > self.U_ia[tPrime]:
                    # 则通过改变资源分配（可能增加资源投入也可能减少）
                    self.c[t] = self.calAddedCov(self.U_ia[tPrime], t, 1)
                    # update ct
                    self.updateC()
                    self.updateU(gameIdx)
                    # self.updateU(gameIdx, display=True, message1="MINCOV为吸引力对象t分配更多资源后，", mincov=True, message2="MINCOV为吸引力对象t分配更多资源后，")
            # 若新方案引导tPrime在满足约束b[gameIdx]的同时还能节省资源则采纳
            neededResources = np.sum(self.c)
            # print('     当前研究的tPrime需要消耗{}的资源'.format(neededResources))
            if self.MOSG.get_U_id(self.U_id, self.U_ia) >= b[gameIdx] and neededResources <= minResources:
                # print('     找到更优的方案，neededResources:{}, tPrime:{}'.format(neededResources, tPrime))
                cStar = copy.copy(self.c)
                minResources = neededResources
        # print('     退出MINCOV后最省资源的方案的资源数为{}'.format(np.sum(cStar)))
        return cStar
    
    def mincov(self, gameIdx, b, next, idx):
        cStar = None
        minResources = self.MOSG.resource
        baseCov = np.sum(self.c)
        for j in range(next):
            feasible = True
            cPrime = copy.copy(self.c)
            tj = idx[j]
            cPrime[tj] = self.calAddedCov(b[gameIdx], tj, 0)
            cPrime[tj] = max(cPrime[tj], self.c[tj])
            if cPrime[tj] > 1:
                if cPrime[tj] - 1 < 1e-5:
                    cPrime[tj] = 1
                else:
                    break
            covSoFar = baseCov + cPrime[tj] - self.c[tj]
            for k in range(self.MOSG.target):
                tk = idx[k]
                # 求Uia(tk), Uia(tj)并回溯
                cTemp = self.c
                self.c = cPrime
                self.updateC()
                self.updateU(gameIdx)
                U_ia_tk = self.U_ia[tk]
                U_ia_tj = self.U_ia[tj]
                self.c = cTemp
                self.updateC()
                self.updateU(gameIdx)
                if tj != tk and U_ia_tk > U_ia_tj:
                    cPrime[tk] = self.calAddedCov(U_ia_tj, tk, 1)
                    if cPrime[tk] < self.c[tk] or cPrime[tk] > 1:
                        feasible = False
                        break
                    covSoFar += cPrime[tk] - self.c[tk]
                    if covSoFar >= minResources:
                        feasible = False
                        break
            if feasible:
                cStar = cPrime
                minResources = covSoFar
        return cStar


    def beginEnd(self, BEGIN=0, END=0, RETURN=0, GATE=0):
        if BEGIN == 1:
            print('     ORIGAMI函数开始')
        elif END == 1:
            print('     ORIGAMI函数结束，求得帕累托最优解：{}'.format(np.round(self.c, 3)))

        if RETURN == 1:
            print('     return NULL\n GATE is {}'.format(GATE))
        elif RETURN == 2:
            print('     return c and c is not NULL')
        elif RETURN == 3:
            print('     pass whose code is not completed')

    # idx refer to (gameidx + 1)
    def recheck(self, idx, b):
        U_d = np.full(b.shape, 0.)
        for i in range(1,b.size):
            self.updateC()
            self.updateU(i)
            U_d[i] = self.MOSG.get_U_id(self.U_id, self.U_ia)
        if len(np.where(U_d[1:] < b[1:])[0]) > 0:
            print('ERROR!!! {}>{}不成立'.format(U_d[1:], b[1:]))
            return True
        return False

    # FIXME updateC函数涉及addCov，可能不能继续用了
    # TODO: update ct
    def do(self, b:np.ndarray)->np.ndarray:
        # self.beginEnd(BEGIN=1)
        gen = 0  # gen refer to the time of adjusting secondary obj_i to satisfy b[i]
        satisfy_n = 0  # count the numpy of secondary obj_i satisfy satisfy b[i]
        while gen < b.size * 4 +1:
            gen += 1
            i = gen % b.size
            # [except for ORIGAMIA] only consider about secondary obj(1<=i<=n-1), not primary obj(i==0)
            if i == 0:
                continue
            self.updateU(i)
            # if U_id(ct,t) is less than b[i], it is needed to allocate more ct(resource)
            # U_id(ct,t) depends on the target_t attacker choosing. but not just max(U_id)
            # debug
            #TODO: recheck b[i] util all secondary objectives satisfy b[i]

            # b[0]会被continue跳过
            if b[i] > self.MOSG.get_U_id(self.U_id, self.U_ia):  # 获取Uid
                satisfy_n = 0

                # NOTE  DIRECTMINCOV
                idx = np.argsort(-self.U_ia)
                self.c = self.MINCOV(i, b, self.MOSG.target, idx)
                if self.c is None:
                    return None
                self.updateC()
                self.updateU(i)
                
            # NOTE for DIRECTMINCOV
            # print('')
            
            # NOTE form all impletation that without optimizing obj_0
            satisfy_n += 1
            if satisfy_n == b.size - 1:
                break
            if gen == (b.size * 4+1):
            # if gen == b.size+1:
                self.errorCount += 1
                return None
        self.leftAllocation(b)
        return self.c

    def debugDo(self, **kwargs):
        for arg, value in kwargs.items():
            print("     {}:{}".format(arg, value))


