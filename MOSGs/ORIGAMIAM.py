import copy

import numpy as np

import tool.algorithm
from securitygame_core.MO_security_game import MOSG

class ORIGAMIM():
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
        if len(np.where(np.abs(self.U_ia) > 10)[0]) > 0 or len(np.where(np.abs(self.U_id) > 10)[0]) > 0:
            # tool.algorithm.saveVari(U_ia=self.U_ia, U_id=self.U_id, payoff=self.MOSG.get_payoff(), gametable=self.MOSG.get_gametable(), ct=self.c, i=np.array([i]))
            print('error1')
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
        lb_idx = np.where(self.c<0)[0]
        ub_idx = np.where(self.c>1)[0]
        if len(lb_idx) > 0 or len(ub_idx) > 0:  
            print('error2')
        self.MOSG.set_ct(self.c)
        self.MOSG.cal_payoff()

    # U_iad refer to U_ia or U_id  player:{0,1} 0:defender 1:attacker
    def calAddedCov(self, U_iad, t, player, flag=False):
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
            addedCov = np.zeros(shape=self.c.shape)
            next = self.getNextLen() + 1
            if np.max(self.U_ica[idx[0:(next - 1)]]) > self.U_ia[idx[next - 1]]:
                x = np.max(self.U_ica[idx[0:(next - 1)]])
                noninducibleNextTarget = True
            else:
                x = self.U_ia[idx[next - 1]]
            # 计算将所有对象和引导对象x看齐需要花费的addedCov
            for j in range(next - 1):
                target = idx[j]
                addedCov[target] = self.calAddedCov(x, target, 1, flag=True) - self.c[target]
            left = self.MOSG.resource - np.sum(self.c)
            if np.sum(addedCov) > left:
                resourcesExceeded = True
                ratio = self.calRatio()[idx[:next - 1]]
                addedCov[idx[:next - 1]] = left * (ratio / np.sum(ratio))  # a vector /in [0, target)
            self.updateC(addedCov)

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

    # TODO: update ct
    def do(self, b):
        # self.beginEnd(BEGIN=1)
        gen = 0  # gen refer to the time of adjusting secondary obj_i to satisfy b[i]
        satisfy_n = 0  # count the numpy of secondary obj_i satisfy satisfy b[i]
        while gen < b.size * 4 +1:
        # while 1:
            gen += 1
            i = gen % b.size
            # [except ORIGAMIA] only consider about secondary obj(1<=i<=n-1), not primary obj(i==0)
            # if i == 0:
            #     continue
            self.updateU(i)
            # if U_id(ct,t) is less than b[i], it is needed to allocate more ct(resource)
            # U_id(ct,t) depends on the target_t attacker choosing. but not just max(U_id)
            # debug
            #TODO: recheck b[i] util all secondary objectives satisfy b[i]

            # print("b[i] {} > U_id {}, >成立说明资源不够满足约束，增加资源".format(np.round(b[i],3), np.round(self.MOSG.get_U_id(self.U_id, self.U_ia),3)))
            # b[0]会被continue跳过(在ORIGAMIA中不跳过)
            if b[i] > self.MOSG.get_U_id(self.U_id, self.U_ia):  # 获取Uid
                satisfy_n = 0
                # print('进入第{}轮'.format(i))
                # for different objective i, update U_ik and U_ijk, which is a vector.
                left = self.MOSG.resource - np.sum(self.c)
                next = 2
                # priority is given to those with large U_a, so sorting target by U_a and save idx.
                idx = np.argsort(-self.U_ia)
                while next <= self.MOSG.target:
                    addedCov = np.full((self.MOSG.target,), 0, dtype=np.float64)
                    # if-else 分支结束都需要更新给攻击集Gamma
                    # 确定要选择的引导对象x
                    # next = size(Gamma) + 1 不是说扩张后Gamma一定变大了1，而是说next-1刚好等于size(Gamma)，下面代码需要
                    next = self.getNextLen() + 1
                    # 若攻击集内某target资源分配满(ct=1)也无法下降到新目标的水平，则返回不可行
                    # if左边代表当前([0,next-1))攻击集，右边代表最有可能假如攻击集的target
                    if np.max(self.U_ica[idx[0:(next-1)]]) > self.U_ia[idx[next-1]]:
                        # 虽然进入该分支代表攻击集不能再继续扩大，但是仍然需要统计x，代表最大能下降的程度
                        x = np.max(self.U_ica[idx[0:(next-1)]])
                        self.noninducibleNextTarget = True
                    else:
                        x = self.U_ia[idx[next-1]]
                    # 计算将所有对象和引导对象x看齐需要花费的addedCov
                    for j in range(next-1):
                        target = idx[j]  # U_ia
                        # 虽然算出来需要多少资源addCov，但是不代表资源够用
                        addedCov[target] = self.calAddedCov(x, target, 1, flag=True) - self.c[target]
                        # self.debugDo(next=next, j=j, target=target, calAddedCov=self.calAddedCov(x, target), addedCov=addedCov)
                    # print('比较sum(addedCov)和left：{} > {}'.format(np.sum(addedCov), left))
                    # 判断资源是否够用，不够用则按比例分配
                    left = self.MOSG.resource - np.sum(self.c)
                    if np.sum(addedCov) > left:
                        # 需要注意的是，next表示攻击集大小，取具体target不能直接target[:next-1]，因为target不是顺序排列的，需要借助idx（排序用下标）
                        self.resourcesExceeded = True
                        ratio = self.calRatio()[idx[:next-1]]
                        addedCov[idx[:next-1]] = left * (ratio / np.sum(ratio)) # a vector /in [0, target)
                    self.updateC(addedCov)
                    self.updateU(i, display=True, message1='do()中添加addedCov后，')
                    # self.updateC(addedCov)后，size(Gamma)会发生变化，因此需要更新攻击集大小
                    next = self.getNextLen()
                    # if-elif-else 判断收益是否达标b[i]，
                    # 若达标且有资源则退出内层while循环，该while负责计算达到b[i]需要多大的攻击集
                    if self.MOSG.get_U_id(self.U_id, self.U_ia) >= b[i]:
                        # print('收益达标！{}>={}'.format(self.MOSG.get_U_id(self.U_id, self.U_ia), b[i]))
                        # debug
                        # break
                        # TODO: MINCOV(i,c,b)
                        # MINCOV参数self.getNextLen()传入size(Gamma)
                        # print('MINCOV进入，因为U_id({})>=b[i]({})'.format(np.round(self.MOSG.get_U_id(self.U_id, self.U_ia), 3), np.round(b[i], 3)))
                        cTemp = copy.copy(self.c)
                        cPrime = self.MINCOV(i, b, next, idx)
                        if cPrime is None:
                            self.c = cTemp
                        else:
                            self.c = cPrime
                        self.updateC()
                        self.updateU(i)
                        break
                    # 若没达标且资源不够了，返回None
                    elif self.resourcesExceeded or self.noninducibleNextTarget:
                        # self.beginEnd(RETURN=1, GATE=1)
                        # print('     resourcesExceeded:{}  noninducibleNextTarget:{}'.format(self.resourcesExceeded, self.noninducibleNextTarget))
                        return None, None # None means infeasible
                    # 若没达标但是资源够则扣除资源继续
                    else:
                        left = self.MOSG.resource - np.sum(self.c)
                        next += 1
                if next == (self.MOSG.target + 1):
                    if left > 0:
                        # self.beginEnd(RETURN=3)
                        # debug
                        # return None
                        # print('MINCOV进入')
                        # TODO: MINCOV(i,c,b)
                        self.c = self.MINCOV(i, b, self.getNextLen(), idx)
                        if self.c is None:
                            return None, None # None means infeasible
                        self.updateC()
                        self.updateU(i)
                    else:
                        # self.beginEnd(RETURN=1, GATE=2)
                        return None, None # None means infeasible
            satisfy_n += 1
            if satisfy_n == b.size:
                break
            # print('gen==',gen)
            # debug
            # if gen == 10:
            #     print('error')
            if gen == (b.size * 4+1):
            # if gen == b.size+1:
                self.errorCount += 1
                return None, None
        # print("c=SGSolver.do(b)  sum(c):", np.round(np.sum(self.c), 3))
        # self.beginEnd(END=1, RETURN=2)
        if self.recheck(b.size, b):
            print('error')

        # print('ct:', self.c)
        # self.MOSG.cal_payoff()
        # self.MOSG.cal_payoff_defender()
        # payoff_d = self.MOSG.get_payoff_defender()
        # print('defender payoff before leftAllocation:', payoff_d)
        # if np.any(payoff_d < b):
        #     print('error, before leftAllocation')

        # 唯一出口，出去前分配剩余资源优化obj1
        # self.leftAllocation(b)

        # print('ct:', self.c)
        # self.MOSG.cal_payoff()
        # self.MOSG.cal_payoff_defender()
        # payoff_d = self.MOSG.get_payoff_defender()
        # print('defender payoff after leftAllocation:', payoff_d)
        # if np.any(payoff_d < b):
        #     print('error, after leftAllocation')
        return self.c, self.errorCount

    def debugDo(self, **kwargs):
        for arg, value in kwargs.items():
            print("     {}:{}".format(arg, value))


