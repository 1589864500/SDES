# 利用cplex解MILPs的demo可见/tool/cplexText

import numpy as np

import cplex
from cplex.exceptions import CplexError

import tool.algorithm

class MILPs():
    # this function is implemented to solve MILPs by cplex tool package. return a resources allocation plan.
    def __init__(self, MOSG=None, b=None):

        # return U_ja/d(t)
        def get_U(player, cover, t, j): # cover:{0,1} player:{1:attack,0:defender}
            j += player*self.N
            return self.MOSG.get_gametable()[t,cover,j]

        self.MOSG = MOSG
        self.T = MOSG.target
        self.N = MOSG.player - 1
        self.variableNum = self.N*2 + self.N*self.T + self.T
        self.b = b
        self.M = 1000.0  # refer to a large number

        # 上下界见configPara()

        self.my_ctype = "C"*self.T + "I"*(self.T*self.N) + "C"*self.N*2               # C:continue I:integer

        # 变量名 # ct, aj_t, dj, kj
        self.my_colnames = []
        self.my_colnames += tool.algorithm.rename(s='c', a=self.T)
        self.my_colnames += tool.algorithm.rename(s='a', a=self.N, b=self.T)
        self.my_colnames += tool.algorithm.rename(s='d', a=self.N)
        self.my_colnames += tool.algorithm.rename(s='k', a=self.N)

        # RHS
        r1_1 = [self.M] * self.N*self.T
        # r1_2 refer to U_jdu(t)
        r1_2 = []
        for t in range(self.T):
            for j in range(self.N):
                r1_2.append(get_U(0, 1, t, j))
        r1 = [x+y for x,y in zip(r1_1, r1_2)]
        r2 = []
        for t in range(self.T):
            for j in range(self.N):
                r2.append(-get_U(1, 1, t, j))
        r3_1 = [-x for x in r2]
        r3 = [x+y for x,y in zip(r1_1, r3_1)]
        r4 = [1.] * self.N
        r5 = [self.MOSG.resource]
        self.my_rhs = r1 + r2 + r3 + r4 + r5

        # 行名
        my_rownames1 = tool.algorithm.rename(s='r1_', a=self.T*self.N)
        my_rownames2 = tool.algorithm.rename(s='r2_', a=self.T*self.N)
        my_rownames3 = tool.algorithm.rename(s='r3_', a=self.T, b=self.N)
        my_rownames4 = tool.algorithm.rename(s='r4_', a=self.N)
        my_rownames5 = ['r5_1']
        self.my_rownames = my_rownames1 + my_rownames2 + my_rownames3 + my_rownames4 + my_rownames5

        self.my_sense = "L"*self.N*self.T*3 + "E"*self.N + "L"  # L:less E:equal G:greater

        def get_my_row1(t, j):
            paraVal = []
            paraName = []
            paraName.append(self.my_colnames[t])                    # ct
            paraName.append(self.my_colnames[self.T + t*self.N+j])  # ajt
            paraName.append(self.my_colnames[self.T*(self.N+1) + j])    # dj
            paraVal.append(get_U(j=j, player=0, cover=1, t=t) - get_U(j=j, player=0, cover=0, t=t))
            paraVal.append(self.M)
            paraVal.append(1.)
            return [paraName, paraVal]

        def get_my_row2(t,j):
            paraVal = []
            paraName = []
            paraName.append(self.my_colnames[t])            # ct
            paraVal.append(-get_U(j=j, player=1, cover=1, t=t) + get_U(j=j, player=1, cover=0, t=t))
            paraName.append(self.my_colnames[self.T*(self.N+1)+self.N + j])   # kj
            paraVal.append(-1.)
            return [paraName, paraVal]

        def get_my_row3(t,j):
            paraVal = []
            paraName = []
            paraName.append(self.my_colnames[t])  # ct
            paraName.append(self.my_colnames[self.T + t*self.N+j])  # ajt
            paraName.append(self.my_colnames[self.T*(self.N+1)+self.N + j])   # kj
            paraVal.append(get_U(j=j, player=1, cover=1, t=t) - get_U(j=j, player=1, cover=0, t=t))
            paraVal.append(self.M)
            paraVal.append(1.)
            return [paraName, paraVal]

        def get_my_row4(j):
            coefficientMatrix = np.full((self.T,self.N), False)
            coefficientMatrix[:, j] = True
            # coefficientMatrixList = coefficientMatrix.reshape(-1,1).tolist()
            paraNameNp = np.array(self.my_colnames)[self.T:self.T*(self.N+1)]
            paraName = paraNameNp[coefficientMatrix.reshape((-1,1)).squeeze()].tolist()
            paraVal = [1.] * self.T
            return [paraName, paraVal]

        def get_my_row5():
            paraVal = [1.] * self.T
            paraName = self.my_colnames[:self.T]
            return [paraName, paraVal]

        # 约束的变量系数矩阵
        #         # demo
        #         # rows = [[["x1", "x2", "x3", "x4"], [-1.0, 1.0, 1.0, 10.0]],  # 配合变量名使用 定义变量系数
        #         #             [["x1", "x2", "x3"], [1.0, -3.0, 1.0]],
        #         #             [["x2", "x4"], [1.0, -3.5]]]
        my_row1 = []
        my_row2 = []
        my_row3 = []
        my_row4 = []
        my_row5 = []
        for t in range(self.T):
            for j in range(self.N):
                my_row1.append(get_my_row1(t,j))
                my_row2.append(get_my_row2(t,j))
                my_row3.append(get_my_row3(t,j))
        for j in range(self.N):
            my_row4.append(get_my_row4(j))
        my_row5.append(get_my_row5())
        self.my_row = my_row1 + my_row2 + my_row3 + my_row4 + my_row5

    # MILPsSolver is an implementation of reference[9] in MOSGs
    # all variable is(and its number): ct(T), a_jt(N*T), dj(N), kj(N)
    # def MILPsSolver(self, b):
        # -*- coding: utf-8 -*-
        # The MIP problem solved in this MILP formulation is:
        #
        #   Maximize  d_lambda
        #   Subject to
        #       constrains  variable  |  value range  |   constrains number
        # 若有N*T行，行顺序为: 1.t=1,j=1; 2.t=1,j=2; ......
        # Ujd/a(ct,t) = ct*U_ja/dc(ct,t) + (1-ct)*U_ja/du(ct,t)
        # U_jau(t)表示dim1=target=t, dim2=c/u=1/2, dim3=a/d=a/a+player_n
        # TODO: U_jau和U_jac
        # r1:       (-U_jdc(t)+U_jdu(t))ct + M*a_jt + dj <= M + U_jdu(t)  (1~N)(t in 1~T)  (1~N)(t in 1~T)
        # DONE:有一个问题，这个约束如何表示 # 第二个问题，如何再cplex调包的过程中更新Ujd(ct, t)
        # k 和 d一样都是不需要规定的变量
        # r2:       (U_jac(t)-U_jau(t))ct + (-1)kj <= -U_jau(t)               (1~N)(t in 1~T)  (1~N)(t in 1~T)
        # r3:       (U_jau(t)-U_jac(t))ct + M*a_jt + kj <= U_jau(t) + M      (1~N)(t in 1~T)
        # r4:       sum(a_jt, axis=t) = 1                                     (j in 1~A)  A/N  # 我有点怀疑大A和N是一个
        # r5:       sum(ct, axis=t) <= m                                       _ 1
        #   Bounds
        #        0 <= ct <= 1                   (t in 1~T)
        #        a_jt = {0, 1}                  (1~N)*(t in 1~T)
        # TODO:需要考虑d_j*如何表示
        #        d_j = d_j*  (1~lambda-1)

        #        d_j >= b_j  (lambda+1~N)
        #   Integers
        #        a_jt

        # demo
        # data common to all populateby functions
        # M = float("inf")
        # T = self.MOSG.target
        # N = self.MOSG.player - 1
        # Lambda = None # 当前研究的i
        # variableNum = N + T + N*T
        # my_ub = np.full((variableNum,), 0)  # 上下界
        # my_lb = []
        # my_ctype = "CCCI"  # C:continue I:integer
        # my_colnames = ["x1", "x2", "x3", "x4"]  # 变量名
        # my_rhs = [20.0, 30.0, 0.0]  # 右侧常数项
        # my_rownames = ["r1", "r2", "r3"]  # 行名
        # my_sense = "LLE"  # L:less E:equal G:greater
        # demo
    def configPara(self, prob, Lambda, dStar):
        # DONE:我们认为研究对象的下标就是lambda，后续还需要验证
        # DONE:bStar到底是什么

        # 目标函数的系数
        # demo: my_obj = [1.0, 2.0, 3.0, 1.0]  # 目标函数系数  在MOSGs中，my_obj是会发生变化的
        epsilon = 10e-5
        gap = 0.05
        self.my_obj = np.full((self.variableNum,), 0.)
        self.my_obj[self.T*(self.N+1)+Lambda] = 1

        # 上下界
        self.my_ub = np.full((self.variableNum,), 1.)    # 上下界
        # DONE: dj(1<=j<Lambda)
        self.my_ub[self.T*(self.N+1):self.T*(self.N+1)+Lambda] = dStar * (1+gap)       # d_j = d_j*  (1~lambda-1)
        # d_Lambda without constraint | d_j >= b_j  (lambda+1~N) | 可能将U_id的上下界缩小能降低时耗
        # TODO:一种文中提到的剪枝加速方法，待验证
        # self.my_ub[self.T * (self.N + 1) + Lambda] = self.b[Lambda]
        self.my_ub[self.T * (self.N + 1) + Lambda] = cplex.infinity
        self.my_ub[self.T*(self.N+1)+Lambda+1:] = cplex.infinity
        self.my_lb = np.full((self.variableNum,), 0.)
        self.my_lb[self.T*(self.N+1):self.T*(self.N+1)+Lambda] = dStar * (1-gap)
        self.my_lb[self.T*(self.N+1):Lambda] = -cplex.infinity
        self.my_lb[self.T*(self.N+1)+Lambda+1:self.T*(self.N+1)+self.N] = self.b[Lambda+1:self.N]
        # DONE:the bound of kj = -cplex.infinity
        self.my_lb[self.T*(self.N+1)+self.N:] = -cplex.infinity     # k

        print('PROBLEM SIZE: VariableNum:{} | ConstrainNum:{}'.format(self.variableNum, self.N*self.T*3+self.N+1))

        prob.objective.set_sense(prob.objective.sense.maximize)  # 交代问题特性？最大化问题

        prob.variables.add(obj=self.my_obj.tolist(), lb=self.my_lb.tolist(), ub=self.my_ub.tolist(), types=self.my_ctype,
                               names=self.my_colnames)

        prob.linear_constraints.add(rhs=self.my_rhs, names=self.my_rownames,
                                        lin_expr=self.my_row, senses=self.my_sense)

    def updateU(self, c):
        self.MOSG.set_ct(c)
        self.MOSG.cal_payoff()
        self.MOSG.cal_payoff_defender()
        return self.MOSG.get_payoff_defender()

    def do(self):
        try:
            c = np.full((self.T,), 0.)
            v = self.updateU(c)
            for Lambda in range(self.N):
                if self.b[Lambda] <= v[Lambda]:
                    continue
                my_prob = cplex.Cplex()
                self.configPara(my_prob, Lambda=Lambda, dStar=v[:Lambda])  # 配置my_prob参数
                my_prob.solve()

                # 打印求解器状态（是否可解），以及解最优值
                # solution.get_status() returns an integer code
                status = my_prob.solution.get_status()
                if status == 101:
                    print('可行解，具体为{} | {} | {}'.format(status, my_prob.solution.status[status],
                                                                 my_prob.solution.get_status_string(status)))
                    print("Solution value  = ", my_prob.solution.get_objective_value())

                    # 打印求解问题的规模（变元数 约束数），以及松弛变量大小
                    # numcols = my_prob.variables.get_num()
                    # numrows = my_prob.linear_constraints.get_num()
                    # print('numcols = {}'.format(numcols))
                    # print('numrows = {}'.format(numrows))
                    # slack = my_prob.solution.get_linear_slacks()  # 松弛部分 富余部分 这里slack=[0,2,0]，2表示达到最优解时约束2还差2
                    # print('松弛变量 = {}'.format(slack))

                    # 打印求解结果
                    c = my_prob.solution.get_values()
                    print('x = {}'.format(c))
                elif status == 103:
                    print('不可行解，具体为{} | {} | {}'.format(status, my_prob.solution.status[status],
                                                        my_prob.solution.get_status_string(status)))
                    return None
                else:
                    print('遇到了没见过的status，具体为{} | {} | {}'.format(status, my_prob.solution.status[status],
                                                                 my_prob.solution.get_status_string(status)))
            return c

        except CplexError as exc:
            print(exc)
            return None


