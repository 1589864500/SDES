# -*- coding: utf-8 -*-
# The MIP problem solved in this example is:
#
#   Maximize  x1 + 2 x2 + 3 x3 + x4
#   Subject to
#      - x1 +   x2 + x3 + 10 x4 <= 20
#        x1 - 3 x2 + x3         <= -30
#               x2      - 3.5x4  = 0
#   Bounds
#        0 <= x1 <= 40
#        0 <= x2
#        0 <= x3
#        2 <= x4 <= 3
#   Integers
#       x4

import cplex
from cplex.exceptions import CplexError

# data common to all populateby functions
my_ub = [40.0, cplex.infinity, cplex.infinity, 3.0]  #上下界
my_lb = [0.0, 0.0, 0.0, 2.0]
# my_ctype = "CCCI"  #  C:continue I:integer
my_ctype = "CCCC"
my_colnames = ["x1", "x2", "x3", "x4"]  #  变量名
my_rhs = [20.0, -30.0, 0.0]  #  右侧常数项
my_rownames = ["r1", "r2", "r3"]  #  行名
my_sense = "LLE"  #  L:less E:equal G:greater


def populatebyrow(prob):
    my_obj = [1.0, 2.0, 3.0, 1.0]  # 目标函数系数  在MOSGs中，my_obj是会发生变化的
    prob.objective.set_sense(prob.objective.sense.maximize)  #  ？？？ 交代问题特性？

    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype,
                       names=my_colnames)

    rows = [[["x1", "x2", "x3", "x4"], [-1.0, 1.0, 1.0, 10.0]],  #  配合变量名使用 定义变量系数
            [["x1", "x2", "x3"], [1.0, -3.0, 1.0]],
            [["x2", "x4"], [1.0, -3.5]]]

    prob.linear_constraints.add(rhs=my_rhs, names=my_rownames,

                                lin_expr=rows, senses=my_sense)


try:
    my_prob = cplex.Cplex()
    populatebyrow(my_prob)
    my_prob.solve()

except CplexError as exc:
    print(exc)

# 打印求解器状态（是否可解），以及解最优值
# solution.get_status() returns an integer code
status = my_prob.solution.get_status()
print("\nSolution status = ", status, ":", end=' ')
# the following line prints the corresponding string
print(my_prob.solution.status[status], ':', end=' ')
print(my_prob.solution.get_status_string(status))
print("Solution value  = ", my_prob.solution.get_objective_value())

# 打印求解问题的规模（变元数 约束数），以及松弛变量大小
numcols = my_prob.variables.get_num()
numrows = my_prob.linear_constraints.get_num()
print('numcols = {}'.format(numcols))
print('numrows = {}'.format(numrows))
slack = my_prob.solution.get_linear_slacks()  #  松弛部分 富余部分 这里slack=[0,2,0]，2表示达到最优解时约束2还差2
print('松弛变量 = {}'.format(slack))

# 打印求解结果
x = my_prob.solution.get_values()
print('x = {}'.format(x))
y = my_prob.solution.get_objective_value()
print('y = {}'.format(y))
