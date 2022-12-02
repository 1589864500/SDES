import numpy as np
from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter

# the theory pf is normalized in [0,1]
# The pareto front of a scaled zdt1 problem
pf = get_problem("zdt1").pareto_front()

# the task obj is minimise
# the result like A is normalized by default
# The result found by an algorithm
A = pf[::10] * 1.1

# plot the result
Scatter(legend=True).add(pf, label="Pareto-front").add(A, label="Result").show()

# pymoo.factory不包含类，下面相当于导入了一个方法
from pymoo.factory import get_performance_indicator

# IGD

igd = get_performance_indicator("igd", pf)
print("IGD", igd.calc(A))

# IGD+

igd_plus = get_performance_indicator("igd+", pf)
print("IGD+", igd_plus.calc(A))

# HV

hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
print("hv", hv.calc(A))
