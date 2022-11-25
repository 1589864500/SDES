from pymoo.algorithms.nsga2_2 import NSGA2 as algorithm1
from pymoo.algorithms.nsga2 import NSGA2 as algorithm2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

problem = get_problem("dtlz7")

algorithm = algorithm1(pop_size=100)

res1 = minimize(problem,
               algorithm,
               ('n_gen', 500),
               seed=1,
               verbose=True)

algorithm = algorithm2(pop_size=100)

res2 = minimize(problem,
               algorithm,
               ('n_gen', 500),
               seed=1,
               verbose=True)

plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res1.F, color="red")
plot.add(res2.F, color='blue')
plot.show()