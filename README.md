# Code of Scaling Multi-Objective Security Games Provably via Space Discretization Based Evolutionary Search (SDES).


## The implementation of SDES and other comparison algorithms.

## Experiment environment.
The experimental environment is consistent with the first work of MOSG, including (i) the randomly-generated security games, i.e., $\mathcal{D}$'s and $\mathcal{A}$'s payoff obey the uniform integral distribution of [1, 10], while the defender and attacker payoffs obey the uniform integral distribution of [-10, -1], (ii) the problem scale is divided into three dimensions: the attacker number $N$, the target needed to be protected number $T$, and the resource ratio $r$, satisfying the following constraint: $N\geq3$, $T=[25, 50, 75, 100, 200, 400, 600, 800, 1000]$, $r=0.2$, the resource equals $r\cdot T$, 
where $r$ is generally much less than 1 due to the limited resources.
*Those setting can be found in pymoo.problems.securitygame.MOSG.py.*

## Implementation of All Algorithms.
This code finishes 7 algorithms, including one implementation of SDES framework based on NSGA-III, 4 algorithms based on $\epsilon$-constraint framework and two general many-objective EAs.
The $\epsilon$-constraint algorithms include ORIGAMI-M and ORIGAMI-A, ORIGAMI-M-BS and DIRECT-MIN-COV. All $\epsilon$-constraint algorithms use step $\epsilon=1$. ORIGAMI-A uses the threshold $\alpha=0.001$ for binary search termination. 
The general many-objective EAs include $c$-code EA with continuous solution representation, $I$-code EA with integral solution representation. Furthermore, in the large-scale scenario, even SOTA many-objective EAs cannot converge well due to Lemma 1 and 2. Therefore, we do not consider those algorithms for comparison.
*The implementation of SDES framework can be found in pymoo.integercode.SDES_NSGA-III.py. The results of SDES_NSGA-III.py have to go through the refinement component, cf. pymoo.integercode.truingAfterGenetic_repeat.py.*
*The implementation of 4 algorithms based on $\epsilon$-constraint framework can be found in MOSGs.epsilon_constraint.py.*
The implementation of $c$-code EA can be found in pymoo.integercode.c-code_EAs.py.
*The implementation of $I$-code EA need uses to tweak the code of pymoo.integercode.MOSG.py, changing the "type" parameter of MOSG.ConflictSolution() from "heuristic" to "random". Then, run pymoo.integercode.SDES_NSGA-III.py.*
*The implementation of NSGA-III is based on pymoo package.*

> Solution Initialization of the implementation of SDES.
As for the method perspective, the discretization component is the core of SDES.  As for the code perspective, the discretization component is very easy for implementation, just use the integeral initialization in the pymoo.integercode.SDES_NSGA-III.py, cf. "get_sampling('int_random')".

## Multi-objective Optimization Problem Indicator
The Pareto dominance relationship is the solution-level criterion, IGD+ and HV are algorithm-level criteria to determine which algorithm yields batter approximate solution set. As the most commonly used MaOPs performance indicators, HV measures the convergence and diversity of the PF, while IGD+ measures the distance from the reference PF to the PF approximation. 
*The implementation of IGD+ and HV is based on pymoo package.*

## Reference PF Construction
To find a well-convergent and well-spaced approximate PF, quantitative indicators HV and IGD+ are selected to evaluate convergence and diversity. However, IGD+ evaluation requires reference PF that the randomly-generated security games cannot provide. A feasible method to construct reference PF is to collect a set of PFs produced by all algorithms. Algorithms involved in the PF building include SDES and all $\epsilon$-constraint algorithms, including ORIGAMI-M, ORIGAMI-A, ORIGAMI-M-BS, DIRECT-MIN-COV.

### End
