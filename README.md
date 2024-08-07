# Code of Scaling Multi-Objective Security Games Provably via Space Discretization Based Evolutionary Search (SDES).

## Declare
Thank you for your timely attention. Compared to the previous version in the `mian` branch of this repository, we have refactored the code to enhance its readability and extensibility. Moreover, we will supplement the MATLAB version of the program as soon as possible. At present, our project is ready for run, and the dependencies we use are very common as shown in `requirements.txt`. If you have any questions, you can submit the issue, and we will reply in the first time. 

## Experiment environment.
The experimental environment is consistent with the first work of MOSG, including (i) the randomly-generated security games, i.e., $\mathcal{D}$'s and $\mathcal{A}$'s payoff obey the uniform integral distribution of [1, 10], while the defender and attacker payoffs obey the uniform integral distribution of [-10, -1], (ii) the problem scale is divided into three dimensions: the attacker number $N$, the target needed to be protected number $T$, and the resource ratio $r$, satisfying the following constraint: $N\geq3$, $T=[25, 50, 75, 100, 200, 400, 600, 800, 1000]$, $r=0.2$, the resource equals $r\cdot T$, 
where $r$ is generally much less than 1 due to the limited resources.
*Those setting can be found in pymoo.problems.securitygame.MOSG.py in the 'main' branch of this repository.*
### End
