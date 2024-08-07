import numpy as np

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from problem_formulation_Icode import SGs1 as SGs_Icode
from problem_formulation_Ccode import SGs1 as SGs_Ccode
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

from tools.algorithms import *

config_file_path = "./config_flexibility.yaml"
config_obj = load_yaml_to_dict(config_file_path)
for param, value in config_obj.items():
    exprs = "{param} = {config_obj}.get('{param}')".format(param=param, config_obj=config_obj)
    exec(exprs)

method = '_'.join(['NSGAIII', code_type])
PROBLEM = {'integer':SGs_Icode(player_num=obj_n+1, target_num=target_n), 'float': SGs_Ccode(player_num=obj_n+1, target_num=target_n)}
# SAMPLING CROSSOVER MUTATION operator
SAMPLING = {'integer': IntegerRandomSampling(), 'float': FloatRandomSampling()}
CROSSOVER = {'integer': SBX(repair=RoundingRepair()), 'float': SBX()}
MUTATION = {'integer': PM(repair=RoundingRepair()), 'float': PM()}



for SEED in range(repetition):
    # Define reference points
    ref_dirs = get_reference_directions("energy", obj_n, ref_num, seed=SEED)
    # SAMPLING CROSSOVER MUTATION operator
    sampling, crossover, mutation = SAMPLING[code_type], CROSSOVER[code_type], MUTATION[code_type]
    problem = PROBLEM[code_type]

    # Get Algorithm
    algorithm = NSGA3(pop_size=pop_size,
                  ref_dirs=ref_dirs,
                  sampling=sampling, crossover=crossover, mutation=mutation,
                  eliminate_duplicates=True
    )

    res = minimize(problem,
                algorithm=algorithm,
                termination=('n_gen', n_gen),
                seed=SEED,
                verbose=False)

    writein_path = os.path.join(writein_dir, f'SEED{SEED}', f'obj{obj_n}target{target_n}')
    if not os.path.exists(writein_path): os.makedirs(writein_path)
    dumpVariPickle(res.F, os.path.join(writein_path, f'{method}.txt'))