'''
PART1:负责将第一阶段的suboptimal solution转化为optimal solution, MIN-COV
PART2:读取PF
PART3"去重，只执行一次
PART4:Indicator of traditional comparison algorithm calculation
PART5:Indicator of PF calculation
PART6:Indicator of ORIGAMIG algorithm calculation
PART7:Indicator of ORIGAMIG without MINCOV algorithm calculation
PART8:Indicator of ORIGAMIG without BSM algorithm calculation
PART9:Indicator of ORIGAMIG without BSM algorithm calculation
PART10:Indicator of Sensitivity calculation
PART11:Indicator of NaiveEA calculation
PART12:合并json for ORIGAMIG
PART13:合并json for TraditionalComparisonAlg_2
PART14:the mean and sdt calculation
'''

from multiprocessing.sharedctypes import Value
import os
from pickle import TRUE
from re import T
import gc


import numpy as np
import math


from typing import *
import sys

# sys.path.append('/home/wuyp/Projects/pymoo/')
sys.path.append('./')

from pymoo.MOSGsGeneticSolver.performance import Performance
from pymoo.MOSGsGeneticSolver.visualization import Visualization
from pymoo.model.result import Result
import pymoo.util.nds.non_dominated_sorting
from pymoo.MOSGsGeneticSolver.performance import Performance


import tool.algorithm
from pymoo.integercode.truing import Truing
# from pymoo.floatscoringmechanism.truing import Truing
# from pymoo.MOSGsGeneticSolver.truing import Truing

# TODO = ['obj3target600']
# TODO = ['obj3target800']
# TODO = ['obj4target25', 'obj4target50', 'obj4target75', 'obj4target100', 'obj4target200']
TODO = ['obj3target1000',]
# DONE = ['obj3target25', 'obj3target50', 'obj3target75', 'obj3target100', 'obj3target200', 'obj3target400', 'obj3target600', 'obj3target800', 'obj3target1000', 
#     'obj4target25', 'obj4target50', 'obj4target75', 'obj4target100', 'obj4target200', 'obj4target400',
#     'obj7target25', 'obj7target50', 'obj8target25']

'''负责将第一阶段的suboptimal solution转化为optimal solution, MIN-COV'''
# suboptimal_path = 'GeneticMOSGinteger-res_dir-GFIndependentRepeatExperiments_SEED0-30.json'
# suboptimal_path = 'GeneticMOSGinteger-res_dir-IndependentRepeatExperiments_SEED0-30.json'
# suboptimal_path = 'GeneticMOSGinteger-res_dir-N=4GFIndependentRepeatExperiments_SEED0-30.json'
# suboptimal_path = tool.algorithm.loadVariJson(path=suboptimal_path)
# res_path = 'GeneticMOSGinteger-res_dir-optimalAfterMINCOV_obj3target600.json'
# res_path = 'GeneticMOSGinteger-res_dir-optimalAfterMINCOV_obj3target800.json'
# res_path = 'GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json_obj4target400'
# res_path = 'GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json_obj3target25-200'
# res_path = 'GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json'
# seed_threshold = 2
# seed_threshold = 4
# seed_threshold = 6
# # seed_threshold = 8
# # seed_threshold = 10
# # seed_threshold = 12
# # seed_threshold = 15
# # seed_threshold = 16
# # seed_threshold = 19
# # seed_threshold = 20
# # seed_threshold = 23
# # seed_threshold = 24
# # seed_threshold = 27
# seed_threshold = 28

# # res_path = 'GeneticMOSGinteger-N=4res_dir-optimalAfterMINCOV.json'
# for para, path_list in suboptimal_path.items():
#     if para not in TODO:
#     # if para in DONE:
#         continue
#     res = {}
#     for path_i in path_list:
#         SEED = int(path_i.split('/')[-3].split('D')[-1])
#         # NOTE 如果有部分SEED数据已处理，则跳过
#         if SEED >= seed_threshold+2 or SEED < seed_threshold:
#         # if False:
#             print(SEED, ' continue~!')
#             continue
#         else:
#             r:Tuple[Result, np.ndarray] = tool.algorithm.loadVariPickle(path_i)
#             truing = Truing(res=r)
#             truing.mosgSearch_pop()
#             r = truing.fit_pf
#             r = np.unique(r, axis=0)
#             fname = os.path.join('Results', 'optimalAfterMINCOV', path_i.split('/')[-1])
#             tool.algorithm.dumpVariPickle(vari=r, name=fname)
#             res[path_i] = fname
#     if os.path.exists(res_path):
#         RES = tool.algorithm.loadVariJson(res_path)
#     else:
#         RES = {}
#     for k,v in res.items():
#         RES[k] = v
#     tool.algorithm.dumpVariJson(vari=RES, name=res_path)

'''PART2:读取PF'''
# # NOTE 读取PF
path_pf = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/PF-final.json')
# TODO = ['obj7target25',] 
# TODO = ['obj7target50']
# TODO = ['obj8target25']
TODO = ['obj3target1000', 'obj3target800']
# # TODO = ['obj3target100', 'obj3target200', 'obj4target25', 'obj4target50', 'obj4target75', 'obj4target100', 'obj4target200']
# # TODO = ['obj3target100', 'obj3target200', 'obj3target400', 'obj3target600', 'obj3target800', 'obj3target1000', 'obj4target25', 'obj4target50']
# # TODO = ['obj5target100', 'obj6target25', 'obj6target50', 'obj6target75', 'obj6target100']
# TODO = ['obj4target400', 'obj3target400', 'obj3target600']

'''PART3"去重，只执行一次'''
# path_pf = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/PF-final.json')
# for k, v in path_pf.items():
#     r = tool.algorithm.loadVariPickle(path=v)
#     r = np.unique(r, axis=0)
#     tool.algorithm.dumpVariPickle(vari=r, name=v)


'''PART4:Indicator of traditional comparison algorithm calculation'''
# path:Dict = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/PF-Comparison.json')
# type = 'N=4TraditionalComparisonAlg'
# # TODO = ['obj3target600', 'obj3target800', 'obj3target1000', 'obj4target25', 'obj4target50', 'obj4target75', 'objt4arget100',
# #     'obj4target200', 'obj4target400', 'obj5target100', 'obj5target200', 'obj6target25', 'obj6target50', 'obj6target75',
# #     'obj6target100', 'obj7target25', 'obj7target50', 'obj8target25',
# # ]
# seed_threshold = 'DIRECTMINCOV'
# for name, path_dict in path.items():
#     # if name != seed_threshold:
#     #     print(name, ' Done~!')
#     #     continue
#     for para, path_i in path_dict.items():
#         if para in TODO:
#             pf = tool.algorithm.loadVariPickle(path_pf[para])
#             pf0 = tool.algorithm.loadVariPickle(path_i).F
#             fname0 = name
#             para_dir = os.path.join('Results', 'pf')
#             Truing(para_dir=para_dir).mulResCompare(pf_total=[pf, pf0],
#             name_total=['pf', fname0],
#             indicator_igdplus=True, indicator_hv=True,
#             dump=True, file_exist=True, fname= para, repeat=name, para_dir=os.path.join('Results', 'IndependentRepeat', 'res', type),
#             ) 
#     else:
#         pass

'''PART5:Indicator of PF calculation'''
# path:Dict = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/PF-final.json')
# type = 'PF'
# # TODO = ['obj3target600', 'obj3target800', 'obj3target1000', 'obj4target25', 'obj4target50', 'obj4target75', 'objt4arget100',
# #     'obj4target200', 'obj4target400', 'obj5target100', 'obj5target200', 'obj6target25', 'obj6target50', 'obj6target75',
# #     'obj6target100', 'obj7target25', 'obj7target50', 'obj8target25',]
# # TODO = ['obj7target50',]
# for para, path_i in path.items():
#     if True:
#         if para in TODO:
#             pf = tool.algorithm.loadVariPickle(path_pf[para])
#             pf0 = tool.algorithm.loadVariPickle(path_i)
#             fname0 = 'PF'
#             para_dir = os.path.join('Results', 'pf')
#             Truing(para_dir=para_dir).mulResCompare(pf_total=[pf, pf0],
#             name_total=['pf', fname0],
#             indicator_igdplus=True, indicator_hv=True,
#             dump=True, file_exist=True, fname= para, repeat=fname0, para_dir=os.path.join('Results', 'IndependentRepeat', 'res', type),
#             ) 
#     else:
#         pass

'''PART6:Indicator of ORIGAMIG algorithm calculation'''
# path:Dict = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-IndependentRepeatExperiments_SEED0-30.json')
# path:Dict = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-N=4GFIndependentRepeatExperiments_SEED0-30.json')
# path_image:Dict[str, str] = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json')
# path_lost = {}
# seed_threshold = range(0, 4)
# for para, path_list in path.items():
#     if para in TODO:
#         pf = tool.algorithm.loadVariPickle(path_pf[para])
#         for i, path_i in enumerate(path_list):
#             # NOTE 假设已经经过Refinement
#             if i in seed_threshold:
#                 continue
#             if path_i not in path_image:
#                 fname = path_i.split('/')[-1]
#                 path_image[path_i] = os.path.json('Results/optimalAfterMINCOV', fname)
#                 path_lost[path_i] = os.path.json('Results/optimalAfterMINCOV', fname)
#             pf0 = tool.algorithm.loadVariPickle(path_image[path_i])
#             # NOTE 假设没经过Refinement
#             # res0 = tool.algorithm.loadVariPickle(path_i)
#             # model1 = Truing(res=res0)
#             # model1.mosgSearch_pop()
#             # pf0 = model1.fit_pf
#             fname0 = path_i
#             para_dir = os.path.join('Results', 'pf')
#             Truing(para_dir=para_dir).mulResCompare(pf_total=[pf, pf0],
#             name_total=['pf', fname0],
#             indicator_igdplus=True, indicator_hv=True,
#             dump=True, file_exist=True, fname= para, repeat=str(i), para_dir=os.path.join('Results', 'IndependentRepeat', 'res', 'ORIGAMIG'),
#             )
#     else:
#         pass
# path_image:Dict[str, str] = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json')
# for k,v in path_lost.items():
#     path_image[k] = v
# tool.algorithm.dumpVariJson(vari=path_image, name='GeneticMOSGinteger-res_dir-optimalAfterMINCOV.json')

'''PART7:Indicator of ORIGAMIG without MINCOV algorithm calculation'''
# type = 'ORIGAMIGwithoutMINCOV'
# path:Dict = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-IndependentRepeatExperiments_SEED0-30.json')
# for para, path_list in path.items():
#     if para in path_pf:
#         pf = tool.algorithm.loadVariPickle(path_pf[para])
#         for i, path_i in enumerate(path_list):
#             # 检查是否已经做过
#             check_path = os.path.join('Results', 'IndependentRepeat', 'res', type, para)
#             if os.path.exists(check_path):
#                 Flag_continue = False
#                 check:Dict[str, Dict] = tool.algorithm.loadVariJson(check_path)
#                 for para_i in check.values():
#                     for name in para_i.keys():
#                         if name == path_i:
#                             Flag_continue = True
#                             break
#                     if Flag_continue:
#                         break
#                 if Flag_continue:
#                     continue
#             # 如果没完成，则读取数据计算指标
#             pf0 = tool.algorithm.loadVariPickle(path_i).pop.get('F')
#             fname0 = path_i
#             para_dir = os.path.join('Results', 'pf')
#             Truing(para_dir=para_dir).mulResCompare(pf_total=[pf, pf0],
#                 name_total=['pf', fname0],
#                 indicator_igdplus=True, indicator_hv=True,
#                 dump=True, file_exist=True, fname= para, repeat=i+100, para_dir=os.path.join('Results', 'IndependentRepeat', 'res', type),
#                 ) 
#     else:
#         pass

'''PART8:Indicator of ORIGAMIG without BSM algorithm calculation'''
# path:Dict = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-AblationStudy_SEED0-30.json')
# for para, path_list in path.items():
#     if para in path_pf:
#         pf = tool.algorithm.loadVariPickle(path_pf[para])
#         for i, path_i in enumerate(path_list):
#             res0 = tool.algorithm.loadVariPickle(path_i)
#             model1 = Truing(res=res0)
#             model1.mosgSearch_pop()
#             pf0 = model1.fit_pf
#             fname0 = path_i
#             para_dir = os.path.join('Results', 'pf')
#             Truing(para_dir=para_dir).mulResCompare(pf_total=[pf, pf0],
#             name_total=['pf', fname0],
#             indicator_igdplus=True, indicator_hv=True,
#             dump=True, file_exist=True, fname= para, repeat=str(i), para_dir=os.path.join('Results', 'IndependentRepeat', 'res', 'ORIGAMIwithoutBSM'),
#             ) 
#     else:
#         pass

'''PART9:Indicator of ORIGAMIG without BSM algorithm calculation'''
# type = 'Icoding'
# path:Dict = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-AblationStudy_SEED0-30.json')
# for para, path_list in path.items():
#     if para in path_pf:
#         pf = tool.algorithm.loadVariPickle(path_pf[para])
#         for i, path_i in enumerate(path_list):
#             pf0 = tool.algorithm.loadVariPickle(path_i).pop.get('F')
#             fname0 = path_i
#             para_dir = os.path.join('Results', 'pf')
#             Truing(para_dir=para_dir).mulResCompare(pf_total=[pf, pf0],
#             name_total=['pf', fname0],
#             indicator_igdplus=True, indicator_hv=True,
#             dump=True, file_exist=True, fname= para, repeat=str(i), para_dir=os.path.join('Results', 'IndependentRepeat', 'res', 'Icoding'),
#             ) 
#     else:
#         pass

'''PART10:Indicator of Sensitivity calculation'''
# path:Dict = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-SensitivityStudy_SEED0-30.json')
# TODO = ['obj5target100',]
# for para, path_list in path.items():
#     if para in path_pf and para in TODO:
#         pf = tool.algorithm.loadVariPickle(path_pf[para])
#         for i, path_i in enumerate(path_list):
#             res0 = tool.algorithm.loadVariPickle(path_i)
#             model1 = Truing(res=res0)
#             model1.mosgSearch_pop()
#             pf0 = model1.fit_pf
#             fname0 = path_i
#             para_dir = os.path.join('Results', 'pf')
#             Truing(para_dir=para_dir).mulResCompare(pf_total=[pf, pf0],
#             name_total=['pf', fname0],
#             indicator_igdplus=True, indicator_hv=True,
#             dump=True, file_exist=True, fname= para, repeat=str(i), para_dir=os.path.join('Results', 'IndependentRepeat', 'res', 'SensitivityN=5'),
#             ) 
#     else:
#         pass
#     print('finish part1')

'''PART11:Indicator of NaiveEA calculation'''
# type = 'NaiveEA'
# path:Dict = tool.algorithm.loadVariJson('GeneticMOSGinteger-res_dir-AblationNaiveEA_SEED0-30.json')
# for para, path_list in path.items():
#     if para in path_pf:
#         pf = tool.algorithm.loadVariPickle(path_pf[para])
#         for i, path_i in enumerate(path_list):
#             # 检查是否已经做过
#             check_path = os.path.join('Results', 'IndependentRepeat', 'res', type, para)
#             if os.path.exists(check_path):
#                 Flag_continue = False
#                 check:Dict[str, Dict] = tool.algorithm.loadVariJson(check_path)
#                 for para_i in check.values():
#                     for name in para_i.keys():
#                         if name == path_i:
#                             Flag_continue = True
#                             break
#                     if Flag_continue:
#                         break
#                 if Flag_continue:
#                     continue
#             # 如果没完成，则读取数据计算指标
#             pf0 = tool.algorithm.loadVariPickle(path_i).pop.get('F')
#             fname0 = path_i
#             para_dir = os.path.join('Results', 'pf')
#             Truing(para_dir=para_dir).mulResCompare(pf_total=[pf, pf0],
#                 name_total=['pf', fname0],
#                 indicator_igdplus=True, indicator_hv=True,
#                 dump=True, file_exist=True, fname= para, repeat=i+100, para_dir=os.path.join('Results', 'IndependentRepeat', 'res', type),
#                 ) 
#     else:
#         pass
# # NOTE PF
# type = 'PF'
# path:Dict = tool.algorithm.loadVariJson('Results/pf/PF')
# for path_list in path.values():
#     for para, path in path_list.items():
#         pf = tool.algorithm.loadVariPickle(path_pf[para])
#         if True:
#             # 如果没完成，则读取数据计算指标
#             pf0 = tool.algorithm.loadVariPickle(path)
#             fname0 = 'pf'
#             para_dir = os.path.join('Results', 'pf')
#             Truing(para_dir=para_dir).mulResCompare(pf_total=[pf, pf0],
#                 name_total=['pf', fname0],
#                 indicator_igdplus=True, indicator_hv=True,
#                 dump=True, fname= para, para_dir=os.path.join('Results', 'IndependentRepeat', 'res', type),
#                 ) 
#     else:
#         pass

'''PART12:合并json for ORIGAMIG'''
# type = 'ORIGAMIG_2'
# RES_DIR = 'Results/IndependentRepeat/res'
# res = {}
# res['obj3target25'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj3target25')
# res['obj3target50'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj3target50')
# res['obj3target75'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj3target75')
# res['obj3target100'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj3target100')
# res['obj3target200'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj3target200')
# res['obj3target400'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj3target400')
# res['obj3target600'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj3target600')
# res['obj3target800'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj3target800')
# res['obj3target1000'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj3target1000')
# res['obj4target25'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj4target25')
# res['obj4target50'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj4target50')
# res['obj4target75'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj4target75')
# res['obj4target100'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj4target100')
# res['obj4target200'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj4target200')
# res['obj4target400'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj4target400')
# res['obj5target100'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj5target100')
# res['obj6target25'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj6target25')
# res['obj6target50'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj6target50')
# res['obj6target75'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj6target75')
# res['obj6target100'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj6target100')
# res['obj7target25'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj7target25')
# res['obj7target50'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj7target50')
# res['obj8target25'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj8target25')
# tool.algorithm.dumpVariJson(vari=res, path=os.path.join(RES_DIR), name=type+'.json')

'''PART13:合并json for TraditionalComparisonAlg_2'''
# type = 'TraditionalComparisonAlg_2'
# RES_DIR = 'Results/IndependentRepeat/res'
# res = {}
# res['obj3target25'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj3target25')
# res['obj3target50'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj3target50')
# res['obj3target75'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj3target75')
# res['obj3target100'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj3target100')
# res['obj3target200'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj3target200')
# res['obj3target400'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj3target400')
# res['obj3target600'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj3target600')
# res['obj3target800'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj3target800')
# res['obj3target1000'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj3target1000')
# res['obj4target25'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj4target25')
# res['obj4target50'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj4target50')
# res['obj4target75'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj4target75')
# res['obj4target100'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj4target100')
# res['obj4target200'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj4target200')
# res['obj4target400'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj4target400')
# res['obj5target100'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj5target100')
# res['obj6target25'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj6target25')
# res['obj6target50'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj6target50')
# res['obj6target75'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj6target75')
# res['obj6target100'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj6target100')
# res['obj7target25'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj7target25')
# res['obj7target50'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj7target50')
# res['obj8target25'] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/N=4TraditionalComparisonAlg/obj8target25')
# tool.algorithm.dumpVariJson(vari=res, path=os.path.join(RES_DIR), name=type+'.json')



'''PART14:the mean and sdt calculation'''
# # NOTE ORIGAMIG
# print('ORIGAMIG========================')
# NOTE 读所有问题规模合并在一个json中的汇总文件
# path:Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG_2.json')
# res_HV:Dict[str, List[float]] = {}  # dict用于存放不同方法，每个方法的value是30个seed下的indicator结果
# res_IGDPlus:Dict[str, List[float]] = {}
# for para, indicators in path.items():
#     for indicator in indicators.values():
#         for path_i, indicator_i in indicator.items():
#             # NOTE 物理地址即path_i
#             # NOTE TODO 接下来用字符串函数提取较短的名字，然后统计结果，存成Dict[name_short, List[Indicator]]
#             if para not in res_HV:
#                 res_HV[para] = []
#             if para not in res_IGDPlus:
#                 res_IGDPlus[para] = []
#             res_HV[para].append(indicator_i['hv'])
#             res_IGDPlus[para].append(indicator_i['igd+'])
#     print(para+':HV:'+str(np.mean(res_HV[para]))+'+-'+str(np.std(res_HV[para])))
#     print(para+':IGD+:'+str(np.mean(res_IGDPlus[para]))+'+-'+str(np.std(res_IGDPlus[para])))
# NOTE 只读一个问题的分文件
print('ORIGAMIG for specific scale==============')
path:Dict[str, Dict[str, Dict[str, List[float]]]] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIG/obj7target25')
res_HV:Dict[str, List[float]] = {}  # dict用于存放不同方法，每个方法的value是30个seed下的indicator结果
res_IGDPlus:Dict[str, List[float]] = {}
for para, indicators in path.items():
    para = para.split('_')[0]
    for path_i, indicator_i in indicators.items():
        # NOTE 物理地址即path_i
        # NOTE TODO 接下来用字符串函数提取较短的名字，然后统计结果，存成Dict[name_short, List[Indicator]]
        if para not in res_HV:
            res_HV[para] = []
        if para not in res_IGDPlus:
            res_IGDPlus[para] = []
        res_HV[para].append(indicator_i['hv'])
        res_IGDPlus[para].append(indicator_i['igd+'])
print(para+':HV:'+str(np.mean(res_HV[para]))+'+-'+str(np.std(res_HV[para])))
print(para+':IGD+:'+str(np.mean(res_IGDPlus[para]))+'+-'+str(np.std(res_IGDPlus[para])))
# # NOTE ORIGAMIGwithoutMINCOV
# print('ORIGAMIGwithoutMINCOV========================')
# path:Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIGwithoutMINCOV.json')
# res_HV:Dict[str, List[float]] = {}  # dict用于存放不同方法，每个方法的value是30个seed下的indicator结果
# res_IGDPlus:Dict[str, List[float]] = {}
# for para, indicators in path.items():
#     for indicator in indicators.values():
#         for path_i, indicator_i in indicator.items():
#             # NOTE 物理地址即path_i
#             # NOTE TODO 接下来用字符串函数提取较短的名字，然后统计结果，存成Dict[name_short, List[Indicator]
#             if para not in res_HV:
#                 res_HV[para] = []
#             if para not in res_IGDPlus:
#                 res_IGDPlus[para] = []
#             res_HV[para].append(indicator_i['hv'])
#             res_IGDPlus[para].append(indicator_i['igd+'])
#     print(para+':HV:'+str(np.mean(res_HV[para]))+'+-'+str(np.std(res_HV[para])))
#     print(para+':IGD+:'+str(np.mean(res_IGDPlus[para]))+'+-'+str(np.std(res_IGDPlus[para])))
# # NOTE ORIGAMIGwithoutBSM
# print('ORIGAMIGwithoutBSM========================')
# path:Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/ORIGAMIGwithoutBSM.json')
# res_HV:Dict[str, List[float]] = {}  # dict用于存放不同方法，每个方法的value是30个seed下的indicator结果
# res_IGDPlus:Dict[str, List[float]] = {}
# for para, indicators in path.items():
#     for indicator in indicators.values():
#         for path_i, indicator_i in indicator.items():
#             # NOTE 物理地址即path_i
#             # NOTE TODO 接下来用字符串函数提取较短的名字，然后统计结果，存成Dict[name_short, List[Indicator]]
#             if para not in res_HV:
#                 res_HV[para] = []
#             if para not in res_IGDPlus:
#                 res_IGDPlus[para] = []
#             res_HV[para].append(indicator_i['hv'])
#             res_IGDPlus[para].append(indicator_i['igd+'])
#     print(para+':HV:'+str(np.mean(res_HV[para]))+'+-'+str(np.std(res_HV[para])))
#     print(para+':IGD+:'+str(np.mean(res_IGDPlus[para]))+'+-'+str(np.std(res_IGDPlus[para])))
# NOTE ORIGAMIGwithoutBSM
# print('Icoding========================')
# path:Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/Icoding.json')
# res_HV:Dict[str, List[float]] = {}  # dict用于存放不同方法，每个方法的value是30个seed下的indicator结果
# res_IGDPlus:Dict[str, List[float]] = {}
# for para, indicators in path.items():
#     for indicator in indicators.values():
#         for path_i, indicator_i in indicator.items():
#             # NOTE 物理地址即path_i
#             # NOTE TODO 接下来用字符串函数提取较短的名字，然后统计结果，存成Dict[name_short, List[Indicator]]
#             if para not in res_HV:
#                 res_HV[para] = []
#             if para not in res_IGDPlus:
#                 res_IGDPlus[para] = []
#             res_HV[para].append(indicator_i['hv'])
#             res_IGDPlus[para].append(indicator_i['igd+'])
#     print(para+':HV:'+str(np.mean(res_HV[para]))+'+-'+str(np.std(res_HV[para])))
#     print(para+':IGD+:'+str(np.mean(res_IGDPlus[para]))+'+-'+str(np.std(res_IGDPlus[para])))
# # NOTE SensitivityStudy
# print('SensitivityN=5========================')
# path:Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/SensitivityN=5.json')
# res_HV:Dict[str, List[float]] = {}  # dict用于存放不同方法，每个方法的value是30个seed下的indicator结果
# res_IGDPlus:Dict[str, List[float]] = {}
# TODO = ['obj5target100']
# for para, indicators in path.items(): # "obj5target25":{}
#     if para not in TODO:
#         continue
#     for indicator in indicators.values(): # 'obj5target25_0':{}
#         for path_i, indicator_i in indicator.items(): # "path":{str, List[float]}
#             # NOTE 物理地址即path_i
#             # NOTE TODO 接下来用字符串函数提取较短的名字，然后统计结果，存成Dict[name_short, List[Indicator]]
#             # NOTE 一般情况下直接将para(e.g., obj5target25)作为标识符，特殊情况下需要自定义标识符
#             popsize = path_i.split('/')[-1].split('-')[-3]
#             tag = para + ':' + popsize
#             if tag not in res_HV:
#                 res_HV[tag] = []
#             if tag not in res_IGDPlus:
#                 res_IGDPlus[tag] = []
#             res_HV[tag].append(indicator_i['hv'])
#             res_IGDPlus[tag].append(indicator_i['igd+'])
# for para in res_HV.keys():
#     print(para+':HV:'+str(np.mean(res_HV[para]))+'+-'+str(np.std(res_HV[para])))
#     print(para+':IGD+:'+str(np.mean(res_IGDPlus[para]))+'+-'+str(np.std(res_IGDPlus[para])))
# # NOTE NaiveEA
# print('NaiveEA========================')
# path:Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = tool.algorithm.loadVariJson('Results/IndependentRepeat/res/NaiveEA.json')
# res_HV:Dict[str, List[float]] = {}  # dict用于存放不同方法，每个方法的value是30个seed下的indicator结果
# res_IGDPlus:Dict[str, List[float]] = {}
# for para, indicators in path.items():
#     for indicator in indicators.values():
#         for path_i, indicator_i in indicator.items():
#             # NOTE 物理地址即path_i
#             # NOTE TODO 接下来用字符串函数提取较短的名字，然后统计结果，存成Dict[name_short, List[Indicator]]
#             if para not in res_HV:
#                 res_HV[para] = []
#             if para not in res_IGDPlus:
#                 res_IGDPlus[para] = []
#             res_HV[para].append(indicator_i['hv'])
#             res_IGDPlus[para].append(indicator_i['igd+'])
#     print(para+':HV:'+str(np.mean(res_HV[para]))+'+-'+str(np.std(res_HV[para])))
#     print(para+':IGD+:'+str(np.mean(res_IGDPlus[para]))+'+-'+str(np.std(res_IGDPlus[para])))
# # NOTE 对比方法或者PF的计算 直接见.json结果
print("Main_finish~@!")