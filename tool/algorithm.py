import datetime
import functools


import numpy as np
from functools import reduce
from typing import *


import pickle
import json
import os

# from torch import isin


def paras2Str(para:Dict[str,Union[float,int]], connecter:str=''):
    para_str = []
    for key, val in para.items():
        para_str.append(key)
        if val is not None:
            para_str.append(str(val))
    return connecter.join(para_str)


'''基本数据结构模块，含比较 数据类型转换等'''
# 处理不定个数参数：
# 将参数转化为 键值对
def addKeyValues(a,**kwargs):
    map = {}
    for arg,value in kwargs.items():
        map[arg] = value

# 比较vector是否小于dic中任意某个元素
def vLessD(vector, dic):
    # if vector greater than anyone in dic, return False; otherwise return True.
    for item in dic:
        item = np.array(list(item))
        if np.all(vector >= item):
            return False
    return True

# 比较vector1与vector2的大小关系
def vLessV(vector1, vector2):
    # if every element in vector1 less than vector2, return True. Otherwise return False.
    return np.all(vector1 < vector2)


# difference set; item in list1 but not in list2
def listWithoutList(list1, list2):
    res = []
    for item in list1:
        if item not in list2:
            res.append(item)
    return res

# 查询某个numpy是否在dic.key()中
def dicQuery(numpy, dic):
    # query numpy from dictionary
    query = tuple(numpy.tolist())
    if query in dic:
        return True
    return False

# 将numpy作为key存储到dict中，以便后续的query
def dicAppendNumpy(numpy, dic):
    # append numpy into dic.
    list = numpy.tolist()
    item = tuple(list)
    dic[item] = 1
    return dic

'''字符串处理模块'''
# 自动化命名，接受多个参数n,m，和公共前缀str，返回命名规则为‘str+num’，num从0到n*m-1
def rename(s, **kwargs):
    # paraMap = {}
    num = 1
    for arg, value in kwargs.items():
#       paraMap[arg] = value
        num *= value
    a = np.arange(num)
    res = list(map(str, a))
    res1 = [s] * num
    res = list(map(lambda a, b: a + b, res1, res))
    return res

# 将字符串转为数组，排序并输出
def v2rank(vari:Union[List[float], str], round=None):
    # 形如[1,2,3,4]
    if isinstance(vari, List):
        if round is None:
            round = 0
        vari = np.array(vari)
        return rank(vari=vari, descending=True, round_num=round)

    # 形如vari = '0.5 0.52 0.57 1.88 0.66'
    if isinstance(vari, str):
        if round is None:
            round = 2
        vari = vari.split(' ')
        vari = [float(v) for v in vari]
        vari = np.array(vari)
        return rank(vari=vari, descending=False, round_num=round)
def excelsplit(s:str):
    vari = s.split()
    return [float(v) for v in vari]

# 从文件路径中读取目录中的参数配置信息，返回Dict
# 例如从'Results/obj5target75/GeneticMOSGinteger-Popsize400Codeintegergen300-2022_06_21_09_26-467197.txt'返回{'obj':5,'target':75}
def path2para(path:str, para_key:List[str])->Dict[str,Union[float, str]]:
    return ;


'''和数学相关模块'''
# 归一化
def normalization(a:np.ndarray):
    # a: 默认Inpt是二维数组

    min = np.amin(a, aixs=0)
    max = np.amax(a, axis=0)
    return (a-min) / max

# 排序，接受1维数组
def rank(vari:np.ndarray, descending=False, round_num=2)->str:
    if descending:
        vari_temp = vari * -1
    else:
        vari_temp = vari
    sorted = np.argsort(vari_temp)
    rank = np.argsort(sorted)
    if round_num == 0:
        return list(map(lambda v, r: str(round(v))+'('+str(r+1)+')', vari, rank))
    return list(map(lambda v, r: str(round(v,round_num))+'('+str(r+1)+')', vari, rank))

'''多目标模块'''
# Compare函数见performance脚本
# def mulResCompare(pf_total:List[np.ndarray], len_total:List[int], name_total:List[str]):
#     pf_total:np.ndarray = np.vstack(pf_total)
#     print(Performance(pf_total, len_total, name_total))

# 演化算法停止指标选择


'''作图相关'''
def getMarker()->List[str]:
# ‘.’：点(point marker)
# ‘,’：像素点(pixel marker)
# ‘o’：圆形(circle marker)
# ‘v’：朝下三角形(triangle_down marker)
# ‘^’：朝上三角形(triangle_up marker)
# ‘<‘：朝左三角形(triangle_left marker)
# ‘>’：朝右三角形(triangle_right marker)
# ‘1’：(tri_down marker)
# ‘2’：(tri_up marker)
# ‘3’：(tri_left marker)
# ‘4’：(tri_right marker)
# ‘s’：正方形(square marker)
# ‘p’：五边星(pentagon marker)
# ‘*’：星型(star marker)
# ‘h’：1号六角形(hexagon1 marker)
# ‘H’：2号六角形(hexagon2 marker)
# ‘+’：+号标记(plus marker)
# ‘x’：x号标记(x marker)
# ‘D’：菱形(diamond marker)
# ‘d’：小型菱形(thin_diamond marker)
# ‘|’：垂直线形(vline marker)
# ‘_’：水平线形(hline marker)
    return ['^', 'P', 'o', 'D', '*', 'x', '.', ',', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', 'h', 'H', '+', '|', '_']

def getColor()->List[str]:
# 'b'          蓝色
# 'g'          绿色
# 'r'          红色
# 'c'          青色
# 'm'          品红
# 'y'          黄色
# 'k'          黑色
# 'w'          白色
    return ['b',
'g',
'r',
'c',
'm',
'y',
'k',
'w']

def getLine()->List[str]:
# ‘-‘：实线(solid line style)
# ‘–‘：虚线(dashed line style)
# ‘-.’：点划线(dash-dot line style)
# ‘:’：点线(dotted line style)
    return ['-', '–', '-.', ':']




'''内存持久化模块-文件处理模块'''
# 读写数据为保证路径的合法性有必要使用os package

# 写入数据时需要注意提供的路径的合法性
# 理想情况下，参数path是已经经过os package处理过的路径

def getTime(format=None):
    if format is None:
        return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    else:
        return datetime.datetime.now().strftime(format)
'''如果path包含完整路径，则name为空，如果name不为空，则说明path只包含文件夹路径'''
# NOTE Pickle
def dumpVariPickle(vari: Any, path:str=None, name:str=None):
    # kwargs.key 文件名, kwargs.value 变量内容
    # path.key 文件名, path.value 变量内容
    if path is None:
        path = os.getcwd()
        path = os.path.join(path, name)
    elif name is not None:
        path = os.path.join(path, name)
    with open(path, 'wb') as f:
        pickle.dump(vari, f)
        f.close()
    print('写入数据：', path)
def loadVariPickle(path:str) ->Any:
    # path.key = vari_name, path.value = vari
    print('入读数据：', path)
    with open(path, 'rb') as f:
        para = pickle.load(f)
        f.close()
        return para
# NOTE Json

def dumpVariJson(vari: Any, path:str=None, name:str=None, indent=4):
    # kwargs.key 文件名, kwargs.value 变量内容
    # path.key 文件名, path.value 变量内容
    if path is None:
        if not os.path.isabs(name):
            path = os.getcwd()
            path = os.path.join(path, name)
        else:
            path = name
    elif name is not None:
        if isinstance(path, list):
            path_i = os.getcwd()
            for dir in path:
                path_i = os.path.join(path_i, dir)
            path = path_i
        path = os.path.join(path, name)
    with open(path, 'w') as f:
        json.dump(vari, f, indent=indent)
        f.close()
    print('写入数据：', path)
def loadVariJson(path:str=None, name:str=None) ->Any:
    # path.key = vari_name, path.value = vari
    if path is None:
        path = name
    if isinstance(path, list):
        path_i = os.getcwd()
        for dir in path:
            path_i = os.path.join(path_i, dir)
        path = path_i
    print('入读数据：', path)
    with open(path, 'r') as f:
        data = f.read()
        para = json.loads(data)
        f.close()
        return para
'''判断文件路径是否存在，不存在则新建'''
def creatDir(dir:str)->None:
    if not os.path.exists(dir):
        os.mkdir(dir)

'''logs'''
def dumpLogs(PRINT:str=None):
    with open(os.path.join('./Results/logs', getTime('%Y_%m_%d')),'a') as file0:
        print(PRINT ,file=file0)

# def readJson(PATH:str)->List
