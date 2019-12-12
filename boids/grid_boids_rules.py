from sys import platform
import os 
if platform == 'linux':

    print('Setting environment variables')
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1"
    
import numpy as np
import numba 
from numba import prange

@numba.njit()
def rule_avoid(one_my_pos, pos_all, radius=30, factor=0.000012):
    
    size = len(pos_all)
    p = np.zeros_like(one_my_pos)
    rel_pos = pos_all - one_my_pos

    for j in range(size):
        dist = np.power(rel_pos[j], 2)  
        if np.all(one_my_pos != pos_all[j]) and np.sqrt(np.sum(dist))< radius:  
            for k in range(len(p)):
                p[k] -= rel_pos[j][k]
              
    return p * factor 


@numba.njit()
def rule_com(one_my_pos, pos_all, radius=30, factor=0.00004):
    
    size = len(pos_all)
    p = np.zeros_like(one_my_pos)
    N = 0
    rel_pos = pos_all - one_my_pos

    for j in range(size):
        dist = np.power(rel_pos[j], 2)  
        if np.all(one_my_pos != pos_all[j]) and np.sqrt(np.sum(dist))< radius:  
            N += 1 
            for k in range(len(p)):
                p[k] += pos_all[j][k]
    if N == 0:
        return p
    else:
        return ((p/N) - one_my_pos) * factor 

@numba.njit()
def rule_match(one_my_pos, one_my_vel, pos_all, vel_all, radius=30, factor=0.00001):
    
    size = len(vel_all)
    p = np.zeros_like(one_my_pos)
    N = 0
    rel_pos = pos_all - one_my_pos

    for j in range(size):
        dist = np.power(rel_pos[j], 2)  
        if np.all(one_my_pos != pos_all[j]) and np.sqrt(np.sum(dist))< radius:  
            N += 1 
            for k in range(len(p)):
                p[k] += vel_all[j][k]
    if N == 0:
        return p 
    else:
        return (-one_my_vel + p/N ) * factor

# @numba.njit()
def rule_wrap(my_pos, box_size):
    my_pos %= box_size
    
