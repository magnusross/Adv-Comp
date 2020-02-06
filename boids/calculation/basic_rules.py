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
def get_co(x_all):
    N = len(x_all)
    co = np.zeros_like(x_all[0])
    for i in range(N):
        co += x_all[i]
    return co/N

@numba.njit()
def rule_avoid(i, pos_all, radius=30, factor=0.002):
    
    size = len(pos_all)
    pos_all = pos_all.astype(np.float64)
    p = np.zeros_like(pos_all[0])

    rel_pos = pos_all - pos_all[i]

    for j in prange(size):
        dist = np.power(rel_pos[j], 2)  
        if i != j and np.sqrt(np.sum(dist))< radius:  
            for k in range(len(p)):
                p[k] -= rel_pos[j][k]
              
    return p * factor 


@numba.njit()
def rule_com(i, pos_all, radius=30, factor=0.01):
    
    size = len(pos_all)
    pos_all = pos_all.astype(np.float64)
    p = np.zeros_like(pos_all[0])
    N = 0
    rel_pos = pos_all - pos_all[i]

    for j in prange(size):
        dist = np.power(rel_pos[j], 2)  
        if i != j and np.sqrt(np.sum(dist))< radius:  
            N += 1 
            for k in range(len(p)):
                p[k] += pos_all[j][k]
    if N == 0:
        return p
    else:
        return ((p/N) - pos_all[i]) * factor 

@numba.njit()
def rule_match(i, pos_all, vel_all, radius=30, factor=0.01):
    
    size = len(vel_all)
    vel_all = vel_all.astype(np.float64)
    p = np.zeros_like(vel_all[0])
    N = 0
    rel_pos = pos_all - pos_all[i]

    for j in prange(size):
        dist = np.power(rel_pos[j], 2)  
        if i != j and np.sqrt(np.sum(dist))< radius:  
            N += 1 
            for k in range(len(p)):
                p[k] += vel_all[j][k]
    if N == 0:
        return p 
    else:
        return (-vel_all[i] + p/N ) * factor

@numba.njit()
def rule_wall_avoid(i, pos_all, pos_obj, radius=30, factor=100):
    '''
    Not working properly 
    '''
    size = len(pos_all)
    pos_all = pos_all.astype(np.float64)
    p = np.zeros_like(pos_all[0])
    
    rel_pos = pos_obj - pos_all[i]
    for j in prange(size):
        dist = np.power(rel_pos[j], 2) 
        if i != j and np.sqrt(np.sum(dist)) < radius:  
            for k in range(len(p)):
                p[k] -= 1/rel_pos[j][k]**100
                print(p[k])
    return p * factor

@numba.njit()
def rule_wall_bounce_2D(i, pos_all, vel_all, width=800., hieght=800.):
    assert pos_all.shape[1] == 2         
    if pos_all[i][0] > width/2 or pos_all[i][0] < -1. * width/2:
        vel_all[i][0] = -1. * vel_all[i][1]
    if pos_all[i][1] > hieght/2 or pos_all[i][1] < -1. * hieght/2:
        vel_all[i][1] = -1. * vel_all[i][1]

    




