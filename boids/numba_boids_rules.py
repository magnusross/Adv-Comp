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
def rule_avoid(i, pos_all, radius=10, factor=0.001):
    
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
def rule_com(i, pos_all, radius=100, factor=0.001):
    
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
def rule_match(i, pos_all, vel_all, radius=10, factor=0.05):
    
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
def rule_wall_avoid(i, pos_all, pos_obj, radius=100, factor=10):
    
    size = len(pos_all)
    pos_all = pos_all.astype(np.float64)
    p = np.zeros_like(pos_all[0])
    
    rel_pos = pos_obj - pos_all[i]

    for j in prange(size):
        dist = np.power(rel_pos[j], 2)  
        if i != j and np.sqrt(np.sum(dist)) < radius:  
            for k in range(len(p)):
                p[k] -= abs(rel_pos[j][k])
    
    return p * factor                


@numba.njit()
def update_boids(pos_all, vel_all,
                 objs=0, pos_obj=np.empty((1,1))):
    '''
    objs option for numba compile 
    '''
    if objs == 1:
        assert pos_all.shape[1] == pos_obj.shape[1]
    
    for i in prange(len(pos_all)):
        v = rule_com(i, pos_all)
        v += rule_avoid(i, pos_all)
        v += rule_match(i, pos_all, vel_all)
        if objs:
            v += rule_wall_avoid(i, pos_all, pos_obj)

        vel_all[i] += v 
        pos_all[i] += vel_all[i]

        
    return pos_all, vel_all  
    

