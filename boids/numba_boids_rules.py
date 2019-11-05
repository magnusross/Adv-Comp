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
def rule_com(i, pos_all, com, factor=0.0003):
    return (com - pos_all[i]) * factor

@numba.njit()
def rule_avoid(i, pos_all, radius=20, factor=0.001):
    
    size = len(pos_all)
    pos_all = pos_all.astype(np.float64)
    p = np.ones_like(pos_all[0])

    #for dealing with objects like walls the boids must avoid
 
    rel_pos = pos_all - pos_all[i]

    for j in prange(size):
        dist = np.power(rel_pos[j], 2)  
        if i != j and np.sqrt(np.sum(dist))< radius:  
            for k in range(len(p)):
                p[k] -= rel_pos[j][k]
              
    return p * factor 

@numba.njit()
def rule_match(i, vel_all, cov, factor=0.001):    
    return (vel_all[i] - cov) * factor

@numba.njit()
def update_boids(pos_all, vel_all):
    
    com = get_co(pos_all)
    cov = get_co(vel_all)

    for i in prange(len(pos_all)):

        v = rule_com(i, pos_all, com)
        v += rule_avoid(i, pos_all)
        v += rule_match(i, vel_all, cov)

        vel_all[i] += v 
        pos_all[i] += vel_all[i]

        
    return pos_all, vel_all  
    

