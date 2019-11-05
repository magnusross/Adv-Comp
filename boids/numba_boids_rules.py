import numpy as np
import numba 
from numba import prange

@numba.njit()
def rule_com(i, pos_all, vel_all, factor=0.1):

    size = len(pos_all)
    pos_all = pos_all.astype(np.float32)
    vel_all = pos_all.astype(np.float32)
    p = np.array([0., 0.], dtype=np.float32)

    for j in prange(size):
        if j != i:
            p[0] += pos_all[j][0]
            p[1] += pos_all[j][1]
    
    return -((pos_all[i] - p)/(size-1)) * factor

@numba.njit()
def rule_avoid(i, pos_all, vel_all, radius=20, factor=2):
    
    size = len(pos_all)
    pos_all = pos_all.astype(np.float64)
    vel_all = pos_all.astype(np.float64)
    p = np.array([0., 0.], dtype=np.float64)

    #for dealing with objects like walls the boids must avoid
 
    rel_pos = pos_all - pos_all[i]

    for j in prange(size):
        if i != j and  rel_pos[j][0]**2 + rel_pos[j][1]**2 < radius:
            p[0] -= rel_pos[j][0]
            p[1] -= rel_pos[j][1]
    return p * factor 

@numba.njit()
def rule_match(i, pos_all, vel_all, factor=0.001):

    size = len(pos_all)
    pos_all = pos_all.astype(np.float32)
    vel_all = pos_all.astype(np.float32)
    p = np.array([0., 0.], dtype=np.float32)

    for j in prange(size):
        if i != j:
            p[0] -= vel_all[j][0]
            p[1] -= vel_all[j][1]
    
    return (p/(size-1) - vel_all[i]) * factor
    

