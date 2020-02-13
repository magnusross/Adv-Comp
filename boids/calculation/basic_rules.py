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
def rule_avoid(i, pos_all, radius=30, factor=0.002):
    """
    rule that stops boids colliding 
    
    Arguments:
        i {int} -- index of boid rule is being applied to 
        pos_all {np array} -- N x D array of positions 
    
    Keyword Arguments:
        radius {float} -- boids field of view distance 
        factor {float} -- weighting of this rule 
    
    Returns:
        numpy array -- velocity update
    """
    size = len(pos_all)
    pos_all = pos_all.astype(np.float64)
    p = np.zeros_like(pos_all[0])

    rel_pos = pos_all - pos_all[i]

    for j in range(size):
        dist = np.power(rel_pos[j], 2)  
        if i != j and np.sqrt(np.sum(dist))< radius:  
            for k in range(len(p)):
                p[k] -= rel_pos[j][k]
              
    return p * factor 


@numba.njit()
def rule_com(i, pos_all, radius=30, factor=0.01):
    """
    rule that guides boids to local center of mass 
    
    Arguments:
        i {int} -- index of boid rule is being applied to 
        pos_all {np array} -- N x D array of positions 
    
    Keyword Arguments:
        radius {float} -- boids field of view distance 
        factor {float} -- weighting of this rule 
    
    Returns:
        numpy array -- velocity update
    """
    
    size = len(pos_all)
    pos_all = pos_all.astype(np.float64)
    p = np.zeros_like(pos_all[0])
    N = 0
    rel_pos = pos_all - pos_all[i]

    for j in range(size):
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
    """
    rule that aligns local velocities 
    
    Arguments:
        i {int} -- index of boid rule is being applied to 
        pos_all {np array} -- N x D array of positions 
        vel_all {np array} -- N x D array of  velocities 
    
    Keyword Arguments:
        radius {float} -- boids field of view distance 
        factor {float} -- weighting of this rule 
    
    Returns:
        numpy array -- velocity update
    """
    
    size = len(vel_all)
    vel_all = vel_all.astype(np.float64)
    p = np.zeros_like(vel_all[0])
    N = 0
    rel_pos = pos_all - pos_all[i]

    for j in range(size):
        dist = np.power(rel_pos[j], 2)  
        if i != j and np.sqrt(np.sum(dist))< radius:  
            N += 1 
            for k in range(len(p)):
                p[k] += vel_all[j][k]
    if N == 0:
        return p 
    else:
        return (-vel_all[i] + p/N ) * factor



    




