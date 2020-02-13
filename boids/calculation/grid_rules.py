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
def rule_avoid(one_my_pos, pos_all, radius=30., factor=0.00001, max_f=0):
    """
    rule that stops boids colliding 
    
    Arguments:
        one_my_pos {np array} -- 1D array with one boids position
        pos_all {np array} -- N x D array of positions 
    
    Keyword Arguments:
        radius {float} -- boids field of view distance 
        factor {float} -- weighting of this rule 
    
    Returns:
        numpy array -- velocity update
    """    
    size = len(pos_all)
    p = np.zeros_like(one_my_pos)
    rel_pos = pos_all - one_my_pos

    for j in np.arange(size):
        dist = np.power(rel_pos[j], 2)  
        if np.all(one_my_pos != pos_all[j]) and np.sqrt(np.sum(dist))< radius:  
            for k in range(len(p)):
                p[k] -= rel_pos[j][k]

    return p * factor


@numba.njit()
def rule_com(one_my_pos, pos_all, radius=30., factor=0.001, max_f=0):
    '''
    rule that guides boids to local center of mass 
    
    Arguments:
        one_my_pos {np array} -- 1D array with one boids position
        pos_all {np array} -- N x D array of positions 
    
    Keyword Arguments:
        radius {float} -- boids field of view distance 
        factor {float} -- weighting of this rule 
    
    Returns:
        numpy array -- velocity update
    '''    
    
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
        v_upd = ((p/N) - one_my_pos) * factor
        return v_upd

@numba.njit()
def rule_match(one_my_pos, one_my_vel, pos_all, vel_all, radius=30., factor=0.00001, max_f=0):
    '''
    rule that aligns local velocities 
    
    Arguments:
        one_my_pos {np array} -- 1D array with one boids position
        pos_all {np array} -- N x D array of positions 
        one_my_vel {np array} -- 1D array with one boids velocity
        vel_all {np array} -- N x D array of velocities 
    
    Keyword Arguments:
        radius {float} -- boids field of view distance 
        factor {float} -- weighting of this rule
    
    Returns:
        numpy array -- velocity update
    '''  

    
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
        v_upd = ((p/N) - one_my_pos) * factor
        return v_upd

    
