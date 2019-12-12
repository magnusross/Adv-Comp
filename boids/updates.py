from sys import platform
import os 
if platform == 'linux':

    print('Setting environment variables')
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1"

import numba
import numpy as np
import numba_boids_rules as r
from numba import prange
import grid_boids_rules as rg 

@numba.njit()
def update_boids(pos_all, vel_all):
    '''
    objs option for numba compile 
    '''
    for i in prange(len(pos_all)):
        v = r.rule_com(i, pos_all)
        v += r.rule_avoid(i, pos_all)
        v += r.rule_match(i, pos_all, vel_all)
        #r.rule_wall_bounce_2D(i, pos_all, vel_all)

        vel_all[i] += v 
        pos_all[i] += vel_all[i]
        
        r.rule_wrap(i, pos_all)
        
    return pos_all, vel_all  

@numba.njit()
def update_my_boids(ind, pos_all, vel_all):

    n_my_b = ind[1] - ind[0]
    dim = pos_all.shape[1]
    my_pos, my_vel = np.empty((n_my_b, dim)), np.empty((n_my_b, dim))

    for i in prange(n_my_b):
        i_a = ind[0] +  i 

        v = r.rule_com(i_a, pos_all)
        v += r.rule_avoid(i_a, pos_all)
        v += r.rule_match(i_a, pos_all, vel_all)

        my_vel[i] = vel_all[i_a] + v
        my_pos[i] = pos_all[i_a] + my_vel[i]
        
        r.rule_wrap(i, my_pos)
    
    return my_pos, my_vel

# @numba.njit()
def grid_update_my_boids(my_boids, all_boids, box_size, radius=30.):
    pos_all, vel_all = all_boids[:, 1], all_boids[:, 2]
    pos_my, vel_my = my_boids[:, 1], my_boids[:, 2]
    
    for i in range(len(pos_my)):
        
        v = rg.rule_com(pos_my[i], pos_all, radius=radius)
        v += rg.rule_avoid(pos_my[i], pos_all, radius=radius)
        v += rg.rule_match(pos_my[i], vel_my[i], pos_all, vel_all, radius=radius)
        
        
        
        vel_my[i] += v
        pos_my[i] += vel_my[i]

    rg.rule_wrap(pos_my, box_size)
    





