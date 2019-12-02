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
    
    return my_pos, my_vel


    