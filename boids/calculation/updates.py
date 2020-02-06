from sys import platform
import os 
if platform == 'linux':

    print('Setting environment variables')
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1"

import numba
import numpy as np
import basic_rules as r
import grid_rules as rg 
import bal_grid_util as butil
from numba import prange


@numba.njit()
def serial_update(pos_all, vel_all):
    '''
    objs option for numba compile 
    '''
    for i in range(len(pos_all)):
        v = r.rule_com(i, pos_all)
        v += r.rule_avoid(i, pos_all)
        v += r.rule_match(i, pos_all, vel_all)
        #r.rule_wall_bounce_2D(i, pos_all, vel_all)

        vel_all[i] += v 
        pos_all[i] += vel_all[i]
        
        r.rule_wrap(i, pos_all)
        
    return pos_all, vel_all  

@numba.njit()
def basic_update(ind, pos_all, vel_all, box_size, radius=30.):

    n_my_b = ind[1] - ind[0]
    dim = pos_all.shape[1]
    my_pos, my_vel = np.empty((n_my_b, dim)), np.empty((n_my_b, dim))

    for i in range(n_my_b):
        i_a = ind[0] +  i 

        v = r.rule_com(i_a, pos_all, radius=radius)
        v += r.rule_avoid(i_a, pos_all, radius=radius)
        v += r.rule_match(i_a, pos_all, vel_all, radius=radius)

        my_vel[i] = vel_all[i_a] + v
        my_pos[i] = pos_all[i_a] + my_vel[i]
        
        my_pos[i] %= box_size
    
    return my_pos, my_vel

@numba.njit()
def grid_update(my_boids, all_boids, box_size, radius=30.):
    pos_all, vel_all = all_boids[:, 1], all_boids[:, 2]
    pos_my, vel_my = my_boids[:, 1], my_boids[:, 2]
    
    for i in range(len(pos_my)):
        
        v = rg.rule_com(pos_my[i], pos_all, radius=radius)
        v += rg.rule_avoid(pos_my[i], pos_all, radius=radius)
        v += rg.rule_match(pos_my[i], vel_my[i], pos_all, vel_all, radius=radius)
        
        vel_my[i] += v
        pos_my[i] += vel_my[i]
    # wrap
    # rg.rule_wrap(pos_my, box_size)
    pos_my %= box_size

@numba.njit()
def bal_grid_update(my_labs, grid_all, pos_all, vel_all, box_size, radius):
    
    for lab in my_labs:
        adj_labs = np.array(butil.get_adj_labs(lab, grid_all))
        adj_pos = pos_all[adj_labs]
        adj_vel = vel_all[adj_labs]
        
        v = rg.rule_com(pos_all[lab], adj_pos, radius=radius)
        v += rg.rule_avoid(pos_all[lab], adj_pos, radius=radius)
        v += rg.rule_match(pos_all[lab], vel_all[lab], adj_pos, adj_vel, radius=radius)

        vel_all[lab] += v
        pos_all[lab] += vel_all[lab]
        
    pos_all %= box_size







