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
def serial_update(pos_all, vel_all, box_size):
    """
    The update function for the initial non parallel naive implementation 
    
    Arguments:
        pos_all {numpy array} -- N x D array of all positions 
        vel_all {numpy array} -- N x D array of all positions 
        box_size {numpy array} -- D x 1 size of simulation region
    
    Returns:
        tuple --  updated pos_all and vel_all 
    """
    for i in range(len(pos_all)):
        v = r.rule_com(i, pos_all)
        v += r.rule_avoid(i, pos_all)
        v += r.rule_match(i, pos_all, vel_all)
        #r.rule_wall_bounce_2D(i, pos_all, vel_all)

        vel_all[i] += v 
        pos_all[i] += vel_all[i]

        pos_all %= box_size
        
        
    return pos_all, vel_all  

@numba.njit()
def basic_update(ind, pos_all, vel_all, box_size, radius=30.):
    """
    The update function for the basic parallel implementation, updates subset of boids 
    from all, correspoiding to the indices in ind. Returns updated subset 
    
    Arguments:
        ind {tuple} -- contains upper and lower ends of range of indices the worker is 
        managing 
        pos_all {numpy array} -- N x D array of all positions 
        vel_all {numpy array} -- N x D array of all velocities  
        box_size {numpy array} -- D x 1 size of simulation region
    
    Returns:
        tuple --  updated subset of pos_all and vel_all 
    """
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
    """
    Update function for Grid implementation, note data structure is slightly different 
    to the other implementations with position and velocity and coords being in one array. Updates 
    boids in my_boids, modifies in place 
    
    Arguments:
        my_boids {numpy array} -- B x 3 x D, contains postions and velocities and grid 
        coords for boids worker owns, postion/velocity/grid second dimension, B is number 
        of boids owned 
        all_boids {numoy array} -- 2 x M x D, contains postions and velocities for boids
        to update relative to, postion/velocity first dimension, M is number of boids
        in vicinity of worker 
        box_size {numpy array} -- D x 1 size of simulation region
    
    Keyword Arguments:
        radius {float} -- boids field of view distance
    """    

    pos_all, vel_all = all_boids[:, 1], all_boids[:, 2]
    pos_my, vel_my = my_boids[:, 1], my_boids[:, 2]
    
    for i in range(len(pos_my)):
        
        v = rg.rule_com(pos_my[i], pos_all, radius=radius)
        v += rg.rule_avoid(pos_my[i], pos_all, radius=radius)
        v += rg.rule_match(pos_my[i], vel_my[i], pos_all, vel_all, radius=radius)
        
        vel_my[i] += v
        pos_my[i] += vel_my[i]
    
    # keeps boids in domain 
    pos_my %= box_size

@numba.njit()
def bal_grid_update(my_labs, grid_all, pos_all, vel_all, box_size, radius):
    """
    Update function for balanced grid method, updates boids with labels in 
    my labs, 
    Arguments:
        my_labs {numpy array} -- array of position indexes to be updated 
        grid_all {numpy array} -- N x D array of all boids grid coordinates
        pos_all {numpy array} -- N x D array of all positions  
        vel_all {numpy array} -- N x D array of all velocities  
        box_size {numpy array} -- D x 1 size of simulation region
        radius {float} -- boids field of view distance
    """    

    for i in range(len(my_labs)):
        lab = my_labs[i]
        adj_labs = np.array(butil.get_adj_labs(lab, grid_all))
        adj_pos = pos_all[adj_labs]
        adj_vel = vel_all[adj_labs]
        
        v = rg.rule_com(pos_all[lab], adj_pos, radius=radius)
        v += rg.rule_avoid(pos_all[lab], adj_pos, radius=radius)
        v += rg.rule_match(pos_all[lab], vel_all[lab], adj_pos, adj_vel, radius=radius)

        vel_all[lab] += v
        pos_all[lab] += vel_all[lab]
   
    # keeps boids in domain     
    pos_all %= box_size







