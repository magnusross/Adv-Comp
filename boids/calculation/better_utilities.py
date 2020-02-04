from sys import platform
import os 
if platform == 'linux':

    print('Setting environment variables')
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np
import numba
import grid_boids_rules as rg 

np.random.seed(200)
'''
[('lab', 'int8'), ('pos', '2float64'), ('vel', '2float64')]

'''


@numba.njit()
def initialise_boids(N_b, box_size, vel=1):
    dim = len(box_size)

    pos = np.random.rand(N_b, dim) * box_size 
    vel = (2*np.random.rand(N_b, dim) - 1) * vel

    return pos, vel

# @numba.njit()
def initialise_grid(pos, vel, box_size, radius):
    grid_list = np.zeros((len(pos), len(box_size)), dtype='int')
    for i in range(len(grid_list)):
        grid_list[i] = grid_from_pos(pos[i], box_size, radius) 

    return grid_list 


@numba.njit()
def grid_from_pos(pos, box_size, radius):
    N_cell = box_size // radius
    grid = np.floor_divide(pos , box_size / N_cell)
    return grid

@numba.njit()
def get_new_grid(upd_labs, grid_list, pos, vel, box_size, radius):
    new_grid = np.copy(grid_list)
    for lab in upd_labs:
        new_grid[lab] = grid_from_pos(pos[lab], box_size, radius) 
    return new_grid

def get_grid_updates(upd_labs, grid_list, pos, vel, box_size, radius):
    
    new_grid = get_new_grid(upd_labs, grid_list, pos, vel, box_size, radius)
    diff_labs = np.where(np.any(new_grid != grid_list, axis=1))
    
    return diff_labs[0], new_grid[diff_labs]

@numba.njit()
def get_adj_labs(upd_lab, grid_list):
    my_grid = grid_list[upd_lab]
    adj = []
    
    for i in range(len(grid_list)):
        if np.any(np.abs(my_grid - grid_list[i])) <= 0:
            adj.append(i)
    
    return adj

@numba.njit()
def better_update_boids(my_labs, grid_all, pos_all, vel_all, box_size, radius):
    
    for lab in my_labs:
        adj_labs = np.array(get_adj_labs(lab, grid_all))
        adj_pos = pos_all[adj_labs]
        adj_vel = vel_all[adj_labs]
        
        v = rg.rule_com(pos_all[lab], adj_pos, radius=radius)
        v += rg.rule_avoid(pos_all[lab], adj_pos, radius=radius)
        v += rg.rule_match(pos_all[lab], vel_all[lab], adj_pos, adj_vel, radius=radius)

        vel_all[lab] += v
        pos_all[lab] += vel_all[lab]
        
    pos_all %= box_size




"""

pos ,vel = initialise_boids(1000, np.array([100., 100.]), 10.)
grid = initialise_grid(pos, vel, np.array([100., 100.]), 10.)



pos[1] = np.array([11., 17.])

print(get_grid_updates([0], grid, pos, vel, np.array([100., 100.]), 1.))

print(get_grid_updates([1], grid, pos, vel, np.array([100., 100.]), 1.))
print(get_adj_labs(3, grid))

print(pos)
better_update_boids([0, 2, 3, 4], grid, pos, vel, np.array([100., 100.]), 10.)
print(pos)


"""
    
    

