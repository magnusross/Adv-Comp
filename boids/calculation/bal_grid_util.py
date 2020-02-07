from sys import platform
import os 
if platform == 'linux':

    print('Setting environment variables')
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numba


np.random.seed(200)



def initialise_boids(N_b, box_size, vel=1.):
    """
    initialises position and velocity
    
    Arguments:
        N_b {int} -- number of boids 
        box_size {np array} -- size of simulation region
    
    Keyword Arguments:
        vel {float} --  veloctiy scale for boids(default: {1})
    
    Returns:
        pos [np array] -- inital positions
        vel [np array] -- inital velocities
    """  

    dim = len(box_size)
    pos = np.random.rand(N_b, dim) * box_size 
    vel = (2*np.random.rand(N_b, dim) - 1) * vel
    return pos, vel


def initialise_grid(pos, box_size, radius):
    """
    initialises grid stucture for boids 
    
    Arguments:
        pos {np array} -- N x D array of all boids positions 
        box_size {np array} -- D x 1 array, size of simulation region 
        radius {float} --  boids field of view distance
    
    Returns:
        {np array} -- N x D grid coords of boids 
    """    


    grid_list = np.zeros((len(pos), len(box_size)), dtype='int')
    for i in range(len(grid_list)):
        grid_list[i] = grid_from_pos(pos[i], box_size, radius) 
    return grid_list 


def make_proc_boid_ind(N_b, N_proc):
    """
    Divides the boids between the processors so that load 
    is spread most evenly 
    
    Arguments:
        N_b {int} -- number of boids 
        N_proc {int} -- number of processors 
    
    Raises:
        ValueError: The number of boids should not be more than
        processirs 
    
    Returns:
        list -- Tuples containing the index range for each
        processor 
    """
    
    if N_b < N_proc:
        raise ValueError('More processors than boids!')
    
    per_proc = N_b // N_proc
    left_over = N_b % N_proc
    group_lens = np.ones(N_proc + 1).astype(np.int) * per_proc
    group_lens[:left_over] += 1
    return [(np.sum(group_lens[:i]), np.sum(group_lens[:i+1])) for i in range(N_proc)]


@numba.njit()
def grid_from_pos(one_pos, box_size, radius):
    '''
    gets grid coordinates from one boids position 
    
    Arguments:
        one_pos {np array} -- D x 1 array, one boids position 
        box_size {np array} -- D x 1 array, size of simulation region 
        radius {float} --  boids field of view distance
    '''

    N_cell = box_size // radius
    grid = np.floor_divide(one_pos , box_size / N_cell)
    return grid

@numba.njit()
def get_new_grid(upd_labs, grid, pos, vel, box_size, radius):
    """
    gets updated grid coords for boids, only updates indexes in upd_labs 
    
    Arguments:
        upd_labs {list} -- list of indexs to update grids for 
        grid {np array} -- array of all boids grid coordinates 
        pos [np array] -- all boids positions
        vel [np array] -- all boids velocities
        box_size {np array} -- D x 1 array, size of simulation region 
        radius {float} --  boids field of view distance
    
    Returns:
        [np array] -- updated grid 
    """    

    new_grid = np.copy(grid)
    for lab in upd_labs:
        new_grid[lab] = grid_from_pos(pos[lab], box_size, radius) 
    return new_grid

def get_grid_updates(upd_labs, grid, pos, vel, box_size, radius):
    """
    gets finds the indices of boids that have moved grid cell and the 
    new cell they're in. This is to avoid sending extra info back
    to master.
    
    Arguments:
        upd_labs {list} -- list of indexs to update grids for 
        grid {np array} -- array of all boids grid coordinates 
        pos [np array] -- all boids positions
        vel [np array] -- all boids velocities
        box_size {np array} -- D x 1 array, size of simulation region 
        radius {float} --  boids field of view distance
    
    Returns:
        [np array] -- indices of boids in new cell
        [np array] -- array of new cells coords that boids are in 
    """  

    
    new_grid = get_new_grid(upd_labs, grid, pos, vel, box_size, radius)
    diff_labs = np.where(np.any(new_grid != grid, axis=1))
    
    return diff_labs[0], new_grid[diff_labs]

@numba.njit()
def get_adj_labs(upd_lab, grid):
    """[summary]
    
    Arguments:
        upd_lab {int} -- index to get adjacents for 
        grid {[type]} -- array of all grid coords 
    
    Returns:
        [list] -- list of adjacent labels 
    """    

    my_grid = grid[upd_lab]
    adj = []
    for i in range(len(grid)):
        if np.any(np.abs(my_grid - grid[i])) <= 0:
            adj.append(i)
    
    return adj






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
    
    

