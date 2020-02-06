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

def init_cells(N_boids, N_cell_ax, box_size=np.array([100., 100., 100.]), vel=1.):
    
    dim = len(box_size)
    pos = (np.random.rand(N_boids, dim)) * box_size 
    vel = (np.random.rand(N_boids, dim)) * vel
    
    boids_owned = np.hstack((np.zeros((len(pos), dim), dtype=np.float64),pos,vel)).reshape(N_boids, 3, dim)
    assign_to_cells(boids_owned, N_cell_ax, box_size)
    return boids_owned

def get_cells_boids(cell_n, boids_owned):
    return boids_owned[np.where(np.all(boids_owned[:, 0].astype('int') == cell_n.astype('int'), axis=1))]

def make_proc_coord_list(N_cell_ax, dim):
    
    a = np.zeros((N_cell_ax**dim, dim))

    if dim == 2:
        arr = np.arange(N_cell_ax**dim).reshape(N_cell_ax, -1)
    else:
        arr = np.arange(N_cell_ax**dim).reshape(N_cell_ax, N_cell_ax, -1)
    
    for i in range(N_cell_ax**dim):
        a[i] = np.argwhere(arr == i) 
    return a 


@numba.njit()    
def get_grid_num(pos_i, N_cell_ax, box_size):

    cell_size = box_size / N_cell_ax
    grid = (pos_i // cell_size)

    return grid

@numba.njit()    
def assign_to_cells(boids_owned,  N_cell_ax, box_size):
    # this can't be jitted because boids grid is gjagged
    for i in range(len(boids_owned)):
        boids_owned[i][0] = get_grid_num(boids_owned[i][1], N_cell_ax, box_size)


def get_neigh(cell_n, N_cell_ax, dim, N_nn=1):
    
    coords = make_proc_coord_list(N_cell_ax, dim)

    nieghs = []
    for i in range(N_cell_ax**dim):
        co = coords[i].astype('int') 
        if np.any(np.abs((co - cell_n)) <= 1) and np.any(co != cell_n):
            nieghs.append(co)

    return np.array(nieghs).reshape(-1, dim)
   