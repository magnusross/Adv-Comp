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

def initialise_boids(N_b, dim, vel=1, width=100):
    pos = (2*np.random.rand(N_b, dim) - 1) * width 
    vel = (2*np.random.rand(N_b, dim) - 1) * vel
    return pos.astype('float64'), vel.astype('float64')

def make_2D_square_walls(a, factor=1):
    
    left = np.zeros((2*a*factor,2))
    left[:, 0] = -a
    left[:, 1] = np.arange(-a, a, step=1/factor)

    right = np.zeros((2*a*factor,2))
    right[:, 0] = a
    right[:, 1] = np.arange(-a, a, step=1/factor)

    top = np.zeros((2*a*factor, 2))
    top[:, 0] = np.arange(-a, a, step=1/factor)
    top[:, 1] = a

    bottom = np.zeros((2*a*factor, 2))
    bottom[:, 0] = np.arange(-a, a, step=1/factor)
    bottom [:, 1] = -a

    return np.vstack((left, right, top, bottom))

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
def get_grid_num(pos_i, N_cell_ax, box_size):

    cell_size = box_size / N_cell_ax
    grid = (pos_i // cell_size)

    return grid


@numba.njit()    
def assign_to_cells(boids_owned,  N_cell_ax, box_size):
    # this can't be jitted because boids grid is gjagged
    for i in range(len(boids_owned)):
        boids_owned[i][0] = get_grid_num(boids_owned[i][1], N_cell_ax, box_size)
    
    # return boids_owned
# @numba.njit()
def get_cells_boids(cell_n, boids_owned):
    return boids_owned[np.where(np.all(boids_owned[:, 0].astype('int') == cell_n.astype('int'), axis=1))]


# @numba.njit()    
def get_neigh(cell_n, N_cell_ax, dim, N_nn=1):
    '''
    grid = np.arange(N_cell_ax**2).reshape(N_cell_ax, N_cell_ax)
    cell_co = np.argwhere(grid == cell_n)
    ''' 
    coords = make_proc_coord_list(N_cell_ax, dim)

    nieghs = []
    for i in range(N_cell_ax**dim):
        co = coords[i].astype('int') 
        if np.any(np.abs((co - cell_n)) <= 1) and np.any(co != cell_n):
            nieghs.append(co)
  
    return np.array(nieghs).reshape(-1, dim)

def make_proc_coord_list(N_cell_ax, dim):
    
    a = np.zeros((N_cell_ax**dim, dim))

    if dim == 2:
        arr = np.arange(N_cell_ax**dim).reshape(N_cell_ax, -1)
    else:
        arr = np.arange(N_cell_ax**dim).reshape(N_cell_ax, N_cell_ax, -1)
    
    for i in range(N_cell_ax**dim):
        a[i] = np.argwhere(arr == i) 
    return a 

# @numba.njit()
def init_cells(N_boids, N_cell_ax, box_size=np.array([100., 100., 100.]), vel=1.):
    
    dim = len(box_size)
    pos = (np.random.rand(N_boids, dim)) * box_size 
    vel = (np.random.rand(N_boids, dim)) * vel
    
    boids_owned = np.hstack((np.zeros((len(pos), dim), dtype=np.float64),pos,vel)).reshape(N_boids, 3, dim)
    assign_to_cells(boids_owned, N_cell_ax, box_size)
    return boids_owned
