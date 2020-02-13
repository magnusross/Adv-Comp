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

def initialise_boids(N_boids, N_cell_ax, box_size, vel=1.):
    """
    initialise data structure for grid method, with random positions 
    velocities 
    
    Arguments:
        N_boids {int} -- number of boids
        N_cell_ax {int} -- number of cells per axis 
        box_size {numpy array} -- D x 1 size of simulation region
    
    Keyword Arguments:
        vel {float} -- velocity scale 
    
    Returns:
        numpy array -- initialized data structure containing psotion velocity 
        and grid coords of all boids 
    """
        
    dim = len(box_size)
    pos = (np.random.rand(N_boids, dim)) * box_size 
    vel = (np.random.rand(N_boids, dim)) * vel
    
    boids = np.hstack((np.zeros((len(pos), dim), dtype=np.float64), pos, vel)).reshape(N_boids, 3, dim)
    assign_to_cells(boids, N_cell_ax, box_size)
    return boids

def get_cells_boids(cell_coord, boids):
    """
    returns subset of boids that are in cell with coords given 
    by cell_coord
    
    Arguments:
        cell_coord {numpy array} -- D x 1 coord of cell 
        boids {numpy array} -- data structure with all boids data in (pos/vel/grid)
    
    Returns:
        numpy array -- sub set of boids in cell
    """    

    return boids[np.where(np.all(boids[:, 0].astype('int') == cell_coord.astype('int'), axis=1))]

def make_proc_coord_list(N_cell_ax, dim):
    """
    makes list of which processor is associated with which grid cell
    i th element is the coords associated with processor i 
    Arguments:
        N_cell_ax {int} -- number of cells per axis 
        dim {int} -- dimension of system 
    
    Returns:
        numpy array
    """ 

    
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
    """
    gets grid coord from spatial postion 
    
    Arguments:
        pos_i {numpy array} -- D x 1 spatial postion
        N_cell_ax {int} -- number of cells per axis 
        box_size {numpy array} -- D x 1 size of simulation region
    
    Returns:
        [type] -- [description]
    """


    cell_size = box_size / N_cell_ax
    grid = (pos_i // cell_size)

    return grid

@numba.njit()    
def assign_to_cells(boids,  N_cell_ax, box_size):
    """
    updates coord values in boids to correct values
    
    Arguments:
        boids {numpy array} -- data structure with all boids data in (pos/vel/grid)
        N_cell_ax {int} -- number of cells per axis 
        box_size {numpy array} -- D x 1 size of simulation region
    """    

    for i in range(len(boids)):
        boids[i][0] = get_grid_num(boids[i][1], N_cell_ax, box_size)


def get_neigh(cell_coord, N_cell_ax, dim):
    """
    gets coords of neighbors of a given cell 
    
    Arguments:
        cell_coord {numpy array} -- D x 1 coord of cell 
        N_cell_ax {int} -- number of cells per axis 
        dim {int} -- dimension of system 
    
    Returns:
        numpy array -- coords of neighbors of cell
    """    

    
    coords = make_proc_coord_list(N_cell_ax, dim)

    nieghs = []
    for i in range(N_cell_ax**dim):
        co = coords[i].astype('int') 
        if np.any(np.abs((co - cell_coord)) <= 1) and np.any(co != cell_coord):
            nieghs.append(co)

    return np.array(nieghs).reshape(-1, dim)
   