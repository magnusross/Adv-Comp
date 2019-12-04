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
    return int(N_cell_ax * grid[0] + grid[1])


@numba.njit()    
def assign_to_cells(boids_owned,  N_cell_ax, box_size):
    # this can't be jitted because boids grid is gjagged
    for i in range(len(boids_owned)):
        boid = boids_owned[i][1:len(box_size)+1]
        grid_num = get_grid_num(boid, N_cell_ax, box_size)
        boids_owned[i][0] = grid_num
    
    return boids_owned[np.argsort(boids_owned[:, 0])]

def get_cells_boids(cell_n, boids_owned):
    return boids_owned[np.where(boids_owned[:, 0].astype('int') == cell_n)]


@numba.njit()    
def get_neigh_2D(cell_n, N_cell_ax, N_nn=1):
    '''
    grid = np.arange(N_cell_ax**2).reshape(N_cell_ax, N_cell_ax)
    cell_co = np.argwhere(grid == cell_n)
    '''
    nieghs = []
    posx, posy = cell_n // N_cell_ax, cell_n % N_cell_ax
    for n in range(N_cell_ax**2):

        if abs(n // N_cell_ax - posx) <= N_nn and abs(n % N_cell_ax - posy) <=1 and n != cell_n:
            nieghs.append(n)
  
    return nieghs




# @numba.njit()
def init_cells_2D(N_boids, N_cell_ax, box_size=np.array([100., 100., 100.]), vel=1.):
    
    dim = len(box_size)
    pos = (np.random.rand(N_boids, dim)) * box_size 
    vel = (np.random.rand(N_boids, dim)) * vel
    boids_owned = np.hstack((np.zeros((len(pos), 1), dtype=np.float64),pos,vel))


    return assign_to_cells(boids_owned, N_cell_ax, box_size)
a =init_cells_2D(100, 2)

'''

# @numba.njit()    
def assign_to_cells(boids_grid,  N_cell_ax, box_size):
    # this can't be jitted because boids grid is gjagged
    boids_grid_new = [ [] for i in range(len(boids_grid))]
    N_per_cell = [len(a) for a in boids_grid]

    for i in range(len(boids_grid)):
        #this is so boids aren't checked 2 times 
        for j in range(N_per_cell[i]):
            boid = np.array(boids_grid[i][j][:len(box_size)])
            grid_num = get_grid_num(boid, N_cell_ax, box_size)
            boids_grid_new[grid_num].append(boids_grid[i][j])

    boids_grid_new = [np.array(boids) for boids in boids_grid_new]
    
    return boids_grid_new
# @numba.njit()
def init_cells_2D(N_boids, N_cell_ax, box_size=np.array([100., 100.]), vel=1.):
    
    dim = len(box_size)
    pos = (np.random.rand(N_boids, dim)) * box_size 
    vel = (np.random.rand(N_boids, dim)) * vel
    boids = np.hstack((pos,vel))
    boids_grid = [ np.array([]) for i in range(N_cell_ax**dim)]
    boids_grid[0] = boids

    return assign_to_cells(boids_grid, N_cell_ax, box_size)


import matplotlib.pyplot as plt
col = ['blue', 'green', 'red', 'yellow', 'blue', 'green', 'red', 'yellow', 'blue', 'green', 'red', 'yellow']
cells  = init_cells_2D(1000, 3)
for i in range(len(cells)):
    plt.scatter(cells[i][:, 0], cells[i][:, 1], color=col[i])
plt.show()
'''