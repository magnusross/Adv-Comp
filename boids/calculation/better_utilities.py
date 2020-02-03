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
'''
[('lab', 'int8'), ('pos', '2float64'), ('vel', '2float64')]

'''


pos_vel2d_dt = np.dtype({'names':['lab', 'pos', 'vel'],
                                 'formats':['int8', '%sfloat64'%(2), '%sfloat64'%(2)]})
pos_vel3d_dt = np.dtype({'names':['lab', 'pos', 'vel'],
                                 'formats':['int8', '%sfloat64'%(3), '%sfloat64'%(3)]})

grid2d_dt = np.dtype({'names':['lab', 'grid'],
                                    'formats':['int8', '%sint8'%(2)]})
grid3d_dt = np.dtype({'names':['lab', 'grid'],
                                    'formats':['int8', '%sint8'%(3)]})

nb_pos_vel_dt = [numba.from_dtype(pos_vel2d_dt), numba.from_dtype(pos_vel3d_dt)]
nb_grid_dt = [numba.from_dtype(grid2d_dt), numba.from_dtype(grid3d_dt)]

# @numba.njit()
def initialise_boids(N_b, box_size, vel=1, dt=nb_grid_dt):
    dim = len(box_size)
  
    pos_vel = np.zeros(N_b, dtype={'names':['lab', 'pos', 'vel'],
                                 'formats':['int8', '%sfloat64'%(dim), '%sfloat64'%(dim)]})

    pos_vel['lab'] = np.arange(N_b)
    pos_vel['pos'] = np.random.rand(N_b, dim) * box_size 
    pos_vel['vel'] = (2*np.random.rand(N_b, dim) - 1) * vel

    return pos_vel

# @numba.njit()
def initialise_grid(pos_vel, box_size, radius):
    grid_list = np.zeros(len(pos_vel), dtype=numba.from_dtype(np.dtype({'names':['lab', 'grid'],
                                    'formats':['int8', '%sint8'%(len(box_size))]})))
    for lab in pos_vel['lab']:
        grid_list['lab'][lab] = lab
        pos = pos_vel['pos'][np.where(lab == pos_vel['lab'])]
        grid_list['grid'][np.where(lab == grid_list['lab'])] = grid_from_pos(pos, box_size, radius) 
    
    return grid_list 




@numba.njit()
def grid_from_pos(pos, box_size, radius):
    N_cell = box_size // radius
    grid = pos // (box_size / N_cell)
    return grid

@numba.njit()
def nb_get_grid_updates(upd_labs, grid_list, pos_vel, box_size, radius):
    # lab_diff = []
    # grid_diff = []
   
    for lab in upd_labs:
        print(pos_vel[1])#[np.where(lab == pos_vel['lab'])]
        
        # current_grid = grid_list['grid'][np.where(lab == grid_list['lab'])] 
        # new_grid = grid_from_pos(pos, box_size, radius) 
        '''
        if np.any(new_grid != current_grid):
            lab_diff.append(lab)
            grid_diff.append(new_grid)
        '''
    # return 1 #lab_diff, grid_diff

def get_grid_updates(upd_labs, grid_list, pos_vel, box_size, radius):
    lab_upd, grid_upd = nb_get_grid_updates(upd_labs, grid_list, pos_vel, box_size, radius)
    
    if lab_upd:
        upd_grid = np.zeros(len(lab_upd), dtype=grid_list.dtype)
        upd_grid['lab'] = np.array(lab_upd)
        upd_grid['grid'] = np.array(grid_upd)
        return upd_grid


pos_vel = initialise_boids(10, np.array([100., 100.]), 1.)
grid = initialise_grid(pos_vel, np.array([100., 100.]), 1.)

pos_vel['pos'][1] = np.array([1., 1.])

print(get_grid_updates([0], grid, pos_vel, np.array([100., 100.]), 1.))

print(nb_get_grid_updates([1], grid, pos_vel, np.array([100., 100.]), 1.))
print(get_grid_updates([1], grid, pos_vel, np.array([100., 100.]), 1.))



    
    

