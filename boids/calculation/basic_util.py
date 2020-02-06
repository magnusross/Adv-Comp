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
    intilaises position and 
    
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

    return pos.astype('float64'), vel.astype('float64')

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