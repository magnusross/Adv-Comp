import numpy as np
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

    


