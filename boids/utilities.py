import numpy as np

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