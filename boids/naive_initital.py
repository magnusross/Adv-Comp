import numpy as np 
import numba_boids_rules as b
import tqdm
from utilities import make_2D_square_walls



#initalise boids
N_b = 300
pos_rand  = (2*np.random.rand(N_b, 2) - 1) * 100 + np.array([300, 300])
vel_rand  = (2*np.random.rand(N_b, 2) - 1) 

walls = make_2D_square_walls(400, factor=3)


#update N times 
N = 700
results = np.zeros((N, 2, N_b, 2))                                                 
for j in tqdm.tqdm(range(N)):
    results[j] = b.update_boids(pos_rand, vel_rand, 
                                objs=1, pos_obj=walls)
np.save('results.npy', results)



            
    