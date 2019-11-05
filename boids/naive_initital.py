import numpy as np 
import numba_boids_rules as b
import tqdm



#initalise boids
N_b = 1000 
pos_rand  = (2*np.random.rand(N_b, 2) - 1) * 100
vel_rand  = np.flip(pos_rand, axis=1)*1e-3


#update N times 
N = 400
results = np.zeros((N, 2, N_b, 2))                                                 
for j in tqdm.tqdm(range(N)):
    results[j] = b.update_boids(pos_rand, vel_rand)
np.save('results.npy', results)



            
    