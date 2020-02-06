import numpy as np 
import updates 
import tqdm
from basic_util import make_2D_square_walls, initialise_boids
np.random.seed(200)



#initalise boids
N_b = 200
dim = 2
pos_all, vel_all = initialise_boids(N_b, dim)
print(pos_all.shape)
#update N times 
N = 10
results = np.zeros((N, 2, N_b, dim))                                           
for j in tqdm.tqdm(range(N)):
    results[j] = updates.serial_update(pos_all, vel_all)
np.save('results_serial.npy', results)



            
    