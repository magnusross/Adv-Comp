import numpy as np 
import updates 
from basic_util import initialise_boids
np.random.seed(200)



#initalise boids
N_b = 1000
dim = 2
box_size = np.array([100., 100.])
pos_all, vel_all = initialise_boids(N_b, box_size)
print(pos_all.shape)
#update N times 
N = 50
results = np.zeros((N, 2, N_b, dim))                                           
for j in range(N):
    results[j] = updates.serial_update(pos_all, vel_all, box_size)
np.save('results_serial.npy', results)



            
    