import numpy as np 
import matplotlib.pyplot as plt
import numba 
import numba_boids_rules as b
import numpy_boids_rules as p 
import tqdm

def update_boids(pos_all, vel_all, rules=[]):
    v = 0.
    for i in range(len(pos_all)):
        for rule in rules:
            v += rule(i, pos_all, vel_all)
        vel_all[i] += v 
        pos_all[i] += vel_all[i]
        v = 0.
        
    return pos_all, vel_all  

#initalise boids
N_b = 50 
pos_rand  = (2*np.random.rand(N_b, 2) - 1) * 100
vel_rand  = (2*np.random.rand(N_b, 2) - 1) * 0.1 + np.array([10., 0])


#update N times 
N = 500
results = np.zeros((N, 2, N_b, 2))                                                 
for j in tqdm.tqdm(range(N)):
    results[j] = update_boids(pos_rand, vel_rand, rules=[b.rule_avoid,
                                                        b.rule_com,
                                                        b.rule_match])
np.save('results.npy', results)



            
    