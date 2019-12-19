from sys import platform
import os 
from subprocess import call 
import tqdm
if platform == 'linux':
    print('Script is for local running!!')
    exit()

rep = 5
max_boids = 1000
procs = 5
dim = 2 
N_it = 50
file_name = 'space_grid_mpi.py'

for i in range(rep):
    for j in tqdm.tqdm(range(0, max_boids, 25)):
        call('mpirun -n %s python ./../calculation/%s --nb %s --d %s'%(procs, file_name, j, dim), shell=True)
        # print(i, j)



