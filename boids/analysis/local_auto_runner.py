from sys import platform
import os 
import numpy as np
from subprocess import call 
import tqdm
if platform == 'linux':
    print('Script is for local running!!')
    exit()

rep = 3
max_boids = 1000
procs = 4
dim = 2 
N_it = 50
file_name = 'basic_mpi.py'


for i in range(rep):
    for j in tqdm.tqdm(range(200, 1000, 100)):
        call('mpirun -n %s python ./../calculation/%s --nb %s --d %s'%(procs, file_name, j, dim), shell=True)
        



