from sys import platform
import os 

if platform == 'linux':
    print('Setting environment variables...')
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
import updates
import utilities
np.random.seed(200)
MASTER = 0
N_IT = 50
N_B = 500
DIM = 2

INDICES_TO_MANAGE = 1
BCAST_POS = 2
BCAST_VEL = 3

comm = MPI.COMM_WORLD
N_proc = comm.Get_size()
task_id = comm.Get_rank()


if task_id == MASTER:

    pos_all, vel_all = utilities.initialise_boids(N_B, DIM)
    
    results = np.zeros((N_IT, 2, N_B, DIM))     

    boids_index = utilities.make_proc_boid_ind(N_B, N_proc - 1)
    for i in range(N_proc-1):
        comm.send(boids_index[i], i+1, tag=INDICES_TO_MANAGE)
    
    for i in range(N_IT):
        print(i)
        results[i] = pos_all, vel_all

        comm.Bcast([pos_all, MPI.DOUBLE], root=MASTER)
        comm.Bcast([vel_all, MPI.DOUBLE], root=MASTER)

        for j in range(N_proc-1):
            proc_len = boids_index[j][1] - boids_index[j][0]
            mini_pos = np.empty((proc_len, DIM))
            mini_vel = np.empty((proc_len, DIM))
            
            tag_pos = int(str(i) + str(j + 1) + '0')
            tag_vel = int(str(i) + str(j + 1) + '1')
            comm.Recv([mini_pos, MPI.DOUBLE], tag=tag_pos)
            comm.Recv([mini_vel, MPI.DOUBLE], tag=tag_vel)

            pos_all[boids_index[j][0]:boids_index[j][1]] = mini_pos
            vel_all[boids_index[j][0]:boids_index[j][1]] = mini_vel
    np.save('results_mpi.npy', results)   



if task_id != MASTER:
    my_inds = comm.recv(tag=INDICES_TO_MANAGE)
    
    for i in range(N_IT):
        pos_all = np.empty((N_B, DIM))
        comm.Bcast([pos_all, MPI.DOUBLE], root=MASTER)

        vel_all = np.empty((N_B, DIM))
        comm.Bcast([vel_all, MPI.DOUBLE], root=MASTER)
        my_upd_pos, my_upd_vel = updates.update_my_boids(my_inds, pos_all, vel_all)

        tag_pos = int(str(i) + str(task_id) + '0')
        tag_vel = int(str(i) + str(task_id) + '1')

        comm.Send([my_upd_pos, MPI.DOUBLE], MASTER, tag=tag_pos)
        comm.Send([my_upd_vel, MPI.DOUBLE], MASTER, tag=tag_vel)
    
        
    






    
    