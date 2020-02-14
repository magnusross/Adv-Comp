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
import basic_util as util 
import argparse
np.random.seed(200)

parser = argparse.ArgumentParser(description='Spatial grid based parallel boids simulation, with MPI.')
parser.add_argument("--n", default=50, type=int, help="Number of iterations")
parser.add_argument("--nb", default=500, type=int, help="Number of boids")
parser.add_argument("--d", default=2, type=int, choices=[2, 3],
                        help="Number of dimensions")
parser.add_argument("--s", default=1000., type=float, help="Box size")
parser.add_argument("--r", default=100., type=float, help="Boids field of view")
parser.add_argument("--f", default='basic_results.txt', help="Results out filename")
parser.add_argument("--w", default=False, type=bool, help="Write results to disk (bool)")
args = parser.parse_args()


N_IT = args.n
N_B = args.nb
DIM = args.d 
BOX_SIZE = np.ones(DIM, dtype=float) * args.s
RADIUS = args.r
FILE_NAME = args.f
SAVE = args.w


MASTER = 0
INDICES_TO_MANAGE = 1
BCAST_POS = 2
BCAST_VEL = 3

comm = MPI.COMM_WORLD
N_proc = comm.Get_size()
task_id = comm.Get_rank()


if task_id == MASTER:
    t1 = MPI.Wtime()
    # initiase data stuctures 
    pos_all, vel_all = util.initialise_boids(N_B, BOX_SIZE)
    
    results = np.zeros((N_IT, 2, N_B, DIM))     

    boids_index = util.make_proc_boid_ind(N_B, N_proc - 1)
    for i in range(N_proc-1):
        # send processors the boids they will manage 
        comm.send(boids_index[i], i+1, tag=INDICES_TO_MANAGE)
    
    for i in range(N_IT):
        results[i] = pos_all, vel_all
        comm.Bcast([pos_all, MPI.DOUBLE], root=MASTER)
        comm.Bcast([vel_all, MPI.DOUBLE], root=MASTER)

        for j in range(N_proc-1):
            # collate data from workers 
            proc_len = boids_index[j][1] - boids_index[j][0]
            mini_pos = np.empty((proc_len, DIM))
            mini_vel = np.empty((proc_len, DIM))
            
            tag_pos = int(str(i) + str(j + 1) + '0')
            tag_vel = int(str(i) + str(j + 1) + '1')

            comm.Recv([mini_pos, MPI.DOUBLE], tag=tag_pos)
            comm.Recv([mini_vel, MPI.DOUBLE], tag=tag_vel)
            # update data structures 
            pos_all[boids_index[j][0]:boids_index[j][1]] = mini_pos
            vel_all[boids_index[j][0]:boids_index[j][1]] = mini_vel
    comm.Barrier()

    t2 = MPI.Wtime()
        
    f = open(FILE_NAME, 'a+')
    f.write('%s %s %s %s %s %s %s\n'%(N_proc, N_IT, N_B, DIM, args.s, RADIUS, t2 - t1))
    print('%s %s %s %s %s %s %s\n'%(N_proc, N_IT, N_B, DIM, args.s, RADIUS, t2 - t1))
    f.close()
    if SAVE:
        np.save('basic_data_%s_%s.npy'%(N_B, N_IT), results)   



if task_id != MASTER:
    my_inds = comm.recv(tag=INDICES_TO_MANAGE)
    
    for i in range(N_IT):
        pos_all = np.empty((N_B, DIM))
        comm.Bcast([pos_all, MPI.DOUBLE], root=MASTER)

        vel_all = np.empty((N_B, DIM))
        comm.Bcast([vel_all, MPI.DOUBLE], root=MASTER)
        my_upd_pos, my_upd_vel = updates.basic_update(my_inds, pos_all, 
                                                        vel_all, BOX_SIZE, radius=RADIUS)

        tag_pos = int(str(i) + str(task_id) + '0')
        tag_vel = int(str(i) + str(task_id) + '1')

        comm.Send([my_upd_pos, MPI.DOUBLE], MASTER, tag=tag_pos)
        comm.Send([my_upd_vel, MPI.DOUBLE], MASTER, tag=tag_vel)
    
    comm.Barrier()
        
    






    
    