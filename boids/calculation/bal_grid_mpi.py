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
import bal_grid_util as util 
import argparse
np.random.seed(200)

parser = argparse.ArgumentParser(description='Spatial grid based parallel boids simulation, with MPI.')
parser.add_argument("--n", default=500, type=int, help="Number of iterations")
parser.add_argument("--nb", default=500, type=int, help="Number of boids")
parser.add_argument("--d", default=2, type=int, choices=[2, 3],
                        help="Number of dimensions")
parser.add_argument("--s", default=500., type=float, help="Box size")
parser.add_argument("--r", default=20., type=float, help="Boids field of view")
parser.add_argument("--f", default='bal_results.txt', help="Results out filename")
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
BCAST_GRID = 4

T_VEL = 10
T_POS = 11
T_LABS = 12
T_GRID = 13
T_N_BOIDS = 14
T_N_SIZE = 15
T_N_GRID = 16 

comm = MPI.COMM_WORLD
N_proc = comm.Get_size()
task_id = comm.Get_rank()

if task_id == MASTER:

    t1 = MPI.Wtime()
    pos_all, vel_all = util.initialise_boids(N_B, BOX_SIZE)
    grid_all = util.initialise_grid(pos_all, BOX_SIZE, RADIUS)

    results = np.zeros((N_IT, 2, N_B, DIM)) 

    boids_index = util.make_proc_boid_ind(N_B, N_proc - 1)
    for i in range(N_proc-1):
        # send processors the boids they will manage 
        comm.send(boids_index[i], i+1, tag=INDICES_TO_MANAGE)
    
    for i in range(N_IT):
        results[i] = pos_all, vel_all

        comm.Bcast([pos_all, MPI.DOUBLE], root=MASTER)
        comm.Bcast([vel_all, MPI.DOUBLE], root=MASTER)
        comm.Bcast([grid_all, MPI.INT], root=MASTER)

        for j in range(N_proc-1):
            proc_len = boids_index[j][1] - boids_index[j][0]

            mini_pos = np.empty((proc_len, DIM))
            mini_vel = np.empty((proc_len, DIM))
            
            comm.Recv([mini_pos, MPI.DOUBLE], source=j+1, tag=T_POS)
            comm.Recv([mini_vel, MPI.DOUBLE], source=j+1, tag=T_VEL)

            pos_all[boids_index[j][0]:boids_index[j][1]] = mini_pos
            vel_all[boids_index[j][0]:boids_index[j][1]] = mini_vel

            upd_len = comm.recv(source=j+1, tag=T_N_GRID)

            diff_labs = np.empty(upd_len, dtype='int')
            new_grid = np.empty((upd_len, DIM), dtype='int')
            
            comm.Recv([diff_labs, MPI.INT], source=j+1, tag=T_LABS)
            comm.Recv([new_grid, MPI.INT], source=j+1, tag=T_GRID)
            # do cell updates 
            grid_all[diff_labs] = new_grid

    
    t2 = MPI.Wtime()
        
    f = open(FILE_NAME, 'a+')
    f.write('%s %s %s %s %s %s %s\n'%(N_proc, N_IT, N_B, DIM, args.s, RADIUS, t2 - t1))
    print('%s %s %s %s %s %s %s\n'%(N_proc, N_IT, N_B, DIM, args.s, RADIUS, t2 - t1))
    f.close()
    if SAVE:
        np.save('bal_grid_data_%s_%s.npy'%(N_B, N_IT) , results)


if task_id != MASTER:
    my_inds_tup = comm.recv(tag=INDICES_TO_MANAGE)
    my_labs = np.arange(my_inds_tup[0], my_inds_tup[1]) 

    
    for i in range(N_IT):
        pos_all = np.empty((N_B, DIM))
        comm.Bcast([pos_all, MPI.DOUBLE], root=MASTER)

        vel_all = np.empty((N_B, DIM))
        comm.Bcast([vel_all, MPI.DOUBLE], root=MASTER)

        grid_all = np.empty((N_B, DIM))
        comm.Bcast([grid_all, MPI.INT])

        updates.bal_grid_update(my_labs, grid_all, pos_all,
                                vel_all, BOX_SIZE, RADIUS)

        my_upd_pos = pos_all[my_labs]
        my_upd_vel = vel_all[my_labs]
        # calculates which boids have moved cell
        diff_labs, new_grid = util.get_grid_updates(my_labs, grid_all, pos_all, 
                                                     vel_all, BOX_SIZE, RADIUS)

        comm.Send([my_upd_pos, MPI.DOUBLE], MASTER, tag=T_POS)
        comm.Send([my_upd_vel, MPI.DOUBLE], MASTER, tag=T_VEL)

        comm.send(len(diff_labs), MASTER, tag=T_N_GRID)
        comm.Send([diff_labs, MPI.INT], MASTER, tag=T_LABS)
        comm.Send([new_grid, MPI.INT], MASTER, tag=T_GRID)

        # comm.Barrier()
        







        


                                                    
    
        




