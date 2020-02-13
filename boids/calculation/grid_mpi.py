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
import grid_util as util
import argparse
np.random.seed(100) #

parser = argparse.ArgumentParser(description='Spatial grid based parrallel boids simulation, with MPI.')
parser.add_argument("--n", default=500, type=int, help="Number of iterations")
parser.add_argument("--nb", default=500, type=int, help="Number of boids")
parser.add_argument("--d", default=2, type=int, choices=[2, 3],
                        help="Number of dimensions")
parser.add_argument("--s", default=300., type=float, help="Box size")
parser.add_argument("--r", default=50., type=float, help="Boids field of view")
parser.add_argument("--f", default='results.txt', help="Results out filename")
args = parser.parse_args()



N_IT = args.n
N_B = args.nb
DIM = args.d 
BOX_SIZE = np.ones(DIM, dtype=float) * args.s
RADIUS = args.r
FILE_NAME = args.f


comm = MPI.COMM_WORLD
N_proc = comm.Get_size()
task_id = comm.Get_rank()

SAVE = True
MASTER = 0
T_SIZE = 10
T_BOIDS = 11
T_N_BOIDS = 12
T_N_SIZE = 13

N_cell_ax = int((N_proc - 1)**(1/DIM))
nearest_pow = N_cell_ax ** DIM

if not nearest_pow == N_proc - 1:
    raise ValueError('Number of procs be square/cube number! + 1 ')

if args.s / N_cell_ax < RADIUS:
    raise ValueError('Boids radius too large, extends past nearest nieghbour.')

coord_list = util.make_proc_coord_list(N_cell_ax, DIM)
rank_to_coord = lambda i: np.array(coord_list[i-1])
coord_to_rank = lambda c: int(np.argwhere(np.all(coord_list == c, axis=1))) + 1


if task_id == MASTER:
    t1 = MPI.Wtime()
    all_boids = util.initialise_boids(N_B, N_cell_ax, BOX_SIZE)
    
    results = np.zeros((N_IT, N_B, 3, DIM)) 
    # send boids
    for i in range(N_IT): 
        N_proc_boids = []
        for j in range(1, N_proc):
            proc_boid = util.get_cells_boids(rank_to_coord(j), all_boids)
            comm.send(len(proc_boid), j, tag=T_SIZE)
            comm.Send([proc_boid, MPI.DOUBLE], j, tag=T_BOIDS)
            N_proc_boids.append(len(proc_boid))

        ind = 0
        for j in range(1, N_proc):
            n = N_proc_boids[j-1]
            comm.Recv([all_boids[ind:ind + n], MPI.DOUBLE], source=j, tag=int(str(T_BOIDS) +  str(j)))
            ind += n

        results[i] = all_boids
        util.assign_to_cells(all_boids, N_cell_ax, BOX_SIZE)
        # print(i)
    comm.Barrier()

    t2 = MPI.Wtime()

    f = open(FILE_NAME, 'a+')
    f.write('%s %s %s %s %s %s %s\n'%(N_proc, N_IT, N_B, DIM, args.s, RADIUS, t2 - t1))
    print('%s %s %s %s %s %s %s\n'%(N_proc, N_IT, N_B, DIM, args.s, RADIUS, t2 - t1))
    f.close()
    if SAVE:
        np.save('data_%s_%s.npy'%(N_B, N_IT), results)



else: 
    my_coords = rank_to_coord(task_id)
    # the cell label starts from 0 wheras working processor
    # ranks sart from 1 so cell_n = task_id - 1 
    my_nns = util.get_neigh(my_coords, N_cell_ax, DIM)
    
    for i in range(N_IT):
        N_my_boids = comm.recv(tag=T_SIZE)

        my_boids = np.zeros((N_my_boids, 3, DIM), dtype='float')
        comm.Recv([my_boids, MPI.DOUBLE], source=MASTER, tag=T_BOIDS)

        all_nn_boids = np.empty((0, 3, DIM))
        # now get neighbour boids 
        N_nn_boids = []
        for j in range(len(my_nns)):
            comm.send(N_my_boids, coord_to_rank(my_nns[j]), tag=T_N_SIZE)

        for j in range(len(my_nns)):
            N_nn_boids_req = comm.recv(source=coord_to_rank(my_nns[j]), tag=T_N_SIZE)
            N_nn_boids.append(N_nn_boids_req)#.wait())
        
        nn_boids = np.zeros((sum(N_nn_boids), 3, DIM)) 
        for j in range(len(my_nns)):
            comm.Isend([my_boids, MPI.DOUBLE], coord_to_rank(my_nns[j]), tag=T_N_BOIDS)
        
        ind = 0
        for j in range(len(my_nns)):
            comm.Irecv([nn_boids[ind:ind + N_nn_boids[j]], MPI.DOUBLE], 
                        source=coord_to_rank(my_nns[j]),tag=T_N_BOIDS)
            ind += N_nn_boids[j] 

        all_my_boids = np.vstack((my_boids, nn_boids))

        updates.grid_update(my_boids, all_my_boids, BOX_SIZE, radius=RADIUS)
        comm.Send([my_boids, MPI.DOUBLE], MASTER, tag=int(str(T_BOIDS) +  str(task_id)))
    
    comm.Barrier()
MPI.Finalize()

