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
np.random.seed(100) # 200 was problem

MASTER = 0
N_IT = 50
N_B = 2000
DIM = 3
BOX_SIZE = np.array([300., 300., 300.])


RADIUS = 70

T_SIZE = 10
T_BOIDS = 11
T_N_BOIDS = 12
T_N_SIZE = 13

comm = MPI.COMM_WORLD
N_proc = comm.Get_size()
task_id = comm.Get_rank()

# NN_Group = comm.group.Excl([0])
# nncomm =  comm.Create_group(NN_Group)

N_cell_ax = int((N_proc - 1)**(1/DIM))

nearest_pow = N_cell_ax ** DIM
if not nearest_pow == N_proc - 1:
    raise ValueError('Number of procs be square/cube number! + 1 ')

coord_list = utilities.make_proc_coord_list(N_cell_ax, DIM)
rank_to_coord = lambda i: np.array(coord_list[i-1])
coord_to_rank = lambda c: int(np.argwhere(np.all(coord_list == c, axis=1))) + 1




if task_id == MASTER:
    all_boids = utilities.init_cells(N_B, N_cell_ax, box_size=BOX_SIZE)
    
    results = np.zeros((N_IT, N_B, 3, DIM)) 
    # send boids
   
    for i in range(N_IT): 
        N_proc_boids = []
        for j in range(1, N_proc):
            proc_boid = utilities.get_cells_boids(rank_to_coord(j), all_boids)
            comm.send(len(proc_boid), j, tag=T_SIZE)
            comm.Send([proc_boid, MPI.DOUBLE], j, tag=T_BOIDS)
            N_proc_boids.append(len(proc_boid))

        ind = 0
        for j in range(1, N_proc):
            n = N_proc_boids[j-1]
            comm.Recv([all_boids[ind:ind + n], MPI.DOUBLE], source=j, tag=int(str(T_BOIDS) +  str(j)))
            ind += n

        results[i] = all_boids
        utilities.assign_to_cells(all_boids, N_cell_ax, BOX_SIZE)
        print(i)

    np.save('res_grid.npy', results)

else: 
    my_coords = rank_to_coord(task_id)
    # the cell label starts from 0 wheras working processor
    # ranks sart from 1 so cell_n = task_id - 1 
    my_nns = utilities.get_neigh(my_coords, N_cell_ax, DIM)
    
    for i in range(N_IT):
        N_my_boids = comm.recv(tag=T_SIZE)

        my_boids = np.zeros((N_my_boids, 3, DIM), dtype='float')
        comm.Recv([my_boids, MPI.DOUBLE], source=MASTER, tag=T_BOIDS)

        all_nn_boids = np.empty((0, 3, DIM))
        # now get neighbour boids 
        N_nn_boids = []
        for j in range(len(my_nns)):
            comm.isend(N_my_boids, coord_to_rank(my_nns[j]), tag=T_N_SIZE)

        for j in range(len(my_nns)):
            N_nn_boids_req = comm.irecv(source=coord_to_rank(my_nns[j]), tag=T_N_SIZE)
            N_nn_boids.append(N_nn_boids_req.wait())
        
        nn_boids = np.zeros((sum(N_nn_boids), 3, DIM)) 
        for j in range(len(my_nns)):
            comm.Isend([my_boids, MPI.DOUBLE], coord_to_rank(my_nns[j]), tag=T_N_BOIDS)
        
        ind = 0
        for j in range(len(my_nns)):
            comm.Irecv([nn_boids[ind:ind + N_nn_boids[j]], MPI.DOUBLE], 
                        source=coord_to_rank(my_nns[j]),tag=T_N_BOIDS)
            ind += N_nn_boids[j] 


        all_my_boids = np.vstack((my_boids, nn_boids))

        updates.grid_update_my_boids(my_boids, all_my_boids, BOX_SIZE, radius=RADIUS)

        # comm.send(N_my_boids, MASTER, tag=int(str(T_SIZE) +  str(task_id)))
        comm.Send([my_boids, MPI.DOUBLE], MASTER, tag=int(str(T_BOIDS) +  str(task_id)))

MPI.Finalize()

