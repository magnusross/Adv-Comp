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
import argparse
np.random.seed(200)

parser = argparse.ArgumentParser(description='Spatial grid based parrallel boids simulation, with MPI.')
parser.add_argument("--n", default=100, type=int, help="Number of iterations")
parser.add_argument("--nb", default=100, type=int, help="Number of boids")
parser.add_argument("--d", default=3, type=int, choices=[2, 3],
                        help="Number of dimensions")
parser.add_argument("--s", default=300., type=float, help="Box size")
parser.add_argument("--r", default=50., type=float, help="Boids field of view")
parser.add_argument("--f", default='results_basic.txt', help="Results out filename")
args = parser.parse_args()


N_IT = args.n
N_B = args.nb
DIM = args.d 
BOX_SIZE = np.ones(DIM, dtype=float) * args.s
RADIUS = args.r
FILE_NAME = args.f
SAVE = False


MASTER = 0
INDICES_TO_MANAGE = 1
BCAST_POS = 2
BCAST_VEL = 3

comm = MPI.COMM_WORLD
N_proc = comm.Get_size()
task_id = comm.Get_rank()
