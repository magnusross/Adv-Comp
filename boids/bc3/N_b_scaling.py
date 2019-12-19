import math
import argparse

def make_runscript(N_proc, N_b,  mpi_path, N_it=100,
                    wall_time='00:05:00',
                    template_path='runjob_temp.sh'):
    
    nodes = math.ceil(N_proc / 16)
    ppn = math.ceil(N_proc / nodes) 

    temp_lines = open(template_path, 'r').readlines()

    temp_lines[1] = '#PBS -l nodes=%s:ppn=%s\n'%(nodes, ppn)
    temp_lines[2] = '#PBS -l walltime=%s\n'%wall_time

    temp_lines[-1] = 'mpiexec -machinefile $CONF -np %s python %s --n %s --nb %s\n'%(N_proc, mpi_path, N_it, N_b)

    return temp_lines

parser = argparse.ArgumentParser(description='Test scaling with increased boids')
parser.add_argument("--sb", default=10, type=int, help="Step of N_b")
parser.add_argument("--mb", default=1000, type=int, help="Max boids")
parser.add_argument("--np", default=9, type=int, help='Number of processors')
parser.add_argument("--dir", default='./NB_Scaling_RS/', type=str, help='Dir to save runscripts in')
parser.add_argument("--mpip", default='./../calculation/space_grid.py', type=str, help='Path to MPI code')

args = parser.parse_args()

MAX_N_B = args.mb
STEP_N_B = args.sb
N_PROC = args.np
DIR = args.dir
MPI_PATH = args.mpip

for i in range(STEP_N_B, MAX_N_B, STEP_N_B):
    rs_lines = make_runscript(N_PROC, i, MPI_PATH)
    f = open(DIR + '%s_%s'%(N_PROC, i), 'w+')
    for l in rs_lines:
        f.write(l)
    f.close()


                    