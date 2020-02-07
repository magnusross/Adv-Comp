import math
import argparse

def make_runscript(N_proc, N_b,  mpi_path, N_it=50,
                    wall_time='00:06:00',
                    template_path='runjob_temp.sh',
                    res_name='results.txt'):
    
    nodes = math.ceil(N_proc / 16)
    ppn = math.ceil(N_proc / nodes) 

    temp_lines = open(template_path, 'r').readlines()

    temp_lines[1] = '#PBS -l nodes=%s:ppn=%s\n'%(nodes, ppn)
    temp_lines[2] = '#PBS -l walltime=%s\n'%wall_time

    temp_lines[-1] = 'mpiexec -machinefile $CONF -np %s python %s --n %s --nb %s --f %s\n'%(N_proc, mpi_path, N_it, N_b, res_name)

    return temp_lines

parser = argparse.ArgumentParser(description='Test scaling with increased boids')

parser.add_argument("--b", default=2000, type=int, help="Num boids")
parser.add_argument("--dir", default='./NB_Scaling_RS/', type=str, help='Dir to save runscripts in')
parser.add_argument("--mpip", default='./../calculation/space_grid_mpi.py', type=str, help='Path to MPI code')
parser.add_argument("--rname", default='results.txt', type=str, help='name of results file')

args = parser.parse_args()

N_B = args.mb
DIR = args.dir
MPI_PATH = args.mpip
RES_NAME = args.rname
proc_list = [5, 10, 17, 26, 37, 50, 65]
for p in proc_list:
    rs_lines = make_runscript(p, N_B, MPI_PATH, res_name=RES_NAME)
    f = open(DIR + '%s_%s'%(p, N_B), 'w+')
    for l in rs_lines:
        f.write(l)
    f.close()