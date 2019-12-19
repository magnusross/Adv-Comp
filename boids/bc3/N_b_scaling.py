import math
def make_runscript(N_proc, N_b, N_it=100, 
                    wall_time='00:05:00',
                    mpi_path='./../calculation/space_grid.py',
                    template_path='runjob_temp.sh'):
    
    nodes = math.ceil(N_proc / 16)
    ppn = math.ceil(N_proc / nodes) 

    temp_lines = open(template_path, 'r').readlines()

    temp_lines[1] = '#PBS -l nodes=%s:ppn=%s'%(nodes, ppn)
    temp_lines[2] = '#PBS -l walltime=%s'%wall_time

    temp_lines[-1]

    

                    