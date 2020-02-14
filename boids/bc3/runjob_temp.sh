                                                                                
#PBS -l nodes=2:ppn=14
#PBS -l walltime=02:00:00                                                                       
# Define the working directory                                                               
export MYDIR="/newhome/mr16064"
cd $PBS_O_WORKDIR
#------------------------------------------------                                            
# Determine which nodes the job has                                                          
# been allocated to and create a                                                             
# machinefile for mpirun                                                                     
#------------------------------------------------                                            
# Don't change anything below this line                                                      
#------------------------------------------------                                            
# Get the job number                                                                         
export JOBNO="`echo $PBS_JOBID | sed s/.master.cm.cluster//`"
# Generate mpirun machinefile ------------------                                             
export CONF="$MYDIR/machines.$JOBNO"
for i in `cat $PBS_NODEFILE`;
do echo $i >> $CONF
done
# Get the number of processors ----------------                                              
export NUMPROC=`cat $PBS_NODEFILE|wc -l`
# Execute the code -----------------------------           
                                  
mpiexec -machinefile $CONF -np $NUMPROC python ./Adv-Comp/boids/space_grid_mpi.py