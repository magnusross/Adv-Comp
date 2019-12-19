#!/bin/bash  

export RS_DIR="./N_b_Scaling_RS/"
export MPI_PATH="./../calculation/space_grid_mpi.py"


mkdir $RS_DIR
rm -f results*
rm -f *.o*
rm -f *.e*


python N_b_scaling.py --sb 10 --mb 40 --np 9 --dir $RS_DIR --mpip $MPI_PATH

for i in $RS_DIR*; do 
    for ((j=0; j<3; j++)); do
        qsub -q teaching $i
    done
done 

rm -r $RS_DIR

