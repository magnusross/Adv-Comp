#!/bin/bash  

export RS_DIR="./N_b_Scaling_RS/"



mkdir $RS_DIR
rm -f results*
rm -f *.o*
rm -f *.e*

export MPI_PATH="./../calculation/basic_mpi.py"
python N_b_scaling.py --sb 1000 --mb 15000 --np 37 --rname basic.txt --dir $RS_DIR --mpip $MPI_PATH

for i in $RS_DIR*; do 
    for ((j=0; j<3; j++)); do
        qsub -q veryshort $i
        sleep 5s 
    done
done 

rm -r $RS_DIR
sleep 10m 
mkdir $RS_DIR

export MPI_PATH="./../calculation/grid_mpi.py"
python N_b_scaling.py --sb 1000 --mb 15000 --np 37 --rname grid.txt --dir $RS_DIR --mpip $MPI_PATH

for i in $RS_DIR*; do 
    for ((j=0; j<3; j++)); do
        qsub -q veryshort $i
        sleep 5s 
    done
done 

rm -r $RS_DIR
sleep 10m 
mkdir $RS_DIR


export MPI_PATH="./../calculation/bal_grid_mpi.py"
python N_b_scaling.py --sb 1000 --mb 15000 --np 37 --rname grid.txt --dir $RS_DIR --mpip $MPI_PATH

for i in $RS_DIR*; do 
    for ((j=0; j<3; j++)); do
        qsub -q veryshort $i
        sleep 5s 
    done
done 

rm -r $RS_DIR
