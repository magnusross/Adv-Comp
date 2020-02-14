Advanced Computational Physics 301: Parallel Boids
==================================================

Below are some instructions for running the code for the coursework for the unit Advanced Computational Physics 301 at the University of Bristol. All scripts mentioned below should be run from the directory they are in. Some of the terms used below are defined in the report, so it may make sense to read that first before running the code. 

Dependencies
------------

### Local

- `python 3.6.10`
- `mpi4py 2.0.0`
- `numpy 1.14.6`
- `numba 0.36.2`
- `scipy 1.3.1`
- `argparse 1.4.0`
- `matplotlib 3.1.1`

For animations:

- `ffmpeg 4.2`

### BC3

The following should be in `.bashrc` on BC3

- `module add languages/python-anaconda-4.2-3.5`

Structure
---------

The project is separated into several different directories the details of which
are listed below. All directoiries are under `boids`

- `calculation`

    This is the most important directory and contains all the code for the 3 parallel boids algorithms. Each algorithms as 2 scripts associated with it, and MPI based script that handles the parallisation, and a utils script that has a set of helper functions. The are also 2 rules scripts that contain the boids rules, one is for the basic method only and one is for the 2 grid bases methods, Grid and Balanced. Additionally there is an updates script that contains the update functions for all algorithms. Finally there is a script that runs the updates with no parallisation. All inner loop functions are compiled with `numba`. The process of generating data with these scripts is listed below. All code in this directory is documented.

- `analysis`

    This directory contains an IPython Notebook which was used to do all the analysis of the results for the report. The notebook is far from perfect, but should be reasonably readable. It also contains a directory `figures` which contains all the figures used in the report. Additionally there a script used to collect local results about the properties of the algorithms.

- `bc3`

    This directory contains shell and python scripts used to run jobs and collect results on BC3. These scripts are not pretty or really commented at all. These may not all work and may need some tweaking. 

- `animation`

    This directory contains 2 scripts that can generate animations of the boids in 2D or 3D. These scripts are quite simple, they are already set up to make an animation from so pre generated data in results. To make one using new data, you can just go in and change the `.npy` file it is loading and change the domain size. The scripts are currently setup for the Balanced method data structure and needs some fiddling with arrays to work with the other methods.

- `results`

    Contains `.txt` files with the data used to generate all the plots for the report, most of these were generated using BC3.

Getting Data
------------

To get data you can do the `mpirun` command in the `calculation` directory, on the method you would like to run. The scripts that run with MPI have the `_mpi` suffix (Balanced=`bal_grid_mpi.py`, Basic=`basic_mpi.py`, Grid=`grid_mpi.py`). There are a number of flags for the scripts which are documented in the scripts, the most important are:
 
- `--nb` Sets number of boids in the simulation
- `--n` Sets number of iterations 
- `--w` If 1 writes a `.npy` file with the boids data for the simulation to disk 
- `--d` Sets dimension of simulation (2D or 3D)

For example if you wanted to run the balanced method with 500 boids, for 100 iterations in 2D,using 4 processors and write the data to disk you would do:

    mpirun -n 4 python bal_grid_mpi.py --nb 500 --n 100 --d 2 --w 1 

The script will print some information about what was run and how long it took when it finishes execution. The output will look something like this:

    4 50 500 2 1000.0 100.0 4.834896087646484

These numbers are in order: the number of processors, the number of iterations, number of boids, the dimension, the domain size the boids field of view, and the execution time. This data is also automatically written to a text file. All data in the `results` directory has this format.

The Grid method is quite constrained in how many processors and what field of view you can use. Only a square number of processors + 1 can be used in 2D and a cube number + 1 in 3D, for example in 3D you could do `-n 28`. There are also constraints on the field of view for the grid method. Running the script with incompatible processor number or field of view will raise a value error.

A note on the three methods
-----------------------

Three different algorithms are implemented here, the details of which can be found in the report. Each implementation uses slightly different data structures, which is admittedly quite confusing. In general, the Grid method is a lot more convoluted and difficult to understand than the Basic and Balanced methods, and is not very well written. It is left in here because it was developed in the course of developing the Balanced method and they share a few similar characteristics. The Balanced and Basic methods use similar data structures, with separate arrays for position and velocity and a similar system for allocation boids to workers. In general they are easier to follow and make more sense.
