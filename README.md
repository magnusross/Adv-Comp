Advanced Computational Physics 301: Parallel Boids
==================================================

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

### BC3

The following should be in `.bashrc` on BC3

- `module add languages/python-anaconda-4.2-3.5`

Structure
---------

The project is separated into several different directories the details of which
are listed below. All directoiries are under `boids`

- `calculation`

    This is the most important directory and contains all the code for the 3 parallel boids algorithms. Each algorithms as 2 scripts associated with it, and MPI based script that handles the parralleisation, and a utils script that has a set of helper functions. The are also 2 rules scripts that conrtain the boids rules, one is for the basic method only and one is for the 2 grid bases methods, Grid and Balanced. Adittionally there is an updates script that contains the update functions for all algorithms. Finally there is a script that runs the updates with no parralleisation. All inner loop functions are compiled with `numba`. The process of generating data with these scripts is listed below. All code in this directory is documented.

- `analysis`

    This directory contains an IPython Notebook which was used to do all the analysis of the results for the report. The notebook is far from perfect, but should be reasonaly readable. It also contains a directory `figures` which contains all the figures used in the report. Aditionallty there a script used to collect local results about the properties of the algorithms.

- `bc3`

    This directory contains shell and python scripts used to run jobs and collect results on BC3. These scripts are not pretty or really commented at all.

- `animation`

    This directory contains 2 scripts that can generate animations of the boids in 2D or 3D the process of generating an animation is below.

- `results`

    Contains `.txt` files with the data used to generate all the plots for the report, most of these were generated using BC3









Data structue a bit different for each implementaion 
all inner loop funtions complied with numba 

scripts must be run from the folder they are in

modules on bc3 

note on the structure of results 
