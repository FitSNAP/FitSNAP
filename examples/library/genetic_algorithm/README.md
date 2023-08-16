# FitSnap3 Ta optimization

### Usage 

Parallel (NOTE: only descriptor design matrix is calculated in parallel) :

    mpirun -n 2 python3 libmod_optimize.py --fitsnap_in SNAP-Ta.in --optimization_style genetic_algorithm

Serial:

    python3 libmod_optimize.py --fitsnap_in SNAP-Ta.in --optimization_style genetic_algorithm

Replace SNAP-Ta.in with ACE-Ta.in to run an ACE fit.


### Files in this directory

- `SNAP-Ta.in` or `ACE-Ta.in`: FitSNAP input files for initial fit and 
template for iteratated fits.

- `script_optimize.py` : python script that sets up and performs
optimization, performing the initial fit and thereafter 
making use of the functions and objects in libmod_optimize.py.

- `libmod_optimize.py` : python script containing functions and
objects to perform hyperparameter optimizations from a 
FitSNAP library mode instance.

- `Seeds.npy` : numpy array of seeds for random number 
generation during the optimization loop.

### Hyperparameter optimization from memory

In this example, the hyperparameters for a SNAP or ACE model of Ta
are optimized for a fixed set of descriptors. That is, the
training matrix for the model is stored in memory, and a Ta
potential is repeatedly refit. The descriptors and initial
hyperparameters are controlled by the FitSNAP input Ta.in.

Currently, a simple cost function is evaluated using a 
modular cost/fitness evaluator object. By default, this is
<i>Q = w<sub>E</sub> E<sub>RMSE</sub> + w<sub>F</sub> F<sub>RMSE</sub></i>
where <i>w<sub>E</sub></i> and <i>w<sub>F</sub></i> are weights for the
respective root mean square error contributions for energy
and forces, respectively. One may also add to the cost 
function with energy difference objectives (e.g. like
adsorption or vacancy energies). No such objectives are 
included in this example, but the functionality may be 
found in the source code. The cost function is 
minimized by varying hyperparameters through optimization
algorithms. Two algorithms are currently implemented and 
may be selected through the 'optimization_style' input 
flag. These algorithms are simulated annealing and genetic
algorithms with input flags: 'simulated_annealing' and 
'genetic_algorithm', respectively. These algorightms
minimize the cost function by varying hyperparameters.
Currently the optimization is performed over the 'group 
weights', but this may easily be extended to other hyper-
parameters that do not change the entries in the descriptor
design matrix, such as regularization penalties.

### Other hyperparameters

For this example, model hyperparameters that do not affect
the SNAP or ACE descriptors (numerical form or number of 
descriptors) are optimized using a modified simulated 
annealing algorithm. For hyperparameters that <i>do</i> 
influence the numerical form and number of SNAP/ACE
descriptors, a separate hyperparameter optimization should
be performed. The optimized cutoffs and other such 
hyperparameters are taken from previous optimizations of 
SNAP/ACE potentials for Ta. While implementing optimization
over these parameters that change the numerical form or
number of SNAP/ACE descriptors would be straightforward, it
would not allow for the optimization to be performed from
descriptors stored in memory. It is recommended to perform
a gridsearch over these hyperparameters or to perform a 
more formal optimization.

### Reproducability

The seeds used to generate random numbers for each step are
saved in the 'seeds.npy' file. By default, the optimizer
will use the seeds in this folder first, before generating 
its own. This allows for the approximate reproducibility of
models from machine to machine. You should be able to 
reproduce, to within numerical precision, the reference for
this potential in 1Aug23_Standard.

### Notes

The initial calculation of the descriptors may be done in
parallel, but the optimization algorithm is serial. There
are some numpy parallelism tricks that may be used to take
advantage of multithreading for the optimization loop, but 
these have not been implemented for this example yet. 

For ACE fits alone: 
If one does not use the local RIDGE solver
for this optimization and instead uses the sklearn RIDGE
solver, some of the sklearn multithreading tools and flags
may be used. For example, setting OMP_NUM_THREADS=4, will 
allow implementations in sklearn that use openMP to access
4 threads. By default, it will often use all available 
thread, <i>but</i> on many HPC clusters, the default
OMP_NUM_THREADS flag is 1.
See https://scikit-learn.org/stable/computing/parallelism.html
for more info.

### Output: Save the STDOUT !

The optimizer will print ouput the final optimized 
potential after reaching a specified number of steps or
after reaching some target threshhold for accuracy. The
hyperparameters that yielded this fit are those last
printed to STDOUT. The output for the potential and 
corresponding parameters will be in standard fitsnap output
formats for SNAP or ACE (relevant coefficient files and .md 
metrics file). Note that none of the procedures will 
overwrite/change your fitsnap_in file, even
after optimization. <i>It will print the optimized group
weights for later use to STDOUT.</i>
NOTE: An optional parameter, "write_to_json", exists in the 
genetic_algorithm class to write the final output to a standard
JSON file. This can be read FitSnap using Python's JSON module. 
This parameter is set to 'True' in the script_optimize.py 
included in this folder, but is turned off by default.

#### Simulated annealing output
The hyperparameters are printed for <i>all</i>
monte carlo steps and written to STDOUT. If the STDOUT is
saved, FitSNAP inputs may be generated for any step in the
optimization. Some of which are local minima in
hyperparameter space and may be useful.

#### Genetic algorithm output

The hyperparameters for the best candidate from each 
generation are printed to SDTOUT. The initial generation of
candidates is generated using a latin hypercube sampling
scheme. These candidates may be reproduced by generating the 
corresponding fitsnap input file with the updated 
hyperparameters.

