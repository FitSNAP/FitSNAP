"""
Python script using library API to:
- Loop over all configurations in parallel and perform the transpose trick C = A^T * A and d = A^T * b.
- Sum these `C` and `d` arrays for all configurations.
- Perform least squares (or ridge regression) fit.

Usage:

    mpirun -np P python example.py

Afterwards, use the `in.run` LAMMPS script to run MD with:

    mpirun -np P lmp -in in.run

NOTE: This workflow is under development and therefore script requires changes.

- `settings` variable can be a dictionary like the example provided, or path to a fitsnap input script.
- `alval`: Ridge regression regularization parameter.
- Comment or uncomment `least_squares` or `ridge` at end of script to choose fitting method.

"""

from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
import numpy as np
from scipy.linalg import lstsq
from sys import float_info as fi
from sklearn.linear_model import Ridge

def least_squares(c, d):
    """
    Normal least squares fit.
    """
    coeffs, residues, rank, s = lstsq(c, d, 1.0e-13)
    return coeffs

def ridge(c, d):
    """
    Least squares fit with ridge regularization.
    """
    alval = 1.e-6
    reg = Ridge(alpha = alval, fit_intercept = False)
    reg.fit(c, d)
    return reg.coef_.T # return transpose if using sklearn ridge

def error_analysis(instance):
    """
    Calculate errors associated with a fitsnap instance that does not have shared arrays for the 
    entire A matrix or b vector, e.g. like we have when doing transpose trick. Here we loop over 
    all configurations and accumulate errors one at a time.

    Args:
        instance: fitsnap instance that contains a valid `fit`.

    Prints total MAE for all configurations in the data set.
    TODO: Organize this to calculate group errors or other kinds of errors.
    """

    # Get total number of atoms and configs across all procs for calculating average errors.
    nconfigs_all = len(fitsnap.pt.shared_arrays["number_of_atoms"].array)
    natoms_all = fitsnap.pt.shared_arrays["number_of_atoms"].array.sum()

    energy_mae = 0.0
    force_mae = 0.0
    stress_mae = 0.0
    for i, configuration in enumerate(instance.data):
        # TODO: Add option to print descriptor calculation progress on single proc.
        # if (i % 1 == 0):
        #    self.pt.single_print(i)
        a,b,w = instance.calculator.process_single(configuration, i)
        aw, bw = w[:, np.newaxis] * a, w * b
        
        pred = a @ coeffs

        # Energy error.
        energy = pred[0,0]
        ediff = np.abs(energy-b[0])
        energy_mae += ediff/nconfigs_all

        # Force error.
        ndim_force = 3
        nrows_force = ndim_force * configuration['NumAtoms']
        force = pred[1:nrows_force+1,0]
        force_mae += np.sum(abs(force-b[1:nrows_force+1]))/(3*natoms_all)

        # Stress error.
        stress = pred[-6:,0]
        sdiff_mae = np.mean(np.abs(stress-b[-6:]))
        stress_mae += sdiff_mae/nconfigs_all
    # Good practice after a large parallel operation is to impose a barrier to wait for all procs to complete.
    instance.pt.all_barrier()

    # Reduce errors across procs.
    energy_mae = np.array([energy_mae])
    force_mae = np.array([force_mae])
    stress_mae = np.array([stress_mae])
    energy_mae_all = np.array([0.0])
    force_mae_all = np.array([0.0])
    stress_mae_all = np.array([0.0])
    comm.Allreduce([energy_mae, MPI.DOUBLE], [energy_mae_all, MPI.DOUBLE])
    comm.Allreduce([force_mae, MPI.DOUBLE], [force_mae_all, MPI.DOUBLE])
    comm.Allreduce([stress_mae, MPI.DOUBLE], [stress_mae_all, MPI.DOUBLE])

    # Print errors.
    if (rank==0):
        print(energy_mae_all[0])
        print(force_mae_all[0])
        print(stress_mae_all[0])


# Declare a communicator (this can be a custom communicator as well).
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# Create an input dictionary containing settings.
settings = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 6,
    "rcutfac": 4.67637,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "Ta",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 0,
    "quadraticflag": 0,
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "stress": 1
    },
"ESHIFT":
    {
    "Ta": 0.0
    },
"SOLVER":
    {
    "solver": "SVD",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"SCRAPER":
    {
    "scraper": "JSON" 
    },
"PATH":
    {
    "dataPath": "../../Ta_Linear_JCP2014/JSON"
    },
"OUTFILE":
    {
    "metrics": "Ta_metrics.md",
    "potential": "Ta_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "hybrid/overlay zero 10.0 zbl 4.0 4.8",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "* * zbl 73 73"
    },
"EXTRAS":
    {
    "dump_descriptors": 1,
    "dump_truth": 1,
    "dump_weights": 1,
    "dump_dataframe": 1
    },
"GROUPS":
    {
    "group_sections": "name training_size testing_size eweight fweight vweight",
    "group_types": "str float float float float float",
    "smartweights": 0,
    "random_sampling": 0,
    "Displaced_A15" :  "1.0    0.0       100             1               1.00E-08",
    "Displaced_BCC" :  "1.0    0.0       100             1               1.00E-08",
    "Displaced_FCC" :  "1.0    0.0       100             1               1.00E-08",
    "Elastic_BCC"   :  "1.0    0.0     1.00E-08        1.00E-08        0.0001",
    "Elastic_FCC"   :  "1.0    0.0     1.00E-09        1.00E-09        1.00E-09",
    "GSF_110"       :  "1.0    0.0      100             1               1.00E-08",
    "GSF_112"       :  "1.0    0.0      100             1               1.00E-08",
    "Liquid"        :  "1.0    0.0       4.67E+02        1               1.00E-08",
    "Surface"       :  "1.0    0.0       100             1               1.00E-08",
    "Volume_A15"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09",
    "Volume_BCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09",
    "Volume_FCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09"
    },
"MEMORY":
    {
    "override": 0
    }
}

# Alternatively, settings could be provided in a traditional input file:
#settings = "../../Ta_Linear_JCP2014/Ta-example.in"

# Create a FitSnap instance using the communicator and settings:
fitsnap = FitSnap(settings, comm=comm, arglist=["--overwrite"])

# Scrape configurations to create and populate the `snap.data` list of dictionaries with structural info.
fitsnap.scrape_configs()

# Allocate `C` and `d` fitting arrays.
a_width = fitsnap.calculator.get_width()
c = np.zeros((a_width,a_width)) # This will also include weights.
d = np.zeros((a_width,1))

# Create fitsnap dictionaries (optional if you want access to distributed lists of groups, etc.)
fitsnap.calculator.create_dicts(len(fitsnap.data))
# Create `C` and `d` arrays for solving lstsq with transpose trick.
a_width = fitsnap.calculator.get_width()
c = np.zeros((a_width,a_width))
d = np.zeros((a_width,1))
c_all = np.zeros((a_width,a_width))
d_all = np.zeros((a_width,1))
for i, configuration in enumerate(fitsnap.data):
    # TODO: Add option to print descriptor calculation progress on single proc.
    if (i % 10 == 0):
        fitsnap.pt.single_print(i)
    a,b,w = fitsnap.calculator.process_single(configuration, i)
    aw, bw = w[:, np.newaxis] * a, w * b

    cm = np.matmul(np.transpose(aw), aw)
    dm = np.matmul(np.transpose(aw), bw[:,np.newaxis])
    c += cm
    d += dm
# Good practice after a large parallel operation is to impose a barrier to wait for all procs to complete.
fitsnap.pt.all_barrier()

# Reduce C and D arrays across procs.
comm.Allreduce([c, MPI.DOUBLE], [c_all, MPI.DOUBLE])
comm.Allreduce([d, MPI.DOUBLE], [d_all, MPI.DOUBLE])

if rank == 0:

    # Perform least squares fit.
    coeffs = least_squares(c_all,d_all)
    #coeffs = ridge(c_all, d_all)

    # Now `coeffs` is owned by all procs, good for parallel error analysis.

    # Calculate errors for this instance (not required).
    # error_analysis(fitsnap)

    # Write LAMMPS files.
    # NOTE: Without error analysis, `fitsnap.solver.errors` is an empty list and will not be written to file.
    fitsnap.output.output(coeffs, fitsnap.solver.errors)
