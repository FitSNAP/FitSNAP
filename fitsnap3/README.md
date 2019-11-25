## FitSNAP3 
A Python Package For Training SNAP Interatomic Potentials for use in LAMMPS Molecular Dynamics

_Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
##

#### A short description of each component part of the FitSNAP package will be provided below along with an overall workflow of the code.
#### This folder should be on your PYTHONPATH, and if you change the name of this folder (fitsnap3) then your command line execution will have to reflect this change: `python3 -m newfoldername [inputfile] [options]`

#### __bispecopt.py__
  - Parses the variables from input file, values are appended to the dictionary 'bispec_options'. Constructs a list of the necessary bispectrum components used in this calculation, this is also added to the options dictionary. The bispec_options dictionary is important because it is passed to nearly all routines of this code, so if you are adding features that are tied to input variables make sure they are appended to this dictionary.

#### __deploy.py__
  - This interfaces with the LAMMPS python library either serially (compute_single) or by binding one LAMMPS per process up to the number of threads defined by the -j# command line switch (compute_partial). Either route returns a sorted list of bispectrum components and reference potential contributions per configuration, these are assembled into the training matrix.

#### __geometry.py__
  - A simple routine that manipulates the training data in order to conform to LAMMPS standards of lattice definition and periodic boundaries. The main function is calculating a proper rotation matrix and applying this to all positions, forces and stress components.

#### __linearfit.py__
  - Constructs the linear least squares problem, solves using a numpy or scikitlearn algorithm defined by user input, and computes the residuals between predicted and training data. There are four key components to constructing the least squares problem: their variable names are A,b,w,and x.
  - A : Matrix of bispectrum components representing the training data imported from json files. Rows of A correspond to the energy, per-atom forces and stress components of each training configuration. Columns of A are determined by the twojmax limit that is defined by user input.
  - b : Column vector of ground truth energies, forces and stress taken from the higher fidelity model (most commonly density funcitonal theory results). If you are using a reference potential (i.e. zbl) then these values are subtracted from the ground truth energy, force, stress read from json training files. Reference potential is supplied by the user in the [REFERENCE] section of the input file, see example cases for more information.
  - w : Row vector of weights to be applied to linear regression problem to favor or suppress certain portions of the training data.
  - x : Column vector which is the solution to the linear problem min||w*A*x-b||

#### __main__.py
  - Central routine for the overall work flow. Parses input files and launches each of the necessary subroutines. The ConfigParser package is used to import variables from input files, please read the documentation of this package if you wish to add features that require new user inputs.

#### __runlammps.py__
  - Constructs the list of LAMMPS commands that are used to calculate bispectrum components as well as reference potential contributions to energy, force and stress. extract_computes() and extract_computes_np() are import functions that accesses the memory used in LAMMPS, this allows for a direct construction of numpy arrays that are used to construct various parts of the linear regression problem.

#### __scrape.py__
  - Parses training data based on what defined in the groupFile input file and constructs dictionaries for each training configuration for easy data access by other routines. The weight vector is also constructed here, with an added wrinkle if you are using the Boltzman weighting or cross validation options (BOLTZT and compute_testerrs).

#### __serialize.py__
  - Prepares data structures for file I/O
