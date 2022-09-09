"""
Demonstrate how to load a model from the python side. This is essentially the same as 
lammps/examples/mliap/in.mliap.pytorch.Ta06A, except that python is the driving program, and lammps
is in library mode.

Serial:

    python mliap_pytorch_Ta.py

Parallel:

    mpirun -np 4 python mliap_pytorch_Ta.py

"""

before_loading =\
"""# Demonstrate MLIAP/PyTorch interface to linear SNAP potential

# Initialize simulation

variable nsteps index 200000
variable nrep equal 6
variable a equal 3.1803
units           metal

# generate the box and atom positions using a BCC lattice

variable nx equal ${nrep}
variable ny equal ${nrep}
variable nz equal ${nrep}

boundary        p p p

lattice         bcc $a
region          box block 0 ${nx} 0 ${ny} 0 ${nz}
create_box      2 box
create_atoms    1 box
mass 1 183.84
mass 2 9.012182

set     group all type/fraction 2 0.05 3590153 # Change 5% of W to He
group   tungsten         type 1
group   beryllium        type 2

#create_atoms 2 random 6 1234 NULL
#create_atoms 2 single 0.75 0.75 0.75
#create_atoms 2 single 3.3 0.75 9.7
#create_atoms 2 single 5 5 5

# choose potential

# include Ta.pytorch.mliap

# DATE: 2014-09-05 UNITS: metal CONTRIBUTOR: Aidan Thompson athomps@sandia.gov CITATION: Thompson, Swiler, Trott, Foiles and Tucker, arxiv.org, 1409.3880 (2014)

# Definition of SNAP potential Ta
# Assumes 1 LAMMPS atom type

variable zblcutinner equal 4
variable zblcutouter equal 4.8
variable zblz equal 73

# Specify hybrid with SNAP, ZBL

pair_style hybrid/overlay &
zbl ${zblcutinner} ${zblcutouter} &
mliap model mliappy LATER &
descriptor sna WBe.mliap.descriptor
pair_coeff 1 1 zbl 74 74
pair_coeff 1 2 zbl 74 4
pair_coeff 2 2 zbl 4 4 
pair_coeff * * mliap W Be
"""
after_loading =\
"""

# Setup output

compute  eatom all pe/atom
compute  energy all reduce sum c_eatom

compute  satom all stress/atom NULL
compute  str all reduce sum c_satom[1] c_satom[2] c_satom[3]
variable press equal (c_str[1]+c_str[2]+c_str[3])/(3*vol)

thermo_style    custom step temp epair c_energy etotal press v_press
thermo          10
thermo_modify norm yes

# Set up NVE run

timestep 0.5e-3
neighbor 1.0 bin
neigh_modify once no every 1 delay 0 check yes

# Run MD

dump 1 all xyz 100 dump.xyz
velocity all create 12000.0 4928459000 rot yes mom yes #loop geom
fix 1 all nvt temp 6000.0 12000.0 0.05
#fix 1 all nve
run             ${nsteps}
undump 1
"""

from mpi4py import MPI
import lammps

lmp = lammps.lammps(cmdargs=['-echo','both'])
me = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()
print("Proc %d out of %d procs has" % (me,nprocs),lmp)

# Before defining the pair style, one must do the following:
import lammps.mliap
lammps.mliap.activate_mliappy(lmp)
# Otherwise, when running lammps in library mode,
# you will get an error:
# "ERROR: Loading MLIAPPY coupling module failure."

# Setup the simulation and declare an empty model
# by specifying model filename as "LATER"
lmp.commands_string(before_loading)

# Define the model however you like. In this example
# we load it from disk:
import torch
#model = torch.load('Ta.mliap.pytorch.pt')
model = torch.load('../FitTorch_Pytorch.pt')

# Connect the PyTorch model to the mliap pair style.
lammps.mliap.load_model(model)
  
# run the simulation with the mliap pair style
lmp.commands_string(after_loading)

MPI.Finalize()
