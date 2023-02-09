"""
Use LAMMPS to loop over configurations and evaluate your model for forces and energies.
Before using this script:
- Edit pair style and pair coeff commands below.
- Edit masses, mass 1, mass 2, etc.

Usage:
    
    python evaluate.py /path/to/descriptor_file /path/to/pt_file

TODO: Make this use FitSNAP library components like the test_error_nofit linear example. 
"""

import sys
import lammps as lammps
import torch
lmp = lammps.lammps()

descriptor_file = str(sys.argv[1]) # mliap descriptor file
pt_file = str(sys.argv[2]) # pytorch FitTorch_Pytorch.pt file

# before defining the pair style, must do the following:

import lammps.mliap
lammps.mliap.activate_mliappy(lmp)

nconfigs = 1000
for m in range(1,nconfigs+1):
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("boundary p p p")
    lmp.command(f"read_data data/DATA_{m}")
    lmp.command("mass 1 69.723")
    lmp.command("mass 2 14.0067")

    # first need to declare empty model:
    lmp.command(f"pair_style mliap model mliappy LATER descriptor sna {descriptor_file}")
    lmp.command("pair_coeff * * Ga N")

    # load model from disk:
    model = torch.load(pt_file)
    
    # connect pytorch model to mliap pair style
    lammps.mliap.load_model(model)
    
    # run calculation
    lmp.command("run 0")

    # get forces and energies from lammps
    forces = lmp.numpy.extract_atom("f") # Nx3 array
    energy = lmp.numpy.extract_compute("PE",0,0)

    lmp.command("clear")

