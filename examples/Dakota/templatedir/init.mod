#NOTE: This script can be modified for different atomic structures, 
# units, etc. See in.elastic for more info.
#

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
variable up equal 1.0e-6
 
# Uncomment one of these blocks, depending on what units
# you are using in LAMMPS and for output

# metal units, elastic constants in eV/A^3
#units		metal
#variable cfac equal 6.2414e-7
#variable cunits string eV/A^3
#variable aunits string A

# metal units, elastic constants in GPa
units		metal
variable cfac equal 1.0e-4
variable cunits string GPa
variable aunits string A

# real units, elastic constants in GPa
#units		real
#variable cfac equal 1.01325e-4
#variable cunits string GPa
#variable aunits string A

# Define minimization parameters
variable etol equal 0.0 
variable ftol equal 1.0e-10
variable maxiter equal 10000
variable maxeval equal 10000
variable dmax equal 1.0e-2

# generate the box and atom positions using a diamond lattice
variable a equal 3.51304

boundary	p p p

lattice         bcc $a
region		box prism 0 1.0 0 1.0 0 1.0 0.0 0.0 0.0
create_box	1 box
create_atoms	1 box
displace_atoms  all move 0.01 0.01 0.01

# Need to set mass to something, just to satisfy LAMMPS
mass 1 1.0e-20
#mass 2 1.0e-20

