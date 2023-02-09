from ase import Atoms
from ase.io import read, write

# read data

frames1 = read("/path/to/*.xyz", index=":")
frames2 = read("/path/to/another/*.xyz", index=":")

frames = frames1 + frames2 

nconfigs = len(frames)
print(f"Found {nconfigs} configs")

count = 1
for atoms in frames:
    write(f"data/DATA_{count}", atoms, format="lammps-data")
    count += 1

