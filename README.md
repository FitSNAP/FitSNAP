<!----------------BEGIN-HEADER------------------------------------>
## FitSNAP3 
A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package

_Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
##

#### Original author: 
    Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
    http://www.cs.sandia.gov/~athomps 

#### Key contributors (alphabetical):
    Mary Alice Cusentino (Sandia National Labs)
    Nicholas Lubbers (Los Alamos National Lab)
    Adam Stephens (Sandia National Labs)
    Mitchell Wood (Sandia National Labs)

#### Additional authors (alphabetical): 
    Elizabeth Decolvenaere (D. E. Shaw Research)
    Stan Moore (Sandia National Labs)
    Steve Plimpton (Sandia National Labs)
    Gary Saavedra (Sandia National Labs)
    Peter Schultz (Sandia National Labs)
    Laura Swiler (Sandia National Labs)
    
<!-----------------END-HEADER------------------------------------->

#### Using this package:
* [Required] This package expects a python 3.6+ version. 
* [Required] Compile LAMMPS (lammps.sandia.gov) as a shared library, detailed instructions can be found here `docs/LAMMPS.md`. If you can open python and run `import lammps; lmp = lammps.lammps()` and it succeeds, you should be good.

#### Installing:
* Put this directory somewhere that you like.
* Add this directory in your Python path.
* See docs/INSTALL.md for more information

#### Running:
* `python -m fitsnap3 [options] infile`
* Command line options can be seen with `python -m fitsnap3 -h`
* Input files are described by `docs/TEMPLATE.in` and `docs/GROUPLIST.template`
* Examples of published SNAP interatomic potentials can be found in `examples/`

#### Version Control:
* `stable` branch is the latest validated stable version of the code. It is recommended for new users. 
* `master` branch reflects all of the latest code changes. It is intended for active developers and requires an up-to-date version of LAMMPS as well. 
