## FitSNAP3
A Python Package For Training SNAP Interatomic Potentials for use in LAMMPS Molecular Dynamics

_Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
##

#### A short description of each parent class of the FitSNAP package will be provided below along with an overall workflow of the code.
_This folder should be on your PYTHONPATH, and if you change the name of this folder (fitsnap3) then your command line execution will have to reflect this change: `python3 -m newfoldername [inputfile] [options]`_

#### __scrapers/__
  - Parent class that parses training data from stored files. 
#### __calculators/__
  - Parent class where feature vectors (descriptors) for the learned model are generated based on configurations parsed from scraper.
#### __solvers/__
  - Parent class that handles how regression (fitting) is carried out such that weights on feature vectors are found to minimize the loss function with respect to ground truth data (i.e. from DFT).
#### __io/__
  - Parent class that handles all input and output functions.

#### __parallel_tools.py__
  - This is the real star of the show in terms of functionality, all of the parent classes will lean on this to distribute tasks and collect data across ranks/nodes. Pull requests that modify this will be given heavy scrutiny, plenty of additional functionality can be more easily made within scrapers/calculators/solvers/io. 

![FitSNAP3 Code Flow Chart](FitSNAP3_CodeFlowChart.png)
