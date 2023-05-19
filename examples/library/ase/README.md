# ASE examples

This directory contains examples for scraping with ASE, or calculating 
descriptors starting from an ASE Atoms object.

### example1.py

Starting with a list of ASE Atoms objects, loop through each configuration 
and calculate descriptors for each separately.

### example2.py

Starting with a list of ASE Atoms objects on each MPI process, where each
process has a different list, calculate all descriptors in parallel and store 
in the shared memory array for all processes.

### example3.py

Using a different ASE format (data not available, just illustration puposes).

### example4.py

Show how to use ASE scraper with groups.