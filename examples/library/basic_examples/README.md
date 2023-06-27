# Basic library examples.

### example1.py

Run with `python example.py` to perform a fit in library mode.

### example2.py

Scrape JSON files and evaluate the fitting matrices for each configuration separately.

### memtest.py

Loop over the `process_configs` to calculate descriptors and observe absence of memory leaks; note 
that memory leak may seem present if using old versions of MPI because of shared array allocation, 
or Python caching on some operating systems. Memory usage should plateau eventually. 
