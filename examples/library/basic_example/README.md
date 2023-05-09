# Basic library examples.

### example.py

Run with `python example.py` to perform a fit in library mode.

### memtest.py

Loop over the `process_configs` to calculate descriptors and observe absence of memory leaks; note 
that memory leak may seem present if using old versions of MPI because of shared array allocation, 
or Python caching on some operating systems. Memory usage should plateau eventually. 
