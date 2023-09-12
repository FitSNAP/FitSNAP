FitSnap
=======

The :code:`FitSnap` class houses all objects needed for performing a fit. These objects 
are instantiated from other core classes described in the rest of the docs. The two main 
inputs to a :code:`FitSnap` instance are (1) settings and (2) an MPI communicator. The 
settings can be a nested dictionary as shown in some examples, while the MPI communicator 
is typically the world communicator containing all resources dictated by the :code:`mpirun` 
command. After instantiating :code:`FitSnap` with these inputs, the instance will contain 
its own instance of :code:`ParallelTools` which houses functions and data structures that 
operate on the resources in the input communicator. The settings will be stored in an instance 
of the :code:`Config` class. The :code:`FitSnap` class is documented below.

.. automodule:: fitsnap3lib.fitsnap
    :members:

:code:`FitSnap` contains instances of two classes that help with MPI communication and 
settings; :code:`ParallelTools` and :code:`Config`, respectively. These two classes are 
explained below. In addition to these classes, there are other core classes 
:code:`Scraper`, :code:`Calculator`, and :code:`Solver` which are explained in later 
sections. 

Parallel Tools
--------------

Parallel Tools are a collection of data structures and functions for transferring data in  
FitSNAP workflow in a massively parallel fashion. 

The :code:`ParallelTools` instance :code:`pt` of a FitSNAP instance :code:`fs` can be used 
to create shared memory arrays, for example::

    # Create a shared array called `a`.
    fs.pt.create_shared_array('a', nrows, ncols)
    # Change the shared array at a rank-dependent element.
    fs.pt.shared_arrays['a'].array[rank,0] = rank
    # Observe that the change happened on all procs.
    print(f"Shared array on rank {rank}: {fs.pt.shared_arrays['a'].array}")

Currently these tools reside in a single file :code:`parallel_tools.py` which houses some 
classes described below.

.. automodule:: fitsnap3lib.parallel_tools
    :members:

Config
------

The Config class is used for storing settings associated with a FitSNAP instance. Throughout 
the code and library examples, you may see code snippets like::

    fs.config.sections["GROUPS"].group_table

where :code:`fs` is the FitSNAP instance being accessed. In this snippet, the :code:`sections` 
attribute contains keys, such as :code:`"GROUPS"`, which contains attributes like the group 
table which we can access. In this regard, :code:`fs.config` stores all the settings relevant 
to a particular FitSNAP instance or fit, which can then be easily accessed anywhere else 
throughout the code. 

.. automodule:: fitsnap3lib.io.input
    :members:
