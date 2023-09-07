FitSnap
=======

.. automodule:: fitsnap3lib.fitsnap
    :members:

:code:`FitSnap` contains instances of two helper classes that help with MPI communication and 
settings; :code:`ParallelTools` and :code:`Config`, respectively. These two classes are 
explained below.

Parallel Tools
--------------

Parallel Tools are a collection of data structures and functions for transferring data in  
FitSNAP workflow in a massively parallel fashion. Currently these tools reside in a single 
file :code:`parallel_tools.py` which houses some classes described below.

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
