Run FitSNAP
===========

**If you want to get started immediately with interactive examples**, please see our 
`Colab Python notebook tutorial <tutorialnotebook_>`_

.. _tutorialnotebook: https://colab.research.google.com/github/FitSNAP/FitSNAP/blob/master/tutorial.ipynb

There are two ways to run FitSNAP: (1) the python executable or (2) as a python library. The 
executable version is reserved for the most basic operation of fitting on a dataset, where data
is scraped from a directory, configurations are input to LAMMPS to calculate descriptors, and the 
machine learning problem is solved. The python library provides more flexibility and allows one to
modify the steps in that fitting process.

Before using the executable, you must set the path to your FitSNAP directoy in your 
:code:`PYTHONPATH` environment variable, e.g.

.. code-block:: console

    export PYTHONPATH="path/to/FitSNAP:$PYTHONPATH"

which is conveniently placed in a :code:`~/.bashrc` or :code:`~/.bash_profile` file. Then fits
can be performed with the executable by doing

.. code-block:: console

   python -m fitsnap3 input.in --option

where :code:`input.in` is a FitSNAP input file and :code:`option` is an acceptable
command line option. Options and input files are explained below.

.. toctree::
   :maxdepth: 1

   run_options
   run_input
   run_outputs
   run_library
