Executable
==========

Here we explain how to modify FitSNAP when running as an `Executable <Run.html#executable>`__. First 
we begin with an explanation of what goes on under the good when running FitSNAP as an executable 
with

.. code-block:: console

   python -m fitsnap3 input.in

There is a certain sequence of functions that is explained here, and coded in
:code:`fitsnap3/__main__.py`. Specifically, the :code:`main()` function uses the FitSNAP library to 
execute the following sequence of functions that perform a fit::

    from fitsnap3lib.parallel_tools import ParallelTools
    pt = ParallelTools()
    from fitsnap3lib.io.input import Config
    config = Config(arguments_lst = ["/path/to/FitSNAP/input/script", "--overwrite"])
    from fitsnap3lib.fitsnap import FitSnap

    def main():
        try:
            initialize_fitsnap_run()
            snap = FitSnap()
            snap.scrape_configs() 
            snap.process_configs()
            pt.all_barrier()
            snap.perform_fit()
            snap.write_output()
        except Exception as e:
            output.exception(e)

From the above code, it is seen that we first run the 
:code:`fitsnap3lib.initialize.initialize_fitsnap_run()` function. This simply prepares necessary 
imports and outputs settings. The rest of the main program execution relies on functions in the 
FitSNAP library. These are accessed by declaring a FitSNAP object with

.. code-block:: console

   snap = FitSNAP()

This can be achieved in any external python script, provided the necessary imports shown above 
are used, and instatiating the :code:`pt` and :code:`config` objects as we did above. This 
:code:`snap` object has functions located in :code:`fitsnap3lib.fitsnap`, and the code that these
functions depends on can be seen by observing :code:`fitsnap3lib/fitsnap.py`. These functions can 
be executed in any order desired by the user. The library also provides a deeper level of control, 
that we will explain in `Library <Library.html>`__. Examples of using the library to perform a variety of 
tasks outside the usual FitSNAP main program execution are located in 
https://github.com/FitSNAP/FitSNAP/tree/master/examples/library. 

Further explanations on how to modify FitSNAP as an executable are explained below.

Data & configuration extraction
-------------------------------

After creating the FitSNAP object in :code:`__main__.py`, the first step is scraping the configs.
Then we process the configs (calculate the descriptors) with :code:`snap.process_configs()`.
This calls the function in :code:`fitsnap.py`::

    def process_configs(self):
        self.calculator.create_a()
        for i, configuration in enumerate(self.data):
            self.calculator.process_configs(configuration, i)
        del self.data
        self.calculator.collect_distributed_lists()
        self.calculator.extras()

The :code:`Calculator` class in :code:`calculators/calculator.py` has a :code:`create_a` method 
which allocates the size of the :code:`a` and :code:`b` matrices, containing data such as 
descriptors and target energies/forces. :code:`calculators/calculator.py` also has a 
:code:`process_configs` method which is overwritten by user-defined derived class, e.g. 
:code:`LammpsSnap` in :code:`lammps_snap.py`. The :code:`calculator.process_configs` method 
therefore gets directed to the method in the derived class, which depends on the particular 
calculator being used.

Modifying the output dataframe
------------------------------

The Pandas dataframe is used for linear solvers to store information about the fit.

The :code:`error_analysis` function in :code:`solvers/solver.py` builds a dataframe containing 
arrays from :code:`pt.shared_arrays` and :code:`pt.fitsnap_dict`. If you want to add your own column 
to the dataframe, it must first be declared/allocated as a :code:`pt.fitsnap_dict` in 
:code:`calculators/calculator.py`, with the :code:`pt.add_2_fitsnap` function. When extracting 
LAMMPS data in a particular calculator subclass, there are loops over energy :code:`bik` rows, force 
rows, and stress rows. These are located in :code:`lammps_snap.py` and :code:`lammps_pace.py`, in 
the :code:`_collect_lammps()` function. There it is seen that data is added to the 
:code:`pt.fitsnap_dict['Column_Name'][indices]` array, where :code:`'Column_Name'` is the name of 
the new column declared earlier, and :code:`'indices'` are the rows of the array.

When adding a new :code:`pt.fitsnap_dict`, realize that it's a :code:`DistributedList`; this means 
that a list of whatever declared size exists on each proc. There is a method 
:code:`collect_distributed_lists` in :code:`calculators/calculator.py` that gathers all these 
distributed lists on the root proc. 

Adding new input file keywords
------------------------------

First you need to choose what section the keyword you're adding is in. For example in the input
file you will see a :code:`CALCULATOR` section. If you want to add a keyword to this section, go to 
:code:`fitsnap3lib/io/sections/calculator_sections/calculator.py`, and use the existing keyword 
examples to add a new keyword. Likewise for other sections such as :code:`SOLVER`, we edit 
:code:`fitsnap3lib/io/sections/solver_sections/solver.py`. If you want to access this keyword later 
in the FitSNAP code somewhere, it is done with :code:`config.sections['SOLVER'].new_keyword` for 
example.  

