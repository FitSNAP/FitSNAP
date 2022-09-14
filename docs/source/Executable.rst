Executable
==========

When running FitSNAP as an executable with

.. code-block:: console

   python -m fitsnap3 input.in

there is a certain sequence of functions that is explained here, and coded in
:code:`fitsnap3/__main__.py`. Specifically, the :code:`main()` function uses the FitSNAP library to 
execute the following sequence of functions that perform a fit.

.. code-block:: console

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
that we will explain in the next section. Examples of using the library to perform a variety of 
tasks outside the usual FitSNAP main program execution are located in 
https://github.com/FitSNAP/FitSNAP/tree/master/examples/library.

