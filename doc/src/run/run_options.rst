Command-line options
====================

At run time, FitSNAP recognizes command-line options which may be used in any order.
Either the full word or a one or two letter abbreviation can be used:

* :ref:`-k or -\-keyword <keyword>`
* :ref:`-l or -\-lammpslog <lammpslog>`
* :ref:`-l or -\-log <log>`
* :ref:`-nf or -\-nofit <nofit>`
* :ref:`-ns or -\-nscreen <nscreen>`
* :ref:`-\-overwrite <overwrite>`
* :ref:`-sc or -\-screen <screen>`
* :ref:`-ps or -\-pscreen <pscreen>`
* :ref:`-r or -\-relative <relative>`
* :ref:`-sc or -\-screen <screen>`
* :ref:`-s2f or -\-screen2file <screen2file>`
* :ref:`-tb or -\-tarball <tarball>`
* :ref:`-v or -\-verbose <verbose>`

For example, the FitSNAP executable might be launched as follows:

.. code-block:: bash

    mpirun -np 4 python -m fitsnap3 input.in --overwrite

----------

.. _keyword:

**-\-keyword GROUP NAME VALUE**

Replace or add input keyword group GROUP, key NAME, with value VALUE. Type carefully; a misspelled 
key name or value may be silently ignored.

----------

.. _lammpslog:

**-\-lammpslog**

Writes a LAMMPS log file for descriptor calculations. Since LAMMPS log files get overwritten with 
configuration that we calculate descriptors for, this will give the last log file that occured. This 
is mainly useful if you want to see the LAMMPS outputs and settings that were used to calculate 
descriptors.

----------

.. _log:

**-\-log FILE**

Write outputs and warnings from Python's native logging facility to a file named FILE. We currently 
do not output any information with the logging facility yet, so the capability mainly exists for 
developers and debugging for now.

----------

.. _nofit:

**-\-nofit**

This will run FitSNAP without performing a fit. Only descriptors will be calculated. This is useful 
when paired with `dump_dataframe = 1 in the [EXTRAS] section <Run_input.html#extras>`__

----------

.. _nscreen:

**-\-nscreen**

Print outputs from the Output class for each node.

----------

.. _overwrite:

**-\-overwrite**

Overwrite output files. Otherwise, output files will not be overwritten and FitSNAP will error. 
This protects existing fits from being overwritten.

----------

.. _pscreen:

**-\-pscreen**

Print outputs from the Output class for each processor.

----------

.. _relative:

**-\-relative**

Put output files in the directory of the FitSNAP input file. For example if the FitSNAP input file
is located at :code:`foo/bar/input.in`, then running FitSNAP like

.. code-block:: bash

    mpirun -np 4 python -m fitsnap3 foo/bar/input.in --relative

will write output files in the directory :code:`foo/bar`. Without the :code:`--relative` option, the 
files will be written in the current directory.

----------

.. _screen:

**-\-screen**

Print outputs to screen.

----------

.. _screen2file:

**-\-screen2file FILE**

Print screen to a file named FILE.

----------

.. _tarball:

**-\-tarball**

Package SNAP fit files into a hashed tarball named :code:`fit-<hash>.tar.gz`. This is only useful 
when fitting with SNAP descriptors since multiple files are used by LAMMPS. The tarball contains 
a SNAP parameter file, a SNAP coefficient file, a :code:`*.mod` file containing the proper pair 
style to use, and an example LAMMPS input file showing how to use this potential.

ACE descriptors, on the other hand, only use :code:`*.yace` files in LAMMPS, therefore no need for a 
tarball.

----------

.. _verbose:

**-\-verbose**

Show more detailed information about processing. Currently there are no verbose outputs, so this 
option exists purely for development and testing purposes.



