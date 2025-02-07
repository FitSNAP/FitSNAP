Quick Instructions
==================

Module names may be changed depending on your system, but the general procedure is the same.

Load modules::

    module purge
    module load gcc
    module load cmake  
    module load openmpi
    module load intel # Sometimes this helps mpi4py work

Create & activate conda environment with dependencies::

    conda create --name fitsnap python=3.10
    conda activate fitsnap
    python -m pip install numpy scipy scikit-learn virtualenv psutil pandas tabulate mpi4py Cython
    # For nonlinear (neural network) fitting:
    python -m pip install torch
    # For fitting ACE:
    python -m pip install sympy pyyaml
    # For contributing to docs:
    python -m pip install sphinx sphinx_rtd_theme sphinxcontrib-napoleon

Now we need to install LAMMPS.
Set the following environment variables::

    LAMMPS_DIR=/path/to/where/you/want/lammps # LAMMPS code will be in $LAMMPS_DIR
    FITSNAP_DIR=/path/to/where/you/want/FitSNAP # FitSNAP code will be in $FITSNAP_DIR

Get & build LAMMPS with Python library::

    git clone https://github.com/lammps/lammps $LAMMPS_DIR
    mkdir $LAMMPS_DIR/build-fitsnap
    cd $LAMMPS_DIR/build-fitsnap
    cmake ../cmake -DBUILD_SHARED_LIBS=yes \
                   -DMLIAP_ENABLE_PYTHON=yes \
                   -DPKG_PYTHON=yes \
                   -DPKG_ML-SNAP=yes \
                   -DPKG_ML-IAP=yes \
                   -DPKG_ML-PACE=yes \
                   -DPKG_SPIN=yes \
                   -DPYTHON_EXECUTABLE:FILEPATH=`which python`
    make
    make install-python

Get & prepare FitSNAP::

    git clone https://github.com/FitSNAP/FitSNAP $FITSNAP_DIR
    export PYTHONPATH=$FITSNAP_DIR:$PYTHONPATH # So you can run FitSNAP as executable

Fit a neural network for tantalum::

    cd $FITSNAP_DIR/examples/Ta_PyTorch_NN
    mpirun -np 2 python -m fitsnap3 Ta-example.in --overwrite

Run high-performance MD with this neural network potential::

    SITE_PACKAGES_DIR=`python -c "import site; print(site.getsitepackages()[0])"`
    export PYTHONPATH=${SITE_PACKAGES_DIR}:$PYTHONPATH # So that ML-IAP package can find torch for MD
    cd MD
    mpirun -np 4 ${LAMMPS_DIR}/fitsnap-build/lmp < in.run


For more details, or if you encounter errors, see `Installation <Installation.html>`__. 