
# <!----------------BEGIN-HEADER------------------------------------>
# ## FitSNAP3
# A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package
#
# _Copyright (2016) Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
# This software is distributed under the GNU General Public License_
# ##
#
# #### Original author:
#     Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
#     http://www.cs.sandia.gov/~athomps
#
# #### Key contributors (alphabetical):
#     Mary Alice Cusentino (Sandia National Labs)
#     Nicholas Lubbers (Los Alamos National Lab)
#     Charles Sievers (Sandia National Labs)
#     Adam Stephens (Sandia National Labs)
#     Mitchell Wood (Sandia National Labs)
#
# #### Additional authors (alphabetical):
#     Elizabeth Decolvenaere (D. E. Shaw Research)
#     Stan Moore (Sandia National Labs)
#     Steve Plimpton (Sandia National Labs)
#     Gary Saavedra (Sandia National Labs)
#     Peter Schultz (Sandia National Labs)
#     Laura Swiler (Sandia National Labs)
#
# <!-----------------END-HEADER------------------------------------->

from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.scrapers.scraper_factory import scraper
from fitsnap3lib.calculators.calculator_factory import calculator
from fitsnap3lib.solvers.solver_factory import solver
#from fitsnap3lib.io.output import output
from fitsnap3lib.io.outputs.output_factory import output
from fitsnap3lib.io.input import Config
from random import random
from mpi4py import MPI
import numpy as np
# Use this to check memory:
from psutil import virtual_memory
import psutil
#config = Config()
#pt = ParallelTools()


class FitSnap:
    """ This classes houses the functions needed for machine learning a potential, start to finish.

    Attributes:
        scraper (:obj:`class` Scraper): instance of the Scraper class for gathering configs
        data (:obj:`list`): list of dictionaries, where each configuration of atoms has its own dictionary
        calculator (:obj:`class` Calculator): instance of the Calculator class for calculating descriptors and fitting data
        solver (:obj:`class` Solver): instance of the Solver class for performing a fit
        fit: numpy array of fitting coefficients from linear models
        delete_data (:obj:`bool`): deletes the data list (if True) after a fit, useful to make False 
                                   if looping over fits.
    """
    def __init__(self, input=None, comm=None, arglist=None):
        """
        Args:
            input (str): Optional path to input file when using library mode.
            comm: Optional MPI communicator when using library mode.
            arglist (list): Optional list of cmd line args when using library mode.
        """
        snapid = id(self)
        print(f"fitsnap id: {snapid}")
        # TODO: Is it okay to create an instance on each proc like this?
        #       Or do we need to create on one proc and broadcast to all others?
        # TODO: Create pt instance on proc 0, then scatter to all other procs in communicator.

        # Basic send:
        """
        self.test = [1,2,3]
        if (comm.Get_rank() == 0):
            comm.send(self.test, dest=1, tag=11)
            print(f"Sent test")
        print(f"rank {comm.Get_rank()} before barrier")
        comm.Barrier()
        if (comm.Get_rank() == 1):
            print("Receiving test")
            self.test = comm.recv(source=0, tag=11)
            print("Recieved")
            self.test[0] = 2
        print(self.test)
        """

        # Trying to send a pt object:
        """
        if (comm.Get_rank() == 0):
            self.pt = ParallelTools(snapid, comm=comm)
            comm.send(self.pt, dest=1, tag=11)
            print(f"Sent pt")
        print(f"rank {comm.Get_rank()} before barrier")
        comm.Barrier()
        if (comm.Get_rank() == 1):
            print("Receiving pt")
            self.pt = comm.recv(source=0, tag=11)
            print("Recieved")
        """

        self.comm = comm

        # Basic shared memory:
        # NOTE: This shows that we can make shared arrays inside the fitsnap object, perhaps 
        #       not needing them in parallel tools?
        # create a shared array of size 1000 elements of type double
        """
        size = 1000 
        itemsize = MPI.DOUBLE.Get_size() 
        if comm.Get_rank() == 0: 
            nbytes = size * itemsize 
        else: 
            nbytes = 0

        # on rank 0, create the shared block
        # on rank 1 get a handle to it (known as a window in MPI speak)
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm) 

        # create a numpy array whose data points to the shared mem
        buf, itemsize = win.Shared_query(0) 
        assert itemsize == MPI.DOUBLE.Get_size() 
        self.ary = np.ndarray(buffer=buf, dtype='d', shape=(size,)) 

        # in process rank 1:
        # write the numbers 0.0,1.0,..,4.0 to the first 5 elements of the array
        if comm.rank == 1: 
            self.ary[:5] = np.arange(5)

        # wait in process rank 0 until process 1 has written to the array
        comm.Barrier() 

        # check that the array is actually shared and process 0 can see
        # the changes made in the array by process 1
        if comm.rank == 0: 
            print(self.ary[:10])

        #print(f"rank {comm.Get_rank()} comm: {comm}")
        #assert(False)
        """

        self.pt = ParallelTools(snapid, comm=comm)
        # Note that each proc in comm will create a different pt object, but the shared arrays should still share 
        # memory within the communicator.
        print(id(self.pt))
        self.config = Config(self.pt, input, arguments_lst=arglist)

        self.scraper = scraper(self.config.sections["SCRAPER"].scraper, self.pt, self.config)
        self.data = []
        self.calculator = calculator(self.config.sections["CALCULATOR"].calculator, self.pt, self.config)
        self.solver = solver(self.config.sections["SOLVER"].solver, self.pt, self.config)
        self.output = output(self.config.sections["OUTFILE"].output_style, self.pt, self.config)

        #assert(False)
        self.fit = None
        self.multinode = 0
        """
        Make False if don't want to delete data object.
        Useful for using library to loop over fits.
        """
        self.delete_data = True
        if self.config.sections["EXTRAS"].only_test:
            self.fit = self.output.read_fit()

        if (hasattr(self.pt, "lammps_version")):
            if (self.config.sections['CALCULATOR'].nonlinear and (self.pt.lammps_version < 20220915) ):
                raise Exception(f"Please upgrade LAMMPS to 2022-09-15 or later to use nonlinear solvers.")
       
    #@pt.single_timeit 
    def scrape_configs(self):
        """Scrapes configurations of atoms and creates a list of configurations called `data`."""

        @self.pt.single_timeit
        def decorated_scrape_configs():
            self.scraper.scrape_groups()
            self.scraper.divvy_up_configs()
            self.data = self.scraper.scrape_configs()
            del self.scraper
        decorated_scrape_configs()

    #@pt.single_timeit 
    def process_configs(self):
        """Calculate descriptors for all configurations in the :code:`data` list"""

        @self.pt.single_timeit
        def decorated_process_configs():

            # preprocess the configs (only if nonlinear)

            if (not self.solver.linear):
                if (self.pt._rank==0): 
                    print("Nonlinear solver, preprocessing configs.")
                self.calculator.preprocess_allocate(len(self.data))
                for i, configuration in enumerate(self.data):
                    self.calculator.preprocess_configs(configuration, i)

            self.calculator.create_a()
            if (self.solver.linear):
                for i, configuration in enumerate(self.data):
                    #self.pt.single_print(i)
                    self.calculator.process_configs(configuration, i)
            else:
                for i, configuration in enumerate(self.data):
                    self.calculator.process_configs_nonlinear(configuration, i)

            if (self.delete_data):
                del self.data
            self.calculator.collect_distributed_lists()

            # calculator.extras() has dataframe processing specific to linear solvers only

            # Test shared array capability.
            # First check that ranks can modify data shared with other ranks:
            """
            if (self.pt._rank==0):
                self.pt.shared_arrays["a"].array[0,0] = 1e9
                self.ary[0] = 100
            #self.comm.Barrier()
            # wait in other ranks until process 1 has written to the array
            self.pt._comm.Barrier()
            #if (self.pt._rank==1):
            #    self.pt.shared_arrays["a"].array[0,0] = 1e12
            print(f"rank {self.pt._rank} array: {self.pt.shared_arrays['a'].array}")
            print(self.ary[0:5])
            """
            # Check the memory
            """
            process = psutil.Process()
            print(f"rank {self.pt._rank} procmem: {process.memory_info().rss}")
            assert(False)
            """
            
            if (self.solver.linear):
                self.calculator.extras()

        decorated_process_configs()

    #@pt.single_timeit 
    def perform_fit(self):
        """Solve the machine learning problem with descriptors as input and energies/forces/etc as 
           targets"""

        @ self.pt.single_timeit
        def decorated_perform_fit():
            if not self.config.args.perform_fit:
                return
            elif self.fit is None:

                if self.solver.linear:
                    self.solver.perform_fit()
                else:
                    # Perform nonlinear fitting on 1 proc only
                    if(self.pt._rank==0):
                        self.solver.perform_fit()
            else:
                self.solver.fit = self.fit
                
        # If not performing a fit, keep in mind that the `configs` list is None 
        # for nonlinear models. Keep this in mind when performing error 
        # analysis.
        
        @self.pt.single_timeit
        def fit_gather():
            self.solver.fit_gather()
        @self.pt.single_timeit
        def error_analysis():
            self.solver.error_analysis()

        decorated_perform_fit()
        fit_gather()
        error_analysis()

    #@pt.single_timeit  
    def write_output(self):
        @self.pt.single_timeit
        def decorated_write_output():
            if not self.config.args.perform_fit:
                return
            self.output.output(self.solver.fit, self.solver.errors)
        decorated_write_output()
