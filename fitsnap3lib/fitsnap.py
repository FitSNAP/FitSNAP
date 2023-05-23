
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
#     Drew Rohskopf (Sandia National Labs)
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
from fitsnap3lib.io.outputs.output_factory import output
from fitsnap3lib.io.input import Config
import random


class FitSnap:
    """ 
    This classes houses the objects needed for machine learning a potential, start to finish.

    Args:
        input (str): Optional dictionary or path to input file when using library mode; defaults to 
                     None for executable use.
        comm: Optional MPI communicator when using library mode; defaults to None.
        arglist (list): Optional list of cmd line args when using library mode.

    Attributes:
        pt (:obj:`class` ParallelTools): Instance of the ParallelTools class for helping MPI 
                                         communication and shared arrays.
        config (:obj:`class` Config): Instance of the Config class for initializing settings, 
                                      initialized with a ParallelTools instance.
        scraper (:obj:`class` Scraper): Instance of the Scraper class for gathering configs.
        data (:obj:`list`): List of dictionaries, where each configuration of atoms has its own 
            dictionary.
        calculator (:obj:`class` Calculator): Instance of the Calculator class for calculating 
            descriptors and fitting data.
        solver (:obj:`class` Solver): Instance of the Solver class for performing a fit.
    """
    def __init__(self, input=None, comm=None, arglist: list=[]):
        self.comm = comm
        # Instantiate ParallelTools and Config instances belonging to this FitSnap instance.
        # NOTE: Each proc in `comm` creates a different `pt` object, but shared arrays still share 
        #       memory within `comm`.
        self.pt = ParallelTools(comm=comm)
        self.pt.all_barrier()
        self.config = Config(self.pt, input, arguments_lst=arglist)
        if self.config.args.verbose:
            self.pt.single_print(f"FitSNAP instance hash: {self.config.hash}")
        # Instantiate other backbone attributes.
        self.scraper = scraper(self.config.sections["SCRAPER"].scraper, self.pt, self.config) \
            if "SCRAPER" in self.config.sections else None
        self.calculator = calculator(self.config.sections["CALCULATOR"].calculator, self.pt, self.config) \
            if "CALCULATOR" in self.config.sections else None
        self.solver = solver(self.config.sections["SOLVER"].solver, self.pt, self.config) \
            if "SOLVER" in self.config.sections else None
        self.output = output(self.config.sections["OUTFILE"].output_style, self.pt, self.config) \
            if "OUTFILE" in self.config.sections else None

        self.fit = None
        self.multinode = 0

        # Optionally read a fit.
        if "EXTRAS" in self.config.sections and self.config.sections["EXTRAS"].only_test:
            self.fit = self.output.read_fit()
        # Check LAMMPS version if using nonlinear solvers.
        if (hasattr(self.pt, "lammps_version")):
            if (self.config.sections['CALCULATOR'].nonlinear and (self.pt.lammps_version < 20220915) ):
                raise Exception(f"Please upgrade LAMMPS to 2022-09-15 or later to use nonlinear solvers.")

        if (self.pt._number_of_nodes > 1 and not self.config.sections["SOLVER"].true_multinode):
            raise Exception(f"Must use ScaLAPACK solver when using > 1 node or you'll fit to 1/nodes of data.")
    
    def __del__(self):
        """Override deletion statement to free shared arrays owned by this instance."""
        self.pt.free()
        del self

    def __setattr__(self, name: str, value):
        """
        Override set attribute statement to prevent overwriting important attributes of an instance.
        """
        protected = ("pt", "config")
        if name in protected and hasattr(self, name):
            raise AttributeError(f"Overwriting {name} is not allowed; instead change {name} in place.")
        else:
            super().__setattr__(name, value)
       
    def scrape_configs(self, delete_scraper: bool = False):
        """
        Scrapes configurations of atoms and creates an instance attribute list of configurations called `data`.
        
        Args:
            delete_scraper: Boolean determining whether the scraper object is deleted or not after scraping. Defaults 
                            to False. Since scraper can retain unwanted memory, we delete it in executable mode.
        """
        @self.pt.single_timeit
        def scrape_configs():
            self.scraper.scrape_groups()
            self.scraper.divvy_up_configs()
            self.data = self.scraper.scrape_configs()
            if delete_scraper:
                del self.scraper
        scrape_configs()

    def process_configs(self, data: list=None, allgather: bool=False, delete_data: bool=False):
        """
        Calculate descriptors for all configurations in the :code:`data` list and stores info in the shared arrays.
        
        Args:
            data: Optional list of data dictionaries to calculate descriptors for. If not supplied, we use the list 
                  owned by this instance.
            allgather: Whether to gather distributed lists to all processes to just to head proc. In some cases, such as 
                       processing configs once and then using that data on multiple procs, we must allgather.
            delete_data: Whether the data list is deleted or not after processing.Since `data` can retain unwanted 
                         memory after processing configs, we delete it in executable mode.
        """

        if data is not None:
            data = data
        elif hasattr(self, "data"):
            data = self.data
        else:
            raise NameError("No list of data dictionaries to process.")

        # Zero distributed index before parallel loop over configs.
        self.calculator.distributed_index = 0

        @self.pt.single_timeit
        def process_configs():
            self.calculator.allocate_per_config(data)
            # Preprocess the configs if nonlinear fitting.
            if (not self.solver.linear):
                if self.config.args.verbose: 
                    self.pt.single_print("Nonlinear solver, preprocessing configs.")
                self.calculator.preprocess_allocate(len(data))
                for i, configuration in enumerate(data):
                    self.calculator.preprocess_configs(configuration, i)
            # Allocate shared memory arrays.
            self.calculator.create_a()
            # Calculate descriptors.
            if (self.solver.linear):
                for i, configuration in enumerate(data):
                    # TODO: Add option to print descriptor calculation progress on single proc.
                    #if (i % 1 == 0):
                    #   self.pt.single_print(i)
                    self.calculator.process_configs(configuration, i)
            else:
                for i, configuration in enumerate(data):
                    self.calculator.process_configs_nonlinear(configuration, i)
            # Delete instance-owned data dictionary to save memory.
            if delete_data and hasattr(self, "data"):
                del self.data
            # Gather distributed lists in `self.pt.fitsnap_dict` to root proc.
            self.calculator.collect_distributed_lists(allgather=allgather)
            # Optional extra steps.
            if self.solver.linear:
                self.calculator.extras()

        process_configs()

    def perform_fit(self):
        """Solve the machine learning problem with descriptors as input and energies/forces/etc as 
           targets"""
        @ self.pt.single_timeit
        def fit():
            if not self.config.args.perform_fit:
                return
            elif self.fit is None:
                if self.solver.linear:
                    self.solver.perform_fit()
                else:
                    # Perform nonlinear fitting on 1 proc only.
                    if(self.pt._rank==0):
                        self.solver.perform_fit()
            else:
                self.solver.fit = self.fit
                
        # If not performing a fit, keep in mind that the `configs` list is None 
        # for nonlinear models. Keep this in mind when performing error 
        # analysis.
        
        def fit_gather():
            self.solver.fit_gather()

        @self.pt.single_timeit
        def error_analysis():
            self.solver.error_analysis()

        fit()
        fit_gather()
        error_analysis()

    def write_output(self):
        @self.pt.single_timeit
        def write_output():
            if not self.config.args.perform_fit:
                return
            self.output.output(self.solver.fit, self.solver.errors)

            #self.output.write_lammps(self.solver.fit)
            #self.output.write_errors(self.solver.errors)
        write_output()
