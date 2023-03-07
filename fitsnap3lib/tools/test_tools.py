import numpy as np
import torch
import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class TestTools():
    """
    Class containing methods that help calculate errors with LAMMPS potentials.

    Attributes
    ----------

    input_script: str
        filename/location of FitSNAP input script

    test_option: str
        option for which test to run, e.g. "FiniteDifference"
    
    """
    def __init__(self, input_script):

        # import parallel tools and create pt object
        from fitsnap3lib.parallel_tools import ParallelTools
        # pt = ParallelTools(comm=comm)
        self.pt = ParallelTools()
        # don't check for existing fitsnap objects since we'll be overwriting things
        self.pt.check_fitsnap_exist = False
        from fitsnap3lib.io.input import Config
        # fitsnap_in = "../examples/Ta_Pytorch_NN/Ta-example.in"
        # fitsnap_in = "Ta-example.in" #ta_example_file.as_posix()
        self.config = Config(arguments_lst = [input_script, "--overwrite"])

    def finite_difference(self, group, config_index=0):
        """
        Calculate finite difference forces for the given group, and compare with model forces.

        Attributes
        ----------

        config_index: int
            index of configuration to test finite difference on
        """

        h = 1e-4 # size of finite difference

        # config.sections['BISPECTRUM'].switchflag = 1 # required for smooth finite difference
        self.config.sections['NETWORK'].manual_seed_flag = 1
        self.config.sections['NETWORK'].dtype = torch.float64
        # only perform calculations on displaced BCC structures
        self.config.sections['GROUPS'].group_table = {group:
                                                     {'training_size': 1.0,
                                                      'testing_size': 0.0,
                                                      'eweight': 100.0,
                                                      'fweight': 1.0,
                                                      'vweight': 1e-08}}
        # create a fitsnap object
        from fitsnap3lib.fitsnap import FitSnap
        self.snap = FitSnap()

        # get config positions
        self.snap.scrape_configs()
        # don't delete the data since we'll use it many times with finite difference
        self.snap.delete_data = False 

        # calculate model forces

        self.snap.process_configs()
        self.pt.all_barrier()
        self.snap.solver.create_datasets()
        (energies_model, forces_model) = self.snap.solver.evaluate_configs(config_idx=None, standardize_bool=True)

        start_indx = config_index

        assert (start_indx < len(self.snap.data)-1)

        errors = []
        for m in range(start_indx,start_indx+1):
            for i in range(0,self.snap.data[m]['NumAtoms']):
                for a in range(0,3):
                    # natoms = self.snap.data[m]['NumAtoms']
                    # calculate model energy with +h (energy1)

                    self.snap.data[m]['Positions'][i,a] += h
                    self.snap.calculator.distributed_index = 0
                    self.snap.calculator.shared_index = 0
                    self.snap.calculator.shared_index_b = 0
                    self.snap.calculator.shared_index_c = 0
                    self.snap.calculator.shared_index_dgrad = 0
                    self.snap.process_configs()
                    self.snap.solver.create_datasets()
                    (energies1, forces1) = self.snap.solver.evaluate_configs(config_idx=m, standardize_bool=False)

                    # calculate model energy with -h (energy2)

                    self.snap.data[m]['Positions'][i,a] -= 2.*h
                    #print(f"position: {snap.data[m]['Positions'][i,a]}")
                    self.snap.calculator.distributed_index = 0
                    self.snap.calculator.shared_index = 0
                    self.snap.calculator.shared_index_b = 0
                    self.snap.calculator.shared_index_c = 0
                    self.snap.calculator.shared_index_dgrad = 0
                    self.snap.process_configs()
                    self.snap.solver.create_datasets()
                    (energies2, forces2) = self.snap.solver.evaluate_configs(config_idx=m, standardize_bool=False)

                    # calculate and compare finite difference force

                    force_fd = -1.0*(energies1[0] - energies2[0])/(2.*h)
                    force_fd = force_fd.item()
                    force_model = forces_model[m][i][a].item()

                    error = force_model - force_fd
                    if (abs(error) > 1e-1):
                        print(f"m i a f_fd f_model: {m} {i} {a} {force_fd} {force_model}")
                        assert(False)
                    errors.append(error)

                    # return position back to normal

                    self.snap.data[m]['Positions'][i,a] += h

        mean_err = np.mean(np.abs(errors))
        max_err = np.max(np.abs(errors))

        print(f"mean max: {mean_err} {max_err}")

        errors = np.abs(errors)

        hist, bins = np.histogram(errors, bins=10)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.hist(errors, bins=logbins)
        plt.xscale('log')
        plt.xlim((1e-12,1e4))
        plt.xlabel(r'Absolute difference (eV/$\AA$)')
        plt.ylabel("Distribution")
        plt.yticks([])
        plt.savefig("fd-force-check.png", dpi=500)

        assert (mean_err < 0.001 and max_err < 0.1)
