from fitsnap3lib.parallel_tools import ParallelTools, double_size, DistributedList, stubs
from fitsnap3lib.io.input import Config
from fitsnap3lib.io.output import output
import numpy as np
import pandas as pd


#config = Config()
#pt = ParallelTools()


class Calculator:

    def __init__(self, name):
        self.pt = ParallelTools()
        self.config = Config()
        self.name = name
        self.number_of_atoms = None
        self.number_of_files_per_node = None
        self.shared_index = None
        self.distributed_index = 0

    def get_width(self):
        pass

    def create_a(self):

        # TODO : Any extra config pulls should be done before this

        self.pt.sub_barrier()
        self.number_of_atoms = self.pt.shared_arrays["number_of_atoms"].array.sum() # total number of atoms in all configs, summed
        self.number_of_files_per_node = len(self.pt.shared_arrays["number_of_atoms"].array)

        # create data matrices for nonlinear pytorch solver

        if (self.config.sections["SOLVER"].solver == "PYTORCH"):
            if (self.pt._sub_size > 1):
                print(f"Using {self.pt._sub_size} procs")
                raise Exception("PyTorch models not yet set up for multiprocs, please use 1 proc.")
            print("----- Creating data arrays for nonlinear fitting in calculator.py")
            self.pt.shared_arrays["number_of_dgradrows"].array = self.dgradrows
            self.number_of_dgradrows = self.pt.shared_arrays["number_of_dgradrows"].array.sum()
            a_len = 0
            b_len = 0 # 1D array of reference energies for each config
            c_len = 0 # (nconfigs*natoms*3) array of reference forces for each config
            dgrad_len = 0
            if self.config.sections["CALCULATOR"].energy:
                energy_rows = self.number_of_files_per_node
                if self.config.sections["CALCULATOR"].per_atom_energy:
                    energy_rows = self.number_of_atoms # total number of atoms in all configs
                a_len += energy_rows
                b_len += self.number_of_files_per_node # total number of configs

            if self.config.sections["CALCULATOR"].force:
                c_len += 3*self.number_of_atoms
                dgrad_len += self.number_of_dgradrows

#            # stress fitting not supported yet.
#            if config.sections["CALCULATOR"].stress:
#                a_len += self.number_of_files_per_node * 6
#                b_len += self.number_of_files_per_node * 6

            a_width = self.get_width()
            assert isinstance(a_width, int)

            # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)

            a_size = a_len * a_width * double_size
            output.screen(">>> Matrix of descriptors takes up ", 
                          "{:.4f}".format(100 * a_size / self.config.sections["MEMORY"].memory),
                          "% of the total memory:", 
                          "{:.4f}".format(self.config.sections["MEMORY"].memory*1e-9), "GB")
            if a_size / self.pt.get_ram() > 0.5 and not self.config.sections["MEMORY"].override:
                raise MemoryError("The descriptor matrix is larger than 50% of your RAM. \n Aborting...!")
            elif a_size / self.pt.get_ram() > 0.5 and self.config.sections["MEMORY"].override:
                output.screen("Warning: I hope you know what you are doing!")

            self.pt.create_shared_array('a', a_len, a_width, 
                                        tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('b', b_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('c', c_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('w', b_len, 2, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('t', a_len, 1, tm=self.config.sections["SOLVER"].true_multinode)

            # TODO: some sort of assertion on the sizes, here or later
            #print("b shape:")
            #print(np.shape(pt.shared_arrays['b'].array))
            #print("c shape:")
            #print(np.shape(pt.shared_arrays['c'].array))

            if self.config.sections["CALCULATOR"].force:
                self.pt.create_shared_array('dgrad', dgrad_len, a_width, 
                                            tm=self.config.sections["SOLVER"].true_multinode)
                self.pt.create_shared_array('dbdrindx', dgrad_len, 3, 
                                            tm=self.config.sections["SOLVER"].true_multinode)
                # create a unique_j_indices array
                # this will house all unique indices (atoms j) in the dbdrindx array
                # so the size is (natoms*nconfigs,)
                self.pt.create_shared_array('unique_j_indices', dgrad_len, 
                                            tm=self.config.sections["SOLVER"].true_multinode)

            self.pt.new_slice_a()
            # an index for which the 'a' array starts on a particular proc
            self.shared_index = self.pt.fitsnap_dict["sub_a_indices"][0] 
            self.pt.new_slice_b()
            # an index for which the 'b' array starts on a particular proc
            self.shared_index_b = self.pt.fitsnap_dict["sub_b_indices"][0] 
            self.pt.new_slice_c()
            # an index for which the 'c' array starts on a particular proc
            self.shared_index_c = self.pt.fitsnap_dict["sub_c_indices"][0] 
            self.pt.new_slice_t() # atom types

            self.pt.new_slice_dgrad()
            # an index for which the 'dgrad' array starts on a particular proc
            self.shared_index_dgrad = self.pt.fitsnap_dict["sub_dgrad_indices"][0]
            # an index for which the 'dbdrindx' array starts on a particular proc 
            self.shared_index_dbdrindx = self.pt.fitsnap_dict["sub_dbdrindx_indices"][0] 
            # index for which the 'unique_j_indices' array starts on a particular proc, 
            # need to add to fitsnap_dict later
            self.shared_index_unique_j = 0 

            # some fitsnap dicts have same size as number of configs for nonlinear fitting

            self.pt.add_2_fitsnap("Groups", DistributedList(self.number_of_files_per_node))
            self.pt.add_2_fitsnap("Configs", DistributedList(self.number_of_files_per_node))
            self.pt.add_2_fitsnap("Row_Type", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Atom_I", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Testing", DistributedList(self.number_of_files_per_node))

        # get data arrays for linear solvers

        else:

            a_len = 0
            if self.config.sections["CALCULATOR"].energy:
                energy_rows = self.number_of_files_per_node
                if self.config.sections["CALCULATOR"].per_atom_energy:
                    energy_rows = self.number_of_atoms
                a_len += energy_rows
            if self.config.sections["CALCULATOR"].force:
                a_len += 3 * self.number_of_atoms
            if self.config.sections["CALCULATOR"].stress:
                a_len += self.number_of_files_per_node * 6

            a_width = self.get_width()
            assert isinstance(a_width, int)

            # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)
            a_size = a_len * a_width * double_size
            output.screen(">>> Matrix of descriptors takes up ", "{:.4f}".format(100 * a_size / self.config.sections["MEMORY"].memory),
                          "% of the total memory:", "{:.4f}".format(self.config.sections["MEMORY"].memory*1e-9), "GB")
            if a_size / self.pt.get_ram() > 0.5 and not self.config.sections["MEMORY"].override:
                raise MemoryError("The descriptor matrix is larger than 50% of your RAM. \n Aborting...!")
            elif a_size / self.pt.get_ram() > 0.5 and self.config.sections["MEMORY"].override:
                output.screen("Warning: I hope you know what you are doing!")

            self.pt.create_shared_array('a', a_len, a_width, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('b', a_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('w', a_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('ref', a_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.new_slice_a()
            self.shared_index = self.pt.fitsnap_dict["sub_a_indices"][0]
            # pt.slice_array('a')

            self.pt.add_2_fitsnap("Groups", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Configs", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Row_Type", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Atom_I", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Testing", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Atom_Type", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))

    def process_configs(self, data, i):
        pass

    def preprocess_configs(self, data, i):
        pass

    def preprocess_allocate(self, nconfigs):
        pass

    @staticmethod
    def collect_distributed_lists():
        pt = ParallelTools()    
        #print("----- calculator.py collect_distributed_lists()")
        #print(pt)
        for key in pt.fitsnap_dict.keys():
            if isinstance(pt.fitsnap_dict[key], DistributedList):
                pt.gather_fitsnap(key)
                if pt.fitsnap_dict[key] is not None and stubs != 1:
                    pt.fitsnap_dict[key] = [item for sublist in pt.fitsnap_dict[key] for item in sublist]
                elif pt.fitsnap_dict[key] is not None:
                    pt.fitsnap_dict[key] = pt.fitsnap_dict[key].get_list()

    #@pt.rank_zero
    def extras(self):
        @self.pt.rank_zero
        def decorated_extras():
            pt = ParallelTools()
            config = Config()
            if config.sections["EXTRAS"].dump_a:
                np.save(config.sections['EXTRAS'].descriptor_file, pt.shared_arrays['a'].array)
            if config.sections["EXTRAS"].dump_b:
                np.save(config.sections['EXTRAS'].truth_file, pt.shared_arrays['b'].array)
            if config.sections["EXTRAS"].dump_w:
                np.save(config.sections['EXTRAS'].weights_file, pt.shared_arrays['w'].array)
            if config.sections["EXTRAS"].dump_dataframe:
                df = pd.DataFrame(pt.shared_arrays['a'].array)
                df['truths'] = pt.shared_arrays['b'].array.tolist()
                df['weights'] = pt.shared_arrays['w'].array.tolist()
                for key in pt.fitsnap_dict.keys():
                    if isinstance(pt.fitsnap_dict[key], list) and len(pt.fitsnap_dict[key]) == len(df.index):
                        df[key] = pt.fitsnap_dict[key]
                df.to_pickle(config.sections['EXTRAS'].dataframe_file)
                del df
        decorated_extras()

        # if not config.sections["SOLVER"].detailed_errors:
        #     print(
        #         ">>>Enable [SOLVER], detailed_errors = 1 to characterize the training/testing split of your output *.npy matricies")
