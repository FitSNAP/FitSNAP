from fitsnap3lib.parallel_tools import ParallelTools, double_size, DistributedList, stubs
from fitsnap3lib.io.input import Config
from fitsnap3lib.io.output import output
import numpy as np
import pandas as pd
#config = Config()
#pt = ParallelTools()

config = Config()
pt = ParallelTools()


class Calculator:

    def __init__(self, name):
        self.name = name
        self.number_of_atoms = None
        self.number_of_files_per_node = None
        self.shared_index = None
        self.distributed_index = 0

    def get_width(self):
        pass

    def create_a(self):
        # TODO : Any extra config pulls should be done before this
        #print("----- create_a in calculator.py")
        pt.sub_barrier()
        self.number_of_atoms = pt.shared_arrays["number_of_atoms"].array.sum() # Total number of atoms in all configs
        #print(f"self.number_of_atoms: {self.number_of_atoms}")
        #print(self.dgradrows)
        #self.number_of_dgradrows = pt.shared_arrays["number_"]
        #pt.create_shared_array('number_of_dgradrows', len(pt.shared_arrays["number_of_atoms"]), tm=config.sections["SOLVER"].true_multinode)
        self.number_of_files_per_node = len(pt.shared_arrays["number_of_atoms"].array)

        if (config.sections["SOLVER"].solver == "PYTORCH"):
            print("----- Creating nonlinear 'a' in calculator.py")
            pt.shared_arrays["number_of_dgradrows"].array = self.dgradrows
            self.number_of_dgradrows = pt.shared_arrays["number_of_dgradrows"].array.sum()
            #print(pt.shared_arrays["number_of_dgradrows"].array)
            #print(f"number of dgradrows: {self.number_of_dgradrows}")
            a_len = 0
            b_len = 0 # 1D array of energies for each config
            c_len = 0 # (nconfigs*natoms*3) array of forces for each config
            dgrad_len = 0
            if config.sections["CALCULATOR"].energy:
                energy_rows = self.number_of_files_per_node
                if config.sections["CALCULATOR"].per_atom_energy:
                    energy_rows = self.number_of_atoms # Total number of atoms in all configs
                a_len += energy_rows
                b_len += self.number_of_files_per_node # Total number of configs

            if config.sections["CALCULATOR"].force:
                #a_len += self.number_of_dgradrows
                c_len += 3*self.number_of_atoms
                dgrad_len += self.number_of_dgradrows
            #b_len += 3 * self.number_of_atoms

            # Stress fitting not supported yet.
            #if config.sections["CALCULATOR"].stress:
            #    a_len += self.number_of_files_per_node * 6
            #    b_len += self.number_of_files_per_node * 6

            #print(f"----- a_len, b_len: {a_len} {b_len}")
            a_width = self.get_width()
            print(f"----- a_width: {a_width}")
            assert isinstance(a_width, int)

            # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)
            a_size = a_len * a_width * double_size
            # output.screen("'a' takes up ", 100 * a_size / pt.get_ram(), "% of the total memory")
            output.screen(">>> Matrix of descriptors takes up ", "{:.4f}".format(100 * a_size / config.sections["MEMORY"].memory),
                          "% of the total memory:", "{:.4f}".format(config.sections["MEMORY"].memory*1e-9), "GB")
            if a_size / pt.get_ram() > 0.5 and not config.sections["MEMORY"].override:
                raise MemoryError("The descriptor matrix is larger than 50% of your RAM. \n Aborting...!")
            elif a_size / pt.get_ram() > 0.5 and config.sections["MEMORY"].override:
                output.screen("Warning: I hope you know what you are doing!")

            pt.create_shared_array('a', a_len, a_width, tm=config.sections["SOLVER"].true_multinode)
            pt.create_shared_array('b', b_len, tm=config.sections["SOLVER"].true_multinode)
            pt.create_shared_array('c', c_len, tm=config.sections["SOLVER"].true_multinode)
            pt.create_shared_array('w', b_len, tm=config.sections["SOLVER"].true_multinode)

            print("b shape:")
            print(np.shape(pt.shared_arrays['b'].array))
            print("c shape:")
            print(np.shape(pt.shared_arrays['c'].array))

            if config.sections["CALCULATOR"].force:
                pt.create_shared_array('dgrad', dgrad_len, a_width, tm=config.sections["SOLVER"].true_multinode)
                pt.create_shared_array('dbdrindx', dgrad_len, 3, tm=config.sections["SOLVER"].true_multinode)
                # Create a unique_j_indices array.
                # This will house all unique indices (atoms j) in the dbdrindx array.
                # So the size is (natoms*nconfigs,)
                pt.create_shared_array('unique_j_indices', dgrad_len, tm=config.sections["SOLVER"].true_multinode)

                print("dgrad shape:")
                print(np.shape(pt.shared_arrays['dgrad'].array))
                print("dbdrindx shape:")
                print(np.shape(pt.shared_arrays['dbdrindx'].array))
                print("unique_j_indices:")
                print(np.shape(pt.shared_arrays['unique_j_indices'].array))

            pt.new_slice_a()
            self.shared_index = pt.fitsnap_dict["sub_a_indices"][0] # An index for which the 'a' array starts on a particular proc.
            pt.new_slice_b()
            self.shared_index_b = pt.fitsnap_dict["sub_b_indices"][0] # An index for which the 'b' array starts on a particular proc.
            # pt.slice_array('a')
            pt.new_slice_c()
            self.shared_index_c = pt.fitsnap_dict["sub_c_indices"][0] # An index for which the 'c' array starts on a particular proc.
            pt.new_slice_dgrad()
            self.shared_index_dgrad = pt.fitsnap_dict["sub_dgrad_indices"][0] # An index for which the 'dgrad' array starts on a particular proc.
            self.shared_index_dbdrindx = pt.fitsnap_dict["sub_dbdrindx_indices"][0] # An index for which the 'dbdrindx' array starts on a particular proc.
            self.shared_index_unique_j = 0 # Index for which the 'unique_j_indices' array starts on a particular proc, need to add to fitsnap_dict later.

            pt.add_2_fitsnap("Groups", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Configs", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Row_Type", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Atom_I", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Testing", DistributedList(pt.fitsnap_dict["sub_a_size"]))

        else:

            a_len = 0
            if config.sections["CALCULATOR"].energy:
                energy_rows = self.number_of_files_per_node
                if config.sections["CALCULATOR"].per_atom_energy:
                    energy_rows = self.number_of_atoms
                a_len += energy_rows
            if config.sections["CALCULATOR"].force:
                a_len += 3 * self.number_of_atoms
            if config.sections["CALCULATOR"].stress:
                a_len += self.number_of_files_per_node * 6

            a_width = self.get_width()
            assert isinstance(a_width, int)

            # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)
            a_size = a_len * a_width * double_size
            # output.screen("'a' takes up ", 100 * a_size / pt.get_ram(), "% of the total memory")
            output.screen(">>> Matrix of descriptors takes up ", "{:.4f}".format(100 * a_size / config.sections["MEMORY"].memory),
                          "% of the total memory:", "{:.4f}".format(config.sections["MEMORY"].memory*1e-9), "GB")
            if a_size / pt.get_ram() > 0.5 and not config.sections["MEMORY"].override:
                raise MemoryError("The descriptor matrix is larger than 50% of your RAM. \n Aborting...!")
            elif a_size / pt.get_ram() > 0.5 and config.sections["MEMORY"].override:
                output.screen("Warning: I hope you know what you are doing!")

            pt.create_shared_array('a', a_len, a_width, tm=config.sections["SOLVER"].true_multinode)
            pt.create_shared_array('b', a_len, tm=config.sections["SOLVER"].true_multinode)
            pt.create_shared_array('w', a_len, tm=config.sections["SOLVER"].true_multinode)
            pt.create_shared_array('ref', a_len, tm=config.sections["SOLVER"].true_multinode)
            pt.new_slice_a()
            self.shared_index = pt.fitsnap_dict["sub_a_indices"][0]
            # pt.slice_array('a')

            pt.add_2_fitsnap("Groups", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Configs", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Row_Type", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Atom_I", DistributedList(pt.fitsnap_dict["sub_a_size"]))
            pt.add_2_fitsnap("Testing", DistributedList(pt.fitsnap_dict["sub_a_size"]))

    def process_configs(self, data, i):
        print("--- process_configs in calculator.py")
        pass

    def preprocess_configs(self, data, i):
        print("--- preprocess_configs in calculator.py")
        pass

    def preprocess_allocate(self, nconfigs):
        print("--- preprocess_allocate in calculator.py")
        pass

    @staticmethod
    def collect_distributed_lists():
        for key in pt.fitsnap_dict.keys():
            if isinstance(pt.fitsnap_dict[key], DistributedList):
                pt.gather_fitsnap(key)
                if pt.fitsnap_dict[key] is not None and stubs != 1:
                    pt.fitsnap_dict[key] = [item for sublist in pt.fitsnap_dict[key] for item in sublist]
                elif pt.fitsnap_dict[key] is not None:
                    pt.fitsnap_dict[key] = pt.fitsnap_dict[key].get_list()

    @pt.rank_zero
    def extras(self):
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

        # if not config.sections["SOLVER"].detailed_errors:
        #     print(
        #         ">>>Enable [SOLVER], detailed_errors = 1 to characterize the training/testing split of your output *.npy matricies")
