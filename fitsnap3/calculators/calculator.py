from ..parallel_tools import pt, double_size, DistributedList, stubs
from ..io.input import config
from ..io.output import output
import numpy as np
import pandas as pd


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
        pt.sub_barrier()
        self.number_of_atoms = pt.shared_arrays["number_of_atoms"].array.sum()
        self.number_of_files_per_node = len(pt.shared_arrays["number_of_atoms"].array)

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
