from ..parallel_tools import pt, double_size
from ..io.input import config
from ..io.output import output
from ..solvers.solver import make_abw
import numpy as np


class Calculator:

    def __init__(self, name):
        self.name = name
        self.number_of_atoms = None
        self.number_of_files_per_node = None

    def get_width(self):
        pass

    def create_a(self):
        # TODO : Any extra config pulls should be done before this
        pt.sub_barrier()
        self.number_of_atoms = pt.shared_arrays["number_of_atoms"].array.sum()
        self.number_of_files_per_node = len(pt.shared_arrays["number_of_atoms"].array)

        elements = 0
        testing = pt.shared_arrays["configs_per_group"].testing
        if testing > 0:
            for i in range(testing):
                if config.sections["CALCULATOR"].energy:
                    energy_rows = 1
                    if config.sections["CALCULATOR"].per_atom_energy:
                        energy_rows = pt.shared_arrays["number_of_atoms"].array[-testing+i]
                    elements += energy_rows
                if config.sections["CALCULATOR"].force:
                    elements += 3 * pt.shared_arrays["number_of_atoms"].array[-testing+i]
                if config.sections["CALCULATOR"].stress:
                    elements += 6
        pt.shared_arrays["configs_per_group"].testing_elements = elements

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
        pt.slice_array('a')

    def process_configs(self, data, i):
        pass

    @pt.rank_zero
    def extras(self):
        length, width = np.shape(pt.shared_arrays['a'].array)
        if config.sections["CALCULATOR"].energy:
            num_energy = 1
            if config.sections["CALCULATOR"].per_atom_energy:
                num_energy = np.array(pt.shared_arrays['a'].num_atoms)
            a_e, b_e, w_e = make_abw(pt.shared_arrays['a'].energy_index, num_energy.tolist())
        else:
            a_e, b_e, w_e = np.zeros((length, width)), np.zeros((length,)), np.zeros((length,))
        if config.sections["CALCULATOR"].force:
            num_forces = np.array(pt.shared_arrays['a'].num_atoms) * 3
            a_f, b_f, w_f = make_abw(pt.shared_arrays['a'].force_index, num_forces.tolist())
        else:
            a_f, b_f, w_f = np.zeros((length, width)), np.zeros((length,)), np.zeros((length,))
        if config.sections["CALCULATOR"].stress:
            a_s, b_s, w_s = make_abw(pt.shared_arrays['a'].stress_index, 6)
        else:
            a_s, b_s, w_s = np.zeros((length, width)), np.zeros((length,)), np.zeros((length,))
        if not config.sections["SOLVER"].detailed_errors:
            print(
                ">>>Enable [SOLVER], detailed_errors = 1 to characterize the training/testing split of your output *.npy matricies")
        if config.sections["EXTRAS"].dump_a:
            # if config.sections["EXTRAS"].apply_transpose:
            #     np.save('Descriptors_Compact.npy', (np.concatenate((a_e,a_f,a_s),axis=0) @ np.concatenate((a_e,a_f,a_s),axis=0).T))
            # else:
            np.save(config.sections['EXTRAS'].descriptor_file,
                    np.concatenate([x for x in (a_e, a_f, a_s) if x.size > 0], axis=0))
        if config.sections["EXTRAS"].dump_b:
            np.save(config.sections['EXTRAS'].truth_file,
                    np.concatenate([x for x in (b_e, b_f, b_s) if x.size > 0], axis=0))
        if config.sections["EXTRAS"].dump_w:
            np.save(config.sections['EXTRAS'].weights_file,
                    np.concatenate([x for x in (w_e, w_f, w_s) if x.size > 0], axis=0))
