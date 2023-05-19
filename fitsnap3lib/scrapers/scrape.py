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
#     Maybe me ¯\_(ツ)_/¯
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

from os import path, listdir, stat
import numpy as np
from random import random, seed, shuffle
from fitsnap3lib.units.units import convert
from copy import copy


class Scraper:

    def __init__(self, name, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.name = name
        self.group_types = {}
        self.group_table = []
        self.files = {}
        self.configs = {} # Originally a dict for `scrape_groups` but gets transformed to list of files.
        self.tests = None
        self.data = {}
        self.test_bool = None
        self.default_conversions = {key: convert(self.config.sections["SCRAPER"].properties[key])
                                    for key in self.config.sections["SCRAPER"].properties}
        self.conversions = {}

        self._init_units()

    def scrape_groups(self):
        """
        Scrape groups of files for a particular fitsnap instance.
        """
        # Reset as empty dict in case running scrape twice.
        self.files = {}

        group_dict = {k: self.config.sections["GROUPS"].group_types[i]
                      for i, k in enumerate(self.config.sections["GROUPS"].group_sections)}
        self.group_table = self.config.sections["GROUPS"].group_table
        size_type = None
        testing_size_type = None
        user_set_random_seed = self.config.sections["GROUPS"].random_seed ## default is 0

        if self.config.sections["GROUPS"].random_sampling:
            #output.screen(f"Random sampling of groups toggled on.")
            self.pt.single_print(f"Random sampling of groups toggled on.")
            if not user_set_random_seed:
                sampling_seed = self.pt.get_seed()
                seed_txt = f"FitSNAP-generated seed for random sampling: {self.pt.get_seed()}"
            else:
                ## groups.py casts random_seed to float, just in case user
                ## uses continuous variable. if user input was originally
                ## an integer, this casts it to int (less confusing for user)
                if user_set_random_seed.is_integer():
                    sampling_seed = int(user_set_random_seed)
                seed_txt = f"User-set seed for random sampling: {sampling_seed}"
            self.pt.single_print(seed_txt)
            seed(sampling_seed)
            self._write_seed_file(seed_txt)

        for key in self.group_table:
            bc_bool = False
            training_size = None
            if 'size' in self.group_table[key]:
                training_size = self.group_table[key]['size']
                bc_bool = True
                size_type = group_dict['size']
            if 'training_size' in self.group_table[key]:
                if training_size is not None:
                    raise ValueError("Do not set both size and training size")
                training_size = self.group_table[key]['training_size']
                size_type = group_dict['training_size']
            if 'testing_size' in self.group_table[key]:
                testing_size = self.group_table[key]['testing_size']
                testing_size_type = group_dict['testing_size']
            else:
                testing_size = 0
            if training_size is None:
                raise ValueError("Please set training size for {}".format(key))

            folder = path.join(self.config.sections["PATH"].datapath, key)
            folder_files = listdir(folder)
            for file_name in folder_files:
                if folder not in self.files:
                    self.files[folder] = []
                self.files[folder].append([folder + '/' + file_name, int(stat(folder + '/' + file_name).st_size)])
            if self.config.sections["GROUPS"].random_sampling:
                shuffle(self.files[folder], random)
            nfiles = len(folder_files)
            if training_size < 1 or (training_size == 1 and size_type == float):
                if training_size == 1:
                    training_size = abs(training_size) * nfiles
                elif training_size == 0:
                    pass
                else:
                    training_size = max(1, int(abs(training_size) * nfiles + 0.5))
                if bc_bool and testing_size == 0:
                    testing_size = nfiles - training_size
            if testing_size != 0 and (testing_size < 1 or (testing_size == 1 and testing_size_type == float)):
                testing_size = max(1, int(abs(testing_size) * nfiles + 0.5))
            training_size = self._float_to_int(training_size)
            testing_size = self._float_to_int(testing_size)
            if nfiles-testing_size-training_size < 0:
                # Force testing_size and training_size to add up to nfiles.
                warnstr = f"\nWARNING: {key} train size {training_size} + test size {testing_size} > nfiles {nfiles}\n"
                warnstr += "         Forcing testing size to add up properly.\n"
                self.pt.single_print(warnstr)
                testing_size = nfiles - training_size
            if (self.config.args.verbose):
                self.pt.single_print(key, ": Detected ", nfiles, " fitting on ", training_size, " testing on ", testing_size)
            if self.tests is None:
                self.tests = {}
            self.tests[folder] = []
            for i in range(nfiles - training_size - testing_size):
                self.files[folder].pop()
            for i in range(testing_size):
                self.tests[folder].append(self.files[folder].pop())

            self.group_table[key]['training_size'] = training_size
            self.group_table[key]['testing_size'] = testing_size
            # self.files[folder] = natsorted(self.files[folder])

    # TODO : Fix divvy up to distribute groups evenly and based on memory
    def divvy_up_configs(self):
        """
        Function to organize groups and allocate shared arrays used in Calculator.
        """

        # Loop over `configs` which is a list of filenames, and organize into groups.
        self.test_bool = []
        groups = []
        group_list = []
        temp_list = []
        test_list = []
        for i, folder in enumerate(self.configs):
            for configuration in self.configs[folder]:
                if isinstance(configuration, list):
                    temp_list.append(configuration[0])
                else:
                    temp_list.append([configuration, folder])
                groups.append(folder)
                self.test_bool.append(0)

        self.configs = temp_list

        if self.tests is not None:
            for i, folder in enumerate(self.tests):
                for configuration in self.tests[folder]:
                    if isinstance(configuration, list):
                        test_list.append(configuration[0])
                    else:
                        test_list.append([configuration, folder])
                    group_list.append(folder)
                    self.test_bool.append(1)
            self.configs += test_list

        # NODES SPLIT UP HERE
        self.configs = self.pt.split_by_node(self.configs)

        self.test_bool = self.pt.split_by_node(self.test_bool)
        groups = self.pt.split_by_node(groups)
        group_list = self.pt.split_by_node(group_list)
        temp_configs = copy(self.configs)

        group_test = list(dict.fromkeys(group_list))
        group_set = list(dict.fromkeys(groups))
        group_counts = np.zeros((len(group_set) + len(group_test),), dtype='i')
        for i, group in enumerate(group_set):
            group_counts[i] = groups.count(group)
        for i, group in enumerate(group_test):
            group_counts[i+len(group_set)] = group_list.count(group)
        for i in range(len(group_test)):
            group_test[i] += '_testing'

        # TODO: `configs_per_group` shared array doesn't seemed to be used anywhere except bcs, 
        #       mcmc, and opt solvers.
        self.pt.create_shared_array('configs_per_group', len(group_counts), dtype='i')
        if self.pt.get_rank() == 0:
            for i in range(len(group_counts)):
                self.pt.shared_arrays['configs_per_group'].array[i] = group_counts[i]
        self.pt.shared_arrays['configs_per_group'].list = group_set + group_test
        self.pt.shared_arrays['configs_per_group'].testing = 0
        if self.tests is not None:
            self.pt.shared_arrays['configs_per_group'].testing = len(test_list)

        # Procs split up here. This is for injecting into the data dictionary in `scrape_configs()`.
        self.test_bool = self.pt.split_within_node(self.test_bool)
        self.configs = self.pt.split_within_node(self.configs)

    def scrape_configs(self):
        raise NotImplementedError("Call to virtual Scraper.scrape_configs method")

    def _init_units(self):
        if self.config.sections["REFERENCE"].units == "real":
            self.kb = 0.00198198665029335
        if self.config.sections["REFERENCE"].units == "metal":
            self.kb = 0.00008617333262145

    def _rotate_coords(self):
        # Transpose here because Lammps stores lattice vectors as columns,
        # QM stores lattice vectors as rows; After transposing lattice vectors are columns
        in_cell = np.asarray(self.data["QMLattice"])
        assert np.linalg.det(in_cell) > 0, "Input cell is not right-handed!"

        # Q matrix of QR decomposition is an orthogonal (rotation-like)
        # matrix whose inverse/transpose makes the input cell upper-diagonal:
        # input cell C = Q C';
        # runlammps-normalized cell C' = Q^T C.
        qmat, rmat = np.linalg.qr(in_cell)

        # Normalize signs of Q matrix to ensure positive diagonals of transformed cell;
        # QR decomposition algorithms don't always return a proper rotation
        ss = np.diagflat(np.sign(np.diag(rmat)))
        rot = ss @ qmat.T

        assert np.allclose(rot @ rot.T, np.eye(3)), "Rotation matrix not orthogonal"
        assert np.allclose(rot.T @ rot, np.eye(3)), "Rotation matrix not orthogonal"
        assert np.linalg.det(rot) > 0, "Rotation matrix is an improper rotation (det<0)"

        # ????
        # Cell transforms on first axis due to runlammps sotring lattice vectors as columns
        out_cell = rot @ in_cell

        # This assert is technically overkill, but checks that the new cell is right-handed
        assert np.linalg.det(out_cell) > 0, "New cell is not right-handed!"

        lower_triangle = out_cell[np.tril_indices(3, k=-1)]
        assert np.allclose(lower_triangle, 0, atol=1e-13), \
            f"Lower triangle of normalized cell has nonzero-elements: {lower_triangle}"

        # Positions and forces transform on the second axis
        # Stress transforms on both the first and second axis.
        self.data["Lattice"] = out_cell
        self.data["Positions"] = self.data["Positions"] * self.conversions["Positions"] @ rot.T
        if self.config.sections["CALCULATOR"].force:
            self.data["Forces"] = self.data["Forces"] * self.conversions["Forces"] @ rot.T
        if self.config.sections["CALCULATOR"].stress:
            self.data["Stress"] = rot @ (self.data["Stress"] * self.conversions["Stress"]) @ rot.T
        self.data["Rotation"] = rot

    def _translate_coords(self):
        cell = self.data["Lattice"]
        position_in = self.data["Positions"]

        # Extra transposes because runlammps uses cells with latttice vectors as columns
        invcell = np.linalg.inv(cell.T).T
        # Fractional coordinates
        frac_coords = position_in @ invcell.T

        # Fix some rounding difficulties in divmod when within machine epsilon of zero
        frac_coords[np.isclose(frac_coords, 0, atol=1e-15)] = 0.

        trans_nums, cell_frac_coords = np.divmod(frac_coords, 1)

        assert (cell_frac_coords < 1).all(), "Fractional coordinates outside cell"
        assert (cell_frac_coords >= 0).all(), "Fractional coordinates outside cell"

        # If no translations are needed, return unmodified positions
        if (trans_nums == 0).all():
            self.data["Positions"] = position_in
            self.data["Translation"] = np.zeros_like(position_in, dtype=float)

        new_pos = cell_frac_coords @ cell.T
        trans_vec = trans_nums @ cell.T
        assert np.allclose(new_pos + trans_vec, position_in), "Translation failed to invert"
        self.data["Positions"] = new_pos
        self.data["Translation"] = trans_vec

    @staticmethod
    def _float_to_int(a_float):
        if a_float == 0:
            return int(a_float)
        if a_float / int(a_float) != 1:
            raise ValueError("Training and Testing Size must be interpretable as integers")
        return int(a_float)

    def _weighting(self, natoms):
        if self.config.sections["GROUPS"].boltz == 0:
            for key in self.group_table[self.data['Group']]:
                # Do not put the word weight in a group table unless you want to use it as a weight
                if 'weight' in key:
                    self.data[key] = self.group_table[self.data['Group']][key]
        else:
            self.data['eweight'] = np.exp(
                (self.group_table[self.data['Group']]['eweight'] - self.data["Energy"] /
                 float(natoms)) / (self.kb * float(self.config.sections["GROUPS"].boltz)))
            for key in self.group_table[self.data['Group']]:
                # Do not put the word weight in a group table unless you want to use it as a weight
                if 'weight' in key and key != 'eweight':
                    self.data[key] = self.data['eweight'] * self.group_table[self.data['Group']][key]

        if self.config.sections["GROUPS"].smartweights:
            for key in self.group_table[self.data['Group']]:
                # Do not put the word weight in a group table unless you want to use it as a weight
                if 'weight' in key:
                    if self.data['test_bool']:
                        self.data[key] /= self.group_table[self.data['Group']]['testing_size']
                    else:
                        try:
                            self.data[key] /= self.group_table[self.data['Group']]['training_size']
                        except ZeroDivisionError:
                            self.data[key] = 0
            if self.config.sections["CALCULATOR"].force:
                self.data['fweight'] /= natoms*3

            if self.config.sections["CALCULATOR"].stress:
                self.data['fweight'] /= 6

    #@pt.rank_zero
    def _write_seed_file(self, txt):
        @self.pt.rank_zero
        def decorated_write_seed_file(txt):
            with open("RandomSamplingSeed.txt", 'w') as f:
                f.write(txt+'\n')
        decorated_write_seed_file(txt)

    # def check_coords(self, cell, pos1, pos2):
    #     """Compares position 1 and position 2 with respect to periodic boundaries defined by cell"""
    #     invcell = np.linalg.inv(np.asarray(cell).T).T
    #
    #     # Fractional coordinates
    #     frac_1 = pos1 @ invcell.T
    #     frac_2 = pos2 @ invcell.T
    #     diff_frac = frac_2 - frac_1
    #
    #     # Assert that diff_frac is very close to an integer
    #     assert np.allclose(
    #         diff_frac,
    #         np.round(diff_frac),
    #         atol=1e-12, rtol=1e-12), "Coordinates are not close after shift." + \
    #                                  "Fractional coordinate Error:{}\nArray:\n{}".format(
    #                                      np.abs(diff_frac).max(), diff_frac)
    #     return True
    #
    # def check_volume(self, lattice, volume):
    #     assert np.allclose(np.linalg.det(lattice), volume, rtol=1e-10, atol=1e-10), \
    #         "Cell volume not equal to supplied volume!"
