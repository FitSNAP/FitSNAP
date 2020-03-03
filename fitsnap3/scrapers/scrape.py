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

from fitsnap3.io.input import config
from pandas import read_csv
from tqdm import tqdm
from os import path, listdir, stat
import numpy as np
from random import shuffle
from fitsnap3.parallel_tools import pt
from natsort import natsorted


class Scraper:

    def __init__(self, name):
        self.name = name
        self.group_types = {}
        self.group_table = []
        self.files = {}
        self.convert = {"Energy": 1.0, "Force": 1.0, "Stress": 1.0, "Distance": 1.0}
        self.data = {}

        self._init_units()

    def scrape_groups(self):
        self.group_types = {'name': str, 'size': float, 'eweight': float, 'fweight': float, 'vweight': float}
        group_names = [key for key in self.group_types]

        self.group_table = read_csv(config.sections["PATH"].group_file,
                                    delim_whitespace=True,
                                    comment='#',
                                    skip_blank_lines=True,
                                    names=group_names,
                                    index_col=False)

        # Remove blank lines ; skip_blank_lines doesn't seem to work.
        self.group_table = self.group_table.dropna()
        self.group_table.index = range(len(self.group_table.index))

        # Convert data types
        self.group_table = self.group_table.astype(dtype=dict(self.group_types))

        for group_info in tqdm(self.group_table.itertuples(),
                               desc="Groups",
                               position=0,
                               total=len(self.group_table),
                               disable=(not config.args.verbose),
                               ascii=True):
            group_name = group_info.name
            folder = path.join(config.sections["PATH"].datapath, group_name)
            folder_files = listdir(folder)
            for file_name in folder_files:
                if folder not in self.files:
                    self.files[folder] = []
                self.files[folder].append([folder + '/' + file_name, int(stat(folder + '/' + file_name).st_size)])

    # TODO : Fix divvy up to distribute groups evenly and based on memory
    def divvy_up_configs(self):
        # self.files = natsorted(self.files)
        for folder in self.files:
            shuffle(self.files[folder], pt.get_seed)

        temp_list = []
        for folder in self.files:
            for file in self.files[folder]:
                temp_list.append(file[0])

        self.files = temp_list

        self.files = pt.split_by_node(self.files)

        number_of_files_per_node = len(self.files)
        pt.create_shared_array('number_of_atoms', number_of_files_per_node, dtype='i')
        pt.slice_array('number_of_atoms')
        self.files = pt.split_within_node(self.files)

    def scrape_configs(self):
        raise NotImplementedError("Call to virtual Scraper.scrape_configs method")

    def _init_units(self):
        if config.sections["REFERENCE"].units == "real":
            self.kb = 0.00198198665029335
        if config.sections["REFERENCE"].units == "metal":
            self.kb = 0.00008617333262145

    def _rotate_coords(self):
        # Transpose here because Lammps stores lattice vectors as columns,
        # QM stores lattice vectors as rows; After transposing lattice vectors are columns
        in_cell = np.asarray(self.data["QMLattice"]).T
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
        self.data["Positions"] = self.data["Positions"] * self.convert["Distance"] @ rot.T
        self.data["Forces"] = self.data["Forces"] * self.convert["Force"] @ rot.T
        self.data["Stress"] = rot @ (self.data["Stress"] * self.convert["Stress"]) @ rot.T
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
        self.data["Postions"] = new_pos
        self.data["Translation"] = trans_vec

    def _stress_conv(self, styles):

        if (config.sections["REFERENCE"].units == "metal" and
                list(styles["Stress"])[0] == "kbar" or list(styles["Stress"])[0] == "kB"):
            self.convert["Stress"] = 1000.0

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
