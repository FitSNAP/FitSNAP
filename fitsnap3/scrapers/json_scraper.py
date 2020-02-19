from fitsnap3.scrapers.scrape import Scraper
from fitsnap3.io.input import config
from json import loads
from fitsnap3.parallel_tools import pt
import numpy as np
from _collections import defaultdict


class Json(Scraper):

    def __init__(self, name):
        super().__init__(name)
        self.style_vars = ['AtomType', 'Stress', 'Lattice', 'Energy', "Positions", "Forces"]
        self.array_vars = ['AtomTypes', 'Stress', 'Lattice', "Positions", "Forces"]
        if config.sections["REFERENCE"].atom_style == "spin":
            self.style_vars.append("Spins")
            self.array_vars.append("Spins")
        if config.sections["REFERENCE"].atom_style == "charge":
            self.style_vars.append("Charges")
            self.array_vars.append("Charges")
        self.styles = defaultdict(lambda: set())
        self.all_data = []
        self.style_info = {}

    def scrape_groups(self):
        super().scrape_groups()

    def scrape_configs(self):
        for i, file_name in enumerate(self.files):
            with open(file_name) as file:
                file.readline()
                try:
                    self.data = loads(file.read(), parse_constant=True)
                except Exception as e:
                    pt.sinlge_print("Trouble Parsing Training Data: ", file_name)
                    pt.error(e)

                assert len(self.data) == 1, "More than one object (dataset) is in this file"

                self.data = self.data['Dataset']

                assert len(self.data['Data']) == 1, "More than one configuration in this dataset"

                self.data['Group'] = file_name.split("/")[-2]
                self.data['File'] = file_name.split("/")[-1]
                if self.group_table.isin([self.data['Group']]).any().any():
                    self.data['GroupIndex'] = \
                        self.group_table.name[self.group_table.name == self.data['Group']].index.tolist()[0]
                else:
                    raise IndexError("{} was not found in dataframe".format(self.data['Group']))

                for sty in self.style_vars:
                    self.styles[sty].add(self.data.pop(sty + "Style", ))

                assert all(k not in self.data for k in self.data["Data"][0].keys()), \
                    "Duplicate keys in dataset and data"

                self.data.update(self.data.pop('Data')[0])  # Move data up one level

                for key in self.array_vars:
                    self.data[key] = np.asarray(self.data[key])

                natoms = np.shape(self.data["Positions"])[0]
                pt.shared_arrays["number_of_atoms"].sliced_array[i] = natoms
                self.data["QMLattice"] = self.data["Lattice"]
                del self.data["Lattice"]  # We will populate this with the lammps-normalized lattice.
                if "Label" in self.data:
                    del self.data["Label"]  # This comment line is not that useful to keep around.

                # possibly due to JSON, some configurations have integer energy values.
                if not isinstance(self.data["Energy"], float):
                    # pt.print(f"Warning: Configuration {all_index}
                    # ({group_name}/{fname_end}) gives energy as an integer")
                    self.data["Energy"] = float(self.data["Energy"])

                self._stress_conv(self.styles)
                self.data["Energy"] *= self.convert["Energy"]

                if hasattr(config.sections["ESHIFT"], 'eshift'):
                    for atom in self.data["AtomTypes"]:
                        self.data["Energy"] += config.sections["ESHIFT"].eshift[atom]

                self._rotate_coords()
                self._translate_coords()

                # TODO : Fix compute testers by adding in amount of files in folder
                nfiles_train = 500
                if config.sections["MODEL"].compute_testerrs and (self.data['GroupIndex'] > nfiles_train):
                    wprefac = 0.0
                else:
                    wprefac = 1.0

                if self.group_table['eweight'][self.data['GroupIndex']] >= 0.0:
                    for wtype in ['eweight', 'fweight', 'vweight']:
                        self.data[wtype] = wprefac * self.group_table[wtype][self.data['GroupIndex']]
                else:
                    self.data['eweight'] = wprefac * np.exp(
                        (self.group_table['eweight'][self.data['GroupIndex']] - self.data["Energy"] /
                         float(natoms)) / (self.kb * float(config.sections["BISPECTRUM"].boltz)))
                    self.data['fweight'] = \
                        wprefac * self.data['eweight'] * self.group_table['fweight'][self.data['GroupIndex']]
                    self.data['vweight'] = \
                        wprefac * self.data['eweight'] * self.group_table['vweight'][self.data['GroupIndex']]

                self.all_data.append(self.data)

        for style_name, style_set in self.styles.items():
            assert len(style_set) == 1, "Multiple styles ({}) for {}".format(len(style_set), style_name)

        self.style_info = {k: v.pop() for k, v in self.styles.items()}

        return self.all_data
