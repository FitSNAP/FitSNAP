from fitsnap3.scrapers.scrape import Scraper, convert
from fitsnap3.io.input import config
from json import loads
from fitsnap3.parallel_tools import pt
from fitsnap3.io.output import output
from copy import copy
import numpy as np


class Json(Scraper):

    def __init__(self, name):
        super().__init__(name)
        self.all_data = []

    def scrape_groups(self):
        super().scrape_groups()
        self.configs = self.files

    def scrape_configs(self):
        self.files = self.configs
        self.conversions = copy(self.default_conversions)
        for i, file_name in enumerate(self.files):
            with open(file_name) as file:
                file.readline()
                try:
                    self.data = loads(file.read(), parse_constant=True)
                except Exception as e:
                    output.screen("Trouble Parsing Training Data: ", file_name)
                    output.exception(e)

                assert len(self.data) == 1, "More than one object (dataset) is in this file"

                self.data = self.data['Dataset']

                assert len(self.data['Data']) == 1, "More than one configuration in this dataset"

                self.data['Group'] = file_name.split("/")[-2]
                self.data['File'] = file_name.split("/")[-1]

                assert all(k not in self.data for k in self.data["Data"][0].keys()), \
                    "Duplicate keys in dataset and data"

                self.data.update(self.data.pop('Data')[0])  # Move data up one level

                for key in self.data:
                    if "Style" in key:
                        if key.replace("Style", "") in self.conversions:
                            temp = config.sections["SCRAPER"].properties[key.replace("Style", "")]
                            temp[1] = self.data[key]
                            self.conversions[key.replace("Style", "")] = convert(temp)

                for key in config.sections["SCRAPER"].properties:
                    if key in self.data:
                        self.data[key] = np.asarray(self.data[key])

                natoms = np.shape(self.data["Positions"])[0]
                pt.shared_arrays["number_of_atoms"].sliced_array[i] = natoms
                self.data["QMLattice"] = self.data["Lattice"] * self.conversions["Lattice"]
                del self.data["Lattice"]  # We will populate this with the lammps-normalized lattice.
                if "Label" in self.data:
                    del self.data["Label"]  # This comment line is not that useful to keep around.

                if not isinstance(self.data["Energy"], float):
                    self.data["Energy"] = float(self.data["Energy"])

                # Currently, ESHIFT should be in units of your training data (note there is no conversion)
                if hasattr(config.sections["ESHIFT"], 'eshift'):
                    for atom in self.data["AtomTypes"]:
                        self.data["Energy"] += config.sections["ESHIFT"].eshift[atom]

                self.data["test_bool"] = self.test_bool[i]

                self.data["Energy"] *= self.conversions["Energy"]

                self._rotate_coords()
                self._translate_coords()

                self._weighting(natoms)

                self.all_data.append(self.data)

        return self.all_data
