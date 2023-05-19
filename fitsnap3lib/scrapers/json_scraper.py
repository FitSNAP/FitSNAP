from fitsnap3lib.scrapers.scrape import Scraper, convert
from json import loads
from copy import copy
import numpy as np


class Json(Scraper):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self.all_data = []

    def scrape_groups(self):
        super().scrape_groups()
        self.configs = self.files

    def scrape_configs(self):
        """
        Loop file files in parallel and populate the data dictionary on this proc.
        Note that `self.configs` at this point includes all filenames on this proc.
        """
        self.all_data = [] # Reset to empty list in case running scraper twice.
        self.files = self.configs
        self.conversions = copy(self.default_conversions)
        data_path = self.config.sections["PATH"].datapath
        for i, file_name in enumerate(self.files):
            if file_name.endswith('.json'):
                with open(file_name) as file:
                    if file.readline()[0]=="{":
                        file.seek(0)
                    try:
                        self.data = loads(file.read(), parse_constant=True)
                    except Exception as e:
                        self.pt.single_print(f"Trouble parsing training data: {file_name}")
                        self.pt.single_print(f"{e}")

                    assert len(self.data) == 1, "More than one object (dataset) is in this file"

                    self.data = self.data['Dataset']

                    assert len(self.data['Data']) == 1, "More than one configuration in this dataset"
                    
                    training_file = file_name.split("/")[-1]
                    self.data['File'] = training_file
                    group_name = file_name.replace(data_path,'').replace(training_file,'')[1:-1] 
                    self.data['Group'] = group_name
                    
                    assert all(k not in self.data for k in self.data["Data"][0].keys()), \
                        "Duplicate keys in dataset and data"

                    # Move data up one level
                    self.data.update(self.data.pop('Data')[0])  

                    for key in self.data:
                        if "Style" in key:
                            if key.replace("Style", "") in self.conversions:
                                temp = self.config.sections["SCRAPER"].properties[key.replace("Style", "")]
                                temp[1] = self.data[key]
                                self.conversions[key.replace("Style", "")] = convert(temp)

                    for key in self.config.sections["SCRAPER"].properties:
                        if key in self.data:
                            self.data[key] = np.asarray(self.data[key])

                    natoms = np.shape(self.data["Positions"])[0]
                    #self.pt.shared_arrays["number_of_atoms"].sliced_array[i] = natoms
                    self.data["QMLattice"] = (self.data["Lattice"] * self.conversions["Lattice"]).T

                    # Populate with LAMMPS-normalized lattice
                    del self.data["Lattice"]  

                    # TODO Check whether "Label" container useful to keep around
                    if "Label" in self.data:
                        del self.data["Label"] 

                    if not isinstance(self.data["Energy"], float):
                        self.data["Energy"] = float(self.data["Energy"])

                    # Insert electronegativities, which are per-atom scalars
                    if (self.config.sections["CALCULATOR"].per_atom_scalar):
                        if not isinstance(self.data["Chis"], float):
                            self.data["Chis"] = self.data["Chis"]

                    # Currently, ESHIFT should be in units of your training data (note there is no conversion)
                    if hasattr(self.config.sections["ESHIFT"], 'eshift'):
                        for atom in self.data["AtomTypes"]:
                            self.data["Energy"] += self.config.sections["ESHIFT"].eshift[atom]

                    self.data["test_bool"] = self.test_bool[i]

                    self.data["Energy"] *= self.conversions["Energy"]

                    self._rotate_coords()
                    self._translate_coords()

                    self._weighting(natoms)

                    self.all_data.append(self.data)
            else:
                self.pt.single_print("Non-json file found: ", file_name)    

        return self.all_data
