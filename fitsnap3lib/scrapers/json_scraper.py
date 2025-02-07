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
        data_path = self.config.sections["PATH"].dataPath
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

                    assert len(self.data) == 1, f"More than one object (dataset) is in this file. \nFile name: {file_name}"

                    self.data = self.data['Dataset']

                    assert len(self.data['Data']) == 1, f"More than one configuration in this dataset. \nFile name: {file_name}"
                    
                    training_file = file_name.split("/")[-1]
                    self.data['File'] = training_file
                    group_name = file_name.replace(data_path,'').replace(training_file,'').replace("/","") 
                    self.data['Group'] = group_name
                    
                    assert all(k not in self.data for k in self.data["Data"][0].keys()), \
                        f"Duplicate keys in dataset and data. \nFile name: {file_name}"

                    # Move data up one level
                    self.data.update(self.data.pop('Data')[0])  

                    for key in self.config.sections["SCRAPER"].properties:
                        if key in self.data:
                            self.data[key] = np.asarray(self.data[key])

                    if not isinstance(self.data["Energy"], float):
                        self.data["Energy"] = float(self.data["Energy"])

                    if "REAXFF" not in self.config.sections:

                        for key in self.data:
                            if "Style" in key:
                                if key.replace("Style", "") in self.conversions:
                                    temp = self.config.sections["SCRAPER"].properties[key.replace("Style", "")]
                                    temp[1] = self.data[key]
                                    self.conversions[key.replace("Style", "")] = convert(temp)

                        natoms = np.shape(self.data["Positions"])[0]
                        self.data["QMLattice"] = (self.data["Lattice"] * self.conversions["Lattice"]).T

                        # Populate with LAMMPS-normalized lattice
                        del self.data["Lattice"]

                        # TODO Check whether "Label" container useful to keep around
                        if "Label" in self.data:
                            del self.data["Label"]

                        # Insert electronegativities, which are per-atom scalars
                        if (self.config.sections["CALCULATOR"].per_atom_scalar):
                            if not isinstance(self.data["Chis"], float):
                                self.data["Chis"] = self.data["Chis"]

                        # Currently, ESHIFT should be in units of your training data (note there is no conversion)
                        if hasattr(self.config.sections["ESHIFT"], 'eshift'):
                            for atom in self.data["AtomTypes"]:
                                self.data["Energy"] += self.config.sections["ESHIFT"].eshift[atom]

                        self._rotate_coords()
                        self._translate_coords()
                        self._weighting(natoms)
                        self.data["Energy"] *= self.conversions["Energy"]
                        # end of non-REAXFF code block

                    self.data["test_bool"] = self.test_bool[i]
                    self.all_data.append(self.data)
            else:
                self.pt.single_print("! WARNING: Non-JSON file found: ", file_name)    

        return self.all_data


    def scrape_configs_reaxff(self):
        """
        Loop file files in parallel and populate the data dictionary on this proc.
        Note that `self.configs` at this point includes all filenames on this proc.
        """

        return self.scrape_configs()
        self.all_data = [] # Reset to empty list in case running scraper twice.
        self.files = self.configs

        #print(f"self.configs={self.configs}")

        self.conversions = copy(self.default_conversions)
        data_path = self.config.sections["PATH"].dataPath
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

                    assert len(self.data) == 1, f"More than one object (dataset) is in this file. \nFile name: {file_name}"

                    self.data = self.data['Dataset']

                    training_file = file_name.split("/")[-1]
                    self.data['File'] = training_file
                    group_name = file_name.replace(data_path,'').replace(training_file,'').replace("/","") 
                    self.data['Group'] = group_name

                    configs = []
                    ground_index = 0
                    ground_reference_energy = 999999.99

                    for i, d in enumerate(self.data["Data"]):

                        for key in self.config.sections["SCRAPER"].properties:
                            if key in d:
                                d[key] = np.asarray(d[key])

                        assert all(k not in self.data for k in d.keys()), \
                            f"Duplicate keys in dataset and data. \nFile name: {file_name}"

                        if not isinstance(d["Energy"], float):
                            d["Energy"] = float(d["Energy"])

                        if ground_reference_energy > d["Energy"]:
                            ground_index = i
                            ground_reference_energy = d["Energy"]

                        if "Weight" not in d: d["Weight"] = 1.0
                        configs.append(d)

                    for i, d in enumerate(configs):
                        d["ground_relative_index"] = ground_index - i
                        d["Energy"] -= ground_reference_energy

                    qm_y = [d["Energy"] for d in configs]
                    auto_weights = np.square(np.max(qm_y)*1.1-np.array(qm_y))

                    self.all_data.append({
                        'ground_index': ground_index,
                        'reference_energy': np.array([c["Energy"] for c in configs]),
                        #'weights': np.array([c["Weight"] for c in configs]),
                        'weights': auto_weights/np.sum(auto_weights),
                        'configs': configs
                    })

            else:
                self.pt.single_print("! WARNING: Non-JSON file found: ", file_name)    

        return self.all_data

