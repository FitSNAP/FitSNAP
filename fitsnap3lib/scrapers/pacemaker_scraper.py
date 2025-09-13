from fitsnap3lib.scrapers.scrape import Scraper, convert
from copy import copy
import pandas as pd
import numpy as np
import gzip
import pickle
from os import path, listdir
from os.path import basename, splitext


class Pacemaker(Scraper):
    """
    Scraper for pacemaker pckl.gzip format
    """

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self.all_data = []
        self.file_to_group_mapping = {}

    def scrape_groups(self):
        """
        Scrape groups from pacemaker pckl.gzip files.
        For pacemaker format, each pckl.gzip file IS the group, containing multiple structures.
        """
        # Reset as empty dict in case running scrape twice.
        self.files = {}
        self.configs = {}
        # Create mapping from file paths to group names
        self.file_to_group_mapping = {}
        
        group_dict = {k: self.config.sections["GROUPS"].group_types[i]
                      for i, k in enumerate(self.config.sections["GROUPS"].group_sections)}
        self.group_table = self.config.sections["GROUPS"].group_table
        size_type = None
        testing_size_type = None
        user_set_random_seed = self.config.sections["GROUPS"].random_seed
        
        if self.config.sections["GROUPS"].random_sampling:
            self.pt.single_print(f"Random sampling of groups toggled on.")
            if not user_set_random_seed:
                sampling_seed = self.pt.get_seed()
                seed_txt = f"FitSNAP-generated seed for random sampling: {self.pt.get_seed()}"
            else:
                if user_set_random_seed.is_integer():
                    sampling_seed = int(user_set_random_seed)
                seed_txt = f"User-set seed for random sampling: {sampling_seed}"
            self.pt.single_print(seed_txt)
            from random import seed, shuffle
            seed(sampling_seed)
            self._write_seed_file(seed_txt)
        
        for key in self.group_table:
            bc_bool = False
            training_size = None
            
            # Handle training and testing size logic
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
                
            # For pacemaker: the "key" should be a pckl.gzip file path
            pckl_file_path = path.join(self.config.sections["PATH"].datapath, key)
            
            if not path.isfile(pckl_file_path):
                raise FileNotFoundError(f"Pacemaker file not found: {pckl_file_path}")
            
            # Load the file to count structures inside it
            try:
                df = pd.read_pickle(pckl_file_path, compression='gzip')
                nconfigs = len(df)
            except Exception as e:
                raise RuntimeError(f"Could not read pacemaker file {pckl_file_path}: {e}")
            
            # Apply size calculations to the structures within the file
            if training_size < 1 or (training_size == 1 and size_type == float):
                if training_size == 1:
                    training_size = abs(training_size) * nconfigs
                elif training_size == 0:
                    pass
                else:
                    training_size = max(1, int(abs(training_size) * nconfigs + 0.5))
                if bc_bool and testing_size == 0:
                    testing_size = nconfigs - training_size
            if testing_size != 0 and (testing_size < 1 or (testing_size == 1 and testing_size_type == float)):
                testing_size = max(1, int(abs(testing_size) * nconfigs + 0.5))
                
            training_size = self._float_to_int(training_size)
            testing_size = self._float_to_int(testing_size)
            
            if nconfigs - testing_size - training_size < 0:
                warnstr = f"\nWARNING: {key} train size {training_size} + test size {testing_size} > nconfigs {nconfigs}\n"
                warnstr += "         Forcing testing size to add up properly.\n"
                self.pt.single_print(warnstr)
                testing_size = nconfigs - training_size
                
            if (self.config.args.verbose):
                self.pt.single_print(key, ": Detected ", nconfigs, " fitting on ", training_size, " testing on ", testing_size)
                
            # Store the file and configuration counts
            # Use a dummy "folder" key since base class expects folder structure
            dummy_folder = path.dirname(pckl_file_path) or "."
            if dummy_folder not in self.files:
                self.files[dummy_folder] = []
            if dummy_folder not in self.configs:
                self.configs[dummy_folder] = []
            
            # Store filename once per folder (like xyz_scraper)
            if pckl_file_path not in [f[0] if isinstance(f, list) else f for f in self.files[dummy_folder]]:
                self.files[dummy_folder].append(pckl_file_path)
            
            # Map file path to group key for later lookup
            self.file_to_group_mapping[pckl_file_path] = key
            
            # Store structure indices for configurations (like xyz_scraper stores file positions)
            structure_indices = list(range(nconfigs))
            if self.config.sections["GROUPS"].random_sampling:
                from random import shuffle
                shuffle(structure_indices)
            
            # Add training structure indices
            for i in range(training_size):
                self.configs[dummy_folder].append(structure_indices[i])
            
            # Handle test structures
            if self.tests is None:
                self.tests = {}
            if dummy_folder not in self.tests:
                self.tests[dummy_folder] = []
                
            # Add testing structure indices 
            for i in range(training_size, training_size + testing_size):
                self.tests[dummy_folder].append(structure_indices[i])
            
            self.group_table[key]['training_size'] = training_size
            self.group_table[key]['testing_size'] = testing_size
        
        # Note: self.configs is already set up properly for divvy_up_configs()
    
    @staticmethod
    def _float_to_int(a_float):
        """Convert float to int with validation"""
        if a_float == 0:
            return int(a_float)
        if a_float / int(a_float) != 1:
            raise ValueError("Training and testing Size must be interpretable as integers")
        return int(a_float)
    
    def _write_seed_file(self, txt):
        """Write random sampling seed to file"""
        @self.pt.rank_zero
        def decorated_write_seed_file(txt):
            with open("RandomSamplingSeed.txt", 'w') as f:
                f.write(txt+'\n')
        decorated_write_seed_file(txt)

    def scrape_configs(self):
        """
        Load and process pacemaker pckl.gzip dataframes
        """
        self.all_data = []  # Reset to empty list in case running scraper twice
        self.conversions = copy(self.default_conversions)
        data_path = self.config.sections["PATH"].datapath

        # After divvy_up_configs, self.configs is a flat list where each entry is [structure_index, folder]
        for i, configuration in enumerate(self.configs):
            structure_index = configuration[0]  # Extract structure index 
            folder = configuration[1]           # Extract folder
            pckl_file_path = self.files[folder][0]  # Get filename from folder
            
            try:
                # Load pacemaker dataframe - pandas automatically handles gzip compression
                df = pd.read_pickle(pckl_file_path, compression='gzip')
                
                # Extract the specific structure (row) from the dataframe
                if structure_index >= len(df):
                    self.pt.single_print(f"Warning: structure index {structure_index} out of range for {pckl_file_path}")
                    continue
                    
                row = df.iloc[structure_index]
                data = self._convert_pacemaker_row(row, pckl_file_path, data_path)
                
                if data is not None:
                    # Use test_bool from divvy_up_configs
                    data["test_bool"] = self.test_bool[i]
                    self.all_data.append(data)
                    
            except Exception as e:
                self.pt.single_print(f"Error reading pacemaker file {pckl_file_path}: {e}")
                import traceback
                self.pt.single_print(f"Traceback: {traceback.format_exc()}")
                continue

        self.pt.single_print(f"Loaded {len(self.all_data)} configurations from pacemaker files")
        return self.all_data

    def _convert_pacemaker_row(self, row, file_name, data_path):
        """
        Convert a single pacemaker dataframe row to FitSNAP format
        """
        try:
            # Extract ASE atoms object
            atoms = row['ase_atoms']
            
            # Create FitSNAP data dictionary
            data = {}
            
            # Store ASE atoms object to avoid recreation in pyace calculator
            data['ase_atoms'] = atoms
            
            # Basic structure info
            training_file = basename(file_name)
            data['File'] = training_file
            
            # Group name from the mapping we created during scrape_groups
            if hasattr(self, 'file_to_group_mapping') and file_name in self.file_to_group_mapping:
                group_name = self.file_to_group_mapping[file_name]
            else:
                # Fallback: derive from filename
                group_name = splitext(splitext(basename(file_name))[0])[0]  # Remove .pckl.gzip
            data['Group'] = group_name
            
            # Positions and atomic numbers
            data["Positions"] = atoms.get_positions()
            data["AtomTypes"] = atoms.get_atomic_numbers()
            data["NumAtoms"] = len(atoms)
            
            # Convert atomic numbers to LAMMPS atom types (1-indexed)
            unique_types = sorted(list(set(data["AtomTypes"])))
            type_mapping = {atomic_num: i + 1 for i, atomic_num in enumerate(unique_types)}
            data["AtomTypes"] = np.array([type_mapping[at] for at in data["AtomTypes"]])
            
            # Lattice vectors
            cell = atoms.get_cell()
            if np.any(cell):
                # Convert to LAMMPS lattice format (transpose)
                data["QMLattice"] = (cell * self.conversions["Lattice"]).T
            else:
                # For non-periodic systems, create a large box
                pos = data["Positions"]
                box_size = np.max(pos, axis=0) - np.min(pos, axis=0) + 20.0
                data["QMLattice"] = np.diag(box_size) * self.conversions["Lattice"]
            
            # Energy (total energy of the structure)
            # Try different energy field names commonly used in pacemaker dataframes
            if 'energy_corrected' in row:
                data["Energy"] = float(row['energy_corrected']) * self.conversions["Energy"]
            elif 'energy' in row:
                data["Energy"] = float(row['energy']) * self.conversions["Energy"]
            elif hasattr(atoms, 'info') and 'energy' in atoms.info:
                data["Energy"] = float(atoms.info['energy']) * self.conversions["Energy"]
            else:
                # Try to get energy from ASE calculator if available
                try:
                    data["Energy"] = atoms.get_potential_energy() * self.conversions["Energy"]
                except:
                    self.pt.single_print(f"Warning: No energy found for structure in {file_name}")
                    data["Energy"] = 0.0
            
            # Forces
            if 'forces' in row:
                data["Forces"] = np.array(row['forces']) * self.conversions["Forces"]
            elif hasattr(atoms, 'arrays') and 'forces' in atoms.arrays:
                data["Forces"] = atoms.arrays['forces'] * self.conversions["Forces"]
            else:
                # Try to get forces from ASE calculator if available
                try:
                    data["Forces"] = atoms.get_forces() * self.conversions["Forces"]
                except:
                    self.pt.single_print(f"Warning: No forces found for structure in {file_name}")
                    data["Forces"] = np.zeros_like(data["Positions"])
            
            # Stress/Virial (if available)
            if 'stress' in row:
                # Convert stress to virial format expected by FitSNAP
                stress = np.array(row['stress'])
                volume = atoms.get_volume()
                # Convert stress to virial: V_ij = -stress_ij * volume
                data["Stress"] = -stress * volume * self.conversions["Stress"]
            elif hasattr(atoms, 'info') and 'stress' in atoms.info:
                stress = np.array(atoms.info['stress'])
                volume = atoms.get_volume()
                data["Stress"] = -stress * volume * self.conversions["Stress"]
            else:
                # No stress available
                data["Stress"] = np.zeros(6)
            
            # Apply energy shifts if configured
            if hasattr(self.config.sections["ESHIFT"], 'eshift'):
                for atom_type in data["AtomTypes"]:
                    if atom_type in self.config.sections["ESHIFT"].eshift:
                        data["Energy"] += self.config.sections["ESHIFT"].eshift[atom_type]
            
            # Apply coordinate transformations
            natoms = len(data["Positions"])
            self.data = data
            self._rotate_coords()
            self._translate_coords()
            self._weighting(natoms)
            
            return self.data
            
        except Exception as e:
            self.pt.single_print(f"Error converting pacemaker row: {e}")
            self.pt.single_print(f"Row data keys: {list(row.keys()) if hasattr(row, 'keys') else 'Unknown'}")
            self.pt.single_print(f"File: {file_name}")
            import traceback
            self.pt.single_print(f"Traceback: {traceback.format_exc()}")
            return None
