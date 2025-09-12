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
        For pacemaker format, each pckl.gzip file IS the group, not a directory containing files.
        """
        # Reset as empty dict in case running scrape twice.
        self.files = {}
        # Create a mapping from file paths to group names
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
            
            # Handle training and testing size logic (copied from base class)
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
                
            # For pacemaker: the "group" key should be a file path or pattern
            pckl_file_path = path.join(self.config.sections["PATH"].datapath, key)
            
            # Check if it's a direct file path
            if path.isfile(pckl_file_path) and pckl_file_path.endswith(('.pckl.gzip', '.pkl.gzip')):
                # Single file case
                folder_files = [path.basename(pckl_file_path)]
                folder = path.dirname(pckl_file_path)
            else:
                # Pattern/directory case - look for pckl.gzip files
                folder = path.dirname(pckl_file_path) if path.dirname(pckl_file_path) else self.config.sections["PATH"].datapath
                if path.isdir(folder):
                    all_files = listdir(folder)
                    # Filter for pckl.gzip files matching the pattern
                    if '*' in key or '?' in key:
                        import fnmatch
                        pattern = path.basename(pckl_file_path)
                        folder_files = [f for f in all_files if fnmatch.fnmatch(f, pattern) and f.endswith(('.pckl.gzip', '.pkl.gzip'))]
                    else:
                        folder_files = [f for f in all_files if f.endswith(('.pckl.gzip', '.pkl.gzip'))]
                else:
                    raise FileNotFoundError(f"Cannot find pacemaker files for group {key} at path {pckl_file_path}")
            
            # Store file paths with sizes and map files to group names
            if folder not in self.files:
                self.files[folder] = []
            
            for file_name in folder_files:
                full_path = path.join(folder, file_name)
                file_size = path.getsize(full_path)
                self.files[folder].append([full_path, file_size])
                # Map this file to its config group name, removing .pckl.gzip extension
                clean_group_name = key
                if clean_group_name.endswith('.pckl.gzip'):
                    clean_group_name = clean_group_name[:-11]  # Remove .pckl.gzip (11 chars)
                elif clean_group_name.endswith('.pkl.gzip'):
                    clean_group_name = clean_group_name[:-10]  # Remove .pkl.gzip (10 chars)
                self.file_to_group_mapping[full_path] = clean_group_name
                
            if self.config.sections["GROUPS"].random_sampling:
                from random import shuffle
                shuffle(self.files[folder])
                
            # Handle size calculations (copied from base class logic)
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
            
            if nfiles - testing_size - training_size < 0:
                warnstr = f"\nWARNING: {key} train size {training_size} + test size {testing_size} > nfiles {nfiles}\n"
                warnstr += "         Forcing testing size to add up properly.\n"
                self.pt.single_print(warnstr)
                testing_size = nfiles - training_size
                
            if (self.config.args.verbose):
                self.pt.single_print(key, ": Detected ", nfiles, " fitting on ", training_size, " testing on ", testing_size)
                
            # Handle test files
            if self.tests is None:
                self.tests = {}
            self.tests[folder] = []
            
            # Remove excess files
            for i in range(nfiles - training_size - testing_size):
                if self.files[folder]:
                    removed_file = self.files[folder].pop()
                    # Remove from mapping too
                    if isinstance(removed_file, list):
                        del self.file_to_group_mapping[removed_file[0]]
                    else:
                        del self.file_to_group_mapping[removed_file]
                    
            # Move testing files
            for i in range(testing_size):
                if self.files[folder]:
                    self.tests[folder].append(self.files[folder].pop())
                    
            self.group_table[key]['training_size'] = training_size
            self.group_table[key]['testing_size'] = testing_size
        
        # Clean up group names in group_table by removing .pckl.gzip extensions
        cleaned_group_table = {}
        for key, value in self.group_table.items():
            clean_key = key
            if clean_key.endswith('.pckl.gzip'):
                clean_key = clean_key[:-11]  # Remove .pckl.gzip (11 chars)
            elif clean_key.endswith('.pkl.gzip'):
                clean_key = clean_key[:-10]  # Remove .pkl.gzip (10 chars)
            cleaned_group_table[clean_key] = value
        self.group_table = cleaned_group_table
        
        # For pacemaker format, each pckl.gzip file is a group
        self.configs = self.files
    
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

        # Handle both dictionary format (before divvy_up_configs) and list format (after divvy_up_configs)
        all_files = []
        
        if isinstance(self.configs, dict):
            # Dictionary format: {folder: [[filepath, filesize], ...]}
            for folder, file_list in self.configs.items():
                for file_entry in file_list:
                    if isinstance(file_entry, list):
                        all_files.append(file_entry[0])  # filepath is first element
                    else:
                        all_files.append(file_entry)
                        
            # Also add test files if they exist
            if hasattr(self, 'tests') and self.tests:
                for folder, test_file_list in self.tests.items():
                    for file_entry in test_file_list:
                        if isinstance(file_entry, list):
                            all_files.append(file_entry[0])  # filepath is first element
                        else:
                            all_files.append(file_entry)
        else:
            # List format (after divvy_up_configs): each entry is a filepath
            all_files = self.configs

        for i, file_name in enumerate(all_files):
            if file_name.endswith('.pckl.gzip') or file_name.endswith('.pkl.gzip'):
                try:
                    # Load pacemaker dataframe - pandas automatically handles gzip compression
                    self.pt.single_print(f"Loading pacemaker file: {file_name}")
                    df = pd.read_pickle(file_name, compression='gzip')
                    
                    # Process each structure in the dataframe
                    for idx, row in df.iterrows():
                        data = self._convert_pacemaker_row(row, file_name, data_path)
                        if data is not None:
                            data["test_bool"] = self.test_bool[i] if i < len(self.test_bool) else False
                            self.all_data.append(data)
                            
                except Exception as e:
                    self.pt.single_print(f"Error reading pacemaker file {file_name}: {e}")
                    import traceback
                    self.pt.single_print(f"Traceback: {traceback.format_exc()}")
                    continue
            else:
                self.pt.single_print(f"! WARNING: Non-pacemaker file found: {file_name}")

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
            
            # Basic structure info
            training_file = basename(file_name)
            data['File'] = training_file
            
            # Group name should match the config group, not derived from filename
            # Use the mapping we created during scrape_groups
            if hasattr(self, 'file_to_group_mapping') and file_name in self.file_to_group_mapping:
                group_name = self.file_to_group_mapping[file_name]
            else:
                # Fallback: try to find which group this file belongs to by checking group_table keys
                group_name = None
                # Clean up the filename for comparison
                clean_file_base = splitext(splitext(basename(file_name))[0])[0]  # Remove .pckl.gzip
                for group_key in self.group_table.keys():
                    if group_key == clean_file_base or group_key in clean_file_base or clean_file_base.startswith(group_key):
                        group_name = group_key
                        break
                if group_name is None:
                    # Last resort: use the first group key if only one group exists
                    if len(self.group_table) == 1:
                        group_name = list(self.group_table.keys())[0]
                    else:
                        raise ValueError(f"Cannot determine group for file {file_name}. Available groups: {list(self.group_table.keys())}")
            data['Group'] = group_name
            
            # Positions and atomic numbers
            data["Positions"] = atoms.get_positions()
            data["AtomTypes"] = atoms.get_atomic_numbers()
            
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
