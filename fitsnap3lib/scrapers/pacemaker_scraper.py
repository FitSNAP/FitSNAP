from fitsnap3lib.scrapers.scrape import Scraper, convert
from copy import copy
import pandas as pd
import numpy as np
import gzip
import pickle
from os import path, listdir


class Pacemaker(Scraper):
    """
    Scraper for pacemaker pckl.gzip format
    """

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self.all_data = []

    def scrape_groups(self):
        """
        Scrape groups from pacemaker pckl.gzip files.
        For pacemaker format, each pckl.gzip file IS the group, not a directory containing files.
        """
        # Reset as empty dict in case running scrape twice.
        self.files = {}
        
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
            
            # Store file paths with sizes
            if folder not in self.files:
                self.files[folder] = []
            
            for file_name in folder_files:
                full_path = path.join(folder, file_name)
                file_size = path.getsize(full_path)
                self.files[folder].append([full_path, file_size])
                
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
                    self.files[folder].pop()
                    
            # Move testing files
            for i in range(testing_size):
                if self.files[folder]:
                    self.tests[folder].append(self.files[folder].pop())
                    
            self.group_table[key]['training_size'] = training_size
            self.group_table[key]['testing_size'] = testing_size
        
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
        self.files = self.configs
        self.conversions = copy(self.default_conversions)
        data_path = self.config.sections["PATH"].datapath

        for i, file_name in enumerate(self.files):
            if file_name.endswith('.pckl.gzip') or file_name.endswith('.pkl.gzip'):
                try:
                    # Load pacemaker dataframe
                    self.pt.single_print(f"Loading pacemaker file: {file_name}")
                    
                    # Handle gzipped pickle files properly
                    with gzip.open(file_name, 'rb') as f:
                        df = pickle.load(f)
                    
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
            training_file = os.path.basename(file_name)
            data['File'] = training_file
            
            # Group name from file path
            group_name = file_name.replace(data_path, '').replace(training_file, '').replace("/", "")
            if not group_name:
                group_name = os.path.splitext(os.path.splitext(training_file)[0])[0]  # Remove .pckl.gzip
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
            if 'energy' in row:
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
            return None
