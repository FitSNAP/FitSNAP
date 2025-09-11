from fitsnap3lib.scrapers.scrape import Scraper, convert
from copy import copy
import pandas as pd
import numpy as np
import gzip
import pickle
import os


class Pacemaker(Scraper):
    """
    Scraper for pacemaker pckl.gzip format
    """

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self.all_data = []

    def scrape_groups(self):
        """
        Scrape groups from pacemaker pckl.gzip files
        """
        super().scrape_groups()
        # For pacemaker format, each pckl.gzip file is a group
        self.configs = self.files

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
                    df = pd.read_pickle(file_name)
                    
                    # Process each structure in the dataframe
                    for idx, row in df.iterrows():
                        data = self._convert_pacemaker_row(row, file_name, data_path)
                        if data is not None:
                            data["test_bool"] = self.test_bool[i] if i < len(self.test_bool) else False
                            self.all_data.append(data)
                            
                except Exception as e:
                    self.pt.single_print(f"Error reading pacemaker file {file_name}: {e}")
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
