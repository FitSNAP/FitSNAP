from fitsnap3lib.scrapers.scrape import Scraper
import numpy as np
import logging
import random
from os import path
from copy import copy

# Suppress logging warnings from PyTorch distributed
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

try:
    from ase import Atoms
    from fairchem.core.datasets import AseDBDataset
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False

# ------------------------------------------------------------------------------------------------

class FAIRChem(Scraper):
    """
    FAIRChem scraper for reading OMat24, OC20, OC22, and other ASE-compatible LMDB datasets.
    Designed for read-only access with MPI parallelization.
    """

    def __init__(self, name, pt, config):
        if not HAS_LMDB:
            raise ImportError("LMDB scraper requires: pip install lmdb fairchem-core")
        
        super().__init__(name, pt, config)
        self.data = []
        
        # Get allowed elements from config
        if "REAXFF" in self.config.sections:
            allowed_elements = self.config.sections["REAXFF"].elements
        elif "ACE" in self.config.sections:
            allowed_elements = self.config.sections["ACE"].types
        elif "BISPECTRUM" in self.config.sections:
            allowed_elements = self.config.sections["BISPECTRUM"].types
        else:
            # Default to common elements if not specified
            allowed_elements = ["H", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca"]
        
        self.allowed_elements = set(allowed_elements)
        
        # LMDB datasets will be determined from group names in scrape_groups
        # No single filename needed since we use multiple subdataset paths
        
        # MPI setup
        if self.pt.stubs == 0:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = pt.get_rank()
            self.size = pt.get_size()
        else:
            self.rank = 0
            self.size = 1
            
        # Configuration options
        self.use_stress = self.config.sections["CALCULATOR"].stress if hasattr(self.config.sections["CALCULATOR"], 'stress') else False
        self.use_forces = self.config.sections["CALCULATOR"].force if hasattr(self.config.sections["CALCULATOR"], 'force') else True
        
        # For handling large datasets, allow subsampling
        self.max_configs_per_rank = getattr(self.config.sections["SCRAPER"], 'max_configs_per_rank', None)
        
        # Always use MDB_NOLOCK for distributed filesystems
        self.use_nolock = True
        
        # Option to skip structures with missing forces/energies
        self.require_energy = getattr(self.config.sections["SCRAPER"], 'require_energy', True)
        self.require_forces = getattr(self.config.sections["SCRAPER"], 'require_forces', self.use_forces)
        
        # Debugging options
        self.verbose = getattr(self.config.sections["SCRAPER"], 'verbose', False)

    def scrape_groups(self, group_names=None):
        """
        Open LMDB datasets and identify available configurations.
        Each group in [GROUPS] maps to a subdataset directory.
        """
        self.group_metadata = {}
        self.local_configs = []
        
        # Get group table from config
        self.group_table = self.config.sections["GROUPS"].group_table
        
        # Build dataset paths for each group and get individual sizes
        dataset_paths = []
        group_to_path = {}
        self.group_index_ranges = {}  # Track which indices belong to which group
        cumulative_size = 0
        
        for group_name in self.group_table.keys():
            dataset_path = path.join(self.config.sections["PATH"].datapath, group_name)
            dataset_paths.append(dataset_path)
            group_to_path[group_name] = dataset_path
            
            # Get size of this individual dataset
            try:
                individual_dataset = AseDBDataset(config=dict(src=dataset_path))
                dataset_size = len(individual_dataset)
                
                # Track index range for this group
                self.group_index_ranges[group_name] = (cumulative_size, cumulative_size + dataset_size)
                cumulative_size += dataset_size
                
                if self.rank == 0:
                    self.pt.single_print(f"Group '{group_name}': {dataset_size} configurations (indices {self.group_index_ranges[group_name][0]}-{self.group_index_ranges[group_name][1]-1})")
                    
            except Exception as e:
                if self.rank == 0:
                    self.pt.single_print(f"Warning: Could not load dataset for group '{group_name}' at {dataset_path}: {e}")
                continue
            
        try:
            # Use fairchem's AseDBDataset with multiple paths
            config_kwargs = {}
            # Note: AseDBDataset may not directly support MDB_NOLOCK
            # This might need to be handled at the LMDB environment level
                
            self.dataset = AseDBDataset(config=dict(src=dataset_paths, **config_kwargs))
            total_dataset_size = len(self.dataset)
            
            if self.rank == 0:
                self.pt.single_print(f"Opened {len(dataset_paths)} LMDB subdatasets with {total_dataset_size} total configurations")
            
            # Determine which configurations this rank will process
            configs_per_rank = total_dataset_size // self.size
            remainder = total_dataset_size % self.size
            
            start_idx = self.rank * configs_per_rank + min(self.rank, remainder)
            end_idx = start_idx + configs_per_rank + (1 if self.rank < remainder else 0)
            
            # Apply max_configs_per_rank limit if specified
            if self.max_configs_per_rank:
                end_idx = min(end_idx, start_idx + self.max_configs_per_rank)
            
            # Store configuration indices for this rank
            self.my_config_indices = list(range(start_idx, end_idx))
            
            if self.rank == 0:
                self.pt.single_print(f"Rank {self.rank} will process configurations {start_idx} to {end_idx-1}")
            
            # Create metadata for each group
            for group_name in self.group_table.keys():
                if group_name in self.group_index_ranges:  # Only if dataset was loaded successfully
                    self.group_metadata[group_name] = {
                        "subset": "train",  # Default to training
                        "eweight": 1.0,
                        "fweight": 1.0,
                        "dataset_path": group_to_path[group_name]
                    }
            
            # Add configurations to local list with group determination
            for idx in self.my_config_indices:
                group_name = self._determine_group_for_index(idx)
                self.local_configs.append((group_name, idx))
                
        except Exception as e:
            raise RuntimeError(f"Failed to open LMDB datasets: {e}")
        
        # Synchronize metadata across ranks
        if self.pt.stubs == 0:
            self.comm.barrier()
            all_meta = self.comm.allgather(self.group_metadata)
            self.group_metadata = {k: v for d in all_meta for k, v in d.items()}

    def divvy_up_configs(self):
        """
        Distribute configurations among MPI ranks.
        For LMDB, this is already done in scrape_groups.
        """
        if self.pt.stubs == 1:
            # For single process, limit to a small number for testing
            self.my_configs = self.local_configs[:min(10, len(self.local_configs))]
        else:
            # Already distributed in scrape_groups
            self.my_configs = self.local_configs
            
        if self.rank == 0:
            self.pt.single_print(f"Total configurations to process across all ranks: {len(self.my_configs) * self.size}")

    def scrape_configs(self):
        """
        Read and process LMDB configurations assigned to this rank.
        """
        self.data = []
        
        try:
            for group_name, config_idx in self.my_configs:
                atoms = self._get_atoms_from_index(config_idx)
                
                if atoms is None:
                    continue
                    
                # Filter by allowed elements
                if not all(symbol in self.allowed_elements for symbol in atoms.get_chemical_symbols()):
                    continue
                
                # Group name was already determined in scrape_groups
                actual_group_name = group_name
                
                # Extract data from atoms object
                data_dict = self._extract_data_from_atoms(atoms, actual_group_name, config_idx)
                
                if data_dict is not None:
                    self.data.append(data_dict)
                elif self.verbose and self.rank == 0:
                    self.pt.single_print(f"Skipped configuration {config_idx} (missing required data or elements)")
                    
        except Exception as e:
            raise RuntimeError(f"Error reading LMDB configurations: {e}")
        
        if self.rank == 0:
            self.pt.single_print(f"Successfully processed {len(self.data)} configurations")
            
        return self.data

    def _get_atoms_from_index(self, idx):
        """
        Get ASE Atoms object from LMDB index.
        """
        try:
            # Use fairchem dataset loader
            return self.dataset.get_atoms(idx)
        except Exception as e:
            logging.warning(f"Failed to read configuration {idx}: {e}")
            return None
    
    def _determine_group_for_index(self, config_idx):
        """
        Determine which group (subdataset) a configuration index belongs to.
        Uses the index ranges we tracked during scrape_groups.
        """
        for group_name, (start_idx, end_idx) in self.group_index_ranges.items():
            if start_idx <= config_idx < end_idx:
                return group_name
        
        # Fallback to first group if no match found
        group_names = list(self.group_metadata.keys())
        if group_names:
            return group_names[0]
        else:
            return "unknown_group"

    def _extract_data_from_atoms(self, atoms, group_name, config_idx):
        """
        Extract FitSNAP-compatible data from ASE Atoms object.
        """
        try:
            # Basic atomic information
            positions = atoms.get_positions()
            cell = atoms.get_cell()
            symbols = atoms.get_chemical_symbols()
            
            # Energy (required)
            energy = None
            if hasattr(atoms, 'get_potential_energy'):
                try:
                    energy = atoms.get_potential_energy()
                except:
                    energy = atoms.info.get('energy', None)
            else:
                energy = atoms.info.get('energy', None)
                
            # Check if energy is required and missing
            if self.require_energy and energy is None:
                return None
            
            # Forces (optional)
            forces = None
            if self.use_forces:
                if hasattr(atoms, 'get_forces'):
                    try:
                        forces = atoms.get_forces()
                    except:
                        forces = atoms.arrays.get('forces', None)
                else:
                    forces = atoms.arrays.get('forces', None)
                    
                # Check if forces are required and missing
                if self.require_forces and forces is None:
                    return None
            
            # Stress (optional)
            stress = None
            if self.use_stress:
                if hasattr(atoms, 'get_stress'):
                    try:
                        stress = atoms.get_stress()
                    except:
                        stress = atoms.info.get('stress', None)
                else:
                    stress = atoms.info.get('stress', None)
            
            # Create lattice matrix (3x3)
            lattice = cell.array.copy() if hasattr(cell, 'array') else np.array(cell)
            
            # Ensure we have a proper 3x3 lattice
            if lattice.shape != (3, 3):
                # Create a large box for non-periodic systems
                max_coord = np.max(np.abs(positions)) + 10.0
                lattice = np.diag([max_coord * 2, max_coord * 2, max_coord * 2])
            
            # Create data dictionary compatible with FitSNAP
            data_dict = {
                "Group": group_name,
                "File": f"{group_name}/{config_idx}",
                "Subset": self.group_metadata[group_name]["subset"],
                "Positions": positions,
                "Energy": float(energy) if energy is not None else 0.0,
                "AtomTypes": symbols,
                "NumAtoms": len(symbols),
                "Lattice": lattice,
                "test_bool": False,  # Default to training
                "eweight": self.group_metadata[group_name]["eweight"],
                "fweight": self.group_metadata[group_name]["fweight"] / len(symbols),
            }
            
            # Add forces if available
            if forces is not None:
                data_dict["Forces"] = forces
            
            # Add stress if available
            if stress is not None:
                data_dict["Stress"] = stress
            
            return data_dict
            
        except Exception as e:
            logging.warning(f"Failed to extract data from atoms object {config_idx}: {e}")
            return None

    def __del__(self):
        """
        Clean up resources on destruction.
        """
        # AseDBDataset handles its own cleanup
        pass
