from fitsnap3lib.scrapers.scrape import Scraper
import numpy as np
import h5py
import logging
import random
from os import path
import numba as nb
from scipy.spatial import KDTree

# ------------------------------------------------------------------------------------------------

def atomic_number_to_symbol(n):
    return {1:"H",6:"C",7:"N",8:"O",9:"F",11:"Na",12:"Mg",15:"P",16:"S",17:"Cl",19:"K",20:"Ca"}[n]

def atomic_symbol_to_number(s):
    return {"H":1,"C":6,"N":7,"O":8,"F":9,"Na":11,"Mg":12,"P":15,"S":16,"Cl":17,"K":19,"Ca":20}[s]


@staticmethod
@nb.jit(nopython=True)
def compute_esp_numba(grid_points, positions, charges, dipoles, quadrupoles, octupoles):
  """Numba-optimized ESP calculation"""
  esp_values = np.zeros(len(grid_points), dtype=np.float32)
  cutoff_dist = 1.4  # Minimum distance cutoff
        
  for i in range(len(grid_points)):
    grid_point = grid_points[i]
    esp = 0.0
            
    # Calculate ESP from all atoms (Numba will optimize this loop)
    for j in range(len(positions)):
      pos = positions[j]
      r_vec = grid_point - pos
      r2 = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
      r = np.sqrt(r2)
                
      # Skip points too close to atoms
      if r < cutoff_dist: continue
                
      # Charge contribution
      esp += charges[j] / r
  
      # Add dipole contribution if available
      if dipoles is not None:
        # μ⋅r / r³
        mu = dipoles[j]
        esp += (mu[0] * r_vec[0] + mu[1] * r_vec[1] + mu[2] * r_vec[2]) / (r**3)
                
      # Add quadrupole contribution if available
      if quadrupoles is not None:
        Q = quadrupoles[j]
        r_inv = 1.0 / r
        r_hat = r_vec * r_inv
                    
        # Quadrupole formula: 1/2 ∑_αβ Q_αβ (3r_α r_β/r² - δ_αβ) / r³
        quad_term = 0.0
        for alpha in range(3):
          for beta in range(3):
            delta_ab = 1.0 if alpha == beta else 0.0
            quad_term += Q[alpha, beta] * (3.0 * r_hat[alpha] * r_hat[beta] - delta_ab)
                    
        esp += 0.5 * quad_term / (r**3)
      
      # Add octupole contribution if available
      if octupoles is not None:
        O = octupoles[j]
        r_hat = r_vec / r
                    
        # Octupole formula
        octu_term = 0.0
        for alpha in range(3):
          for beta in range(3):
            for gamma in range(3):
              main_term = 5.0 * r_hat[alpha] * r_hat[beta] * r_hat[gamma]
              delta_bg = 1.0 if beta == gamma else 0.0
              delta_ag = 1.0 if alpha == gamma else 0.0
              delta_ab = 1.0 if alpha == beta else 0.0
              correction = r_hat[alpha] * delta_bg + r_hat[beta] * delta_ag + r_hat[gamma] * delta_ab
              octu_term += O[alpha, beta, gamma] * (main_term - correction)
                    
        esp += octu_term / (6.0 * r**4)
            
    esp_values[i] = esp
  
  return esp_values

# ------------------------------------------------------------------------------------------------

class HDF5(Scraper):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self.data = []
        logging.getLogger("h5py._conv").setLevel(logging.WARNING)
        self._lattice_margin = 10
        
        if( "REAXFF" in self.config.sections ):
          allowed_elements = self.config.sections["REAXFF"].elements
        elif( "ACE" in self.config.sections ):
          allowed_elements = self.config.sections["ACE"].types
          
        print(f"*** allowed_elements {allowed_elements}")
        
        
        self.allowed_atomic_numbers = set(atomic_symbol_to_number(e) for e in allowed_elements)
        self.hdf5_path = path.join(self.config.sections["PATH"].dataPath, \
          self.config.sections["SCRAPER"].filename)

        if self.pt.stubs == 0:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = pt.get_rank()
            self.size = pt.get_size()
        else:
            self.rank = 0
            self.size = 1

    # --------------------------------------------------------------------------------------------

    def scrape_groups(self, group_names=None):
        self.group_metadata = {}
        self.local_configs = []

        file_kwargs = {"driver": "mpio", "comm": self.comm} if self.pt.stubs == 0 else {}

        with h5py.File(self.hdf5_path, "r", **file_kwargs) as f:

            if group_names is None:
                if self.pt.stubs == 1:
                    group_names = list(f.keys())[:2]
                else:
                    groups = list(f.keys()) if self.rank == 0 else None
                    group_names = self.comm.bcast(groups, root=0)

            for i in range(self.rank, len(group_names), self.size):
                group_name = group_names[i]
                group = f[group_name]
                atomic_numbers = group["atomic_numbers"][()]

                if not np.all(np.isin(atomic_numbers, list(self.allowed_atomic_numbers))):
                    continue

                conformations = group["conformations"][()]
                formation_energy = group["formation_energy"][()]
                subset = group["subset"].asstr()[0]
                is_solvated = "solvated" in subset.lower() or "water" in subset.lower()

                bounds_min = conformations.min(axis=(0, 1)) - 10.0
                bounds_max = conformations.max(axis=(0, 1)) + 10.0
                lattice = [
                    [bounds_max[0] - bounds_min[0], 0.0, 0.0],
                    [0.0, bounds_max[1] - bounds_min[1], 0.0],
                    [0.0, 0.0, bounds_max[2] - bounds_min[2]],
                ]

                # Scale all to ~comparable range using expected magnitude
                norm = {
                    "energy":     1.0 / 100.0,     # normalize ~100 kcal/mol
                    "force":      1.0 / 5000.0,    # normalize ~5000 kcal/mol/Å
                    "charge":     1.0 / 250.0,     # normalize large sum(q_i)^2
                    "dipole":     1.0 / 30.0,      # normalize ~30 e·Å
                    "quadrupole": 1.0 / 600.0,     # normalize ~600 e·Å²
                    "esp":        1.0,             # normalize later by mean(esp)
                }

                importance = {
                    "energy":     0.1,
                    "force":      100.0,
                    "charge":     0.05,
                    "dipole":     50.0,
                    "quadrupole": 20.0,
                    "esp":        20.0,
                }

                weights = {k: norm[k] * importance[k] for k in norm}

                self.group_metadata[group_name] = {
                    "subset": subset,
                    "is_solvated": is_solvated,
                    "eweight": weights["energy"],
                    "fweight": weights["force"],
                    "cweight": weights["charge"] if is_solvated else 0.0,
                    "dweight": weights["dipole"] if is_solvated else 0.0,
                    "qweight": weights["quadrupole"] if is_solvated else 0.0,
                    "gweight": weights["esp"], # if is_solvated else 0.0,
                    "bounds": (bounds_min, bounds_max),
                    "lattice": lattice
                }

                #print(f"*** {group_name} formation_energy {formation_energy} weights {weights} {[atomic_number_to_symbol(n) for n in atomic_numbers]}")
                #print(f"*** {group_name} {[atomic_number_to_symbol(n) for n in atomic_numbers]}")

                for j in range(int(conformations.shape[0]*0.8)):
                    self.local_configs.append((group_name, j))

        if self.pt.stubs==0:
            all_meta = self.comm.allgather(self.group_metadata)
            self.group_metadata = {k: v for d in all_meta for k, v in d.items()}

    # --------------------------------------------------------------------------------------------

    def divvy_up_configs(self):

        if self.pt.stubs==1:
            self.my_configs = self.local_configs[:3]
        else:
            all_configs = self.comm.allgather(self.local_configs)
            flat_configs = [cfg for sub in all_configs for cfg in sub]
            flat_configs.sort()

            #total = len(flat_configs)
            total = 1*(self.size)

            base = total // (self.size)

            # make sure that all ranks have same number of configs
            # remainder extra configs can be used for validation testing
            #remainder = total % (self.size-1)
            remainder = 0

            start = self.rank * base + min(self.rank, remainder)
            stop = start + base + (1 if self.rank < remainder else 0)
            expected = base + (1 if self.rank < remainder else 0)
            self.my_configs = flat_configs[start:stop]
            actual = len(self.my_configs)
            #print(f"[Rank {self.rank}] Expected {expected} configs, got {actual}. total {total} base {base}, remainder {remainder}")
            if actual != expected:
                raise RuntimeError(f"[Rank {self.rank}] Expected {expected} configs, got {actual}")

    # --------------------------------------------------------------------------------------------

    def scrape_configs(self):
        self.data = []

        #print(f"*** self.my_configs {self.my_configs}")

        file_kwargs = {"driver": "mpio", "comm": self.comm} if self.pt.stubs == 0 else {}

        from collections import defaultdict
        grouped = defaultdict(list)
        for g, i in self.my_configs:
            grouped[g].append(i)

        BOHR_TO_ANGSTROM = 0.52917721092
        HARTREE_TO_KCAL_MOL = 627.509474
        FORCE_CONV = HARTREE_TO_KCAL_MOL / BOHR_TO_ANGSTROM

        import logging
        logging.getLogger("numba").setLevel(logging.WARNING)

        with h5py.File(self.hdf5_path, "r", **file_kwargs) as f:
            for group_name in sorted(grouped):
                print(f"*** group_name {group_name}", flush=True)
                group = f[group_name]
                meta = self.group_metadata[group_name]
                conformations = group["conformations"][()] * BOHR_TO_ANGSTROM
                formation_energy = group["formation_energy"][()] * HARTREE_TO_KCAL_MOL
                dft_total_gradient = group["dft_total_gradient"][()] * FORCE_CONV
                mbis_charges = group["mbis_charges"][()]
                mbis_dipoles = group["mbis_dipoles"][()]
                mbis_quadrupoles = group["mbis_quadrupoles"][()]
                mbis_octupoles = group["mbis_octupoles"][()]
                scf_dipoles = group["scf_dipole"][()] * BOHR_TO_ANGSTROM
                scf_quadrupole = group["scf_quadrupole"][()]  * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM
                mayer_indices = group["mayer_indices"][()]
                atomic_numbers = group["atomic_numbers"][()]
                for i in grouped[group_name]:

                    charges = mbis_charges[i]
                    sum_charges = np.sum(charges)
                    charges[0,0] += (np.round(sum_charges) - sum_charges)

                    # Calculate ESP grid
                    esp_grid = self.calculate_esp_grid_optimized(
                        positions=conformations[i],
                        charges=mbis_charges[i],
                        dipoles=mbis_dipoles[i] if "mbis_dipoles" in group else None,
                        quadrupoles=mbis_quadrupoles[i] if "mbis_quadrupoles" in group else None,
                        octupoles=mbis_octupoles[i] if "mbis_octupoles" in group else None,
                        bounds=meta["bounds"],
                        spacing=self.config.sections["CALCULATOR"].spacing
                    ) if self.config.sections["CALCULATOR"].esp else None

                    # if np.round(sum_charges) != 0.0: continue

                    self.data.append({
                        "Group": group.name,
                        "File": f"{group.name}/{i}",
                        "Subset": meta["subset"],
                        "Positions": conformations[i],
                        "Energy": formation_energy[i],
                        "Forces": dft_total_gradient[i],
                        "Charges": charges,
                        "Dipole": scf_dipoles[i],
                        "Quadrupole": scf_quadrupole[i],
                        "ESP": esp_grid,
                        "BondOrder": mayer_indices[i],
                        "AtomTypes": [atomic_number_to_symbol(n) for n in atomic_numbers],
                        "NumAtoms": len(atomic_numbers),
                        "Lattice": meta["lattice"],
                        "Bounds": meta["bounds"],
                        "test_bool": random.choices([True, False], weights=[0.7, 0.3])[0],
                        "eweight": meta["eweight"],
                        "fweight": meta["fweight"] / len(atomic_numbers),
                        "vweight": 0.0,
                        "cweight": meta["cweight"],
                        "dweight": meta["dweight"],
                        "qweight": meta["qweight"],
                        "gweight": meta["gweight"] # /np.mean(esp_grid)
                    })

        print(self.data)
        return self.data

    # --------------------------------------------------------------------------------------------

    def calculate_esp_grid(self, positions, charges, dipoles=None, quadrupoles=None,
                          octupoles=None, bounds=None, spacing=0.3):
        """Calculate ESP grid from multipole data"""
        
        # Use the bounds that already include margin
        bounds_min, bounds_max = bounds
        
        # Create grid with specified spacing
        nx = max(1, int(np.ceil((bounds_max[0] - bounds_min[0]) / spacing)))
        ny = max(1, int(np.ceil((bounds_max[1] - bounds_min[1]) / spacing)))
        nz = max(1, int(np.ceil((bounds_max[2] - bounds_min[2]) / spacing)))
        
        # Initialize ESP grid
        esp_grid = np.zeros((nz, ny, nx))
        
        # Unit conversion constants
        BOHR_TO_ANGSTROM = 0.52917721092
        
        # Populate grid with ESP values
        for iz in range(nz):
            z = bounds_min[2] + (iz + 0.5) * spacing
            for iy in range(ny):
                y = bounds_min[1] + (iy + 0.5) * spacing
                for ix in range(nx):
                    x = bounds_min[0] + (ix + 0.5) * spacing
                    grid_point = np.array([x, y, z])
                    
                    # Calculate ESP from multipoles
                    esp = 0.0
                    for i, pos in enumerate(positions):
                        r_vec = grid_point - pos
                        r = np.linalg.norm(r_vec)
                        
                        # Skip points too close to atoms
                        if r < 1.4: continue
                        
                        # Charge contribution
                        esp += charges[i][0] / r
                        
                        # Add dipole contribution if available
                        if dipoles is not None:
                            # μ⋅r / r³ with unit conversion
                            mu = dipoles[i] * BOHR_TO_ANGSTROM  # Convert e·bohr → e·Å
                            esp += np.dot(mu, r_vec) / (r**3)
                        
                        # Add quadrupole contribution if available
                        if quadrupoles is not None:
                            # Convert to 3x3 tensor with unit conversion
                            Q = np.zeros((3, 3))
                            conversion = BOHR_TO_ANGSTROM**2  # Convert e·bohr² → e·Å²
                            Q[0, 0] = quadrupoles[i][0, 0] * conversion
                            Q[1, 1] = quadrupoles[i][1, 1] * conversion
                            Q[2, 2] = quadrupoles[i][2, 2] * conversion
                            Q[0, 1] = Q[1, 0] = quadrupoles[i][0, 1] * conversion
                            Q[0, 2] = Q[2, 0] = quadrupoles[i][0, 2] * conversion
                            Q[1, 2] = Q[2, 1] = quadrupoles[i][1, 2] * conversion
                            
                            # Correct quadrupole formula: 1/2 ∑_αβ Q_αβ (3r_α r_β/r² - δ_αβ) / r³
                            r_hat = r_vec / r
                            quad_term = 0
                            for alpha in range(3):
                                for beta in range(3):
                                    delta_ab = 1 if alpha == beta else 0
                                    quad_term += Q[alpha, beta] * (3 * r_hat[alpha] * r_hat[beta] - delta_ab)
                            esp += 0.5 * quad_term / (r**3)
                        
                        # Add octupole contribution if available
                        if octupoles is not None:
                            # Convert with proper units
                            conversion = BOHR_TO_ANGSTROM**3  # Convert e·bohr³ → e·Å³
                            O = octupoles[i] * conversion
                            
                            # Correct octupole formula
                            r_hat = r_vec / r
                            octu_term = 0
                            for alpha in range(3):
                                for beta in range(3):
                                    for gamma in range(3):
                                        main_term = 5 * r_hat[alpha] * r_hat[beta] * r_hat[gamma]
                                        
                                        delta_bg = 1 if beta == gamma else 0
                                        delta_ag = 1 if alpha == gamma else 0
                                        delta_ab = 1 if alpha == beta else 0
                                        
                                        correction = r_hat[alpha] * delta_bg + r_hat[beta] * delta_ag + r_hat[gamma] * delta_ab
                                        
                                        octu_term += O[alpha, beta, gamma] * (main_term - correction)
                            
                            esp += octu_term / (6 * r**4)
                    
                    esp_grid[iz, iy, ix] = esp
                    # print(f"*** esp_grid[{iz}, {iy}, {ix}] {esp}")

        return esp_grid.flatten()










    def calculate_esp_grid_optimized(self, positions, charges, dipoles=None, quadrupoles=None,
                          octupoles=None, bounds=None, spacing=0.3):
        """Calculate ESP grid from multipole data - optimized version"""
        
        # Use the bounds that already include margin
        bounds_min, bounds_max = bounds
        
        # Create grid with specified spacing
        nx = max(1, int(np.ceil((bounds_max[0] - bounds_min[0]) / spacing)))
        ny = max(1, int(np.ceil((bounds_max[1] - bounds_min[1]) / spacing)))
        nz = max(1, int(np.ceil((bounds_max[2] - bounds_min[2]) / spacing)))
        
        # Convert input data to more efficient formats
        positions_array = np.array(positions, dtype=np.float32)
        charges_array = np.array([c[0] for c in charges], dtype=np.float32)
        
        # Precompute unit conversions
        BOHR_TO_ANGSTROM = 0.52917721092
        
        # Prepare dipoles in optimized format
        dipoles_array = None
        if dipoles is not None:
            dipoles_array = np.array(dipoles, dtype=np.float32) * BOHR_TO_ANGSTROM
        
        # Prepare quadrupoles in optimized format
        quad_tensors = None
        if quadrupoles is not None:
            conversion = BOHR_TO_ANGSTROM**2
            quad_tensors = np.zeros((len(quadrupoles), 3, 3), dtype=np.float32)
            for i, q in enumerate(quadrupoles):
                quad_tensors[i, 0, 0] = q[0, 0] * conversion
                quad_tensors[i, 1, 1] = q[1, 1] * conversion
                quad_tensors[i, 2, 2] = q[2, 2] * conversion
                quad_tensors[i, 0, 1] = quad_tensors[i, 1, 0] = q[0, 1] * conversion
                quad_tensors[i, 0, 2] = quad_tensors[i, 2, 0] = q[0, 2] * conversion
                quad_tensors[i, 1, 2] = quad_tensors[i, 2, 1] = q[1, 2] * conversion
        
        # Prepare octupoles in optimized format
        octu_tensors = None
        if octupoles is not None:
            conversion = BOHR_TO_ANGSTROM**3
            octu_tensors = np.array(octupoles, dtype=np.float32) * conversion
        
        # Create grid points
        grid_points = np.zeros((nx * ny * nz, 3), dtype=np.float32)
        idx = 0
        for iz in range(nz):
            z = bounds_min[2] + (iz + 0.5) * spacing
            for iy in range(ny):
                y = bounds_min[1] + (iy + 0.5) * spacing
                for ix in range(nx):
                    x = bounds_min[0] + (ix + 0.5) * spacing
                    grid_points[idx] = [x, y, z]
                    idx += 1
        
        # Calculate ESP values using numba-optimized function
        esp_values = compute_esp_numba(grid_points, positions_array, charges_array, dipoles_array, quad_tensors, octu_tensors)
        return esp_values.reshape(nz, ny, nx).flatten()

