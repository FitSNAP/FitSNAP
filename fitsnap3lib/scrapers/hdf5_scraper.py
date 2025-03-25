from fitsnap3lib.scrapers.scrape import Scraper
import numpy as np
import h5py
import logging
from os import path
from scipy.special import logsumexp

# ------------------------------------------------------------------------------------------------

def atomic_number_to_symbol(n):
    return {1:"H",6:"C",7:"N",8:"O",9:"F",11:"Na",12:"Mg",15:"P",16:"S",17:"Cl",19:"K",20:"Ca"}[n]

def atomic_symbol_to_number(s):
    return {"H":1,"C":6,"N":7,"O":8,"F":9,"Na":11,"Mg":12,"P":15,"S":16,"Cl":17,"K":19,"Ca":20}[s]

# ------------------------------------------------------------------------------------------------

class HDF5(Scraper):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self.data = []
        logging.getLogger("h5py._conv").setLevel(logging.WARNING)
        self._lattice_margin = 10
        allowed_elements = self.config.sections["REAXFF"].elements
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

    def scrape_groups(self):
        self.group_metadata = {}
        self.local_configs = []

        file_kwargs = {"driver": "mpio", "comm": self.comm} if self.pt.stubs == 0 else {}

        with h5py.File(self.hdf5_path, "r", **file_kwargs) as f:
            if self.pt.stubs == 1:
                all_group_names = list(f.keys())
            else:
                groups = list(f.keys())[:3] if self.rank == 0 else None
                all_group_names = self.comm.bcast(groups, root=0)

            for i in range(self.rank, len(all_group_names), self.size):
                group_name = all_group_names[i]
                group = f[group_name]
                atomic_numbers = group["atomic_numbers"][()]

                if not np.all(np.isin(atomic_numbers, list(self.allowed_atomic_numbers))):
                    continue

                #if atomic_symbol_to_number('S') not in atomic_numbers:
                #    continue

                conformations = group["conformations"][()]
                formation_energy = group["formation_energy"][()]
                subset = group["subset"].asstr()[0]

                is_solvated = "solvated" in subset.lower() or "water" in subset.lower()
                cweight_val = 60.0 if is_solvated else 0.0
                dweight_val = 20.0 if is_solvated else 0.0

                bounds_min = conformations.min(axis=(0, 1)) - 10.0
                bounds_max = conformations.max(axis=(0, 1)) + 10.0
                region = f"region box block {bounds_min[0]} {bounds_max[0]} {bounds_min[1]} {bounds_max[1]} {bounds_min[2]} {bounds_max[2]}"
                lattice = [
                    [bounds_max[0] - bounds_min[0], 0.0, 0.0],
                    [0.0, bounds_max[1] - bounds_min[1], 0.0],
                    [0.0, 0.0, bounds_max[2] - bounds_min[2]],
                ]

                min_energy = formation_energy.min()
                delta_E = formation_energy.max() - min_energy
                kbT = max(delta_E / 5, 0.5)  # adaptive smoothing to avoid flat weights
                exponents = -(formation_energy - min_energy) / kbT
                log_weights = exponents - logsumexp(exponents)
                weights = np.exp(log_weights)
                # Optional: zero out very small weights and re-normalize
                # weights[weights < 1e-8] = 0.0
                weights /= weights.sum()

                self.group_metadata[group_name] = {
                    "subset": subset,
                    "is_solvated": is_solvated,
                    "cweight": cweight_val,
                    "dweight": dweight_val,
                    "region": region,
                    "lattice": lattice,
                    "weights": weights.tolist()
                }

                #print(f"*** {group_name} formation_energy {formation_energy} weights {weights} {[atomic_number_to_symbol(n) for n in atomic_numbers]}")
                #print(f"*** {group_name} {[atomic_number_to_symbol(n) for n in atomic_numbers]}")

                for j in range(conformations.shape[0]):
                    self.local_configs.append((group_name, j))

        all_meta = self.comm.allgather(self.group_metadata)
        self.group_metadata = {k: v for d in all_meta for k, v in d.items()}

    def divvy_up_configs(self):
        all_configs = self.comm.allgather(self.local_configs)
        flat_configs = [cfg for sub in all_configs for cfg in sub]
        flat_configs.sort()

        total = len(flat_configs)
        base = total // self.size
        remainder = total % self.size

        start = self.rank * base + min(self.rank, remainder)
        stop = start + base + (1 if self.rank < remainder else 0)

        self.my_configs = flat_configs[start:stop]

        expected = base + (1 if self.rank < remainder else 0)
        actual = len(self.my_configs)
        if actual != expected:
            raise RuntimeError(f"[Rank {self.rank}] Expected {expected} configs, got {actual}")

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

        with h5py.File(self.hdf5_path, "r", **file_kwargs) as f:
            for group_name in sorted(grouped):
                group = f[group_name]
                meta = self.group_metadata[group_name]
                conformations = group["conformations"][()] * BOHR_TO_ANGSTROM
                formation_energy = group["formation_energy"][()] * HARTREE_TO_KCAL_MOL
                dft_total_gradient = group["dft_total_gradient"][()] * FORCE_CONV
                mbis_charges = group["mbis_charges"][()]
                scf_dipoles = group["scf_dipole"][()] * BOHR_TO_ANGSTROM
                atomic_numbers = group["atomic_numbers"][()]
                for i in grouped[group_name]:
                    self.data.append({
                        "Group": group.name,
                        "File": f"{group.name}/{i}",
                        "Subset": meta["subset"],
                        "Positions": conformations[i],
                        "Energy": formation_energy[i],
                        "Forces": dft_total_gradient[i],
                        "Charges": mbis_charges[i],
                        "Dipole": scf_dipoles[i],
                        "AtomTypes": [atomic_number_to_symbol(n) for n in atomic_numbers],
                        "NumAtoms": len(atomic_numbers),
                        "Lattice": meta["lattice"],
                        "Region": meta["region"],
                        "eweight": float(meta["weights"][i] * 100),
                        "fweight": 50.0,
                        "vweight": 0.0,
                        "cweight": meta["cweight"],
                        "dweight": meta["dweight"],
                        "test_bool": np.random.rand() < 0.2
                    })

        return self.data

    # --------------------------------------------------------------------------------------------
