from fitsnap3lib.scrapers.scrape import Scraper
import numpy as np
import h5py

# ----------------------------------------------------------------

def atomic_number_to_symbol(n):
    return {
        1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
        11: "Na", 12: "Mg", 15: "P", 16: "S",
        17: "Cl", 19: "K", 20: "Ca"
    }[n]

# ----------------------------------------------------------------

class HDF5(Scraper):

    # ----------------------------------------------------------------

    def __init__(self, name, pt, config, filename="/Users/mitch/Dropbox/github/hmx/spice/spice_test.hd5"):
        super().__init__(name, pt, config)
        self.data = []
        self.filename = filename
        if self.pt.stubs == 0:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = pt.get_rank()
            self.size = pt.get_size()
        else:
            self.rank = 0
            self.size = 1
        self.exclude_atomic_numbers = [3, 35, 53]
        self._lattice_margin = 10
        self.allowed_symbols = {
            1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
            11: "Na", 12: "Mg", 15: "P", 16: "S",
            17: "Cl", 19: "K", 20: "Ca"
        }

    # ----------------------------------------------------------------

    def scrape_groups(self):
        if self.pt.stubs == 1:
            with h5py.File(self.filename, "r") as f:
                self.all_group_names = list(f.keys())
        else:
            with h5py.File(self.filename, "r", driver="mpio", comm=self.comm) as f:
                groups = list(f.keys()) if self.rank == 0 else None
                self.all_group_names = self.comm.bcast(groups, root=0)

    # ----------------------------------------------------------------

    def divvy_up_configs(self):
        local_configs = []

        with h5py.File(self.filename, "r", driver="mpio", comm=self.comm) as f:
            for i in range(self.rank, len(self.all_group_names), self.size):
                g = self.all_group_names[i]
                atomic_numbers = f[g]["atomic_numbers"][()]
                if not np.any(np.isin(atomic_numbers, self.exclude_atomic_numbers)):
                    M = f[g]["conformations"].shape[0]
                    for j in range(M):
                        local_configs.append((g, j))

        all_configs = self.comm.allgather(local_configs)
        flat_configs = [cfg for sub in all_configs for cfg in sub]

        total = len(flat_configs)
        count = total // self.size
        remainder = total % self.size
        start = self.rank * count + min(self.rank, remainder)
        stop = start + count + (1 if self.rank < remainder else 0)

        self.my_configs = flat_configs[start:stop]

    # ----------------------------------------------------------------

    def scrape_configs(self):
        if self.pt.stubs == 1:
            with h5py.File(self.filename, "r") as f:
                for g, i in self.my_configs:
                    self._scrape_group(f[g], i)
        else:
            with h5py.File(self.filename, "r", driver="mpio", comm=self.comm) as f:
                for g, i in self.my_configs:
                    self._scrape_group(f[g], i)
        return self.data

    # ----------------------------------------------------------------

    def _scrape_group(self, group, i):
        BOHR_TO_ANGSTROM = 0.52917721092
        HARTREE_TO_KCAL_MOL = 627.509474
        FORCE_CONV = HARTREE_TO_KCAL_MOL / BOHR_TO_ANGSTROM

        positions = group["conformations"][i] * BOHR_TO_ANGSTROM
        min_bounds = positions.min(axis=0) - self._lattice_margin
        max_bounds = positions.max(axis=0) + self._lattice_margin
        ax, bx, cx = max_bounds[0] - min_bounds[0], 0.0, 0.0
        ay, by, cy = 0.0, max_bounds[1] - min_bounds[1], 0.0
        az, bz, cz = 0.0, 0.0, max_bounds[2] - min_bounds[2]

        atomic_numbers = group["atomic_numbers"][()]
        formation_energy = group["formation_energy"][()]
        dft_total_gradient = group["dft_total_gradient"][()]
        mbis_charges = group["mbis_charges"][()]
        scf_dipoles = group["scf_dipole"][()]

        self.data.append({
            "Group": group.name,
            "File": f"{group.name}/{i}",
            "Positions": positions,
            "Energy": formation_energy[i] * HARTREE_TO_KCAL_MOL,
            "Forces": dft_total_gradient[i] * FORCE_CONV,
            "Charges": mbis_charges[i].squeeze().tolist(),
            "Dipole": scf_dipoles[i] * BOHR_TO_ANGSTROM,
            "AtomTypes": [atomic_number_to_symbol(n) for n in atomic_numbers],
            "NumAtoms": len(atomic_numbers),
            "Lattice": [[ax, bx, cx], [ay, by, cy], [az, bz, cz]],
            "eweight": 1.0,
            "fweight": 50.0,
            "vweight": 0.0,
            "test_bool": np.random.rand() < 0.2
        })

    # ----------------------------------------------------------------
