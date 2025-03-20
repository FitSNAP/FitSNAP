from fitsnap3lib.scrapers.scrape import Scraper
import numpy as np
import h5py, json, sys

from pprint import pprint

# ----------------------------------------------------------------

def atomic_number_to_symbol(n):
    return {1:"H", 6:"C", 7:"N", 8:"O", 9:"F", 11:"Na", 12:"Mg",
            15:"P", 16:"S", 17:"Cl", 19:"K", 20:"Ca"}[n]

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
        self.allowed_symbols = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F",
                                11:"Na", 12:"Mg", 15:"P", 16:"S",
                                17:"Cl", 19:"K", 20:"Ca"}

    # ----------------------------------------------------------------

    def scrape_groups(self):
        if self.pt.stubs == 1:
            with h5py.File(self.filename, "r") as f:
                self.all_group_names = list(f.keys())
        else:
            with h5py.File(self.filename, "r", driver="mpio", comm=self.comm) as f:
                groups = list(f.keys()) if self.rank == 0 else None
                self.all_group_names = self.comm.bcast(groups, root=0)

        #pprint(self.all_group_names)

    # ----------------------------------------------------------------

    def divvy_up_configs(self):
        if self.pt.stubs == 1:
            self.local_group_names = self.all_group_names
            return
        with h5py.File(self.filename, "r", driver="mpio", comm=self.comm) as f:
            local_group_names = [g for i, g in enumerate(self.all_group_names) if i % self.size == self.rank]
            local_groups_info = []
            for g in local_group_names:
                atomic_numbers = f[g]["atomic_numbers"][()]
                if np.any(np.isin(atomic_numbers, self.exclude_atomic_numbers)):
                    continue
                M = f[g]["conformations"].shape[0]
                N = atomic_numbers.shape[0]
                local_groups_info.append((g, M * N))
            all_groups_info = [item for sublist in self.comm.allgather(local_groups_info) for item in sublist]
            all_groups_info.sort(key=lambda x: x[1], reverse=True)
            assignments = {i: [] for i in range(self.size)}
            totals = {i: 0 for i in range(self.size)}
            for group_name, MN in all_groups_info:
                min_rank = min(totals, key=totals.get)
                assignments[min_rank].append(group_name)
                totals[min_rank] += MN
            self.local_group_names = assignments[self.rank]

        pprint(self.local_group_names)

    # ----------------------------------------------------------------

    def scrape_configs(self):
        if self.pt.stubs == 1:
            with h5py.File(self.filename, "r") as f:
                for g in self.local_group_names:
                    self._scrape_group(f[g])
        else:
            with h5py.File(self.filename, "r", driver="mpio", comm=self.comm) as f:
                for g in self.local_group_names:
                    self._scrape_group(f[g])

        print(f"|self.data| {len(self.data)}")
        return self.data
        
    # ----------------------------------------------------------------

    def _scrape_group(self, group):
        BOHR_TO_ANGSTROM = 0.52917721092
        HARTREE_TO_KCAL_MOL = 627.509474
        FORCE_CONV = HARTREE_TO_KCAL_MOL / BOHR_TO_ANGSTROM
        conformations = group["conformations"][()]
        formation_energy = group["formation_energy"][()]
        dft_total_gradient = group["dft_total_gradient"][()]
        mbis_charges = group["mbis_charges"][()]
        scf_dipoles = group["scf_dipole"][()]
        atomic_numbers = group["atomic_numbers"][()]
        for i in range(conformations.shape[0]):
            self.data.append({
                "Positions": conformations[i] * BOHR_TO_ANGSTROM,
                "Energy": formation_energy[i] * HARTREE_TO_KCAL_MOL,
                "Forces": dft_total_gradient[i] * FORCE_CONV,
                "Dipole": scf_dipoles[i] * BOHR_TO_ANGSTROM,
                "AtomTypes": [atomic_number_to_symbol(n) for n in atomic_numbers],
            })

# ----------------------------------------------------------------

def custom_json_dumps(o, indent=0, indent_step=2):
    spaces = " " * indent
    if isinstance(o, dict):
        items = []
        for k, v in o.items():
            items.append(f'{spaces}{" " * indent_step}"{k}": {custom_json_dumps(v, indent + indent_step, indent_step)}')
        return '{\n' + ',\n'.join(items) + '\n' + spaces + '}'
    elif isinstance(o, list):
        if all(isinstance(item, (int, float, bool, str)) or item is None for item in o):
            return json.dumps(o)
        else:
            items = [custom_json_dumps(item, indent + indent_step, indent_step) for item in o]
            return '[\n' + ',\n'.join(items) + '\n' + spaces + ']'
    elif isinstance(o, np.ndarray):
        return custom_json_dumps(o.tolist(), indent, indent_step)
    elif isinstance(o, bytes):
        return json.dumps(o.decode('utf-8'))
    else:
        return json.dumps(o)

# ----------------------------------------------------------------

import h5py
import numpy as np



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 hdf5_scraper.py <path_to_hdf5_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    class DummyPT:
        def __init__(self, stubs=0):
            self.stubs = stubs
            if stubs==0:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                self._rank = comm.Get_rank()
                self._size = comm.Get_size()
            else:
                self._rank = 0
                self._size = 1
        def get_rank(self):
            return self._rank
        def get_size(self):
            return self._size
        def single_print(self, *args):
            print(*args)
    class DummyConfig:
        sections = {"PATH":type("obj",(object,),{"datapath":""}),
                    "SCRAPER":type("obj",(object,),{"properties":{}}),
                    "CALCULATOR":type("obj",(object,),{"per_atom_scalar":False}),
                    "REFERENCE":type("obj",(object,),{"units":"real"})}
        default_conversions = {"Energy":1.0}
    pt = DummyPT(stubs=0)
    config = DummyConfig()
    scraper = HDF5("HDF5Scraper", pt, config, file_path)
    scraper.scrape_groups()
    scraper.divvy_up_configs()
    scraper.scrape_configs()
    print(f"rank {scraper.rank}")
    print(custom_json_dumps(scraper.data, indent=0, indent_step=2))
    with open(f"data_{scraper.rank}.json", "w") as f:
        f.write(custom_json_dumps(scraper.data, indent=0, indent_step=2))
